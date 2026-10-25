"""Failure review regressions: fake HTTP, synthetic sources and temporary databases only."""
import copy
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.maintenance import backup_database,integrity,repair_missing_indexes
from agent_memory.protocol import ContractError,media_size
from agent_memory.provider_http import BudgetedJSONTransport,OPENAI_ENDPOINT,ProviderFailure
from agent_memory.providers import GPT4oMiniCards,TextEmbeddingV4
from agent_memory.store import MemoryStore
from agent_memory.system import AgentMemorySystem,RetrievalConfig
from test_provider_http import FakeHTTP,reply,embedding_reply,card_reply,EMBED


class FailureInjectionTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def payload(self,text='source',request='r',user='u',session='same'):
        return dict(request_id=request,user_id=user,session_id=session,messages=[dict(role='user',content=text)])
    def query(self,text='source',user='u',k=100):return dict(query=text,user_id=user,top_k=k)
    def client(self,http,model='gpt-4o-mini',name='budget.sqlite',input_rate='1.5'):
        ledger=BudgetLedger(self.root/name)
        transport=BudgetedJSONTransport(model=model,endpoint=OPENAI_ENDPOINT if model=='gpt-4o-mini' else EMBED,
            ledger=ledger,rate=VerifiedRate(model,input_rate,'6','synthetic',datetime.now(timezone.utc).isoformat()),
            mode='offline_fake',http=http)
        return transport,ledger
    def assert_status(self,status,fn,*args):
        with self.assertRaises(ContractError) as caught:fn(*args)
        self.assertEqual(caught.exception.status,status)

    def test_metadata_snapshot_with_concurrent_fact_commit(self):
        path=self.root/'snapshot.sqlite';searcher=AgentMemorySystem(path);writer=AgentMemorySystem(path)
        searcher.add(self.payload('FACT entity=mira attribute=drink value=tea event=2026-01-01T00:00:00Z'))
        # WAL is confined to this temporary test DB, allowing deterministic interleaving.
        with searcher.connect() as db:db.execute('PRAGMA journal_mode=WAL')
        original=searcher.connect;fired=[]
        @contextmanager
        def injected():
            with original() as db:
                class Proxy:
                    def execute(inner,sql,*args):
                        if sql.startswith('SELECT card_id,entity,attribute') and not fired:
                            fired.append(True)
                            writer.add(self.payload('FACT entity=mira attribute=drink value=coffee event=2026-02-01T00:00:00Z','new'))
                        return db.execute(sql,*args)
                yield Proxy()
        searcher.connect=injected
        found=searcher.search(self.query('mira current drink'))
        self.assertEqual(len(found['data']),1);self.assertIn('value=tea',found['data'][0]['content'][0]['text'])
        self.assertEqual(writer.inspect_user('u')['memories'],2)

    def test_lazy_source_replacement_is_rejected(self):
        path=self.root/'source.sqlite';searcher=AgentMemorySystem(path);writer=AgentMemorySystem(path)
        searcher.add(self.payload('original source'));original=searcher.connect;fired=[]
        @contextmanager
        def injected():
            with original() as db:
                class Proxy:
                    def execute(inner,sql,*args):
                        if sql.startswith('SELECT m.content,m.created_at') and not fired:
                            fired.append(True);writer.purge_user('u');writer.add(self.payload('replacement source'))
                        return db.execute(sql,*args)
                yield Proxy()
        searcher.connect=injected
        self.assert_status(503,searcher.search,self.query('original'))
        self.assertIn('replacement',writer.search(self.query('replacement'))['data'][0]['content'][0]['text'])

    def test_image_occurrences_and_request_keys_are_user_scoped(self):
        store=AgentMemorySystem(self.root/'images.sqlite')
        content=json.loads((Path(__file__).resolve().parent.parent/'fixtures/add.json').read_text())['messages'][0]['content']
        repeated=copy.deepcopy(content)+[copy.deepcopy(content[1])]
        store.add(self.payload(repeated,'r','u','first'))
        store.add(self.payload(content,'r2','u','second'))
        store.add(self.payload(content,'r','v','first'))
        u=store.search(self.query([content[1]],'u'));v=store.search(self.query([content[1]],'v'))
        self.assertEqual(len(u['data']),2);self.assertEqual(len(v['data']),1)
        self.assertFalse({r['id'] for r in u['data']}&{r['id'] for r in v['data']})
        with store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*),count(DISTINCT sha256),count(DISTINCT memory_id) FROM evidence').fetchone(),(4,1,3))
        u['data'][0]['content'][0]['text']='mutated response'
        self.assertNotIn('mutated response',json.dumps(store.search(self.query([content[1]],'u'))))
        self.assertEqual(store.search(self.query([content[1]],'unknown')),{'data':[]})

    def test_pending_owner_conflicts_and_other_user_can_progress(self):
        entered=threading.Event();release=threading.Event()
        def slow(_):entered.set();self.assertTrue(release.wait(3));return reply(card_reply())
        http=FakeHTTP([slow,reply(card_reply())]);transport,ledger=self.client(http)
        path=self.root/'claims.sqlite';first=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
        second=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport));source=self.payload()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(first.add,source)
            try:
                self.assertTrue(entered.wait(3))
                self.assert_status(503,second.add,source)
                self.assert_status(409,second.add,self.payload('different'))
                self.assertTrue(second.add(self.payload(user='v'))['success'])
            finally:release.set()
            self.assertTrue(future.result(3)['success'])
        self.assertEqual(len(http.requests),2);self.assertEqual(ledger.totals()['attempts'],2)
        self.assertEqual(first.inspect_user('u')['memories'],1);self.assertEqual(first.inspect_user('v')['memories'],1)

    def test_unknown_timeout_and_5xx_survive_restart_without_retry(self):
        for index,step in enumerate((TimeoutError('SYNTHETIC_ERROR_SECRET'),reply({},503))):
            with self.subTest(index=index):
                http=FakeHTTP([step,reply(card_reply())]);transport,ledger=self.client(http,name='b'+str(index)+'.sqlite')
                path=self.root/('unknown'+str(index)+'.sqlite');source=self.payload()
                system=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
                self.assertRaises(ProviderFailure,system.add,source)
                held=ledger.totals()['accounted_micro_cny'];self.assertGreater(held,0)
                restarted=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
                self.assert_status(503,restarted.add,source)
                self.assertEqual(len(http.requests),1);self.assertEqual(ledger.totals()['accounted_micro_cny'],held)
                self.assertEqual(restarted.inspect_user('u')['memories'],0)

    def test_pending_cannot_bypass_models_and_survives_backup_repair(self):
        path=self.root/'pending.sqlite';http=FakeHTTP([reply({},503)]);transport,_=self.client(http)
        system=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport));source=self.payload()
        self.assertRaises(ProviderFailure,system.add,source)
        self.assert_status(503,MemoryStore(path).add,source)
        self.assert_status(503,AgentMemorySystem(path).add,source)
        self.assert_status(409,AgentMemorySystem(path).add,self.payload('different'))
        self.assertRaises(ValueError,repair_missing_indexes,path)
        backup=self.root/'restored.sqlite';backup_database(path,backup)
        self.assert_status(503,AgentMemorySystem(backup).add,source)
        self.assertTrue(integrity(backup)['integrity_ok'])

    def test_interrupted_preparing_is_never_expired_or_retried(self):
        def interrupt(_):raise KeyboardInterrupt()
        http=FakeHTTP([interrupt]);transport,ledger=self.client(http)
        path=self.root/'interrupt.sqlite';system=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
        self.assertRaises(KeyboardInterrupt,system.add,self.payload())
        with system.connect() as db:db.execute("UPDATE add_operations SET state='preparing',updated_at='2000-01-01T00:00:00Z'")
        self.assert_status(503,AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport)).add,self.payload())
        self.assertEqual(len(http.requests),1);self.assertGreater(ledger.totals()['accounted_micro_cny'],0)

    def test_lost_ack_after_commit_replay_reads_durable_success(self):
        http=FakeHTTP([reply(card_reply())]);transport,_=self.client(http)
        path=self.root/'lost-ack.sqlite';system=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
        original=system.connect;committed=[];injected=[]
        @contextmanager
        def lost_ack():
            with original() as db:
                class Proxy:
                    def execute(inner,sql,*args):
                        if sql.startswith("UPDATE add_operations SET state='complete'"):committed.append(True)
                        return db.execute(sql,*args)
                yield Proxy()
            if committed and not injected:injected.append(True);raise OSError('synthetic lost acknowledgement')
        system.connect=lost_ack
        self.assertRaises(OSError,system.add,self.payload())
        restarted=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
        self.assertTrue(restarted.add(self.payload())['success']);self.assertEqual(len(http.requests),1)
        with restarted.connect() as db:self.assertEqual(db.execute('SELECT state FROM add_operations').fetchone()[0],'complete')

    def test_purge_revokes_inflight_owner_and_cleans_only_one_scope(self):
        path=self.root/'purge.sqlite';holder={}
        def purge(_):holder['system'].purge_user('u');return reply(card_reply())
        http=FakeHTTP([purge]);transport,_=self.client(http)
        system=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport));holder['system']=system
        AgentMemorySystem(path).add(self.payload(user='v'))
        self.assert_status(503,system.add,self.payload())
        self.assertEqual(system.inspect_user('u')['memories'],0);self.assertEqual(system.inspect_user('v')['memories'],1)
        with system.connect() as db:self.assertEqual(db.execute("SELECT count(*) FROM add_operations WHERE user_id='u'").fetchone()[0],0)

    def test_huge_usage_freezes_all_entry_points_and_survives_reopen(self):
        data=embedding_reply(1);data['usage']['total_tokens']=2**63
        http=FakeHTTP([reply(data)]);transport,ledger=self.client(http,'text-embedding-v4')
        self.assertRaises(ProviderFailure,TextEmbeddingV4(EMBED,transport,64).embed,['source'])
        with ledger.connect() as db:
            self.assertEqual(db.execute('SELECT status,input_tokens FROM attempts').fetchone(),('needs_reconciliation',None))
        self.assertRaises(ValueError,BudgetLedger(ledger.path).reserve,transport.rate,1)
        self.assertEqual(len(http.requests),1)
        for method in ('settle','record_usage_keep_hold'):
            other=BudgetLedger(self.root/(method+'.sqlite'));attempt=other.reserve(transport.rate,1);other.mark_dispatched(attempt)
            self.assertRaises(ValueError,getattr(other,method),attempt,2**63,0)
            self.assertRaises(ValueError,other.reserve,transport.rate,1)

    def test_concurrent_unknown_calls_respect_total_500_cap(self):
        http=FakeHTTP([TimeoutError('synthetic timeout')]*12)
        transport,ledger=self.client(http,'text-embedding-v4',input_rate='24414.0625')
        def attempt(_):
            try:TextEmbeddingV4(EMBED,transport,64).embed(['source'])
            except ProviderFailure:return 'dispatched'
            except ValueError:return 'blocked'
        with ThreadPoolExecutor(max_workers=8) as pool:outcomes=list(pool.map(attempt,range(12)))
        self.assertEqual(outcomes.count('dispatched'),2);self.assertEqual(len(http.requests),2)
        self.assertEqual(ledger.totals()['accounted_micro_cny'],400_000_000)
        self.assertEqual(ledger.totals()['remaining_micro_cny'],100_000_000)

    def test_dense_migration_and_concurrent_scope_change_do_not_ack_success(self):
        path=self.root/'dense.sqlite';local=AgentMemorySystem(path);local.add(self.payload())
        class UnitEmbedding:
            model='text-embedding-v4'
            def embed(inner,texts):return [[1.0,0.0] for _ in texts]
        dense=AgentMemorySystem(path,RetrievalConfig(dense=True),embedding=UnitEmbedding())
        self.assert_status(503,dense.add,self.payload())
        self.assert_status(503,dense.add,self.payload('new','new'))
        second=self.root/'mixed.sqlite';writer=AgentMemorySystem(second)
        class RacingEmbedding(UnitEmbedding):
            def embed(inner,texts):writer.add(self.payload('local-only','foreign-mode'));return super().embed(texts)
        mixed=AgentMemorySystem(second,RetrievalConfig(dense=True),embedding=RacingEmbedding())
        self.assert_status(503,mixed.add,self.payload())
        self.assertEqual(mixed.inspect_user('u')['memories'],1)

    def test_media_prefix_preserves_whole_original_parts(self):
        system=AgentMemorySystem(self.root/'prefix.sqlite')
        content=json.loads((Path(__file__).resolve().parent.parent/'fixtures/add.json').read_text())['messages'][0]['content']
        for i in range(3):system.add(self.payload(content,str(i)))
        full=system.search(self.query('north gate'))['data']
        limit=sum(media_size(r['content']) for r in full[:2])
        with patch('agent_memory.system.MEDIA_LIMIT',limit):limited=system.search(self.query('north gate'))['data']
        self.assertEqual([r['id'] for r in limited],[r['id'] for r in full[:2]])
        self.assertTrue(all(r['content']==content for r in limited))

    def test_failure_logs_and_journal_contain_no_source_or_token(self):
        http=FakeHTTP([TimeoutError('SYNTHETIC_SECRET_SENTINEL')]);transport,ledger=self.client(http)
        path=self.root/'logs.sqlite';system=AgentMemorySystem(path,card_adapter=GPT4oMiniCards(transport))
        output=io.StringIO()
        with redirect_stdout(output),redirect_stderr(output):
            with self.assertRaises(ProviderFailure) as caught:system.add(self.payload('PRIVATE_SOURCE_SENTINEL'))
        self.assertNotIn('SYNTHETIC_SECRET_SENTINEL',output.getvalue()+str(caught.exception))
        self.assertNotIn('PRIVATE_SOURCE_SENTINEL',output.getvalue()+str(caught.exception))
        self.assertNotIn(b'PRIVATE_SOURCE_SENTINEL',path.read_bytes())
        self.assertNotIn(b'PRIVATE_SOURCE_SENTINEL',Path(ledger.path).read_bytes())
