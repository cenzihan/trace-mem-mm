import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_memory.protocol import ContractError
from agent_memory.providers import GPT4oMiniCards,TextEmbeddingV4,ProviderBlocked,CandidateReranker
from agent_memory.store import MemoryStore
from agent_memory.system import AgentMemorySystem,RetrievalConfig


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'db.sqlite'
        self.store=AgentMemorySystem(self.path)

    def tearDown(self): self.tmp.cleanup()

    def payload(self,text,req='r1',user='u',session='shared'):
        return dict(request_id=req,user_id=user,session_id=session,messages=[dict(role='user',content=text)])

    def query(self,text,user='u',k=100): return dict(query=text,user_id=user,top_k=k)

    def test_current_and_historical_out_of_order(self):
        new=self.payload('FACT entity=Mira attribute=drink value=coffee event=2026-01-20T00:00:00Z','new',session='s2')
        old=self.payload('FACT entity=Mira attribute=drink value=tea event=2026-01-10T00:00:00Z','old',session='s1')
        self.store.add(new);self.store.add(old)
        self.assertIn('value=coffee',self.store.search(self.query('Mira current drink'))['data'][0]['content'][0]['text'])
        self.assertIn('value=tea',self.store.search(self.query('Mira drink as of 2026-01-15'))['data'][0]['content'][0]['text'])
        self.assertEqual(self.store.inspect_user('u')['facts'],2)

    def test_atomic_reject_bad_event_and_adapter(self):
        self.assertRaises(ContractError,self.store.add,self.payload('FACT entity=Mira attribute=drink value=tea event=2026-99-99T00:00:00Z'))
        self.assertEqual(self.store.inspect_user('u')['memories'],0)
        class BadCards:
            model='gpt-4o-mini'
            def extract(self,p): return {'summary':'x','evidence_quotes':['invented']}
        bad=AgentMemorySystem(self.path,card_adapter=BadCards())
        self.assertRaises(ValueError,bad.add,self.payload('source text'))
        self.assertEqual(self.store.inspect_user('u')['memories'],0)

    def test_namespace_links_never_cross_users(self):
        self.store.add(self.payload('LINK entity=Mira related=box42'))
        self.store.add(self.payload('box42 location loft','r2',session='s2'))
        self.store.add(self.payload('box42 location PRIVATE_FOREIGN','r1','other'))
        result,trace=self.store.retrieve(self.query('Mira location'))
        self.assertTrue(trace['linked_candidates'])
        self.assertNotIn('PRIVATE_FOREIGN',json.dumps(result))
        self.assertEqual(self.store.search(self.query('box42','unknown')),{'data':[]})

    def test_restart_concurrent_idempotent_and_purge(self):
        payload=self.payload('source text')
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda _:self.store.add(payload),range(4)))
        self.assertEqual(AgentMemorySystem(self.path).inspect_user('u')['cards'],1)
        self.store.purge_user('u')
        self.assertEqual(self.store.inspect_user('u'),dict(memories=0,cards=0,facts=0,entity_links=0))

    def test_failed_index_transaction_rolls_back_raw(self):
        with self.store.connect() as db:
            db.execute("CREATE TRIGGER reject_card BEFORE INSERT ON cards BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
        self.assertRaises(Exception,self.store.add,self.payload('text'))
        self.assertEqual(self.store.inspect_user('u')['memories'],0)
        with self.store.connect() as db: self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0)

    def test_replay_repairs_legacy_source(self):
        payload=self.payload('legacy record')
        MemoryStore(self.path).add(payload)
        self.assertRaises(ContractError,self.store.search,self.query('legacy'))
        self.store.add(payload)
        self.assertEqual(len(self.store.search(self.query('legacy'))['data']),1)

    def test_dense_is_fail_closed_no_mock_substitution(self):
        self.assertRaises(ValueError,AgentMemorySystem,self.path,RetrievalConfig(dense=True))
        self.assertRaises(ProviderBlocked,TextEmbeddingV4().embed,['text'])
        self.assertRaises(ProviderBlocked,GPT4oMiniCards().extract,[{'type':'text','text':'x'}])
        self.assertRaises(ProviderBlocked,CandidateReranker)

    def test_embedding_adapter_batch_order_dimensions_model(self):
        calls=[]
        def fake(endpoint,body):
            calls.append(body)
            return {'model':'text-embedding-v4','data':[{'index':i,'embedding':[float(i)]*64} for i in reversed(range(len(body['input'])))]}
        adapter=TextEmbeddingV4('https://synthetic.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings',fake,64)
        result=adapter.embed(['synthetic text']*11)
        self.assertEqual([len(c['input']) for c in calls],[10,1])
        self.assertTrue(all(c['model']=='text-embedding-v4' for c in calls))
        self.assertEqual(result[1],[1.0]*64)

    def test_dense_injected_test_vectors_are_labelled_only_unit_test(self):
        class UnitVectors:
            model='text-embedding-v4'
            def embed(self,texts):return [[1.0,0.0] for _ in texts]
        store=AgentMemorySystem(self.path,RetrievalConfig(dense=True),embedding=UnitVectors())
        store.add(self.payload('unit source'))
        self.assertTrue(store.retrieve(self.query('unit'))[1]['dense_enabled'])

    def test_search_does_not_persist_questions_or_change_index(self):
        self.store.add(self.payload('ordinary source'))
        before=self.path.read_bytes()
        self.store.search(self.query('PRIVATE_QUERY_NOT_A_MEMORY'))
        self.assertEqual(before,self.path.read_bytes())
