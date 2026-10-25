"""Explicit fake HTTP bytes only; no real credential, socket, provider or GPU use."""
import contextlib
import copy
import io
import json
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.provider_http import (BudgetedJSONTransport,HTTPReply,NativeHTTPS,OPENAI_ENDPOINT,
                                        ProviderFailure,RetryPolicy,validate_endpoint)
from agent_memory.providers import GPT4oMiniCards,TextEmbeddingV4,ProviderBlocked
from agent_memory.protocol import ContractError
from agent_memory.provider_runtime import build_provider_system
from agent_memory.system import AgentMemorySystem,RetrievalConfig

EMBED='https://synthetic.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings'
SYNTHETIC_SECRET='SYNTHETIC_KEY_NEVER_LOG_THIS'


def embedding_reply(count,dimension=64):
    return dict(model='text-embedding-v4',data=[dict(index=i,embedding=[float(i+1)]*dimension) for i in reversed(range(count))],
                usage=dict(total_tokens=3*count))


def card_reply(summary='synthetic summary',quotes=None):
    return dict(model='gpt-4o-mini-2024-07-18',usage=dict(prompt_tokens=40,completion_tokens=10,total_tokens=50),
                choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content=json.dumps(dict(summary=summary,evidence_quotes=quotes or []))))])


def reply(data,status=200,headers=None):return HTTPReply(status,json.dumps(data).encode(),headers or {})


class FakeHTTP:
    is_fake_http=True
    def __init__(self,steps=None):self.steps=list(steps or []);self.requests=[]
    def exchange(self,endpoint,body,headers,timeout,max_response_bytes):
        payload=json.loads(body);self.requests.append(dict(endpoint=endpoint,payload=payload,timeout=timeout,limit=max_response_bytes))
        assert headers['Authorization']=='Bearer '+SYNTHETIC_SECRET
        step=self.steps.pop(0) if self.steps else reply(embedding_reply(len(payload['input'])))
        if isinstance(step,Exception):raise step
        return step(payload) if callable(step) else step


class ProviderHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.ledger=BudgetLedger(self.root/'budget.sqlite')
    def tearDown(self):self.tmp.cleanup()
    def client(self,http,model='text-embedding-v4',**options):
        rate=VerifiedRate(model,'1.5','6','synthetic price; NOT provider quotation',datetime.now(timezone.utc).isoformat())
        return BudgetedJSONTransport(model=model,endpoint=EMBED if model=='text-embedding-v4' else OPENAI_ENDPOINT,
                    ledger=self.ledger,rate=rate,credential_reader=lambda:SYNTHETIC_SECRET,http=http,mode='offline_fake',**options)
    def observations(self):
        with self.ledger.connect() as db:
            db.row_factory=sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM provider_observations ORDER BY observed_at,attempt_id')]
    def embed(self,client,texts=None):return TextEmbeddingV4(EMBED,client,64).embed(texts or ['source text'])

    def test_embedding_real_wire_schema_batch_order_and_reopen_usage(self):
        http=FakeHTTP();client=self.client(http);vectors=self.embed(client,['text']*21)
        self.assertEqual([len(r['payload']['input']) for r in http.requests],[10,10,1])
        self.assertEqual(vectors[1],[2.0]*64)
        self.assertTrue(all(r['timeout']==30 and r['payload']['model']=='text-embedding-v4' for r in http.requests))
        self.assertEqual(len(self.observations()),3)
        self.assertEqual(sum(r['input_tokens'] for r in self.observations()),63)
        self.assertTrue(all(r['mode']=='offline_fake' for r in self.observations()))
        self.assertEqual(BudgetLedger(self.root/'budget.sqlite').totals()['attempts'],3)

    def test_gpt_multimodal_order_caption_and_usage_usd(self):
        parts=json.loads((Path(__file__).resolve().parent.parent/'fixtures/add.json').read_text())['messages'][0]['content']
        http=FakeHTTP([reply(card_reply()),reply(card_reply('image summary'))]);client=self.client(http,'gpt-4o-mini',usd_rates=('0.15','0.60'))
        adapter=GPT4oMiniCards(client)
        adapter.extract(parts);self.assertEqual(adapter.caption(parts),'image summary')
        body=http.requests[0]['payload']
        self.assertEqual(body['messages'][1]['content'],parts)
        self.assertFalse(body['store']);self.assertEqual(body['max_completion_tokens'],512)
        self.assertEqual(self.observations()[0]['estimated_usd'],'0.000012')
        self.assertEqual(self.observations()[0]['input_tokens'],40)

    def test_card_batch_is_ordered_bounded_synchronous(self):
        http=FakeHTTP([reply(card_reply('one')),reply(card_reply('two'))]);adapter=GPT4oMiniCards(self.client(http,'gpt-4o-mini'))
        result=adapter.extract_many([[dict(type='text',text='first')],[dict(type='text',text='second')]])
        self.assertEqual([v['summary'] for v in result],['one','two'])
        self.assertRaises(ValueError,adapter.extract_many,[[dict(type='text',text='x')]]*17)
        self.assertEqual(len(http.requests),2)

    def test_live_default_blocks_before_credential_or_dispatch(self):
        # A mocked root constraint is only for constructing the disabled live client in isolation.
        rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat());reads=[]
        with patch('agent_memory.provider_http.SHARED_LEDGER',Path(self.ledger.path)):
            client=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=self.ledger,rate=rate,credential_reader=lambda:reads.append(1))
        self.assertRaises(ProviderBlocked,GPT4oMiniCards(client).extract,[dict(type='text',text='x')])
        self.assertFalse(reads);self.assertEqual(self.ledger.totals()['attempts'],0)

    def test_fake_never_reads_credentials_or_uses_real_shared_ledger(self):
        client=self.client(FakeHTTP())
        def forbidden_reader():raise AssertionError('must never read credentials for fake HTTP')
        client.credential_reader=forbidden_reader
        self.embed(client)
        with patch('agent_memory.provider_http.SHARED_LEDGER',Path(self.ledger.path)):
            self.assertRaises(ValueError,self.client,FakeHTTP())

    def test_unconfirmed_live_prices_block_before_reader_and_http(self):
        rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat());reads=[]
        with patch('agent_memory.provider_http.SHARED_LEDGER',Path(self.ledger.path)):
            client=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=self.ledger,rate=rate,
                                         enabled=True,prices_confirmed=False,credential_reader=lambda:reads.append(1))
        self.assertRaises(ProviderBlocked,GPT4oMiniCards(client).extract,[dict(type='text',text='x')])
        self.assertFalse(reads);self.assertEqual(self.ledger.totals()['attempts'],0)

    def test_preflight_rejects_model_endpoint_input_limits_without_charge(self):
        http=FakeHTTP();client=self.client(http)
        body=dict(model='text-embedding-v4',input=['x'],dimensions=64,encoding_format='float')
        bad=copy.deepcopy(body);bad['model']='qwen-embedding'
        self.assertRaises(ValueError,client,EMBED,bad)
        bad=copy.deepcopy(body);bad['input']=['x'*8193]
        self.assertRaises(ValueError,client,EMBED,bad)
        bad=copy.deepcopy(body);bad['input']=['x']*11
        self.assertRaises(ValueError,client,EMBED,bad)
        for endpoint in ('https://evil.example/v1/embeddings',EMBED+'?key=secret',EMBED.replace('https:','http:')):
            self.assertRaises(ValueError,validate_endpoint,'text-embedding-v4',endpoint)
        self.assertEqual(self.ledger.totals()['attempts'],0);self.assertFalse(http.requests)

    def test_gpt_preflight_text_and_image_limits(self):
        http=FakeHTTP();adapter=GPT4oMiniCards(self.client(http,'gpt-4o-mini'))
        self.assertRaises(ValueError,adapter.extract,[dict(type='text',text='x'*32769)])
        part=json.loads((Path(__file__).resolve().parent.parent/'fixtures/add.json').read_text())['messages'][0]['content'][1]
        self.assertRaises(ValueError,adapter.extract,[part]*9)
        self.assertEqual(self.ledger.totals()['attempts'],0)

    def test_429_default_no_retry_retains_hold(self):
        http=FakeHTTP([reply({'error':SYNTHETIC_SECRET},429)]);client=self.client(http)
        with self.assertRaises(ProviderFailure) as caught:self.embed(client)
        self.assertEqual(caught.exception.category,'rate_limited');self.assertEqual(len(http.requests),1)
        self.assertGreater(self.ledger.totals()['accounted_micro_cny'],0)
        self.assertIsNone(self.observations()[0]['input_tokens'])

    def test_explicit_429_retry_new_reservation_and_delay_cap(self):
        http=FakeHTTP([reply({},429,{'retry-after':'999999'}),reply(embedding_reply(1))]);delays=[]
        self.embed(self.client(http,retry=RetryPolicy(2,True,2),sleep=delays.append))
        self.assertEqual(delays,[2]);self.assertEqual(len(http.requests),2)
        with self.ledger.connect() as db:rows=db.execute('SELECT id,retry_of,status FROM attempts ORDER BY created_at').fetchall()
        self.assertIsNone(rows[0][1]);self.assertEqual(rows[1][1],rows[0][0]);self.assertEqual(rows[0][2],'uncertain')

    def test_retry_cannot_overrun_budget(self):
        self.ledger=BudgetLedger(self.root/'tiny.sqlite',cap_cny='.02')
        http=FakeHTTP([reply({},429)]);client=self.client(http,retry=RetryPolicy(2,True,0),sleep=lambda _:None)
        self.assertRaises(ValueError,self.embed,client)
        self.assertEqual(len(http.requests),1);self.assertEqual(self.ledger.totals()['attempts'],1)

    def test_concurrent_fake_http_dispatch_cannot_overrun_reserved_cap(self):
        self.ledger=BudgetLedger(self.root/'tiny.sqlite',cap_cny='.02')
        http=FakeHTTP([TimeoutError('synthetic timeout')]);client=self.client(http)
        def attempt(_):
            try:self.embed(client);return 'unexpected success'
            except ProviderFailure:return 'uncertain'
            except ValueError:return 'budget blocked'
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(attempt,range(4)))
        self.assertEqual(results.count('uncertain'),1);self.assertEqual(results.count('budget blocked'),3)
        self.assertEqual(len(http.requests),1);self.assertEqual(self.ledger.totals()['attempts'],1)

    def test_5xx_timeout_network_and_permissions_do_not_retry(self):
        for step,category in [(reply({},503),'server_error'),(TimeoutError(SYNTHETIC_SECRET),'timeout'),
                              (OSError(SYNTHETIC_SECRET),'network_error'),(reply({},401),'authentication'),
                              (reply({},403),'permission_or_region'),(reply({},302,{'location':'https://evil.example'}),'redirect')]:
            with self.subTest(category=category):
                http=FakeHTTP([step]);client=self.client(http,retry=RetryPolicy(2,True,0))
                with self.assertRaises(ProviderFailure) as caught:self.embed(client)
                self.assertEqual(caught.exception.category,category);self.assertEqual(len(http.requests),1)
                self.assertNotIn(SYNTHETIC_SECRET,str(caught.exception))

    def test_connectivity_check_cannot_enable_retry(self):
        self.assertRaises(ValueError,self.client,FakeHTTP(),retry=RetryPolicy(2,True,0),purpose='connectivity_check')
        self.assertRaises(ValueError,self.client,FakeHTTP(),retry=RetryPolicy(3,True,0))

    def test_invalid_model_freezes_ledger_and_prevents_substitution(self):
        data=embedding_reply(1);data['model']='another-model';http=FakeHTTP([reply(data)])
        self.assertRaises(ProviderFailure,self.embed,self.client(http))
        self.assertEqual(self.observations()[0]['category'],'model_mismatch')
        self.assertIsNone(self.observations()[0]['returned_model'])
        self.assertRaises(ValueError,self.embed,self.client(http));self.assertEqual(len(http.requests),1)

    def test_usage_over_bound_freezes_future_attempts(self):
        data=embedding_reply(1);data['usage']['total_tokens']=8193;http=FakeHTTP([reply(data)])
        self.assertRaises(ProviderFailure,self.embed,self.client(http))
        self.assertEqual(self.observations()[0]['input_tokens'],8193)
        self.assertRaises(ValueError,self.embed,self.client(http));self.assertEqual(len(http.requests),1)

    def test_bad_usage_json_and_oversize_keep_reservations(self):
        bad=embedding_reply(1);bad.pop('usage')
        for index,response in enumerate((reply(bad),HTTPReply(200,b'{secret invalid'),HTTPReply(200,b'x'*1025))):
            self.ledger=BudgetLedger(self.root/('isolated-invalid-'+str(index)+'.sqlite'))
            http=FakeHTTP([response]);self.assertRaises(ProviderFailure,self.embed,self.client(http,max_response_bytes=1024))
        self.assertTrue(all(r['input_tokens'] is None for r in self.observations()))

    def test_response_content_failure_accounts_usage_before_rejecting(self):
        data=embedding_reply(1);data['data'][0]['embedding']=[1.0]*63
        http=FakeHTTP([reply(data)]);self.assertRaises(ProviderFailure,self.embed,self.client(http))
        observation=self.observations()[0];self.assertEqual(observation['category'],'invalid_contract');self.assertEqual(observation['input_tokens'],3)

    def test_atomic_source_and_index_when_second_embedding_batch_fails(self):
        http=FakeHTTP([reply(embedding_reply(10)),reply({},503)]);embedding=TextEmbeddingV4(EMBED,self.client(http),64)
        system=AgentMemorySystem(self.root/'memory.sqlite',RetrievalConfig(dense=True),embedding=embedding)
        payload=dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content='source '+str(i)) for i in range(11)])
        self.assertRaises(ProviderFailure,system.add,payload)
        self.assertEqual(system.inspect_user('u'),dict(memories=0,cards=0,facts=0,entity_links=0))
        self.assertEqual(self.ledger.totals()['attempts'],2)
        self.assertEqual([r['category'] for r in self.observations()],['success','server_error'])

    def test_full_fake_card_dense_add_search_and_restart(self):
        http=FakeHTTP([reply(card_reply('red square at north gate',quotes=['red square'])),
                       reply(embedding_reply(1)),reply(embedding_reply(1))])
        cards=GPT4oMiniCards(self.client(http,'gpt-4o-mini'))
        embedding=TextEmbeddingV4(EMBED,self.client(http),64)
        path=self.root/'memory.sqlite'
        system=AgentMemorySystem(path,RetrievalConfig(dense=True),embedding=embedding,card_adapter=cards)
        content=json.loads((Path(__file__).resolve().parent.parent/'fixtures/add.json').read_text())['messages'][0]['content']
        payload=dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content=content)])
        self.assertTrue(system.add(payload)['success'])
        self.assertTrue(system.add(payload)['success']);self.assertEqual(len(http.requests),2)
        found=system.search(dict(query='north gate',user_id='u',top_k=100))
        self.assertEqual(found['data'][0]['content'],content)
        restarted=AgentMemorySystem(path,RetrievalConfig(dense=True),embedding=embedding,card_adapter=cards)
        self.assertEqual(restarted.inspect_user('u')['cards'],1)
        with self.ledger.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM provider_observations').fetchone()[0],3)

    def test_factory_composes_and_labels_fake_without_network_or_key_lookup(self):
        settings=dict(embedding_endpoint=EMBED,dense=True,cards=True,dimensions=64)
        rates={model:VerifiedRate(model,'1.5','6','synthetic',datetime.now(timezone.utc).isoformat()) for model in ('text-embedding-v4','gpt-4o-mini')}
        system=build_provider_system(self.root/'factory.sqlite',settings=settings,
            credential_readers={m:lambda:SYNTHETIC_SECRET for m in rates},rates=rates,http=FakeHTTP(),mode='offline_fake',ledger=self.ledger)
        self.assertEqual(system.capabilities()['mode'],'offline_fake_providers')
        self.assertEqual(self.ledger.totals()['attempts'],0)
        self.assertRaises(ProviderBlocked,build_provider_system,self.root/'blocked.sqlite',settings=settings,credential_readers={},rates={},http=FakeHTTP(),mode='offline_fake',ledger=self.ledger)
        bad=dict(settings,api_key='not allowed here')
        self.assertRaises(ValueError,build_provider_system,self.root/'bad.sqlite',settings=bad,credential_readers={},rates={})

    def test_model_add_capacity_and_late_invalid_input_do_not_dispatch(self):
        http=FakeHTTP();cards=GPT4oMiniCards(self.client(http,'gpt-4o-mini'))
        system=AgentMemorySystem(self.root/'memory.sqlite',card_adapter=cards)
        payload=dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content='text')]*17)
        self.assertRaises(ContractError,system.add,payload)
        adapter=TextEmbeddingV4(EMBED,self.client(http),64)
        self.assertRaises(ValueError,adapter.embed,['text']*10+['x'*8193])
        self.assertEqual(self.ledger.totals()['attempts'],0);self.assertFalse(http.requests)

    def test_grounding_rejection_keeps_paid_card_usage(self):
        http=FakeHTTP([reply(card_reply(quotes=['invented quote']))])
        system=AgentMemorySystem(self.root/'memory.sqlite',card_adapter=GPT4oMiniCards(self.client(http,'gpt-4o-mini')))
        payload=dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content='source only')])
        self.assertRaises(ValueError,system.add,payload)
        self.assertEqual(system.inspect_user('u')['memories'],0);self.assertEqual(self.observations()[0]['input_tokens'],40)

    def test_gpt_truncation_refusal_and_nonjson_reject_without_retry(self):
        a=card_reply();a['choices'][0]['finish_reason']='length'
        b=card_reply();b['choices'][0]['message']['refusal']='not processed'
        c=card_reply();c['choices'][0]['message']['content']='not JSON'
        for data in (a,b,c):
            http=FakeHTTP([reply(data)]);adapter=GPT4oMiniCards(self.client(http,'gpt-4o-mini'))
            self.assertRaises(ProviderFailure,adapter.extract,[dict(type='text',text='source')])
            self.assertEqual(len(http.requests),1)

    def test_no_body_token_or_error_logs_or_ledger_payload(self):
        output=io.StringIO();http=FakeHTTP([reply({'error':SYNTHETIC_SECRET+' PRIVATE_SOURCE'},500)])
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            with self.assertRaises(ProviderFailure) as caught:self.embed(self.client(http),['PRIVATE_SOURCE'])
        self.assertNotIn(SYNTHETIC_SECRET,output.getvalue()+repr(caught.exception))
        self.assertNotIn('PRIVATE_SOURCE',output.getvalue()+repr(caught.exception))
        raw=(self.root/'budget.sqlite').read_bytes()
        self.assertNotIn(SYNTHETIC_SECRET.encode(),raw);self.assertNotIn(b'PRIVATE_SOURCE',raw)

    def test_native_https_uses_tls_no_redirect_and_closes(self):
        class Sock:
            def settimeout(self,value):pass
        class Response:
            status=302
            def getheader(self,*args):return ''
            def read1(self,n):return b''
        class Connection:
            instances=[]
            def __init__(self,host,**kwargs):self.host=host;self.kwargs=kwargs;self.sock=Sock();self.closed=False;self.calls=[];self.instances.append(self)
            def connect(self):pass
            def request(self,*args,**kwargs):self.calls.append(args)
            def getresponse(self):return Response()
            def close(self):self.closed=True
        with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
            result=NativeHTTPS().exchange(OPENAI_ENDPOINT,b'{}',{},1,1024)
        self.assertEqual(result.status,302);self.assertEqual(len(Connection.instances),1)
        conn=Connection.instances[0];self.assertEqual(conn.host,'api.openai.com');self.assertEqual(conn.calls,[('POST','/v1/chat/completions')]);self.assertTrue(conn.closed)
        self.assertEqual(conn.kwargs['context'].verify_mode,2)

    def test_native_deadline_prevents_late_post(self):
        calls=[]
        class Connection:
            def __init__(self,*args,**kwargs):pass
            def connect(self):time.sleep(.08)
            def request(self,*args,**kwargs):calls.append(1)
            def close(self):pass
        with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
            with self.assertRaises(ProviderFailure) as caught:NativeHTTPS().exchange(OPENAI_ENDPOINT,b'{}',{},.01,1024)
            self.assertEqual(caught.exception.category,'timeout');time.sleep(.1)
        self.assertFalse(calls)

    def test_duplicate_observation_and_future_schema_fail_closed(self):
        self.embed(self.client(FakeHTTP()));observed=self.observations()[0]
        self.assertRaises(ValueError,self.ledger.observe_provider,observed['attempt_id'],mode='offline_fake',category='success')
        with self.ledger.connect() as db:db.execute('PRAGMA user_version=99')
        self.assertRaises(ValueError,BudgetLedger,self.root/'budget.sqlite')
