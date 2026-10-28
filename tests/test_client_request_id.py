"""OpenAI attempt correlation uses only an already-durable anonymous ledger ID."""
import json
import re
import tempfile
import unittest
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply, NativeHTTPS, OPENAI_ENDPOINT, ProviderFailure
from agent_memory.providers import GPT4oMiniCards, TextEmbeddingV4
from agent_memory.joint_selection import SelectionServiceTransport, selection_payload
from agent_memory.prepaid_online import PrepaidOnlineTransport
from agent_memory.protocol import content_parts
from test_chat_stream import events, wire
import test_service_transport as fixtures
import test_bailian_rerank as rerank_fixtures

EMBED = 'https://synthetic.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings'
SECRET = 'SYNTHETIC_SECRET_NOT_A_REAL_KEY'
PATTERN = re.compile(r'attempt_[0-9a-f]{32}\Z')


class RecordingHTTP:
    is_fake_http = True

    def __init__(self, ledger, *, failure=False):
        self.ledger, self.failure = ledger, failure
        self.ids, self.models, self.persisted_before_dispatch = [], [], []

    def exchange(self, endpoint, body, headers, timeout, limit):
        payload = json.loads(body)
        request_id = headers.get('X-Client-Request-Id')
        self.ids.append(request_id)
        self.models.append(payload['model'])
        if request_id is not None:
            with self.ledger.connect() as db:
                row = db.execute('SELECT model,status FROM attempts WHERE id=?', (request_id,)).fetchone()
            self.persisted_before_dispatch.append(row)
        if self.failure:
            raise TimeoutError('synthetic transport failure')
        if payload['model'] == 'text-embedding-v4':
            value = dict(model='text-embedding-v4', data=[dict(index=0, embedding=[.1]*64)], usage=dict(total_tokens=3))
        else:
            value = dict(model='gpt-4o-mini-2024-07-18', usage=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30),
                         choices=[dict(index=0,finish_reason='stop',message=dict(content=json.dumps(dict(summary='synthetic',evidence_quotes=[]))))])
        return HTTPReply(200, json.dumps(value).encode())


class ClientRequestIDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = BudgetLedger(Path(self.tmp.name)/'budget.sqlite')

    def client(self, http, model, *, observer=None):
        endpoint = OPENAI_ENDPOINT if model == 'gpt-4o-mini' else EMBED
        rate = VerifiedRate(model, '1.5', '6', 'synthetic', datetime.now(timezone.utc).isoformat())
        return BudgetedJSONTransport(model=model, endpoint=endpoint, ledger=self.ledger, rate=rate,
            http=http, mode='offline_fake', fixed_reserve_cny='.25' if model=='gpt-4o-mini' else None,
            attempt_observer=observer)

    def cards(self, client):
        return GPT4oMiniCards(client).extract([dict(type='text',text='untrusted synthetic source')])

    def test_ids_are_anonymous_unique_and_already_bound_before_post(self):
        http = RecordingHTTP(self.ledger)
        observed = []
        client = self.client(http, 'gpt-4o-mini', observer=observed.append)
        self.cards(client); self.cards(client)
        self.assertEqual(len(http.ids), 2)
        self.assertNotEqual(http.ids[0], http.ids[1])
        self.assertEqual(http.ids, observed)
        self.assertTrue(all(PATTERN.fullmatch(value) for value in http.ids))
        self.assertEqual(http.persisted_before_dispatch, [('gpt-4o-mini','dispatched')]*2)
        self.assertTrue(all('synthetic' not in value.lower() and SECRET not in value for value in http.ids))

    def test_embedding_receives_no_openai_specific_header(self):
        http = RecordingHTTP(self.ledger)
        TextEmbeddingV4(EMBED, self.client(http, 'text-embedding-v4'), 64).embed(['synthetic'])
        self.assertEqual(http.ids, [None])

    def test_hosted_reranker_receives_no_openai_specific_header(self):
        fixture=rerank_fixtures.HostedTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        headers_seen=[];original_exchange=fixture.http.exchange
        def capture(endpoint,body,headers,timeout,limit):
            headers_seen.append(set(headers))
            return original_exchange(endpoint,body,headers,timeout,limit)
        fixture.http.exchange=capture
        fixture.call()
        self.assertEqual(len(headers_seen),1)
        self.assertNotIn('X-Client-Request-Id',headers_seen[0])

    def test_native_http1_passes_header_to_the_single_post(self):
        captured=[]
        class Sock:
            def settimeout(self,value):pass
        class Response:
            status=200
            def read1(self,n):return b''
            def getheader(self,*args):return ''
        class Connection:
            def __init__(self,*args,**kwargs):self.sock=Sock()
            def connect(self):pass
            def request(self,method,path,body=None,headers=None):captured.append((method,path,headers))
            def getresponse(self):return Response()
            def close(self):pass
        identifier='attempt_'+'a'*32
        with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
            NativeHTTPS().exchange(OPENAI_ENDPOINT,b'{}',{'X-Client-Request-Id':identifier},1,1024)
        self.assertEqual(len(captured),1)
        self.assertEqual(captured[0],('POST','/v1/chat/completions',{'X-Client-Request-Id':identifier}))

    def test_parallel_calls_do_not_share_or_reuse_header_values(self):
        http = RecordingHTTP(self.ledger)
        client = self.client(http, 'gpt-4o-mini')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: self.cards(client), range(4)))
        self.assertEqual(len(http.ids), 4)
        self.assertEqual(len(set(http.ids)), 4)
        self.assertTrue(all(PATTERN.fullmatch(value) for value in http.ids))
        self.assertEqual(http.persisted_before_dispatch, [('gpt-4o-mini','dispatched')]*4)

    def test_observer_failure_never_dispatches_and_preserves_hold(self):
        http = RecordingHTTP(self.ledger)
        def fail(_):raise RuntimeError('synthetic private error')
        with self.assertRaises(ProviderFailure) as error:
            self.cards(self.client(http, 'gpt-4o-mini', observer=fail))
        self.assertEqual(error.exception.category, 'network_error')
        self.assertNotIn('synthetic private error', str(error.exception))
        self.assertEqual(http.ids, [])
        with self.ledger.connect() as db:
            row = db.execute('SELECT status,reserved,accounted FROM attempts').fetchone()
        self.assertEqual(row, ('uncertain',250000,250000))

    def test_timeout_keeps_correlation_and_no_retry_or_hold_release(self):
        http = RecordingHTTP(self.ledger, failure=True)
        with self.assertRaises(ProviderFailure) as error:
            self.cards(self.client(http, 'gpt-4o-mini'))
        self.assertEqual(error.exception.category, 'timeout')
        self.assertEqual(http.ids, [error.exception.attempt_id])
        with self.ledger.connect() as db:
            row=db.execute('SELECT status,reserved,accounted FROM attempts WHERE id=?', (http.ids[0],)).fetchone()
        self.assertEqual(row, ('uncertain',250000,250000))

    def test_invalid_ledger_identifier_never_reaches_network(self):
        injected = 'x\r\nInjected: y'
        self.assertIn('\r\n', injected)
        for bad in ('G'*32, injected, '安'*32, 'a'*513):
            with self.subTest(kind=len(bad)):
                ledger = BudgetLedger(Path(self.tmp.name)/('bad-'+str(len(bad))+str(ord(bad[0]))+'.sqlite'))
                self.ledger = ledger
                http = RecordingHTTP(ledger)
                client = self.client(http, 'gpt-4o-mini')
                with patch('agent_memory.budget.uuid.uuid4',return_value=SimpleNamespace(hex=bad)):
                    with self.assertRaises(ProviderFailure) as error:
                        self.cards(client)
                self.assertEqual(error.exception.category,'invalid_contract')
                self.assertEqual(http.ids, [])
                with ledger.connect() as db:row=db.execute('SELECT status,reserved,accounted FROM attempts').fetchone()
                self.assertEqual(row,('uncertain',250000,250000))


class ServiceCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ServiceTransportTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.payload = selection_payload(content_parts('synthetic query'), [dict(
            content=[dict(type='text',text='synthetic source')],session_group=0,
            created_at='2020-01-01T00:00:00Z')])

    def test_receipt_and_original_budget_are_bound_before_post_and_timeout_stays_blocked(self):
        f=self.fixture
        class HTTP:
            is_fake_http=True
            def __init__(self):self.calls=0;self.ids=[]
            def exchange(inner, endpoint, body, headers, timeout, limit):
                inner.calls+=1;inner.ids.append(headers.get('X-Client-Request-Id'))
                with transport.journal.connect() as db:
                    row=db.execute('select state,attempt_id from operations').fetchone()
                with f.ledger.connect() as db:
                    budget=db.execute('select status,reserved,accounted from attempts where id=?',(inner.ids[-1],)).fetchone()
                self.assertEqual(tuple(row),('inflight',inner.ids[-1]))
                self.assertEqual(budget,('dispatched',250000,250000))
                raise TimeoutError('synthetic timeout')
        http=HTTP()
        transport=SelectionServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,
            f.p/'receipts',{'gpt-4o-mini':lambda:SECRET},http=http,offline_test=True)
        with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,self.payload)
        with transport.journal.connect() as db:row=db.execute('select state,attempt_id from operations').fetchone()
        self.assertEqual(tuple(row),('unknown',http.ids[0]))
        self.assertTrue(PATTERN.fullmatch(http.ids[0]))
        with self.assertRaises(ValueError):transport(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(http.calls,1)
        with f.ledger.connect() as db:
            self.assertEqual(db.execute('select reserved,accounted from attempts').fetchone(),(250000,250000))

    def test_success_has_bound_id_and_does_not_persist_prompt_or_secret(self):
        f=self.fixture
        class HTTP:
            is_fake_http=True
            def __init__(self):self.ids=[]
            def exchange(inner, endpoint, body, headers, timeout, limit):
                request_id=headers['X-Client-Request-Id'];inner.ids.append(request_id)
                with transport.journal.connect() as db:row=db.execute('select state,attempt_id from operations').fetchone()
                self.assertEqual(tuple(row),('inflight',request_id))
                return HTTPReply(200,wire(events('{"selected":[0]}')),
                                 {'content-type':'text/event-stream'})
        http=HTTP()
        transport=SelectionServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,
            f.p/'receipts',{'gpt-4o-mini':lambda:SECRET},http=http,offline_test=True)
        transport(OPENAI_ENDPOINT,self.payload)
        transport.ready()
        with transport.journal.connect() as db:row=db.execute('select state,attempt_id,body,manifest from operations').fetchone()
        self.assertEqual(row['state'],'complete');self.assertEqual(row['attempt_id'],http.ids[0]);self.assertIsNone(row['body'])
        self.assertNotIn('synthetic query',row['manifest']);self.assertNotIn(SECRET,row['manifest'])
        self.assertNotIn(SECRET.encode(),(f.p/'budget.sqlite').read_bytes())

    def test_journal_attachment_failure_stops_before_network(self):
        f=self.fixture
        class HTTP:
            is_fake_http=True
            calls=0
            def exchange(inner,*args):inner.calls+=1;raise AssertionError('network must not run')
        http=HTTP()
        transport=SelectionServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,
            f.p/'receipts',{'gpt-4o-mini':lambda:SECRET},http=http,offline_test=True)
        with patch.object(transport.journal,'attach',side_effect=RuntimeError('synthetic attachment failure')):
            with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(http.calls,0)
        with transport.journal.connect() as db:row=db.execute('select state,attempt_id from operations').fetchone()
        self.assertEqual(tuple(row),('unknown',None))
        with f.ledger.connect() as db:row=db.execute('select reserved,accounted,status from attempts').fetchone()
        self.assertEqual(row,(250000,250000,'uncertain'))
        with self.assertRaises(ValueError):transport.ready()

    def test_mismatched_journal_binding_stops_before_network(self):
        f=self.fixture
        class HTTP:
            is_fake_http=True
            calls=0
            def exchange(inner,*args):inner.calls+=1;raise AssertionError('network must not run')
        http=HTTP()
        transport=SelectionServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,
            f.p/'receipts',{'gpt-4o-mini':lambda:SECRET},http=http,offline_test=True)
        original_attach=transport.journal.attach
        with patch.object(transport.journal,'attach',
                side_effect=lambda fingerprint,attempt:original_attach(fingerprint,'attempt_'+'f'*32)):
            with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(http.calls,0)
        with transport.journal.connect() as db:row=db.execute('select state from operations').fetchone()
        self.assertEqual(row['state'],'unknown')
        with f.ledger.connect() as db:row=db.execute('select reserved,accounted,status from attempts').fetchone()
        self.assertEqual(row,(250000,250000,'uncertain'))
        with self.assertRaises(ValueError):transport.ready()


class PrepaidCompatibilityTests(unittest.TestCase):
    def make_transport(self,root,*,fail=False):
        ledger=BudgetLedger(root/'synthetic-parent.sqlite')
        now=datetime.now(timezone.utc)
        rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',now.isoformat())
        parent_id=ledger.reserve(rate,128000,512,fixed_reserve_cny='.25')
        grant=dict(version=1,grant_id='a'*32,price_checked_at=now.isoformat(),
            expires_at=(now+timedelta(hours=2)).isoformat(),slots=[dict(ordinal=1,
            model='gpt-4o-mini',parent_attempt_id=parent_id,reserved_micro_cny=250000)])
        data=json.dumps(grant).encode();path=root/'grant.json';path.write_bytes(data)
        class HTTP:
            is_fake_http=True
            def __init__(inner):inner.ids=[]
            def exchange(inner,endpoint,body,headers,timeout,limit):
                inner.ids.append(headers.get('X-Client-Request-Id'))
                with transport.journal.connect() as db:row=db.execute('select state,attempt_id from operations').fetchone()
                self.assertEqual(tuple(row),('inflight',parent_id))
                if fail:raise ProviderFailure('timeout')
                response=dict(model='gpt-4o-mini-2024-07-18',
                    usage=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30),
                    choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(summary='synthetic',evidence_quotes=[]))))])
                return HTTPReply(200,json.dumps(response).encode())
        http=HTTP()
        transport=PrepaidOnlineTransport(path,hashlib.sha256(data).hexdigest(),root/'receipts',
            {'gpt-4o-mini':lambda:SECRET},http=http,offline_test=True)
        return ledger,transport,http,parent_id

    def test_prepaid_gpt_preserves_bound_parent_id_without_new_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger,transport,http,parent_id=self.make_transport(Path(temp))
            before=ledger.totals()
            GPT4oMiniCards(transport).extract([dict(type='text',text='synthetic source')])
            self.assertEqual(http.ids,[parent_id])
            self.assertEqual(ledger.totals(),before)
            self.assertEqual(transport.summary()[0]['state'],'complete')

    def test_prepaid_timeout_preserves_parent_and_prevents_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger,transport,http,parent_id=self.make_transport(Path(temp),fail=True)
            before=ledger.totals()
            cards=GPT4oMiniCards(transport)
            with self.assertRaises(ProviderFailure):cards.extract([dict(type='text',text='synthetic source')])
            self.assertEqual(http.ids,[parent_id])
            self.assertEqual(transport.summary()[0]['state'],'unknown')
            with self.assertRaises(ValueError):cards.extract([dict(type='text',text='synthetic source')])
            self.assertEqual(http.ids,[parent_id])
            self.assertEqual(ledger.totals(),before)


if __name__ == '__main__':unittest.main()
