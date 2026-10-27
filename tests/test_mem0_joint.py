"""Independent synthetic evidence, actual Mem0 SDK, no network or official inputs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.joint_selection import (selection_payload, selected_indices, strict_json,
    SelectionServiceTransport, SelectionBudgetedTransport, candidate_count)
from agent_memory.mem0_joint import Mem0JointSystem, VERSION
from agent_memory.protocol import ContractError, content_parts, text_of
from agent_memory.provider_http import HTTPReply, OPENAI_ENDPOINT, ProviderFailure
from test_mem0_native import FakeEmbedding, FakeCards, FakeTransport
from test_mem0_evidence import image
from test_chat_stream import wire, events
import test_service_transport as fixtures


def source(text='synthetic source', picture=False):
    return dict(content=content_parts(text)+([image()] if picture else []),
                created_at='2020-01-01T00:00:00Z', session_group=0)


class SelectionContractTests(unittest.TestCase):
    def test_explicit_activation_and_original_scope(self):
        from agent_memory.mem0_app import create_joint_app
        with self.assertRaises(ValueError):create_joint_app({})
        self.assertEqual(SelectionServiceTransport.receipt_scope, 'native-mem0-raw-v4')

    def test_bounded_unicode_views_do_not_mutate_originals(self):
        original = source('独立证据'*2000, True); before = json.dumps(original)
        payload = selection_payload(content_parts('查询'), [original])
        header = strict_json(payload['messages'][1]['content'][0]['text'])
        self.assertLessEqual(len(header['candidates'][0]['text_prefix'].encode()), 1500)
        self.assertTrue(header['candidates'][0]['text_truncated'])
        self.assertEqual(json.dumps(original), before)

    def test_total_images_includes_query_and_missing_views_are_disclosed(self):
        payload = selection_payload([image()], [source(str(i), True) for i in range(32)])
        content = payload['messages'][1]['content']
        self.assertEqual(sum(p['type']=='image_url' for p in content), 8)
        rows = strict_json(content[0]['text'])['candidates']
        self.assertEqual(sum(r['image_preview_attached'] for r in rows), 7)
        self.assertTrue(all(r['has_image'] for r in rows))
        self.assertLess(len(json.dumps(payload).encode()), 3*1024*1024)

    def test_oversized_query_and_pool_rejected(self):
        with self.assertRaises(ContractError):selection_payload(content_parts('x'*4001), [source()])
        for values in ([], [source()]*33):
            with self.assertRaises(ValueError):selection_payload(content_parts('query'), values)

    def test_unknown_duplicate_bool_and_negative_indices_rejected(self):
        for values in ([2],[-1],[True],[0,0],[1.0],['0'],list(range(9))):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):selected_indices({'selected':values},2)

    def test_strict_object_no_answers_extra_fields_or_duplicate_keys(self):
        for raw in ('{"selected":[],"answer":"invented"}', '{"selected":[],"selected":[0]}',
                    '{"selected":null}', '{"selected":[NaN]}', '[]'):
            with self.assertRaises(ValueError):selected_indices(strict_json(raw),1)
        self.assertEqual(selected_indices(strict_json('{"selected":[]}'),1), [])

    def test_bounds_require_fixed_prompt_and_candidate_indices(self):
        payload=selection_payload(content_parts('query'), [source()])
        self.assertEqual(candidate_count(payload), 1)
        payload['messages'][0]['content']='changed'
        with self.assertRaises(ValueError):candidate_count(payload)

    def test_model_identity_is_exact(self):
        transport=object.__new__(SelectionBudgetedTransport);transport.model='gpt-4o-mini'
        self.assertEqual(transport._response_model({'model':'gpt-4o-mini-2024-07-18'}),
                         'gpt-4o-mini-2024-07-18')
        for name in ('gpt-4o-mini','other'):
            with self.assertRaises(ProviderFailure):transport._response_model({'model':name})


class SelectionReceiptTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ServiceTransportTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.payload=selection_payload(content_parts('synthetic query'), [source()])

    def transport(self, http):
        f=self.fixture
        return SelectionServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,
            f.p/'joint-receipts',{'gpt-4o-mini':lambda:'SYNTHETIC'},http=http,offline_test=True)

    def http(self, text='{"selected":[0]}', done=True, failure=False):
        class FakeHTTP:
            is_fake_http=True
            def __init__(self):self.calls=0
            def exchange(self,endpoint,body,*args):
                self.calls+=1
                payload=json.loads(body)
                assert payload['stream_options']=={'include_usage':True}
                if failure:raise ProviderFailure('network_error')
                return HTTPReply(200,wire(events(text),done),{'content-type':'text/event-stream'})
        return FakeHTTP()

    def test_complete_selection_minimized_original_ledger_receipt(self):
        http=self.http();transport=self.transport(http)
        result=transport(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(strict_json(result['choices'][0]['message']['content']), {'selected':[0]})
        with transport.journal.connect() as db:row=db.execute('select * from operations').fetchone()
        self.assertEqual(row['state'],'complete');self.assertIsNone(row['body'])
        self.assertEqual(row['scope'],'native-mem0-raw-v4')
        self.assertNotIn('synthetic query',row['manifest'])
        self.assertEqual(transport.ledger.totals()['attempts'],1)
        self.assertEqual(http.calls,1)

    def test_invalid_selection_is_known_failed_and_never_retried(self):
        http=self.http('{"selected":[999]}');transport=self.transport(http)
        with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,self.payload)
        with transport.journal.connect() as db:row=db.execute('select state,status,body from operations').fetchone()
        self.assertEqual(row['state'],'failed');self.assertEqual(row['status'],200);self.assertTrue(row['body'])
        with self.assertRaises(ValueError):transport(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(http.calls,1)

    def test_incomplete_stream_not_empty_selection_or_success(self):
        http=self.http('{"selected":[]}',done=False);transport=self.transport(http)
        with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,self.payload)
        with self.assertRaises(ValueError):transport.ready()
        self.assertEqual(http.calls,1)

    def test_unknown_retains_full_money_and_blocks_same_scope_after_restart(self):
        http=self.http(failure=True);transport=self.transport(http)
        before=transport.ledger.totals()['accounted_micro_cny']
        with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,self.payload)
        with transport.journal.connect() as db:row=db.execute('select state,status from operations').fetchone()
        self.assertEqual(row['state'],'unknown');self.assertIsNone(row['status'])
        self.assertEqual(transport.ledger.totals()['accounted_micro_cny']-before,250000)
        reopened=self.transport(http)
        with self.assertRaises(ValueError):reopened(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(http.calls,1)

    def test_insufficient_original_budget_rejected_before_dispatch(self):
        http=self.http();transport=self.transport(http)
        with patch.object(transport.ledger,'totals',return_value={'remaining_micro_cny':0}):
            with self.assertRaises(ValueError):transport(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(http.calls,0)


class SelectFake(FakeTransport):
    def __init__(self):super().__init__();self.pick=[0]
    def __call__(self, endpoint, payload):
        self.calls.append(payload)
        if self.failed:raise RuntimeError('synthetic unavailable')
        return {'choices':[{'message':{'content':json.dumps({'selected':self.pick})}}]}


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0 dependency')
class JointNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.e,self.t,self.c=FakeEmbedding(),SelectFake(),FakeCards()
        self.store=Mem0JointSystem(Path(self.temp.name)/'memory.sqlite',
            embedding=self.e,transport=self.t,card_adapter=self.c)
        self.addCleanup(self.close)

    def close(self):
        self.store.close()
        conn=getattr(self.store.native.db,'connection',None)
        if conn:conn.close()

    def add(self, text, uid='alice', request='r', session='s'):
        return self.store.add(dict(user_id=uid,request_id=request,session_id=session,
            messages=[dict(role='user',content=c) for c in text]))

    def search(self, query, uid='alice', top=100):
        return self.store.search(dict(user_id=uid,query=query,top_k=top))['data']

    def test_add_no_gpt_caption_or_generated_memory_and_search_one_call(self):
        self.add(['cobalt bridge original'])
        self.assertEqual(self.t.calls,[]);self.assertEqual(self.c.calls,0)
        rows=self.search('cobalt bridge')
        self.assertEqual(text_of(rows[0]['content']),'cobalt bridge original')
        self.assertEqual(len(self.t.calls),1)
        self.assertEqual(self.store.capabilities()['version'],VERSION)

    def test_selected_complete_original_and_same_add_neighbors(self):
        original='cobalt '+'原始证据'*2000
        self.add(['earlier support',original,'later support'])
        rows=self.search('cobalt')
        self.assertEqual([text_of(r['content']) for r in rows],[original,'earlier support','later support'])
        self.assertEqual(len(self.t.calls),1)

    def test_empty_selection_is_empty_not_raw_tail(self):
        self.add(['unrelated source']);self.t.pick=[]
        self.assertEqual(self.search('missing information'),[])

    def test_exact_image_retained_and_original_bytes_unchanged(self):
        original=[dict(type='text',text='original image'),image()]
        self.add([original]);self.t.pick=[]
        rows=self.search([image()])
        self.assertEqual(rows[0]['content'],original)
        self.assertEqual(len(self.t.calls),1)

    def test_cross_user_sources_never_enter_selection_prompt(self):
        self.add(['alice cobalt']);self.add(['PRIVATE_OTHER_SOURCE'],uid='bob')
        self.search('cobalt')
        raw=json.dumps(self.t.calls[-1])
        self.assertNotIn('PRIVATE_OTHER_SOURCE',raw)
        self.assertNotIn('user_id',raw)
        self.assertNotIn('request_id',raw)
        before=len(self.t.calls);self.assertEqual(self.search('cobalt',uid='absent'),[])
        self.assertEqual(len(self.t.calls),before)

    def test_cross_session_metadata_and_joint_selection(self):
        self.add(['bridge first'],session='one');self.add(['bridge second'],request='two',session='two')
        self.t.pick=[0,1]
        rows=self.search('bridge')
        self.assertEqual(len(rows),2)
        candidates=strict_json(self.t.calls[-1]['messages'][1]['content'][0]['text'])['candidates']
        self.assertEqual({r['session_group'] for r in candidates},{0,1})

    def test_bad_model_index_cannot_access_other_source_or_fallback(self):
        self.add(['source']);self.t.pick=[1234]
        with self.assertRaises(ValueError):self.search('source')
        self.assertEqual(len(self.t.calls),1)

    def test_provider_exception_is_not_empty_success_or_hidden_rerank_fallback(self):
        self.add(['source'])
        with patch.object(self.store.selector,'select',side_effect=RuntimeError('unavailable')) as call:
            with self.assertRaises(RuntimeError):self.search('source')
            self.assertEqual(call.call_count,1)

    def test_query_never_archived_and_top_k_retained(self):
        self.add(['source one','source two'])
        self.assertEqual(len(self.search('source NEW_QUERY_SENTINEL',top=1)),1)
        with self.store.connect() as db:
            rows=db.execute('select indexed_text from mem0_source_index').fetchall()
        self.assertNotIn('NEW_QUERY_SENTINEL',str(rows))

    def test_query_capacity_rejected_before_embedding(self):
        self.add(['source']);before=len(self.e.calls)
        with self.assertRaises(ContractError):self.search('q'*4001)
        self.assertEqual(len(self.e.calls),before);self.assertEqual(self.t.calls,[])
