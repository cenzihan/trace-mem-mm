"""Synthetic source images and real Mem0 SDK; no provider network."""
import base64
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.bailian_rerank import MODEL,HostedRerankTransport
from agent_memory.visual_rerank import (VisualHostedTransport,VisualRerankJSONTransport,
    VisualScorer,visual_payloads,query_views,source_views,view_bound,MAX_BATCHES,MAX_REQUEST_BYTES)
from agent_memory.mem0_visual import Mem0VisualSystem,VERSION
from agent_memory.mem0_raw import Mem0RawTransport
from agent_memory.protocol import content_parts,ContractError
from agent_memory.provider_http import ProviderFailure,BAILIAN_RERANK_ENDPOINT
import test_bailian_rerank as hosted_tests
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding,FakeTransport,FakeCards


class VisualPayloadTests(unittest.TestCase):
    def test_preview_is_separate_and_preserves_original_order(self):
        parts=[{'type':'text','text':'before'},image(),{'type':'text','text':'after'}]
        before=json.dumps(parts)
        views=source_views(parts)
        self.assertEqual(views[0],{'text':'before after'})
        self.assertTrue(views[1]['image'].startswith('data:image/jpeg;base64,'))
        self.assertNotEqual(views[1]['image'],parts[1]['image_url']['url'])
        self.assertEqual(json.dumps(parts),before)

    def test_text_prefix_is_unicode_safe_and_explicit(self):
        source='源'*3000
        result=source_views(content_parts(source))
        self.assertEqual(len(result[0]['text'].encode()),6000)
        self.assertEqual(source,'源'*3000)

    def test_query_views_have_text_and_first_image_without_caption(self):
        parts=[{'type':'text','text':'teal object'},image(),image()]
        views=query_views(parts)
        self.assertEqual(len(views),2)
        self.assertEqual(views[0],{'text':'teal object'})
        self.assertEqual(len(query_views([image()])),1)

    def test_query_capacity_rejects_before_dispatch(self):
        with self.assertRaises(ContractError):query_views(content_parts('q'*4001))

    def test_all_documents_batched_without_dropping_or_duplication(self):
        docs=[{'text':str(i)+'x'*5996} for i in range(80)]
        query={'text':'q'*4000}
        batches=visual_payloads(query,docs)
        self.assertEqual([d for b in batches for d in b['input']['documents']],docs)
        self.assertTrue(all(len(b['input']['documents'])<=8 for b in batches))
        for batch in batches:
            self.assertEqual(VisualRerankJSONTransport._bounds(None,batch),(120000,0))
            self.assertLessEqual(len(json.dumps(batch,ensure_ascii=False).encode()),MAX_REQUEST_BYTES)

    def test_all_image_pairs_fit_reviewed_batch_and_search_ceiling(self):
        view=query_views([image()])[0]
        batches=visual_payloads(view,[view]*80)
        self.assertLessEqual(2*len(batches),MAX_BATCHES)
        for batch in batches:VisualRerankJSONTransport._bounds(None,batch)

    def test_external_url_and_unbounded_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):view_bound({'image':'https://example.invalid/a.png'})
        from PIL import Image
        buf=io.BytesIO();Image.new('RGB',(513,10)).save(buf,format='JPEG')
        with self.assertRaises(ValueError):
            view_bound({'image':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()})

    def test_invalid_modality_model_and_raw_response_request_are_rejected(self):
        for view in ({'video':'x'},{'text':'x','image':'x'},{'text':''}):
            with self.assertRaises(ValueError):view_bound(view)
        payload=visual_payloads({'text':'q'},[{'text':'source'}])[0]
        payload['parameters']['return_documents']=True
        with self.assertRaises(ValueError):VisualRerankJSONTransport._bounds(None,payload)

    def test_explicit_activation_and_original_embedding_scope(self):
        from agent_memory.mem0_app import create_visual_app
        with self.assertRaises(ValueError):create_visual_app({})
        self.assertEqual(Mem0RawTransport.receipt_scope,'native-mem0-raw-v4')


# Reuse only the temporary offline harness, not inherited text-only assertions.
class VisualTransportTests(unittest.TestCase):
    def setUp(self):hosted_tests.HostedTests.setUp(self)
    def make(self):
        return VisualHostedTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},
            call_limit=self.cap,http=self.http,offline_test=True)

    def test_visual_response_maps_all_view_and_query_scores_to_originals(self):
        transport=self.make();scorer=VisualScorer(transport)
        parts=[{'type':'text','text':'query'},image()]
        sources=[{'content':content_parts('source')},{'content':[image()]}]
        scores=scorer.score(parts,sources)
        self.assertEqual(scores,[1/3,2/3])
        self.assertEqual(len(self.http.calls),2)
        with transport.journal.connect() as db:
            self.assertEqual(db.execute('select state,body,redacted from operations').fetchall()[0][:],
                             ('complete',None,1))

    def test_same_immutable_guard_and_ledger_survive_transport_switch(self):
        transport=self.make()
        transport(BAILIAN_RERANK_ENDPOINT,visual_payloads({'text':'q'},[{'text':'a'}])[0])
        old=HostedRerankTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},
            call_limit=self.cap,http=self.http,offline_test=True)
        with old.journal.connect() as db:self.assertEqual(db.execute('select count(*) from operations').fetchone()[0],1)
        self.cap+=1
        with self.assertRaises(ValueError):self.make()

    def test_full_call_plan_rejected_before_first_request(self):
        self.cap=1;transport=self.make()
        payload=visual_payloads({'text':'q'},[{'text':'a'}])[0]
        with self.assertRaises(ValueError):transport.preflight([payload,payload])
        self.assertEqual(self.http.calls,[])

    def test_whole_search_money_guard_does_not_create_a_new_budget(self):
        transport=self.make();payload=visual_payloads({'text':'q'},[{'text':'a'}])[0]
        with patch.object(transport.ledger,'totals',return_value={'remaining_micro_cny':1}):
            with self.assertRaises(ValueError):transport.preflight([payload])
        with self.ledger.connect() as db:self.assertEqual(db.execute('select count(*) from attempts').fetchone()[0],0)

    def test_unknown_keeps_full_hold_and_blocks_visual_and_original_transports(self):
        transport=self.make();payload=visual_payloads({'text':'q'},[{'text':'a'}])[0]
        self.http.failure=ProviderFailure('timeout')
        with self.assertRaises(ProviderFailure):transport(BAILIAN_RERANK_ENDPOINT,payload)
        self.http.failure=None
        with self.assertRaises(ValueError):transport.preflight([payload])
        with self.ledger.connect() as db:
            self.assertEqual(db.execute('select status,accounted from attempts').fetchone(),('uncertain',60000))
        self.assertEqual(len(self.http.calls),1)

    def test_bad_index_fails_without_partial_scores(self):
        self.http.mutate=lambda v:dict(v,output={'results':[{'index':5,'relevance_score':.5}]})
        with self.assertRaises(ProviderFailure):
            VisualScorer(self.make()).score(content_parts('q'),[{'content':[image()]}])

    def test_two_query_passes_keep_source_indices_across_batch_boundaries(self):
        self.cap=8
        sources=[{'content':content_parts('source'+str(i))} for i in range(10)]
        scores=VisualScorer(self.make()).score([{'type':'text','text':'q'},image()],sources)
        self.assertEqual(scores,[i/9 for i in range(1,9)]+[1/3,2/3])
        self.assertEqual(len(self.http.calls),4)


class FakeScorer:
    def __init__(self):self.calls=[];self.failed=False
    def ready(self):
        if self.failed:raise ValueError('synthetic scoring unavailable')
    def score(self,parts,sources):
        self.calls.append((parts,sources))
        return [1.0 if any(p['type']=='image_url' for p in s['content']) else .1 for s in sources]


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned dependency')
class VisualNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'memory.sqlite'
        self.e,self.t,self.c,self.sc=FakeEmbedding(),FakeTransport(),FakeCards(),FakeScorer()
        self.store=Mem0VisualSystem(self.path,embedding=self.e,transport=self.t,card_adapter=self.c,scoring_adapter=self.sc)
    def tearDown(self):
        self.assertEqual(self.t.calls,[]);self.assertEqual(self.c.calls,0)
        self.store.close();self.store.native.db.connection.close();self.temp.cleanup()
    def add(self,contents,uid='u',req='r'):
        return self.store.add(dict(user_id=uid,session_id='s',request_id=req,
            messages=[{'role':'user','content':x} for x in contents]))
    def search(self,query,uid='u'):
        return self.store.search(dict(user_id=uid,query=query,top_k=100))['data']

    def test_text_query_can_retrieve_image_only_original_without_native_vectors(self):
        original=[image()]
        self.add([original])
        rows=self.search('a visible teal object')
        self.assertEqual(rows[0]['content'],original)
        self.assertEqual(self.e.calls,[])
        self.assertEqual(len(self.sc.calls),1)
        self.assertEqual(self.store.capabilities()['version'],VERSION)

    def test_visual_rank_changes_order_and_preserves_all_original_parts(self):
        self.add(['other item',[{'type':'text','text':'source note'},image()]])
        rows=self.search('a visible item')
        self.assertEqual(rows[0]['content'][1],image())
        self.assertEqual({r['id'] for r in rows},set(s['id'] for s in self.sc.calls[-1][1]))

    def test_absent_user_no_embedding_or_reranker_and_cross_user_isolation(self):
        self.add([[image()]],uid='private')
        before=len(self.e.calls)
        self.assertEqual(self.search('teal',uid='missing'),[])
        self.assertEqual(len(self.e.calls),before);self.assertEqual(self.sc.calls,[])

    def test_oversized_query_stops_before_any_query_call(self):
        self.add(['source'])
        before=len(self.e.calls)
        with self.assertRaises(ContractError):self.search('x'*4001)
        self.assertEqual(len(self.e.calls),before);self.assertEqual(self.sc.calls,[])

    def test_rerank_failure_blocks_search_without_invented_empty_success(self):
        self.add(['source'])
        self.sc.failed=True
        with self.assertRaises(ValueError):self.search('source')
        self.assertEqual(self.sc.calls,[])

    def test_no_query_or_preview_is_persisted_in_original_archive(self):
        original=[image()]
        self.add([original])
        self.search('private-query-sentinel')
        with self.store.connect() as db:
            rows=db.execute('select content from memories').fetchall()
        self.assertEqual(json.loads(rows[0][0]),original)
        self.assertNotIn('private-query-sentinel',str(rows))

