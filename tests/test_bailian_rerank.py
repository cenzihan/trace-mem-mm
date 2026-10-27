"""Offline contracts only. Synthetic source strings, no real credentials or API."""
from datetime import datetime,timezone,timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest

from agent_memory.bailian_rerank import (MODEL,MAX_INPUT,payloads,BailianScorer,
    RerankJSONTransport,HostedRerankTransport)
from agent_memory.bailian_app import BailianServingSystem,CombinedReadiness,create_app
from agent_memory.budget import BudgetLedger
from agent_memory.provider_http import HTTPReply,ProviderFailure,BAILIAN_RERANK_ENDPOINT,validate_endpoint
from agent_memory.service_transport import history_digest
from agent_memory.online_candidate import OnlineDenseRerankSystem


class FakeHTTP:
    is_fake_http=True
    def __init__(self):self.calls=[];self.failure=None;self.mutate=lambda value:value
    def exchange(self,endpoint,body,headers,timeout,limit):
        self.calls.append(json.loads(body))
        if self.failure:raise self.failure
        count=len(self.calls[-1]['input']['documents'])
        value={'output':{'results':[{'index':i,'relevance_score':(i+1)/(count+1)} for i in reversed(range(count))]},
               'usage':{'total_tokens':120},'request_id':'synthetic'}
        return HTTPReply(200,json.dumps(self.mutate(value)).encode())


class HostedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.p=Path(self.tmp.name);self.http=FakeHTTP();self.ledger=BudgetLedger(self.p/'budget.sqlite')
        self.authority=self.p/'authority.json';self.prices=self.p/'prices.json'
        self.authority.write_text(json.dumps({'active_owner':'server','cap_micro_cny':500000000,
            'history':{table:{'rows':0,'sha256':history_digest([])} for table in ('attempts','provider_observations')}}))
        self.config={'checked_at':datetime.now(timezone.utc).isoformat(),'models':{MODEL:{
            'input_cny_per_million':'.5','output_cny_per_million':'0',
            'price_source':'https://help.aliyun.com/zh/model-studio/qwen3-vl-rerank'}}}
        self.prices.write_text(json.dumps(self.config));self.cap=4
        self.payload=payloads('synthetic query',['source a','source b'])[0]

    def make(self):
        return HostedRerankTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},
                                     call_limit=self.cap,http=self.http,offline_test=True)

    def call(self,transport=None):return (transport or self.make())(BAILIAN_RERANK_ENDPOINT,self.payload)

    def test_success_accounts_original_ledger_and_minimizes_receipt(self):
        self.call();transport=self.make()
        with self.ledger.connect() as db:
            self.assertEqual(db.execute('SELECT model,status,reserved,accounted FROM attempts').fetchone(),(MODEL,'settled',60000,60))
            self.assertIsNone(db.execute('SELECT returned_model FROM provider_observations').fetchone()[0])
        with transport.journal.connect() as db:
            row=db.execute('SELECT state,body,redacted,receipt_policy,manifest FROM operations').fetchone()
            self.assertEqual(tuple(row[:4]),('complete',None,1,'usage_only'))
            self.assertNotIn('synthetic query',row[4])

    def test_sorted_response_reassembles_original_document_order(self):
        scorer=BailianScorer(self.make())
        self.assertEqual(scorer.score([{'type':'text','text':'synthetic query'}],
                                     [{'retrieval_text':'source a'},{'retrieval_text':'source b'}],[0,0]),[1/3,2/3])

    def test_whole_document_batching_no_loss_or_extra_model(self):
        docs=['a'*7000+str(i) for i in range(40)]
        batches=payloads('q'*1000,docs)
        self.assertGreater(len(batches),1)
        self.assertEqual([d for b in batches for d in b['input']['documents']],docs)
        for batch in batches:self.assertEqual(RerankJSONTransport._bounds(None,batch),(MAX_INPUT,0))
        scores=BailianScorer(self.make()).score([{'type':'text','text':'q'*1000}],
                                               [{'retrieval_text':d} for d in docs],[0]*40)
        self.assertEqual(len(scores),40);self.assertEqual(len(self.http.calls),len(batches))

    def test_lifetime_limit_survives_restart_and_cannot_be_raised(self):
        self.cap=1;self.call()
        with self.assertRaisesRegex(ValueError,'exhausted'):self.call()
        self.cap=2
        with self.assertRaisesRegex(ValueError,'ceiling cannot change'):self.make()
        self.assertEqual(len(self.http.calls),1)

    def test_removed_guard_after_activity_is_rejected(self):
        self.call();(self.p/'receipts/hosted-call-limit.json').unlink()
        with self.assertRaisesRegex(ValueError,'missing hosted call limit'):self.make()

    def test_unknown_retains_hold_and_blocks_replay(self):
        self.http.failure=ProviderFailure('timeout')
        with self.assertRaises(ProviderFailure):self.call()
        self.http.failure=None
        with self.assertRaisesRegex(ValueError,'unresolved'):self.call()
        with self.ledger.connect() as db:self.assertEqual(db.execute('SELECT status,accounted FROM attempts').fetchone(),('uncertain',60000))
        self.assertEqual(len(self.http.calls),1)

    def test_duplicate_indices_fail_and_stop(self):
        self.http.mutate=lambda v:dict(v,output={'results':[{'index':0,'relevance_score':.5}]*2})
        with self.assertRaises(ProviderFailure):self.call()
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(len(self.http.calls),1)

    def test_overbound_usage_freezes_shared_ledger(self):
        self.http.mutate=lambda v:dict(v,usage={'total_tokens':MAX_INPUT+1})
        with self.assertRaises(ProviderFailure):self.call()
        with self.ledger.connect() as db:self.assertEqual(db.execute('SELECT status FROM attempts').fetchone()[0],'needs_reconciliation')

    def test_reported_wrong_model_is_not_accepted(self):
        self.http.mutate=lambda v:dict(v,model='different')
        with self.assertRaisesRegex(ProviderFailure,'model_mismatch'):self.call()

    def test_no_raw_returned_document_is_accepted(self):
        def add_document(value):value['output']['results'][0]['document']={'text':'not requested'};return value
        self.http.mutate=add_document
        with self.assertRaisesRegex(ProviderFailure,'invalid_contract'):self.call()

    def test_stale_and_changed_prices_do_not_dispatch(self):
        self.config['checked_at']=(datetime.now(timezone.utc)-timedelta(days=3)).isoformat()
        self.prices.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):self.call()
        self.config['checked_at']=datetime.now(timezone.utc).isoformat()
        self.config['models'][MODEL]['input_cny_per_million']='.6'
        self.prices.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.http.calls,[])

    def test_invalid_payload_has_no_journal_claim(self):
        self.payload['parameters']['return_documents']=True
        with self.assertRaises(ValueError):self.call()
        with self.make().journal.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM operations').fetchone()[0],0)

    def test_model_and_region_are_explicit(self):
        validate_endpoint(MODEL,BAILIAN_RERANK_ENDPOINT)
        validate_endpoint(MODEL,'https://ws-example.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank')
        for endpoint in [BAILIAN_RERANK_ENDPOINT+'?x=1',BAILIAN_RERANK_ENDPOINT.replace('dashscope.','dashscope-intl.'),'http://127.0.0.1/rerank']:
            with self.assertRaises(ValueError):validate_endpoint(MODEL,endpoint)

    def test_no_image_or_truncation_in_text_only_candidate(self):
        for query,docs in [('q'*4001,['x']),('q',[{'image':'data:image/png;base64,private'}]),('q',['x'*8001])]:
            with self.assertRaises(ValueError):payloads(query,docs)

    def test_activation_is_separate_and_default_off(self):
        with self.assertRaisesRegex(ValueError,'activation'):create_app({})
        self.assertEqual(OnlineDenseRerankSystem.reranker_model,'Qwen/Qwen3-VL-Reranker-2B')
        self.assertEqual(BailianServingSystem.reranker_model,MODEL)

    def test_readiness_has_no_network_call(self):
        transport=self.make();CombinedReadiness(transport,transport).ready()
        self.assertEqual(self.http.calls,[])

    def test_complete_memory_path_preserves_sources_and_isolation(self):
        from test_online_candidate import Embedding,Cards
        from agent_memory.protocol import text_of
        embedding=Embedding();cards=Cards();scorer=BailianScorer(self.make())
        system=BailianServingSystem(self.p/'memory.sqlite',embedding=embedding,card_adapter=cards,scoring_adapter=scorer)
        original=['blue original evidence','red original evidence']
        add={'user_id':'u','session_id':'s','request_id':'r','messages':[{'role':'user','content':v} for v in original]}
        system.add(add);system.add(add)
        query={'user_id':'u','query':'needle','top_k':100}
        first=system.search(query)
        self.assertEqual({text_of(row['content']) for row in first['data']},set(original))
        calls=(len(embedding.calls),cards.calls,len(self.http.calls))
        self.assertEqual(system.search(dict(query,user_id='other')),{'data':[]})
        self.assertEqual((len(embedding.calls),cards.calls,len(self.http.calls)),calls)
        reopened=BailianServingSystem(self.p/'memory.sqlite',embedding=embedding,card_adapter=cards,scoring_adapter=scorer)
        self.assertEqual(reopened.search(query),first)
        self.assertTrue(all('derived description' not in text_of(row['content']) for row in first['data']))

    def test_old_service_does_not_silently_accept_hosted_model(self):
        from test_online_candidate import Embedding,Cards
        with self.assertRaises(ValueError):
            OnlineDenseRerankSystem(self.p/'old.sqlite',embedding=Embedding(),card_adapter=Cards(),scoring_adapter=BailianScorer(self.make()))
