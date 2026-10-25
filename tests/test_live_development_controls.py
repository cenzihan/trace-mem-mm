"""Synthetic HTTP and temporary ledgers only; no credential reads or external calls."""
import json
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.development_run import DevelopmentRun,ScopedTransport
from agent_memory.provider_http import BudgetedJSONTransport,HTTPReply,OPENAI_ENDPOINT,BAILIAN_BEIJING_ENDPOINT,validate_endpoint
from agent_memory.providers import GPT4oMiniCards,TextEmbeddingV4
from agent_memory.system import AgentMemorySystem,RetrievalConfig
import base64
def png(scene,variant=False):
    # Unit-only byte identity fixture; no performance scenes or labels.
    return b'\x89PNG\r\n\x1a\n'+(b'unit-variant' if variant else b'unit-original')
def uri(raw):return 'data:image/png;base64,'+base64.b64encode(raw).decode('ascii')


class FakeHTTP:
    is_fake_http=True
    def __init__(self):self.payloads=[]
    def exchange(self,endpoint,body,headers,timeout,limit):
        body=json.loads(body);self.payloads.append(body)
        value=dict(model='gpt-4o-mini-2024-07-18',usage=dict(prompt_tokens=2900,completion_tokens=50,total_tokens=2950),
                   choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(summary='red circle left of blue square',evidence_quotes=[]))))])
        return HTTPReply(200,json.dumps(value).encode())


class DevelopmentControls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def card_client(self,hold='.25'):
        self.ledger=BudgetLedger(self.root/'budget.sqlite');self.http=FakeHTTP()
        rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat(),conversion_basis='budget_assumption_not_market_fx')
        return BudgetedJSONTransport(model=rate.model,endpoint=OPENAI_ENDPOINT,ledger=self.ledger,rate=rate,
                                     http=self.http,mode='offline_fake',fixed_reserve_cny=hold,usd_rates=('.15','.60'))
    def test_official_legacy_beijing_endpoint_is_explicitly_allowed(self):
        validate_endpoint('text-embedding-v4',BAILIAN_BEIJING_ENDPOINT)
        TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT,dimensions=256)
        for endpoint in (BAILIAN_BEIJING_ENDPOINT+'?x=1',BAILIAN_BEIJING_ENDPOINT.replace('dashscope.','dashscope-intl.'),
                         BAILIAN_BEIJING_ENDPOINT.replace('https:','http:'),BAILIAN_BEIJING_ENDPOINT.replace('.com/','.com:444/')):
            self.assertRaises(ValueError,validate_endpoint,'text-embedding-v4',endpoint)
            self.assertRaises(ValueError,TextEmbeddingV4,endpoint)
    def test_fixed_assumption_retains_hold_with_observed_usage(self):
        client=self.card_client();GPT4oMiniCards(client).extract([dict(type='text',text='synthetic')])
        self.assertEqual(self.ledger.totals()['accounted_micro_cny'],250000)
        with self.ledger.connect() as db:
            self.assertEqual(db.execute('SELECT estimated_usd FROM provider_observations').fetchone()[0],'0.000465')
    def test_insufficient_fixed_hold_never_dispatches(self):
        client=self.card_client('.01')
        self.assertRaises(ValueError,GPT4oMiniCards(client).extract,[dict(type='text',text='synthetic')])
        self.assertEqual(self.ledger.totals()['attempts'],0);self.assertFalse(self.http.payloads)
    def test_low_detail_is_provider_only_and_does_not_change_original(self):
        client=self.card_client();parts=[dict(type='image_url',image_url=dict(url=uri(png((('circle','square'),('red','blue'),'horizontal')))))]
        GPT4oMiniCards(client,image_detail='low').extract(parts)
        self.assertEqual(set(parts[0]['image_url']),{'url'})
        self.assertEqual(self.http.payloads[0]['messages'][1]['content'][0]['image_url']['detail'],'low')
    def test_bad_detail_rejected(self):
        self.assertRaises(ValueError,GPT4oMiniCards,self.card_client(),image_detail='original')
    def test_missing_fixed_assumption_rejected_before_any_read(self):
        self.assertRaises(ValueError,self.card_client,None)
    def test_run_survives_restart_with_request_cap(self):
        run=DevelopmentRun(self.root/'run.sqlite',create=True,max_requests=2,max_reservation_cny='1')
        for _ in range(2):
            ordinal=run.claim('gpt-4o-mini','.25');run.finish(ordinal,'gpt-4o-mini-2024-07-18')
        reopened=DevelopmentRun(run.path)
        self.assertRaises(ValueError,reopened.claim,'gpt-4o-mini','.25')
        self.assertEqual(reopened.summary()['request_claims'],2)
    def test_uncertain_run_blocks_further_calls_and_completion(self):
        run=DevelopmentRun(self.root/'run.sqlite',create=True);run.claim('gpt-4o-mini','.25')
        self.assertRaises(ValueError,DevelopmentRun(run.path).claim,'gpt-4o-mini','.25')
        self.assertRaises(ValueError,run.close)
    def test_cumulative_preflight_cap_not_released_on_success(self):
        run=DevelopmentRun(self.root/'run.sqlite',create=True,max_reservation_cny='.25')
        ordinal=run.claim('gpt-4o-mini','.25');run.finish(ordinal,'gpt-4o-mini')
        self.assertRaises(ValueError,run.claim,'text-embedding-v4','.01')
    def test_scope_success_closes_no_automatic_second_run(self):
        run=DevelopmentRun(self.root/'run.sqlite',create=True)
        self.assertRaises(FileExistsError,DevelopmentRun,run.path,create=True)
        scoped=ScopedTransport(self.card_client(),run,'.25')
        GPT4oMiniCards(scoped).extract([dict(type='text',text='synthetic')]);run.close()
        self.assertEqual(run.summary()['state'],'complete')
        self.assertRaises(ValueError,run.claim,'gpt-4o-mini','.25')
    def test_visual_query_requires_specified_model(self):
        self.assertRaises(ValueError,AgentMemorySystem,self.root/'memory.sqlite',RetrievalConfig(visual_query_cards=True))
    def test_visual_query_transient_no_unknown_user_cost_or_query_persistence(self):
        class Cards:
            model='gpt-4o-mini'
            def __init__(self):self.calls=0
            def extract(self,parts):self.calls+=1;return dict(summary='red circle left of blue square',evidence_quotes=[])
        cards=Cards();store=AgentMemorySystem(self.root/'memory.sqlite',RetrievalConfig(visual_query_cards=True),card_adapter=cards)
        scene=(('circle','square'),('red','blue'),'horizontal')
        store.add(dict(user_id='u',session_id='s',request_id='r',messages=[dict(role='user',content=[dict(type='text',text='original'),dict(type='image_url',image_url=dict(url=uri(png(scene))))])]))
        query=dict(user_id='unknown',query=[dict(type='image_url',image_url=dict(url=uri(png(scene,True))))],top_k=3)
        self.assertEqual(store.search(query)['data'],[]);self.assertEqual(cards.calls,1)
        query['user_id']='u';result,trace=store.retrieve(query)
        self.assertEqual(len(result['data']),1);self.assertFalse(trace['image_matches']);self.assertTrue(trace['query_summary'])
        self.assertEqual(store.inspect_user('u')['memories'],1);self.assertEqual(cards.calls,2)


if __name__=='__main__':unittest.main()
