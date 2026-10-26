import io
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, HTTPReply, ProviderFailure
from agent_memory.service_transport import ServiceTransport, ServiceBudgetedTransport, history_digest, omit_nontext_quote_metadata
from agent_memory.service_app import ServiceApplication, ServingSystem
from agent_memory.protocol import ContractError
from agent_memory.store import MemoryStore


class FakeHTTP:
    is_fake_http=True
    def __init__(self):self.calls=0;self.failure=False;self.model='text-embedding-v4'
    def exchange(self,endpoint,body,headers,timeout,limit):
        self.calls+=1
        if self.failure:raise ProviderFailure('timeout')
        value=dict(model=self.model,usage=dict(total_tokens=3),data=[dict(index=0,embedding=[.1]*256)])
        return HTTPReply(200,json.dumps(value).encode())


class ServiceTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.p=Path(self.temp.name)
        self.ledger=BudgetLedger(self.p/'budget.sqlite');self.http=FakeHTTP()
        self.authority=self.p/'authority.json';self.prices=self.p/'prices.json'
        self.authority.write_text(json.dumps(dict(active_owner='server',cap_micro_cny=500000000,
            history={table:dict(rows=0,sha256=history_digest([])) for table in ('attempts','provider_observations')})))
        self.config=dict(checked_at=datetime.now(timezone.utc).isoformat(),models={
            'gpt-4o-mini':dict(input_cny_per_million='1.5',output_cny_per_million='6',price_source='https://developers.openai.com/api/docs/models/gpt-4o-mini',conversion_basis='budget_assumption_not_market_fx'),
            'text-embedding-v4':dict(input_cny_per_million='.5',output_cny_per_million='0',price_source='https://help.aliyun.com/zh/model-studio/embedding')})
        self.prices.write_text(json.dumps(self.config))
        self.payload=dict(model='text-embedding-v4',input=['Synthetic unit fixture; not performance data.'],dimensions=256,encoding_format='float')

    def make(self):
        return ServiceTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},http=self.http,offline_test=True)

    def call(self,transport=None):return (transport or self.make())(BAILIAN_BEIJING_ENDPOINT,self.payload)

    def test_existing_ledger_required(self):
        with self.assertRaises(ValueError):ServiceTransport(self.p/'absent.sqlite',self.authority,self.prices,self.p/'receipts',{},http=self.http,offline_test=True)
        self.assertFalse((self.p/'absent.sqlite').exists());self.assertEqual(self.http.calls,0)

    def test_success_removes_content_and_survives_restart(self):
        self.call();self.call()
        with self.make().journal.connect() as db:rows=db.execute('SELECT state,body,redacted,receipt_policy,manifest FROM operations').fetchall()
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(tuple(row[:4]),('complete',None,1,'usage_only'))
            self.assertNotIn('Synthetic unit fixture',row[4])
        self.assertEqual(self.ledger.totals()['attempts'],2)
        self.assertEqual(self.http.calls,2)

    def test_unknown_never_replays_after_restart(self):
        self.http.failure=True
        with self.assertRaises(ProviderFailure):self.call()
        self.http.failure=False
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.http.calls,1)

    def test_historical_prefix_mismatch_is_rejected(self):
        value=json.loads(self.authority.read_text());value['history']['attempts']['sha256']='0'*64;self.authority.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.make()
        self.assertEqual(self.http.calls,0)

    def test_ownership_change_is_rejected(self):
        transport=self.make();self.authority.write_text(self.authority.read_text()+' ')
        with self.assertRaises(ValueError):self.call(transport)
        self.assertEqual(self.http.calls,0)

    def test_expired_price_has_no_dispatch_or_journal_claim(self):
        transport=self.make();self.config['checked_at']=(datetime.now(timezone.utc)-timedelta(days=3)).isoformat();self.prices.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):self.call(transport)
        self.assertEqual(self.http.calls,0)
        with transport.journal.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM operations').fetchone()[0],0)

    def test_insufficient_budget_blocks_before_dispatch(self):
        rate=VerifiedRate('text-embedding-v4','.5','0','https://help.aliyun.com/zh/model-studio/embedding',self.config['checked_at'])
        self.ledger.reserve(rate,1,fixed_reserve_cny='496')
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.http.calls,0)

    def test_invalid_input_does_not_poison_journal(self):
        transport=self.make();self.payload['dimensions']=128
        with self.assertRaises(ValueError):self.call(transport)
        self.payload['dimensions']=256;self.call(transport)
        self.assertEqual(self.http.calls,1)

    def test_model_substitution_stops_future_requests(self):
        self.http.model='different-model'
        with self.assertRaises(ProviderFailure):self.call()
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.http.calls,1)

    def test_price_change_requires_new_review_before_dispatch(self):
        self.config['models']['gpt-4o-mini']['input_cny_per_million']='2'
        self.prices.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):self.call()
        self.assertEqual(self.http.calls,0)


class ServiceApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        class Transport:
            fail=False
            calls=0
            def ready(self):
                self.calls+=1
                if self.fail:raise ValueError('private reason')
        self.transport=Transport();self.app=ServiceApplication(MemoryStore(Path(self.temp.name)/'memory.sqlite'),self.transport,'fixture-token',scorer_health=lambda:None)

    def call(self,path='/health',auth=None):
        env=dict(PATH_INFO=path,REQUEST_METHOD='GET' if path=='/health' else 'POST',CONTENT_TYPE='application/json',CONTENT_LENGTH='2',**{'wsgi.input':io.BytesIO(b'{}')})
        if auth:env['HTTP_AUTHORIZATION']=auth
        values=[];body=b''.join(self.app(env,lambda status,headers:values.append((status,dict(headers)))))
        return values[0],json.loads(body)

    def test_unhealthy_readiness_is_not_advertised_as_ok(self):
        self.transport.fail=True
        meta,value=self.call();self.assertTrue(meta[0].startswith('503'));self.assertNotIn('private reason',str(value))

    def test_unauthorized_does_not_touch_model_transport(self):
        meta,value=self.call('/search');self.assertTrue(meta[0].startswith('401'));self.assertEqual(self.transport.calls,0)

    def test_busy_returns_bounded_retry_after(self):
        self.app.active.acquire()
        try:
            meta,value=self.call('/search','Bearer fixture-token');self.assertTrue(meta[0].startswith('429'));self.assertEqual(meta[1]['Retry-After'],'5')
        finally:self.app.active.release()

    def test_expired_lease_never_dispatches(self):
        self.app.expires_at=1
        meta,value=self.call('/search','Bearer fixture-token')
        self.assertTrue(meta[0].startswith('503'));self.assertEqual(self.transport.calls,0)


class AdmissionTests(unittest.TestCase):
    def test_oversize_source_rejected_before_any_model_or_store_access(self):
        store=object.__new__(ServingSystem)
        payload=dict(user_id='fixture',session_id='fixture',request_id='fixture',messages=[dict(role='user',content='x'*6001,timestamp=1767225600000)])
        with self.assertRaises(ContractError) as exc:store.add(payload)
        self.assertEqual(exc.exception.status,413)


class AuxiliaryMetadataTests(unittest.TestCase):
    def test_only_nontext_auxiliary_entries_are_omitted_and_raw_is_unchanged(self):
        value=dict(model='gpt-4o-mini',choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(summary='unchanged model summary',evidence_quotes=[{'quote':'invented'},'source',5]))))])
        original=json.dumps(value);view=omit_nontext_quote_metadata(value)
        card=json.loads(view['choices'][0]['message']['content'])
        self.assertEqual(card['summary'],'unchanged model summary')
        self.assertEqual(card['evidence_quotes'],['source'])
        self.assertEqual(card['omitted_nontext_quote_count'],2)
        self.assertEqual(json.dumps(value),original)
        validator=object.__new__(ServiceBudgetedTransport);validator.model='gpt-4o-mini'
        validator._contract(view,{})

    def test_nonarray_quotes_remain_invalid(self):
        value=dict(choices=[dict(message=dict(content=json.dumps(dict(summary='x',evidence_quotes={'quote':'x'}))))])
        with self.assertRaises(ProviderFailure):omit_nontext_quote_metadata(value)

    def test_oversize_query_rejected_before_model_or_store_access(self):
        store=object.__new__(ServingSystem)
        with self.assertRaises(ContractError) as exc:store.search(dict(user_id='fixture',query='x'*8193,top_k=20))
        self.assertEqual(exc.exception.status,413)


if __name__=='__main__':unittest.main()
