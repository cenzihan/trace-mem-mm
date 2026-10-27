import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.budget import BudgetLedger
from agent_memory.mem0_bridge import Mem0System,utf8_chunks,native_scope
from agent_memory.protocol import ContractError
from agent_memory.mem0_transport import Mem0BudgetedTransport
from agent_memory.provider_http import ProviderFailure


class FakeEmbedding:
    def __init__(self):self.calls=[];self.fail=False
    def embed(self,texts):
        self.calls.append(texts)
        if self.fail:raise RuntimeError('synthetic unavailable')
        result=[]
        for text in texts:
            v=[0.01]*256
            for word in text.lower().split():v[sum(word.encode())%256]+=1
            result.append(v)
        return result


class FakeTransport:
    def __init__(self):self.calls=[];self.failed=False
    def ready(self):
        if self.failed:raise ValueError('unresolved synthetic call')
    def __call__(self,endpoint,payload):
        self.calls.append(payload)
        return {'choices':[{'message':{'content':json.dumps({'memory':[{'text':'Alice keeps a teal notebook in the workshop.'}]})}}]}


class FakeCards:
    def __init__(self):self.calls=0
    def extract(self,parts):self.calls+=1;return {'summary':'A teal notebook in a workshop.','evidence_quotes':[]}


class BudgetConsentTests(unittest.TestCase):
    def test_additive_consent_preserves_original_and_does_not_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'money.sqlite';ledger=BudgetLedger(path)
            ledger.authorize_increase('800','night','explicit human cumulative increase')
            ledger.authorize_increase('800','night','explicit human cumulative increase')
            self.assertEqual(BudgetLedger(path).totals()['cap_micro_cny'],800000000)
            with ledger.connect() as db:
                self.assertEqual(db.execute('SELECT cap FROM budget').fetchone()[0],500000000)
                self.assertEqual(db.execute('SELECT count(*) FROM budget_consents').fetchone()[0],1)
            with self.assertRaises(ValueError):ledger.authorize_increase('700','night','changed')
            with self.assertRaises(ValueError):ledger.authorize_increase('801','second','unapproved')

    def test_consented_reserve_uses_all_original_spending(self):
        from agent_memory.budget import VerifiedRate
        with tempfile.TemporaryDirectory() as temp:
            ledger=BudgetLedger(Path(temp)/'money.sqlite')
            rate=VerifiedRate('text-embedding-v4','0.5','0','synthetic','2026-10-27T00:00:00Z')
            first=ledger.reserve(rate,1,fixed_reserve_cny='499')
            with self.assertRaises(ValueError):ledger.reserve(rate,1,fixed_reserve_cny='2')
            ledger.authorize_increase('800','consent','explicit human cumulative increase')
            ledger.reserve(rate,1,fixed_reserve_cny='300')
            self.assertEqual(ledger.totals()['remaining_micro_cny'],1000000)
            with self.assertRaises(ValueError):ledger.reserve(rate,1,fixed_reserve_cny='2')
            with ledger.connect() as db:self.assertEqual(db.execute('SELECT accounted FROM attempts WHERE id=?',(first,)).fetchone()[0],499000000)


class NativeContractTests(unittest.TestCase):
    def test_required_native_shape_and_truncation_are_not_silently_empty(self):
        validator=object.__new__(Mem0BudgetedTransport);validator.model='gpt-4o-mini'
        payload={'response_format':{'type':'json_object'}}
        def response(value,finish='stop'):
            return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(value)}}]}
        validator._contract(response({'memory':[]}),payload)
        validator._contract(response({'memory':[{'text':'A synthetic fact.'}]}),payload)
        for value in ({},{'memory':'invalid'},{'memory':[{'text':3}]},{'memory':[None]}):
            with self.assertRaises(ProviderFailure):validator._contract(response(value),payload)
        with self.assertRaises(ProviderFailure):validator._contract(response({'memory':[]},'length'),payload)


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0 dependency')
class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'raw.sqlite'
        self.e=FakeEmbedding();self.t=FakeTransport();self.c=FakeCards()
        self.store=Mem0System(self.path,embedding=self.e,transport=self.t,card_adapter=self.c)
        self.payload=dict(user_id='alice',request_id='request',session_id='s1',messages=[dict(role='user',content='I keep a teal notebook in the workshop.')])
    def tearDown(self):
        self.store.close()
        # Native history manager owns a SQLite connection; release it for Windows.
        conn=getattr(self.store.native.db,'connection',None)
        if conn:conn.close()
        self.temp.cleanup()
    def test_real_sdk_add_search_idempotence_and_isolation(self):
        self.store.add(self.payload);before=len(self.t.calls)
        self.assertTrue(self.store.add(self.payload)['success']);self.assertEqual(len(self.t.calls),before)
        result=self.store.search(dict(user_id='alice',query='teal notebook',top_k=100))
        self.assertTrue(result['data']);self.assertIn('teal notebook',result['data'][0]['content'])
        self.assertEqual(self.store.search(dict(user_id='other',query='notebook',top_k=100)),{'data':[]})
        with self.assertRaises(ContractError):self.store.add(dict(self.payload,messages=[dict(role='user',content='changed')]))
        self.assertNotEqual(native_scope(' alice '),native_scope('alice'))
        self.assertEqual(self.c.calls,0)
    def test_unknown_sdk_fallback_never_replays(self):
        self.e.fail=True
        with self.assertRaises(Exception):self.store.add(self.payload)
        calls=len(self.e.calls)
        with self.assertRaises(ContractError):self.store.add(self.payload)
        with self.assertRaises(ContractError):self.store.search(dict(user_id='alice',query='notebook',top_k=100))
        self.assertEqual(len(self.e.calls),calls)

    def test_restart_preserves_native_and_original_stores(self):
        self.store.add(self.payload);self.store.close()
        self.store=Mem0System(self.path,embedding=self.e,transport=self.t,card_adapter=self.c)
        before=len(self.t.calls);self.store.add(self.payload)
        self.assertEqual(before,len(self.t.calls))
        self.assertTrue(self.store.search(dict(user_id='alice',query='teal notebook',top_k=100))['data'])
    def test_lossless_unicode_chunks(self):
        original='一二三 abc '*5000
        parts=list(utf8_chunks(original,8000));self.assertEqual(''.join(parts),original)
        self.assertTrue(all(len(x.encode())<=8000 for x in parts))
        vectors=self.store.native.embedding_model.embed_batch([original])
        self.assertEqual(len(vectors),1);self.assertEqual(len(vectors[0]),256)
        self.assertTrue(all(len(t.encode())<=8000 for batch in self.e.calls for t in batch))
    def test_search_image_gets_semantic_caption(self):
        self.store.add(self.payload)
        from base64 import b64encode
        import io
        from PIL import Image
        buffer=io.BytesIO();Image.new('RGB',(40,30),'teal').save(buffer,format='PNG')
        uri='data:image/png;base64,'+b64encode(buffer.getvalue()).decode()
        result=self.store.search(dict(user_id='alice',query=[{'type':'image_url','image_url':{'url':uri}}],top_k=100))
        self.assertEqual(self.c.calls,1);self.assertTrue(result['data'])
