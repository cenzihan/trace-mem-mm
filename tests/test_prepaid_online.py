import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, timezone
import unittest
from agent_memory.prepaid_online import PrepaidOnlineTransport, MODELS
from agent_memory.provider_http import HTTPReply, ProviderFailure

MODEL='text-embedding-v4'
ENDPOINT=MODELS[MODEL][0]

class HTTP:
    is_fake_http=True
    def __init__(self,result):self.result,self.calls=result,0
    def exchange(self,*args):
        self.calls+=1
        if isinstance(self.result,Exception):raise self.result
        return HTTPReply(200,json.dumps(self.result).encode())

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);now=datetime.now(timezone.utc)
        self.grant=dict(version=1,grant_id='a'*32,price_checked_at=now.isoformat(),expires_at=(now+timedelta(hours=2)).isoformat(),slots=[dict(ordinal=1,model=MODEL,parent_attempt_id='attempt_'+'b'*32,reserved_micro_cny=50000)])
        self.payload=dict(model=MODEL,input=['source'],dimensions=256,encoding_format='float')
        self.response=dict(model=MODEL,data=[dict(index=0,embedding=[1.0]+[0.0]*255)],usage=dict(total_tokens=2))
    def transport(self,result):
        raw=json.dumps(self.grant).encode();p=self.root/'grant.json';p.write_bytes(raw);h=HTTP(result)
        return PrepaidOnlineTransport(p,hashlib.sha256(raw).hexdigest(),self.root/'receipts',{MODEL:lambda:'SYNTHETIC_KEY_ONLY'},http=h,offline_test=True),h
    def test_success_exhaustion_and_restart(self):
        t,h=self.transport(self.response);self.assertEqual(t(ENDPOINT,self.payload),self.response)
        with self.assertRaises(ValueError):t(ENDPOINT,self.payload)
        self.assertEqual(h.calls,1)
        t,h=self.transport(self.response)
        with self.assertRaises(ValueError):t(ENDPOINT,self.payload)
        self.assertEqual(h.calls,0)
    def test_timeout_never_replayed(self):
        t,h=self.transport(TimeoutError())
        with self.assertRaises(TimeoutError):t(ENDPOINT,self.payload)
        with self.assertRaises(ValueError):t(ENDPOINT,self.payload)
        self.assertEqual(h.calls,1);self.assertEqual(t.summary()[0]['state'],'unknown')
    def test_receipt_before_schema_rejection(self):
        self.response['data'][0]['embedding']=[1]
        t,h=self.transport(self.response)
        with self.assertRaises(ProviderFailure):t(ENDPOINT,self.payload)
        self.assertEqual(t.summary()[0]['state'],'failed');self.assertTrue(t.summary()[0]['body_sha'])
        with self.assertRaises(ValueError):t(ENDPOINT,self.payload)
        self.assertEqual(h.calls,1)
    def test_modified_grant_blocks(self):
        t,h=self.transport(self.response);t.path.write_text('{}')
        with self.assertRaises(ValueError):t(ENDPOINT,self.payload)
        self.assertEqual(h.calls,0)
    def test_bounds_and_endpoint_before_credentials(self):
        t,h=self.transport(self.response);t.readers={MODEL:lambda:self.fail('credential read')}
        with self.assertRaises(ValueError):t('https://example.com',self.payload)
        self.payload['input']=['x'*8193]
        with self.assertRaises(ValueError):t(ENDPOINT,self.payload)
        self.assertEqual(h.calls,0)
    def test_usage_overflow_retains_receipt(self):
        self.response['usage']['total_tokens']=81921;t,h=self.transport(self.response)
        with self.assertRaises(ProviderFailure):t(ENDPOINT,self.payload)
        obs=json.loads(t.summary()[0]['online_observation']);self.assertEqual(obs['category'],'invalid_usage');self.assertEqual(obs['input_tokens'],81921)
    def test_no_hosted_rerank_substitution(self):
        t,h=self.transport(self.response)
        with self.assertRaises(ValueError):t(ENDPOINT,dict(model='qwen3-vl-rerank'))
        self.assertEqual(h.calls,0)

if __name__=='__main__':unittest.main()
