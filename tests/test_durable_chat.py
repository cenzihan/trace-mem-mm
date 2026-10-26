import json
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.development_run import DevelopmentRun,ScopedTransport
from agent_memory.durable_chat import DurableChat,DurableChatBlocked
from agent_memory.provider_http import BudgetedJSONTransport,HTTPReply,OPENAI_ENDPOINT


class FakeHTTP:
    is_fake_http=True
    def __init__(self,fail=False): self.calls=0; self.fail=fail
    def exchange(self,*args):
        self.calls+=1
        if self.fail: raise TimeoutError()
        content=dict(summary='synthetic reader',evidence_quotes=[],answer='synthetic answer')
        return HTTPReply(200,json.dumps(dict(model='gpt-4o-mini-2024-07-18',
            choices=[dict(finish_reason='stop',message=dict(content=json.dumps(content)))],
            usage=dict(prompt_tokens=20,completion_tokens=20,total_tokens=40))).encode())


class DurableChatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='chat-receipt-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ledger=BudgetLedger(self.root/'budget.sqlite')
        self.run=DevelopmentRun(self.root/'run.sqlite',create=True,max_requests=10,max_reservation_cny='3')
        self.http=FakeHTTP()
        self.payload=dict(model='gpt-4o-mini',messages=[dict(role='system',content='Return JSON.'),dict(role='user',content='Synthetic task')],
                          max_completion_tokens=128,store=False,n=1,response_format=dict(type='json_object'))
    def executor(self,hook=None):
        def factory(http,observer):
            base=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=self.ledger,
                rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat()),
                http=http,mode='offline_fake',fixed_reserve_cny='.25',attempt_observer=observer)
            return ScopedTransport(base,self.run,'.25')
        return DurableChat(self.root/'receipts','frozen-test',self.http,factory,self.run,offline_test=True,after_received=hook)
    def test_success_restart_uses_same_receipt_without_charge(self):
        first=self.executor().execute('q1-dense',self.payload)
        second=self.executor().execute('q1-dense',self.payload)
        self.assertEqual(first['receipt_sha256'],second['receipt_sha256'])
        self.assertTrue(second['recovered']);self.assertEqual(self.http.calls,1)
        self.assertEqual(self.ledger.totals()['attempts'],1)
    def test_changed_payload_same_id_cannot_replay(self):
        self.executor().execute('q1-dense',self.payload)
        changed={**self.payload,'messages':[dict(role='system',content='Return JSON.'),dict(role='user',content='Changed synthetic task')]}
        self.assertRaises(RuntimeError,self.executor().execute,'q1-dense',changed)
        self.assertEqual(self.http.calls,1)
    def test_timeout_blocks_retry(self):
        self.http.fail=True
        self.assertRaises(Exception,self.executor().execute,'q1',self.payload)
        self.assertRaises(DurableChatBlocked,self.executor().execute,'q1',self.payload)
        self.assertEqual(self.http.calls,1)
    def test_crash_after_receipt_does_not_resend_or_settle(self):
        def crash(): raise KeyboardInterrupt()
        self.assertRaises(KeyboardInterrupt,self.executor(crash).execute,'q1',self.payload)
        self.assertRaises(DurableChatBlocked,self.executor().execute,'q1',self.payload)
        self.assertEqual(self.http.calls,1)
        with self.run.connect() as db:
            self.assertEqual(db.execute('SELECT state FROM calls').fetchone()[0],'claimed')
    def test_receipt_tamper_detected(self):
        executor=self.executor();result=executor.execute('q1',self.payload)
        with executor.journal.connect() as db:
            db.execute('UPDATE operations SET body=? WHERE fingerprint=?',(b'{}',result['fingerprint']))
        self.assertRaises(DurableChatBlocked,self.executor().execute,'q1',self.payload)
        self.assertEqual(self.http.calls,1)
    def test_missing_read_only_recovery_never_claims_or_sends(self):
        self.assertRaises(DurableChatBlocked,self.executor().recover,'missing',self.payload)
        self.assertEqual(self.http.calls,0)
        self.assertEqual(self.run.summary()['request_claims'],0)


if __name__=='__main__': unittest.main()
