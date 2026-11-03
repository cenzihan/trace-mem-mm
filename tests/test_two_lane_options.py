import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import test_service_transport as fixture
import test_service_queue as queue_fixture
from agent_memory.two_lane_options import TwoLaneRetryTransport
from agent_memory.service_app import ServiceApplication
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT
from agent_memory.scoped_unknown import request_owner


class TwoLaneTests(unittest.TestCase):
    setUp = fixture.ServiceTransportTests.setUp

    def make(self, factory=fixture.FakeHTTP):
        t = TwoLaneRetryTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},
            http=self.http,http_factory=factory,offline_test=True)
        t.retry_delays=(0,)*5
        self.addCleanup(t.close)
        return t

    def call(self,t):
        with request_owner('synthetic'):return t(BAILIAN_BEIJING_ENDPOINT,self.payload)

    def test_only_two_model_calls_overlap(self):
        gate=threading.Barrier(2)
        active=peak=0
        lock=threading.Lock()
        class HTTP(fixture.FakeHTTP):
            def exchange(self,*args):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                try:
                    gate.wait(5)
                    return super().exchange(*args)
                finally:
                    with lock:active-=1
        t=self.make(HTTP)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in [pool.submit(self.call,t) for _ in range(8)]:f.result(20)
        self.assertEqual(peak,2)
        self.assertEqual(self.ledger.totals()['attempts'],8)

    def test_exclusive_worker_rejects_second_owner(self):
        t=self.make()
        with self.assertRaises(RuntimeError):self.make()
        self.call(t)
        self.assertEqual(self.ledger.totals()['attempts'],1)

    def test_restart_acknowledges_pristine_ready_without_erasing_row(self):
        t=self.make();t.journal.install('synthetic-ready',t.receipt_scope,{'keys':['synthetic-ready']})
        old=t.journal.row('synthetic-ready')
        with self.assertRaises(ValueError):t.ready() # cannot adopt after serving starts
        t.close();new=self.make();new.ready()
        self.assertEqual(old,new.journal.row('synthetic-ready'))
        self.call(new)
        self.assertEqual(self.ledger.totals()['attempts'],1)

    def test_killed_dispatch_keeps_full_charge_and_allows_fresh_call(self):
        t=self.make()
        # Create a durable reservation and a dispatch without pretending it returned.
        from agent_memory.budget import VerifiedRate
        rate=VerifiedRate(model='text-embedding-v4',checked_at=self.config['checked_at'],
                          **self.config['models']['text-embedding-v4'])
        attempt=self.ledger.reserve(rate,8192)
        self.ledger.mark_dispatched(attempt)
        t.journal.install('synthetic-inflight',t.receipt_scope,{'keys':['synthetic-inflight']})
        t.journal.start('synthetic-inflight');t.journal.attach('synthetic-inflight',attempt)
        old=t.journal.row('synthetic-inflight');before=self.ledger.totals()
        t.close();new=self.make();new.ready()
        self.assertEqual(old,new.journal.row('synthetic-inflight'))
        self.assertEqual(before,self.ledger.totals())
        self.call(new)
        with self.ledger.connect() as db:
            self.assertEqual(db.execute('SELECT status,reserved,accounted FROM attempts WHERE id=?',(attempt,)).fetchone(),('dispatched',4096,4096))
            self.assertIsNone(db.execute('SELECT 1 FROM provider_observations WHERE attempt_id=?',(attempt,)).fetchone())
        new.close();again=self.make();again.ready()
        self.assertEqual(old,again.journal.row('synthetic-inflight'))
        self.assertEqual(self.ledger.totals()['attempts'],2)

    def test_recovery_cannot_cover_new_unowned_operation_or_changed_receipt(self):
        t=self.make();t.journal.install('before',t.receipt_scope,{'keys':['before']});t.close()
        new=self.make();new.ready()
        new.journal.install('after',new.receipt_scope,{'keys':['after']})
        with self.assertRaises(ValueError):new.ready()
        with new.journal.connect() as db:
            db.execute("UPDATE operations SET state='failed' WHERE fingerprint='before'")
        with self.assertRaises(ValueError):new.ready()


class TwoLaneAdmissionTests(unittest.TestCase):
    setUp=queue_fixture.QueueTests.setUp
    call=queue_fixture.QueueTests.call

    def test_sixteen_requests_complete_with_two_execution_slots(self):
        self.app=ServiceApplication(self.store,self.transport,'synthetic-token',scorer_health=lambda:None,
            admission_capacity=16,execution_capacity=2,queue_wait_seconds=120)
        barrier=threading.Barrier(2)
        def search(payload):
            barrier.wait(8)
            return {'data':[]}
        self.store.search=search
        with ThreadPoolExecutor(max_workers=16) as pool:
            result=[f.result(20) for f in [pool.submit(self.call) for _ in range(16)]]
        self.assertTrue(all(r[0][0].startswith('200') for r in result))


if __name__=='__main__':unittest.main()
