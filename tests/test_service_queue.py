"""Concurrent admission checks with synthetic local storage and no providers."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from agent_memory.service_app import ServiceApplication
from agent_memory.store import MemoryStore


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MemoryStore(Path(self.temp.name) / 'memory.sqlite')
        self.transport = Mock()
        self.app = ServiceApplication(self.store, self.transport, 'synthetic-token',
            scorer_health=lambda: None, admission_capacity=16, queue_wait_seconds=2)

    def call(self, path='/search', auth=True, payload=None):
        raw = json.dumps(payload or {'user_id':'synthetic-user','query':'synthetic','top_k':1}).encode()
        env = {'PATH_INFO':path,'REQUEST_METHOD':'GET' if path=='/health' else 'POST',
               'CONTENT_TYPE':'application/json','CONTENT_LENGTH':str(len(raw)), 'wsgi.input':io.BytesIO(raw)}
        if auth: env['HTTP_AUTHORIZATION']='Bearer synthetic-token'
        response=[]
        body=b''.join(self.app(env,lambda status,headers:response.append((status,dict(headers)))))
        return response[0],json.loads(body)

    def test_sixteen_simultaneous_requests_finish_without_overlapping_model_work(self):
        gate=threading.Barrier(16)
        lock=threading.Lock()
        active=peak=completed=0
        def search(payload):
            nonlocal active,peak,completed
            with lock: active+=1;peak=max(peak,active)
            time.sleep(.01)
            with lock: active-=1;completed+=1
            return {'data':[]}
        self.store.search=search
        def run():gate.wait(timeout=5);return self.call()
        with ThreadPoolExecutor(max_workers=16) as pool:
            results=list(pool.map(lambda _:run(),range(16)))
        self.assertTrue(all(r[0][0].startswith('200') for r in results))
        self.assertEqual((peak,completed),(1,16))

    def test_full_admission_rejects_before_readiness_or_payload_and_health_stays_available(self):
        for _ in range(16):self.assertTrue(self.app.admissions.acquire(False))
        try:
            meta,_=self.call()
            self.assertTrue(meta[0].startswith('429'));self.assertEqual(meta[1]['Retry-After'],'5')
            self.transport.ready.assert_not_called()
            self.assertTrue(self.call(auth=False)[0][0].startswith('401'))
            self.assertTrue(self.call('/health')[0][0].startswith('200'))
        finally:
            for _ in range(16):self.app.admissions.release()

    def test_queue_timeout_releases_admission_without_dispatch(self):
        self.app.queue_wait_seconds=.01;self.app.active.acquire()
        try:self.assertTrue(self.call()[0][0].startswith('429'))
        finally:self.app.active.release()
        self.transport.ready.assert_not_called()
        self.assertTrue(self.call()[0][0].startswith('200'))
        for _ in range(16):self.assertTrue(self.app.admissions.acquire(False))
        self.assertFalse(self.app.admissions.acquire(False))

    def test_lease_is_rechecked_after_waiting(self):
        self.app.active.acquire()
        entered=threading.Event()
        original=self.app.admissions.acquire
        def admission(*args,**kwargs):
            result=original(*args,**kwargs);entered.set();return result
        self.app.admissions.acquire=admission
        with ThreadPoolExecutor(max_workers=1) as pool:
            task=pool.submit(self.call)
            try:
                self.assertTrue(entered.wait(2));self.app.expires_at=1
            finally:self.app.active.release()
            self.assertTrue(task.result(timeout=2)[0][0].startswith('503'))
        self.transport.ready.assert_not_called()

    def test_failed_readiness_releases_both_locks_and_never_calls_store(self):
        self.transport.ready.side_effect=RuntimeError('private reason')
        self.store.search=Mock()
        meta,body=self.call();self.assertTrue(meta[0].startswith('503'))
        self.assertNotIn('private reason',str(body));self.store.search.assert_not_called()
        self.transport.ready.side_effect=None;self.store.search.return_value={'data':[]}
        self.assertTrue(self.call()[0][0].startswith('200'))

    def test_invalid_limits_rejected_before_startup(self):
        for capacity,wait in [(0,0),(17,0),(True,0),(16,31),(16,float('nan'))]:
            with self.subTest(capacity=capacity,wait=wait),self.assertRaises(ValueError):
                ServiceApplication(self.store,self.transport,'synthetic',admission_capacity=capacity,queue_wait_seconds=wait)


if __name__=='__main__':unittest.main()
