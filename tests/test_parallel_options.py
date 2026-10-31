import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import test_service_transport as fixture
from test_mem0_native import FakeEmbedding, FakeTransport, FakeCards
from agent_memory.parallel_options import ParallelRetryTransport, ParallelOptionsSystem
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, ProviderFailure
from agent_memory.scoped_unknown import request_owner
from agent_memory.protocol import ContractError
from agent_memory.service_app import ServiceApplication
import test_service_queue as queue_fixture


class ParallelAdmissionTests(unittest.TestCase):
    setUp = queue_fixture.QueueTests.setUp
    call = queue_fixture.QueueTests.call

    def test_sixteen_http_requests_execute_four_at_a_time(self):
        self.app = ServiceApplication(self.store, self.transport, 'synthetic-token',
            scorer_health=lambda:None, admission_capacity=16, execution_capacity=4,
            queue_wait_seconds=120)
        barrier = threading.Barrier(4)
        def search(payload):
            barrier.wait(timeout=8)
            return {'data':[]}
        self.store.search = search
        with ThreadPoolExecutor(max_workers=16) as pool:
            fs = [pool.submit(self.call) for _ in range(16)]
            results = [f.result(timeout=20) for f in fs]
        self.assertTrue(all(r[0][0].startswith('200') for r in results))

    def test_invalid_parallel_capacity_and_unbounded_wait_are_rejected(self):
        for capacity, wait in [(0,0),(17,0),(True,0),('four',0),(4,121)]:
            with self.assertRaises(ValueError):
                ServiceApplication(self.store, self.transport, 'synthetic-token',
                    admission_capacity=16, execution_capacity=capacity, queue_wait_seconds=wait)


class ParallelTransportTests(unittest.TestCase):
    setUp = fixture.ServiceTransportTests.setUp

    def make(self, factory=fixture.FakeHTTP):
        t = ParallelRetryTransport(self.ledger.path, self.authority, self.prices,
            self.p/'receipts', {}, http=self.http, http_factory=factory, offline_test=True)
        t.retry_delays = (0,)*5
        return t

    def call(self, t, owner):
        with request_owner(owner):return t(BAILIAN_BEIJING_ENDPOINT, self.payload)

    def test_four_actual_http_calls_overlap_and_receipts_keep_owners(self):
        barrier = threading.Barrier(4)
        class HTTP(fixture.FakeHTTP):
            def exchange(self, *args):
                barrier.wait(timeout=8)
                # Normal owned in-flight work must not turn health into 503.
                t.ready()
                return super().exchange(*args)
        t = self.make(HTTP)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.call, t, str(i)) for i in range(8)]
            for future in futures:future.result(timeout=20)
        self.assertEqual(self.ledger.totals()['attempts'], 8)
        with t.journal.connect() as db:
            rows = db.execute('SELECT state,manifest FROM operations').fetchall()
        self.assertEqual({r[0] for r in rows}, {'complete'})
        self.assertEqual(len({json.loads(r[1])['owner_sha256'] for r in rows}), 8)
        self.assertFalse(t._live)
        self.assertEqual(t._lanes.qsize(), 4)

    def test_orphan_is_not_mistaken_for_live_process_work(self):
        t = self.make()
        t.journal.install('orphan', t.receipt_scope, {'keys':['orphan']})
        with self.assertRaises(ValueError):t.ready()
        with self.assertRaises(ValueError):self.call(t, 'new')
        self.assertEqual(self.ledger.totals()['attempts'], 0)
        self.assertEqual(t._lanes.qsize(), 4)

    def test_all_busy_lanes_have_bounded_wait_without_a_bill(self):
        t = self.make();t.lane_wait_seconds = .01
        lanes = [t._lanes.get() for _ in range(4)]
        try:
            with self.assertRaises(ContractError) as raised:self.call(t, 'new')
            self.assertEqual(raised.exception.status, 429)
            self.assertEqual(self.ledger.totals()['attempts'], 0)
        finally:
            for lane in lanes:t._lanes.put(lane)

    def test_unfinished_worker_keeps_its_lane_after_caller_exits(self):
        clients = []
        class HTTP(fixture.FakeHTTP):
            def __init__(self):
                super().__init__();self._slot = threading.Lock();clients.append(self)
            def exchange(self, *args):
                self._slot.acquire()
                raise ProviderFailure('authentication')  # no retry; simulate late worker
        t = self.make(HTTP)
        try:
            with self.assertRaises(ProviderFailure):self.call(t, 'first')
            self.assertEqual(t._lanes.qsize(), 3)
        finally:
            clients[0]._slot.release()
        # Queue.get synchronizes with the drain thread without a timing assertion.
        held = [t._lanes.get(timeout=3) for _ in range(4)]
        for lane in held:t._lanes.put(lane)
        self.assertEqual(self.ledger.totals()['attempts'], 1)

    def test_one_failed_lane_retries_without_poisoning_other_lanes(self):
        barrier = threading.Barrier(4)
        class HTTP(fixture.FakeHTTP):
            def exchange(self, *args):
                if self.calls == 0:
                    barrier.wait(timeout=8)
                    self.calls += 1
                    raise ProviderFailure('timeout')
                return super().exchange(*args)
        t = self.make(HTTP)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.call, t, str(i)) for i in range(4)]
            for f in futures:f.result(timeout=20)
        with t.journal.connect() as db:
            states = [r[0] for r in db.execute('SELECT state FROM operations')]
        self.assertEqual(states.count('unknown'), 4)
        self.assertEqual(states.count('complete'), 4)
        with self.ledger.connect() as db:
            failed = db.execute("SELECT accounted,reserved FROM attempts WHERE status!='settled'").fetchall()
        self.assertTrue(all(a == r for a,r in failed))
        t.ready()


@unittest.skipUnless(importlib.util.find_spec('mem0'), 'pinned Mem0 required')
class ParallelNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.embedding = FakeEmbedding()
        self.store = ParallelOptionsSystem(Path(self.temp.name)/'memory.sqlite',
            embedding=self.embedding, transport=FakeTransport(), card_adapter=FakeCards())

    def tearDown(self):
        self.store.close();self.store.native.db.close();self.temp.cleanup()

    def payload(self, request, user='u'):
        return dict(user_id=user, request_id=request, session_id='s',
                    messages=[dict(role='user', content='Synthetic notebook '+request)])

    def test_same_user_adds_embed_in_parallel_and_commit_once_each(self):
        barrier = threading.Barrier(4)
        original = self.embedding.embed
        def embed(texts):
            barrier.wait(timeout=8)
            return original(texts)
        with patch.object(self.embedding, 'embed', side_effect=embed):
            with ThreadPoolExecutor(max_workers=4) as pool:
                fs = [pool.submit(self.store.add, self.payload(str(i))) for i in range(4)]
                for f in fs:f.result(timeout=20)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT count(*) FROM add_operations WHERE state='complete'").fetchone()[0], 4)
        vs = self.store.native.vector_store
        self.assertEqual(vs.client.count(collection_name=vs.collection_name).count, 4)
        self.assertEqual(len(self.embedding.calls), 4)
        self.assertFalse(self.store._operations.entries)

    def test_concurrent_duplicate_add_is_idempotent_and_conflict_unbilled(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(self.store.add, [self.payload('same')]*4))
        self.assertTrue(all(a == answers[0] for a in answers))
        self.assertEqual(len(self.embedding.calls), 1)
        changed = self.payload('same');changed['messages'][0]['content']='Different synthetic text'
        with self.assertRaises(ContractError):self.store.add(changed)
        self.assertEqual(len(self.embedding.calls), 1)

    def test_failed_preparation_does_not_write_or_poison_other_request(self):
        barrier = threading.Barrier(2);original = self.embedding.embed
        def embed(texts):
            barrier.wait(timeout=8)
            if 'bad' in texts[0]:raise RuntimeError('synthetic unavailable')
            return original(texts)
        with patch.object(self.embedding, 'embed', side_effect=embed):
            with ThreadPoolExecutor(max_workers=2) as pool:
                bad = pool.submit(self.store.add, self.payload('bad'))
                good = pool.submit(self.store.add, self.payload('good'))
                with self.assertRaises(RuntimeError):bad.result(timeout=15)
                good.result(timeout=15)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0], 1)
        self.store.ready();self.store.add(self.payload('bad'))

    def test_search_model_selection_overlaps_with_snapshot_isolation(self):
        self.store.add(self.payload('alpha', 'a'))
        self.store.add(self.payload('beta', 'b'))
        barrier = threading.Barrier(2)
        seen = [];guard = threading.Lock()
        def select(payload, parts, sources):
            with guard:seen.append((payload['user_id'], json.dumps(sources)))
            barrier.wait(timeout=8)
            return [0]
        with patch.object(self.store, '_select_sources', side_effect=select):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs = [pool.submit(self.store.search, dict(user_id=u, query='Synthetic notebook', top_k=100)) for u in ('a','b')]
                results = [f.result(timeout=20) for f in fs]
        self.assertTrue(all(r['data'] for r in results))
        for user, sources in seen:
            self.assertIn('alpha' if user == 'a' else 'beta', sources)
            self.assertNotIn('beta' if user == 'a' else 'alpha', sources)

    def test_commit_failure_cleans_only_uncommitted_vectors(self):
        self.store.add(self.payload('saved'))
        with patch.object(self.store, '_persist_descriptions', side_effect=RuntimeError('synthetic commit failure')):
            with self.assertRaises(RuntimeError):self.store.add(self.payload('failed'))
        vs = self.store.native.vector_store
        self.assertEqual(vs.client.count(collection_name=vs.collection_name).count, 1)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0], 1)
        self.store.add(self.payload('failed'))
