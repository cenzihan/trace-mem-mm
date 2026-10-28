"""Offline checks for owner isolation; no official inputs or model calls."""
import copy
import json
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from agent_memory.scoped_unknown import bound_request, owner_digest, request_owner, verify_unknown


class ScopedUnknownTests(unittest.TestCase):
    def test_parallel_owners_do_not_bleed_between_threads(self):
        barrier=Barrier(2)
        def work(uid):
            with request_owner(uid):
                before=owner_digest();barrier.wait(timeout=3);return before,owner_digest()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(work,'user-a');b=pool.submit(work,'user-b');av=a.result();bv=b.result()
        self.assertEqual(av[0],av[1]);self.assertEqual(bv[0],bv[1]);self.assertNotEqual(av[0],bv[0])

    def test_context_is_nested_and_restored_on_exception(self):
        with self.assertRaises(ValueError): owner_digest()
        with request_owner('user-a'):
            outer = owner_digest()
            with self.assertRaises(RuntimeError):
                with request_owner('user-b'):
                    self.assertNotEqual(owner_digest(), outer)
                    raise RuntimeError('synthetic')
            self.assertEqual(owner_digest(), outer)
        with self.assertRaises(ValueError): owner_digest()

    def test_only_bound_unknown_with_full_reserve_and_observation_is_accepted(self):
        row = dict(state='unknown', status=None, body=None, transport_completed=0,
                   attempt_id='a', manifest=json.dumps(dict(owner_sha256='a'*64, payload_sha256='b'*64)))
        attempt = dict(id='a', status='uncertain', reserved=4096, accounted=4096)
        observation = dict(attempt_id='a', http_status=None, category='network_error')
        self.assertEqual(verify_unknown(row, attempt, observation), ('a'*64, 'b'*64))
        for target, key, value in ((row,'manifest','{}'), (row,'transport_completed',1),
                                   (attempt,'accounted',0), (attempt,'id','b'),
                                   (observation,'http_status',200), (observation,'category','success')):
            r,a,o = copy.deepcopy((row,attempt,observation))
            chosen = r if target is row else a if target is attempt else o
            chosen[key] = value
            with self.assertRaises(ValueError): verify_unknown(r,a,o)

    def test_sdk_failure_latch_clears_only_for_proven_scoped_unknown(self):
        class Transport:
            scoped_unknown_enabled=True
            proof=False
            def has_scoped_unknown(self, owner, *, model=None):
                self.last=(owner,model);return self.proof
            def ready(self): pass
        class Store:
            transport=Transport()
            native=SimpleNamespace(embedding_model=SimpleNamespace(failed=True))
            @bound_request
            def add(self,payload): raise RuntimeError('synthetic')
        store=Store()
        with self.assertRaises(RuntimeError):store.add({'user_id':'u'})
        self.assertTrue(store.native.embedding_model.failed)
        store.transport.proof=True
        with self.assertRaises(RuntimeError):store.add({'user_id':'u'})
        self.assertFalse(store.native.embedding_model.failed)
        self.assertEqual(store.transport.last[1],'text-embedding-v4')

    def test_readiness_failure_never_clears_sdk_latch(self):
        class Transport:
            scoped_unknown_enabled=True
            def has_scoped_unknown(self, owner, *, model=None): return True
            def ready(self): raise ValueError('unscoped failure')
        class Store:
            transport=Transport()
            native=SimpleNamespace(embedding_model=SimpleNamespace(failed=True))
            @bound_request
            def add(self,payload): raise RuntimeError('synthetic')
        store=Store()
        with self.assertRaises(RuntimeError):store.add({'user_id':'u'})
        self.assertTrue(store.native.embedding_model.failed)


if __name__ == '__main__': unittest.main()
