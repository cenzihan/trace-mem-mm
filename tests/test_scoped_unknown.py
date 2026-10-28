"""Offline checks for owner isolation; no official inputs or model calls."""
import copy
import json
import unittest

from agent_memory.scoped_unknown import owner_digest, request_owner, verify_unknown


class ScopedUnknownTests(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()
