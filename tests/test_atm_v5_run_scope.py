"""Bound the single V5 run; old completed guards never get expanded."""
import tempfile
import unittest
from pathlib import Path

from agent_memory.development_run import DevelopmentRun


class V5ScopeTests(unittest.TestCase):
    def test_explicit_scope_is_bounded_and_existing_run_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.sqlite'
            run = DevelopmentRun(path, create=True, max_requests=1209,
                                 max_reservation_cny='60.45',
                                 approval_scope='atm_v5_embedding_1209_6045')
            ordinal = run.claim('text-embedding-v4', '.05')
            run.finish(ordinal, 'text-embedding-v4')
            self.assertEqual(DevelopmentRun(path).summary()['request_claims'], 1)
            with self.assertRaises(FileExistsError):
                DevelopmentRun(path, create=True, max_requests=1209,
                               max_reservation_cny='60.45',
                               approval_scope='atm_v5_embedding_1209_6045')

    def test_new_scope_cannot_raise_either_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            for requests, cap in ((1210, '60.45'), (1209, '60.46')):
                path = Path(temporary) / f'{requests}-{cap}.sqlite'
                with self.assertRaises(ValueError):
                    DevelopmentRun(path, create=True, max_requests=requests,
                                   max_reservation_cny=cap,
                                   approval_scope='atm_v5_embedding_1209_6045')
                self.assertFalse(path.exists())

    def test_old_scope_still_rejects_new_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                DevelopmentRun(Path(temporary) / 'old.sqlite', create=True,
                               max_requests=1209, max_reservation_cny='60.45',
                               approval_scope='native_100_20')


if __name__ == '__main__':
    unittest.main()
