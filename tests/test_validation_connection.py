import sqlite3
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from test_gpu6_app import GPU6Tests
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT
from agent_memory.scoped_unknown import request_owner


class ValidationConnectionTests(unittest.TestCase):
    """Accounting stays live; only connection setup is amortized."""
    setUp = GPU6Tests.setUp
    make = GPU6Tests.make

    def seed_failure(self, t):
        with request_owner('synthetic-seed'):
            t(BAILIAN_BEIJING_ENDPOINT, self.payload)
        with t.journal.connect() as db:
            row = dict(db.execute('SELECT * FROM operations').fetchone())
            db.execute("UPDATE operations SET state='unknown' WHERE fingerprint=?", (row['fingerprint'],))
        with t.ledger.connect() as db:
            db.execute("UPDATE attempts SET status='uncertain',accounted=reserved WHERE id=?", (row['attempt_id'],))
            db.execute("UPDATE provider_observations SET category='client_error',http_status=400 WHERE attempt_id=?", (row['attempt_id'],))
        with t.journal.connect() as db:
            return dict(db.execute('SELECT * FROM operations').fetchone())

    def test_reused_reader_observes_accounting_changes_without_caching(self):
        t = self.make();row = self.seed_failure(t)
        with t.ledger.validation_scope():
            with t.ledger.validation_connect() as first:
                self.assertTrue(t._accounted_terminal_failure(row))
                with t.ledger.validation_scope(), t.ledger.validation_connect() as second:
                    self.assertIs(first, second)
                with t.ledger.connect() as writer:
                    writer.execute('UPDATE attempts SET accounted=0 WHERE id=?', (row['attempt_id'],))
                self.assertFalse(t._accounted_terminal_failure(row))
                with self.assertRaises(sqlite3.OperationalError):
                    first.execute('UPDATE attempts SET accounted=0')
        with self.assertRaises(sqlite3.ProgrammingError):first.execute('SELECT 1')
        self.assertIsNone(t.ledger._validation_db.get())

    def test_exception_closes_reader_and_other_threads_get_distinct_connections(self):
        t = self.make();barrier = threading.Barrier(2)
        def read(_):
            with t.ledger.validation_scope(), t.ledger.validation_connect() as db:
                barrier.wait(5)
                return id(db)
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(read, range(2)))
        self.assertEqual(len(set(ids)), 2)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with t.ledger.validation_scope(), t.ledger.validation_connect() as db:
                raise RuntimeError('test')
        with self.assertRaises(sqlite3.ProgrammingError):db.execute('SELECT 1')
        self.assertIsNone(t.ledger._validation_db.get())

    def test_ready_and_dispatch_reject_released_or_removed_accounting(self):
        t = self.make();row = self.seed_failure(t)
        t.ready()
        with t.ledger.connect() as db:
            original = db.execute('SELECT accounted FROM attempts WHERE id=?',(row['attempt_id'],)).fetchone()[0]
            db.execute('UPDATE attempts SET accounted=0 WHERE id=?', (row['attempt_id'],))
        with self.assertRaises(ValueError):t.ready()
        with t.ledger.connect() as db:
            db.execute('UPDATE attempts SET accounted=? WHERE id=?', (original,row['attempt_id']))
        t.ready()
        with t.ledger.connect() as db:
            db.execute('DELETE FROM provider_observations WHERE attempt_id=?', (row['attempt_id'],))
        with self.assertRaises(ValueError):t.ready()
        before = t.ledger.totals()['attempts']
        with request_owner('another-synthetic'), self.assertRaises(ValueError):
            t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        self.assertEqual(t.ledger.totals()['attempts'], before)
        self.assertIsNone(t.ledger._validation_db.get())


if __name__ == '__main__':unittest.main()
