"""Durable embedding dispatch with temporary synthetic data and fake HTTP only."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.development_run import DevelopmentRun, ScopedTransport
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache
from agent_memory.embedding_executor import DurableEmbeddingExecutor, EmbeddingExecutorBlocked
from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply, ProviderFailure, BAILIAN_BEIJING_ENDPOINT, RetryPolicy


ROOT = Path(__file__).resolve().parents[1]


def response(count=2):
    return dict(model='text-embedding-v4', usage=dict(total_tokens=count * 3),
                data=[dict(index=i, embedding=[float(i + 1)] + [0.0] * 63) for i in reversed(range(count))])


class FakeHTTP:
    is_fake_http = True

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = 0

    def exchange(self, endpoint, body, headers, timeout, limit):
        self.calls += 1
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply or HTTPReply(200, json.dumps(response(len(json.loads(body)['input']))).encode())


def factory(ledger, run=None):
    def build(http, observer):
        transport = BudgetedJSONTransport(model='text-embedding-v4', endpoint=BAILIAN_BEIJING_ENDPOINT,
            ledger=ledger, rate=VerifiedRate('text-embedding-v4', '.5', '0', 'synthetic',
            datetime.now(timezone.utc).isoformat()), http=http, mode='offline_fake',
            fixed_reserve_cny='.05', attempt_observer=observer,
            credential_reader=lambda: (_ for _ in ()).throw(AssertionError('no credentials in fake tests')))
        return ScopedTransport(transport, run, '.05') if run else transport
    return build


def setup_at(root):
    scope = CacheScope('synthetic-user', 'synthetic-revision', 'executor-test', dimensions=64)
    cache = SourceEmbeddingCache(root / 'cache.sqlite', scope, offline_test=True)
    cache.prepare_sources([dict(id='a', text='synthetic original alpha'), dict(id='b', text='synthetic original beta')])
    ledger = BudgetLedger(root / 'budget.sqlite')
    return cache, ledger


class EmbeddingExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='embedding-executor-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache, self.ledger = setup_at(self.root)
        self.batch = self.cache.missing_batches()[0]

    def executor(self, http, hook=None, build=None):
        return DurableEmbeddingExecutor(self.cache, self.root / 'journal', http, build or factory(self.ledger),
                                        offline_test=True, after_received=hook)

    def test_success_bound_receipt_and_restart_reuses_cache(self):
        http = FakeHTTP()
        first = self.executor(http).execute(self.batch)
        self.assertEqual(http.calls, 1)
        self.assertFalse(first['run_accounting_check_required'])
        self.assertEqual(self.cache.missing_batches(), [])
        snapshot = self.cache.publish()
        proof = snapshot['chunks'][0]['provenance']
        self.assertEqual(proof['budget_attempt_id'], first['attempt_id'])
        self.assertEqual(proof['receipt_sha256'], first['receipt_sha256'])
        self.assertEqual([chunk['vector'][0] for chunk in snapshot['chunks']], [1.0, 2.0])
        def forbidden(*args):
            self.fail('cached batch must not build transport or reserve')
        second = self.executor(http, build=forbidden).execute(self.batch)
        self.assertTrue(second['recovered'])
        self.assertFalse(second['run_accounting_check_required'])
        self.assertEqual(self.ledger.totals()['attempts'], 1)

    def test_receipt_precedes_parser_and_bad_response_never_replays(self):
        bad = response()
        bad['data'].pop()
        http = FakeHTTP(HTTPReply(200, json.dumps(bad).encode()))
        executor = self.executor(http)
        self.assertRaises(ProviderFailure, executor.execute, self.batch)
        fingerprint = executor._prepare(self.batch)[0]
        row = executor.journal.row(fingerprint)
        self.assertEqual(row['state'], 'received')
        self.assertEqual(hashlib.sha256(row['body']).hexdigest(), row['body_sha'])
        self.assertRaises(EmbeddingExecutorBlocked, executor.execute, self.batch)
        self.assertEqual(executor.journal.row(fingerprint)['state'], 'invalid_response')
        self.assertEqual(http.calls, 1)

    def test_unknown_and_repartition_never_redispatch(self):
        http = FakeHTTP(TimeoutError())
        executor = self.executor(http)
        self.assertRaises(ProviderFailure, executor.execute, self.batch)
        self.assertRaises(EmbeddingExecutorBlocked, executor.execute, self.batch)
        self.assertRaises(EmbeddingExecutorBlocked, executor.execute, list(reversed(self.batch)))
        self.assertRaises(EmbeddingExecutorBlocked, executor.execute, self.batch[:1])
        self.assertEqual(http.calls, 1)
        self.assertEqual(self.ledger.totals()['attempts'], 1)

    def test_receipt_recovery_does_not_complete_interrupted_run_claim(self):
        run = DevelopmentRun(self.root / 'run.sqlite', create=True, max_requests=10, max_reservation_cny='1')
        http = FakeHTTP()
        def interrupt():
            raise KeyboardInterrupt()
        executor = self.executor(http, hook=interrupt, build=factory(self.ledger, run))
        self.assertRaises(KeyboardInterrupt, executor.execute, self.batch)
        recovered = self.executor(http, build=factory(self.ledger, run)).execute(self.batch)
        self.assertTrue(recovered['run_accounting_check_required'])
        self.assertEqual(recovered['run_path'], str((self.root / 'run.sqlite').resolve()))
        self.assertEqual(recovered['run_ordinal'], 1)
        self.assertEqual(self.cache.missing_batches(), [])
        with run.connect() as db:
            self.assertEqual(db.execute('SELECT state FROM calls').fetchone()[0], 'claimed')
        with self.ledger.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM provider_observations').fetchone()[0], 0)
        self.assertEqual(http.calls, 1)

    def test_successful_scoped_fake_binds_the_exact_guard_claim(self):
        run = DevelopmentRun(self.root / 'run.sqlite', create=True, max_requests=10, max_reservation_cny='1')
        http = FakeHTTP()
        executor = self.executor(http, build=factory(self.ledger, run))
        result = executor.execute(self.batch)
        self.assertEqual(result['run_path'], str((self.root / 'run.sqlite').resolve()))
        self.assertEqual(result['run_ordinal'], 1)
        self.assertFalse(result['run_accounting_check_required'])
        row = executor.journal.row(result['fingerprint'])
        self.assertEqual(row['attempt_id'], result['attempt_id'])
        self.assertEqual(row['run_ordinal'], 1)
        with run.connect() as db:
            self.assertEqual(db.execute('SELECT state FROM calls WHERE ordinal=1').fetchone()[0], 'complete')
        self.assertEqual(self.ledger.totals()['attempts'], 1)

    def test_child_exit_after_receipt_can_import_without_http(self):
        child = self.root / 'child'
        child.mkdir()
        code = '''import os,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_embedding_executor import *
r=Path(sys.argv[1]); c,l=setup_at(r)
DurableEmbeddingExecutor(c,r/'journal',FakeHTTP(),factory(l),offline_test=True,
    after_received=lambda:os._exit(43)).execute(c.missing_batches()[0])
'''
        completed = subprocess.run([sys.executable, '-B', '-c', code, str(child)], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        self.assertEqual(completed.returncode, 43)
        cache, ledger = setup_at(child)
        http = FakeHTTP()
        result = DurableEmbeddingExecutor(cache, child / 'journal', http, factory(ledger), offline_test=True).execute(cache.missing_batches()[0])
        self.assertTrue(result['run_accounting_check_required'])
        self.assertEqual(http.calls, 0)
        self.assertEqual(ledger.totals()['attempts'], 1)

    def test_cache_write_failure_recovers_receipt_without_new_request(self):
        http = FakeHTTP()
        executor = self.executor(http)
        with patch.object(self.cache, 'import_batch', side_effect=OSError('synthetic disk failure')):
            self.assertRaises(OSError, executor.execute, self.batch)
        recovered = self.executor(http).execute(self.batch)
        self.assertFalse(recovered['run_accounting_check_required'])
        self.assertEqual(http.calls, 1)
        self.assertEqual(self.cache.missing_batches(), [])

    def test_receipt_write_failure_becomes_unknown_not_replayable(self):
        http = FakeHTTP()
        executor = self.executor(http)
        with patch.object(executor.journal, 'received', side_effect=OSError('synthetic disk failure')):
            self.assertRaises(ProviderFailure, executor.execute, self.batch)
        self.assertRaises(EmbeddingExecutorBlocked, executor.execute, self.batch)
        self.assertEqual(http.calls, 1)

    def test_batch_source_change_rejected_before_factory(self):
        changed = [dict(row) for row in self.batch]
        changed[0]['text'] += ' changed'
        http = FakeHTTP()
        self.assertRaises(ValueError, self.executor(http).execute, changed)
        self.assertEqual(http.calls, 0)
        self.assertEqual(self.ledger.totals()['attempts'], 0)

    def test_retrying_or_unobserved_transport_is_rejected(self):
        http = FakeHTTP()
        def build(wrapped, observer):
            transport = factory(self.ledger)(wrapped, observer)
            transport.retry = RetryPolicy(max_attempts=2, retry_429=True)
            return transport
        self.assertRaises(EmbeddingExecutorBlocked, self.executor(http, build=build).execute, self.batch)
        self.assertEqual(http.calls, 0)
        self.assertEqual(self.ledger.totals()['attempts'], 0)

    def test_secret_echoes_are_scrubbed_and_source_not_in_journal(self):
        value = response()
        value['echo'] = 'SYNTHETIC_KEY_NEVER_LOG_THIS'
        value['headers'] = {'Authorization': 'Bearer SYNTHETIC_KEY_NEVER_LOG_THIS'}
        executor = self.executor(FakeHTTP(HTTPReply(200, json.dumps(value).encode())))
        executor.execute(self.batch)
        raw = executor.journal.path.read_bytes()
        self.assertNotIn(b'SYNTHETIC_KEY_NEVER_LOG_THIS', raw)
        self.assertNotIn(b'synthetic original alpha', raw)
        self.assertNotIn(b'synthetic original beta', raw)

    def test_receipt_hash_tamper_blocks_without_request(self):
        http = FakeHTTP()
        executor = self.executor(http)
        result = executor.execute(self.batch)
        with executor.journal.connect() as db:
            db.execute('UPDATE operations SET body=?', (b'{}',))
        self.assertRaises(EmbeddingExecutorBlocked, executor.execute, self.batch)
        self.assertEqual(http.calls, 1)


if __name__ == '__main__':
    unittest.main()
