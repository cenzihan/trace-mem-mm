"""Synthetic additive-judge runner tests; no native data or real HTTP/ledger.

Keep the real freeze, historical hashing, closed old guard, new guard, provider
accounting, receipt journal and judge decoder. All artifacts live in a test temp.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_memory.budget import BudgetLedger
from agent_memory.development_run import DevelopmentRun
from agent_memory.durable_chat import DurableChat
from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply
from scripts import run_atm_v6_reader as original_runner
from scripts import run_atm_v6_recovery_judge as runner


class FakeHTTP:
    is_fake_http = True

    def __init__(self):
        self.calls = 0
        self.timeout = False

    def exchange(self, endpoint, body, headers, timeout, limit):
        self.calls += 1
        if self.timeout:
            raise TimeoutError('synthetic unknown outcome')
        payload = json.loads(body)
        if payload['model'] != 'gpt-4o-mini' or payload['max_completion_tokens'] != 1536:
            raise AssertionError('unexpected synthetic judge request')
        content = dict(summary='Synthetic judge batch', evidence_quotes=[], judgments=[
            dict(item_id=index, label='correct', score=1, reason='Synthetic rationale') for index in range(8)])
        response = dict(model='gpt-4o-mini-2024-07-18', choices=[dict(
            finish_reason='stop', message=dict(content=json.dumps(content)))],
            usage=dict(prompt_tokens=20, completion_tokens=20, total_tokens=40))
        return HTTPReply(200, json.dumps(response).encode())


class ATMV6RecoveryLiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='v6-recovery-live-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / 'runtime/atm-v6-reader-recovery'
        self.old = self.root / 'runtime/atm-v6-reader'
        self.work.mkdir(parents=True)
        (self.old / 'receipts').mkdir(parents=True)
        (self.root / 'reports').mkdir()
        self.ledger = BudgetLedger(self.root / 'synthetic-budget.sqlite')
        # Bulk-load synthetic completed historical accounting without any calls.
        with self.ledger.connect() as db:
            db.executemany('''INSERT INTO attempts
                (id,retry_of,model,status,created_at,max_input,max_output,input_rate,output_rate,
                 price_source,price_checked_at,reserved,input_tokens,output_tokens,accounted,amount_kind)
                VALUES (?,NULL,'gpt-4o-mini','settled','synthetic',20,20,'1.5','6',
                        'synthetic','synthetic',250000,20,20,150,'usage_estimate')''',
                           [('historical-' + str(index),) for index in range(1451)])
        old_guard = DevelopmentRun(self.old / 'run.sqlite', create=True, max_requests=80,
                                   max_reservation_cny='20', approval_scope='native_100_20')
        with old_guard.connect() as db:
            db.executemany('INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)', [
                (index, 'gpt-4o-mini', 250000, 'complete', 'synthetic', 'synthetic', 'gpt-4o-mini')
                for index in range(1, 75)])
        old_guard.close()
        for relative in ('receipts/private-embedding.sqlite', 'FROZEN_LIVE.json',
                         'reader-answers.jsonl', 'judge-results.jsonl'):
            (self.old / relative).write_text('synthetic preserved artifact', encoding='utf-8')
        (self.root / 'reports/atm-v6-reader-scores.json').write_text('synthetic old score', encoding='utf-8')
        (self.root / 'synthetic-code.py').write_text('# synthetic immutable code\n', encoding='utf-8')
        self.old_hashes = {path: runner.sha(path) for path in self.old.rglob('*') if path.is_file()}
        score = self.root / 'reports/atm-v6-reader-scores.json'
        self.old_hashes[score] = runner.sha(score)
        self.requests = [dict(request_id='synthetic-judge-' + str(index), payload=dict(
            model='gpt-4o-mini', max_completion_tokens=1536, store=False, n=1,
            response_format=dict(type='json_object'), messages=[
                dict(role='system', content='Grade synthetic examples and return JSON.'),
                dict(role='user', content='Synthetic blinded eight-item batch ' + str(index))])) for index in range(5)]
        self.http = FakeHTTP()
        self.credentials = Mock(side_effect=AssertionError('credentials must never be read'))

        def fake_transport(**kwargs):
            kwargs['mode'] = 'offline_fake'
            return BudgetedJSONTransport(**kwargs)

        def fake_executor(*args, **kwargs):
            kwargs['offline_test'] = True
            return DurableChat(*args, **kwargs)

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.multiple(runner, ROOT=self.root, WORK=self.work, OLD=self.old,
            FROZEN=self.work / 'FROZEN_RECOVERY_LIVE.json', JUDGE=self.work / 'FROZEN_RECOVERY_JUDGE.json',
            OUTPUT=self.work / 'judge-results.jsonl', REPORT=self.root / 'reports/recovery-judge-run.json',
            PROGRESS=self.root / 'reports/recovery-progress.json', SHARED_LEDGER=Path(self.ledger.path),
            CODE=('synthetic-code.py',), singleton=nullcontext, BudgetedJSONTransport=fake_transport,
            DurableChat=fake_executor, NativeHTTPS=lambda: self.http, read_key=self.credentials))
        # historical_digest is imported from the frozen original module and uses
        # that module's global. Redirect it to this test's ledger, never the real one.
        stack.enter_context(patch.object(original_runner, 'SHARED_LEDGER', Path(self.ledger.path)))
        self.write_prepared()
        self.historical_hash = runner.historical_digest(1451)

    def write_prepared(self):
        requests_path = self.work / 'judge-requests.jsonl'
        requests_path.write_text(''.join(json.dumps(row) + '\n' for row in self.requests), encoding='utf-8')
        runner.save(runner.JUDGE, dict(judge_requests=5, model='gpt-4o-mini', input_hashes={},
            code_hashes={'synthetic-code.py': runner.sha(self.root / 'synthetic-code.py')},
            requests_file=str(requests_path.relative_to(self.root)), requests_sha256=runner.sha(requests_path)))

    def invoke(self):
        with patch.object(runner.sys, 'argv', ['run_atm_v6_recovery_judge.py', '--live']), redirect_stdout(io.StringIO()):
            runner.main()

    def run_guard(self):
        return DevelopmentRun(self.work / 'run.sqlite')

    def assert_preserved_and_accounted(self, count, state='active'):
        self.assertEqual(self.http.calls, count)
        self.assertEqual(self.ledger.totals()['attempts'], 1451 + count)
        self.assertEqual(runner.historical_digest(1451), self.historical_hash)
        for path, digest in self.old_hashes.items():
            self.assertEqual(runner.sha(path), digest, str(path))
        summary = self.run_guard().summary()
        self.assertEqual(summary['state'], state)
        self.assertEqual(summary['request_claims'], count)
        self.assertEqual(summary['max_requests'], 5)
        self.assertEqual(summary['max_reservation_micro_cny'], 1250000)
        self.assertEqual(summary['cumulative_preflight_micro_cny'], count * 250000)
        self.assertLessEqual(74 + count, 80)
        self.assertLessEqual(18500000 + summary['cumulative_preflight_micro_cny'], 20000000)
        self.credentials.assert_not_called()

    def fail_before_append(self):
        with patch.object(runner, 'append', side_effect=OSError('synthetic pre-append crash')) as append:
            with self.assertRaises(SystemExit):
                self.invoke()
            append.assert_called_once()
        self.assertEqual(runner.load(runner.PROGRESS)['error_type'], 'OSError')

    def test_append_crash_recovers_completed_receipt_no_duplicate_call(self):
        self.fail_before_append()
        self.assertEqual(runner.lines(runner.OUTPUT), [])
        self.assert_preserved_and_accounted(1)
        frozen_hash = runner.sha(runner.FROZEN)
        self.invoke()
        rows = runner.lines(runner.OUTPUT)
        self.assertEqual([row['run_ordinal'] for row in rows], list(range(1, 6)))
        self.assertEqual(len({row['attempt_id'] for row in rows}), 5)
        self.assertEqual(runner.sha(runner.FROZEN), frozen_hash)
        report = runner.load(runner.REPORT)
        self.assertEqual(report['aggregate_v6_calls'], 79)
        self.assertEqual(report['original_reader_calls_repeated'], 0)
        self.assert_preserved_and_accounted(5, 'complete')

    def test_unknown_timeout_never_replays_or_finishes_guard(self):
        self.http.timeout = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.http.timeout = False
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(runner.lines(runner.OUTPUT), [])
        self.assert_preserved_and_accounted(1)
        with self.run_guard().connect() as db:
            self.assertEqual(db.execute('SELECT state FROM calls').fetchone()[0], 'claimed')

    def test_closed_guard_cannot_dispatch_missing_outputs(self):
        self.fail_before_append()
        self.run_guard().close()
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(runner.lines(runner.OUTPUT), [])
        self.assert_preserved_and_accounted(1, 'complete')

    def test_saved_response_tamper_blocks_resume(self):
        self.invoke()
        rows = runner.lines(runner.OUTPUT)
        content = json.loads(rows[0]['response']['choices'][0]['message']['content'])
        content['summary'] = 'Tampered synthetic summary'
        rows[0]['response']['choices'][0]['message']['content'] = json.dumps(content)
        runner.OUTPUT.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assert_preserved_and_accounted(5, 'complete')

    def test_closed_guard_can_repair_report_without_new_call(self):
        original_save = runner.save

        def crash_report(path, value, exclusive=False):
            if Path(path) == runner.REPORT:
                raise OSError('synthetic report crash after close')
            return original_save(path, value, exclusive=exclusive)

        with patch.object(runner, 'save', crash_report):
            with self.assertRaises(SystemExit):
                self.invoke()
        self.assertEqual(runner.load(runner.PROGRESS)['error_type'], 'OSError')
        self.assertFalse(runner.REPORT.exists())
        self.assert_preserved_and_accounted(5, 'complete')
        output_hash = runner.sha(runner.OUTPUT)
        self.invoke()
        self.assertEqual(runner.sha(runner.OUTPUT), output_hash)
        self.assertEqual(runner.load(runner.REPORT)['results_sha256'], output_hash)
        self.assert_preserved_and_accounted(5, 'complete')

    def test_utf8_1536_and_text_only_schema_rejected_before_guard_creation(self):
        mutations = (
            lambda p: p.update(max_completion_tokens=1536.0),
            lambda p: p.update(max_completion_tokens=1537),
            lambda p: p.update(model='other'),
            lambda p: p.update(store=True),
            lambda p: p.update(n=True),
            lambda p: p.update(response_format={'type': 'text'}),
            lambda p: p['messages'][1].update(content='原' * 11000),
            lambda p: p['messages'][1].update(content=[{'type': 'text', 'text': 'synthetic'}]),
        )
        original = copy.deepcopy(self.requests)
        for mutation in mutations:
            self.requests = copy.deepcopy(original)
            mutation(self.requests[0]['payload'])
            self.write_prepared()
            with self.subTest(payload=self.requests[0]['payload']), self.assertRaises(ValueError):
                self.invoke()
            self.assertFalse((self.work / 'run.sqlite').exists())
            self.assertEqual(self.http.calls, 0)
            self.assertEqual(self.ledger.totals()['attempts'], 1451)

    def test_frozen_history_code_and_old_artifacts_are_checked_before_dispatch(self):
        runner.freeze(self.ledger)
        paths = (self.old / 'reader-answers.jsonl', self.root / 'synthetic-code.py', runner.JUDGE)
        for path in paths:
            original = path.read_bytes()
            try:
                path.write_bytes(original + b' ')
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.invoke()
                self.assertEqual(self.http.calls, 0)
            finally:
                path.write_bytes(original)
        with self.ledger.connect() as db:
            db.execute("UPDATE attempts SET accounted=151 WHERE id='historical-0'")
        with self.assertRaisesRegex(ValueError, 'historical budget rows changed'):
            self.invoke()
        self.assertEqual(self.http.calls, 0)
        self.assertFalse((self.work / 'run.sqlite').exists())

    def test_aggregate_and_existing_guard_scope_cannot_expand(self):
        old = runner.old_summary()
        runner.aggregate_cap(old)
        for changed, count, held in ((dict(old, count=75), 5, 1250000),
                                    (dict(old, state='active'), 5, 1250000),
                                    (old, 6, 1500000), (old, 5, 1500000)):
            with self.subTest(old=changed, count=count, held=held), self.assertRaises(ValueError):
                runner.aggregate_cap(changed, count, held)
        self.fail_before_append()
        with self.run_guard().connect() as db:
            db.execute('UPDATE run SET max_requests=6')
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.http.calls, 1)
        self.assertEqual(self.ledger.totals()['attempts'], 1452)


if __name__ == '__main__':
    unittest.main()
