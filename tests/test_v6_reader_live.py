"""Synthetic runner fault tests: temporary accounting, fake HTTP, no native QA.

The actual guard, ledger, receipt journal and answer decoder remain in the path.
Only input admission, credentials, Windows locking and external HTTP are replaced.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_memory.atm_reader import VARIANTS, pack_atm_evidence, query_id, reader_payload
from agent_memory.budget import BudgetLedger
from agent_memory.development_run import DevelopmentRun
from agent_memory.durable_chat import DurableChat
from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply
from scripts import run_atm_v6_reader as runner


class FakeHTTP:
    is_fake_http = True

    def __init__(self):
        self.calls = 0

    def exchange(self, endpoint, body, headers, timeout, limit):
        self.calls += 1
        payload = json.loads(body)
        if payload['model'] != 'gpt-4o-mini':
            raise AssertionError('unexpected model in synthetic transport')
        content = dict(summary='atm reader output', evidence_quotes=[],
                       answer='A synthetic supported fact', answer_ids=[],
                       cited_ids=['email0001'], confidence='supported')
        response = dict(model='gpt-4o-mini-2024-07-18',
                        choices=[dict(finish_reason='stop', message=dict(content=json.dumps(content)))],
                        usage=dict(prompt_tokens=20, completion_tokens=20, total_tokens=40))
        return HTTPReply(200, json.dumps(response).encode())


class V6ReaderLiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='v6-reader-runner-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / 'runtime/atm-v6-reader'
        self.work.mkdir(parents=True)
        (self.root / 'reports').mkdir()
        self.ledger = BudgetLedger(self.root / 'fake-budget.sqlite')
        self.http = FakeHTTP()
        self.answers = self.work / 'reader-answers.jsonl'
        self.judge_results = self.work / 'judge-results.jsonl'
        self.judge_report = self.root / 'reports/atm-v6-reader-judge-run.json'
        self.packs = []
        for dataset in ('regular', 'hard'):
            for index in range(8):
                identity = 'synthetic-' + str(index)
                question = dict(id=identity, question='What synthetic fact is supported?', qtype='open_end')
                packed = pack_atm_evidence(question, ['email0001'],
                                           {'email0001': 'A synthetic supported fact.'}, len)
                for variant in VARIANTS:
                    self.packs.append(dict(query_id=query_id(dataset, identity), dataset=dataset,
                                           id=identity, qtype='open_end', variant=variant, pack=packed))
        self.reader_frozen = dict(state='reader_inputs_frozen_no_calls', planned_reader_answers=64)
        self.live_frozen = dict(historical_attempts=0, baseline_budget=self.ledger.totals())
        runner.save(self.work / 'FROZEN_READER.json', self.reader_frozen)
        runner.save(self.work / 'FROZEN_LIVE.json', self.live_frozen)
        self.credentials = Mock(side_effect=AssertionError('credentials must never be read'))

        def fake_transport(**kwargs):
            kwargs['mode'] = 'offline_fake'
            return BudgetedJSONTransport(**kwargs)

        def fake_executor(*args, **kwargs):
            kwargs['offline_test'] = True
            return DurableChat(*args, **kwargs)

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.multiple(
            runner, ROOT=self.root, WORK=self.work,
            READER_FROZEN=self.work / 'FROZEN_READER.json',
            LIVE_FROZEN=self.work / 'FROZEN_LIVE.json',
            PROGRESS=self.root / 'reports/atm-v6-reader-live-progress.json',
            SHARED_LEDGER=Path(self.ledger.path), singleton=nullcontext,
            verify_reader=lambda: (self.reader_frozen, self.packs),
            live_freeze=lambda ledger: self.live_frozen,
            BudgetedJSONTransport=fake_transport, DurableChat=fake_executor,
            NativeHTTPS=lambda: self.http, read_key=self.credentials))
        # Any accidental reader-phase access to judge inputs fails before HTTP.
        self.original_load = runner.load

    def invoke(self, phase='reader', limit=None):
        argv = ['run_atm_v6_reader.py', phase, '--live']
        if limit is not None:
            argv += ['--max-requests', str(limit)]

        def guarded_load(path):
            if phase == 'reader' and Path(path).name == 'FROZEN_JUDGE.json':
                raise AssertionError('reader accessed judge/gold inputs')
            return self.original_load(path)

        with patch.object(runner.sys, 'argv', argv), patch.object(runner, 'load', guarded_load), redirect_stdout(io.StringIO()):
            runner.main()

    def run_guard(self):
        return DevelopmentRun(self.work / 'run.sqlite')

    def assert_accounting(self, count, state='active'):
        self.assertEqual(self.http.calls, count)
        self.assertEqual(self.ledger.totals()['attempts'], count)
        summary = self.run_guard().summary()
        self.assertEqual(summary['request_claims'], count)
        self.assertEqual(summary['state'], state)
        self.assertEqual(summary['max_requests'], 80)
        self.assertEqual(summary['max_reservation_micro_cny'], 20_000_000)
        self.assertEqual(summary['cumulative_preflight_micro_cny'], count * 250_000)
        self.credentials.assert_not_called()

    def test_append_failure_restart_recovers_receipt_without_repeating_call(self):
        with patch.object(runner, 'append', side_effect=OSError('synthetic crash before answer append')):
            with self.assertRaises(SystemExit):
                self.invoke(limit=1)
        self.assertEqual(runner.lines(self.answers), [])
        self.assert_accounting(1)
        # Recovery appends the first response; only the second request consumes
        # this invocation's one-new-request allowance.
        self.invoke(limit=1)
        rows = runner.lines(self.answers)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['run_ordinal'] for row in rows], [1, 2])
        self.assertEqual(len({row['request_id'] for row in rows}), 2)
        self.assertTrue(all(row['answer']['schema_status'] == 'valid' for row in rows))
        self.assert_accounting(2)
        progress = runner.load(runner.PROGRESS)
        self.assertEqual(progress['state'], 'invocation_limit_guard_remains_active')
        self.assertEqual(progress['new_this_invocation'], 1)

    def test_saved_answer_tamper_is_rejected_before_any_new_dispatch(self):
        self.invoke(limit=1)
        rows = runner.lines(self.answers)
        rows[0]['answer']['answer'] = 'Tampered synthetic answer'
        self.answers.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'saved answer differs'):
            self.invoke(limit=1)
        self.assert_accounting(1)

    def test_complete_guard_cannot_dispatch_missing_reader_requests(self):
        self.invoke(limit=1)
        self.run_guard().close()
        with self.assertRaisesRegex(ValueError, 'completed V6 guard cannot dispatch'):
            self.invoke(limit=1)
        self.assertEqual(len(runner.lines(self.answers)), 1)
        self.assert_accounting(1, 'complete')

    def test_judge_report_failure_after_close_can_finalize_without_dispatch(self):
        # Synthetic isolated judge input: no native reference answer is opened.
        request_path = self.work / 'judge-requests.jsonl'
        request = dict(request_id='synthetic-judge-batch-0', payload=reader_payload(self.packs[0]['pack']))
        runner.append(request_path, request)
        runner.save(self.work / 'FROZEN_JUDGE.json', dict(input_hashes={},
                    requests_file=str(request_path.relative_to(self.root)),
                    requests_sha256=runner.sha(request_path)))
        original_save = runner.save

        def crash_at_report(path, value, exclusive=False):
            if Path(path) == self.judge_report:
                raise OSError('synthetic crash after guard close')
            return original_save(path, value, exclusive=exclusive)

        with patch.object(runner, 'save', crash_at_report):
            with self.assertRaises(SystemExit):
                self.invoke('judge')
        self.assertFalse(self.judge_report.exists())
        rows = runner.lines(self.judge_results)
        self.assertEqual(len(rows), 1)
        self.assert_accounting(1, 'complete')
        self.invoke('judge')
        self.assertEqual(runner.lines(self.judge_results), rows)
        report = runner.load(self.judge_report)
        self.assertEqual(report['state'], 'judge_complete')
        self.assertEqual(report['completed_requests'], 1)
        self.assertEqual(report['results_sha256'], runner.sha(self.judge_results))
        self.assertFalse(report['official_score'])
        self.assert_accounting(1, 'complete')


if __name__ == '__main__':
    unittest.main()
