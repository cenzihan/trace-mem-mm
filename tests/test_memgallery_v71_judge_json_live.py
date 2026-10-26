"""Synthetic live-entry contract tests: no real data, credential or HTTP.

These seven runner tests originally passed under test_memgallery_v71_judge_json
before that shared filename was replaced by the independent scorer tests.
They have their own filename so both suites remain available.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_memory.budget import BudgetLedger
from agent_memory.development_run import DevelopmentRun
from agent_memory.provider_http import HTTPReply
from scripts import run_memgallery_v71_judge_json as entry


def payload(number=0):
    return dict(model='gpt-4o-mini', temperature=0, max_completion_tokens=1536,
        store=False, n=1, response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': 'Score synthetic items.\nReturn JSON.'},
                  {'role': 'user', 'content': [{'type': 'text', 'text': 'Synthetic batch ' + str(number)}]}])


class FakeHTTP:
    is_fake_http = True

    def __init__(self, status=200):
        self.payloads = []; self.status = status

    def exchange(self, endpoint, body, headers, timeout, max_response_bytes):
        self.payloads.append(json.loads(body))
        card = dict(summary='Synthetic batch only.', evidence_quotes=[], scores=[{'item_id': 'fake', 'score': 4}])
        response = dict(model='gpt-4o-mini-2024-07-18',
            usage=dict(prompt_tokens=20, completion_tokens=12, total_tokens=32),
            choices=[dict(finish_reason='stop', message={'content': json.dumps(card)})])
        return HTTPReply(self.status, json.dumps(response).encode(), {})


class JudgeJSONRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='v71-json-judge-offline-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.work = self.root / 'runtime/memgallery-v7/judge-json-v71'
        self.work.mkdir(parents=True)
        self.progress = self.root / 'reports/progress.json'
        self.ledger = BudgetLedger(self.root / 'synthetic-budget.sqlite')
        self.http = FakeHTTP()
        self.credential = Mock(side_effect=AssertionError('credentials forbidden in tests'))
        self.no_http = self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',
            side_effect=AssertionError('real HTTP forbidden')))
        self.enterContext(patch.object(entry, 'SHARED_LEDGER', Path(self.ledger.path)))
        self.enterContext(patch.object(entry, 'BASELINE', 0))
        original_factory = entry.make_executor
        self.received = []
        def factory(work, live_sha, ledger, run):
            def after_received():
                with executor.journal.connect() as db:
                    last = db.execute('SELECT state,body,attempt_id FROM operations ORDER BY rowid DESC LIMIT 1').fetchone()
                self.assertEqual(last[0], 'received'); self.assertIsInstance(last[1], bytes)
                self.assertTrue(last[2]); self.received.append(True)
            executor = original_factory(work, live_sha, ledger, run, http=self.http,
                offline_test=True, credential_reader=self.credential, after_received=after_received)
            return executor
        self.enterContext(patch.object(entry, 'make_executor', side_effect=factory))
        for name in entry.CODE:
            destination = self.root / name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((entry.ROOT / name).read_bytes())
        oldrun = DevelopmentRun(self.root / entry.OLD / 'run.sqlite', create=True,
            max_requests=72, max_reservation_cny='18', approval_scope='native_100_20')
        with oldrun.connect() as db:
            db.executemany('INSERT INTO calls VALUES(?,?,?,?,?,?,?)',
                [(n, 'gpt-4o-mini', 250000, 'complete' if n <= 64 else 'rejected_response',
                  'synthetic', 'synthetic', 'gpt-4o-mini') for n in range(1, 66)])
            db.execute("UPDATE run SET state='complete'")
        for name in (entry.OLD + '/receipts/private-embedding.sqlite', entry.OLD + '/FROZEN_JUDGE.json',
                     entry.OLD + '/READER_COMPLETED.json', entry.AUDIT):
            destination = self.root / name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{}', encoding='utf-8')
        self.requests = [{'request_id': 'synthetic-' + str(i), 'payload': payload(i)} for i in range(8)]
        self.freeze()

    def freeze(self):
        path = self.work / 'judge-requests.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in self.requests), encoding='utf-8')
        names = (entry.OLD + '/run.sqlite', entry.OLD + '/receipts/private-embedding.sqlite',
                 entry.OLD + '/FROZEN_JUDGE.json', entry.OLD + '/READER_COMPLETED.json', entry.AUDIT,
                 path.relative_to(self.root).as_posix())
        frozen = dict(input_hashes={n: entry.sha(self.root / n) for n in names},
            code_hashes={entry.CODE[0]: entry.sha(self.root / entry.CODE[0])},
            requests_file=path.relative_to(self.root).as_posix(), requests_sha256=entry.sha(path))
        entry.save(self.work / 'FROZEN_JUDGE.json', frozen)

    def execute(self, limit=None):
        with redirect_stdout(io.StringIO()):
            entry.execute(limit, root=self.root, work=self.work, progress_file=self.progress)

    def test_full_receipt_chain_bounded_one_then_seven_and_completed_reuse(self):
        old_hashes = entry.load(self.work / 'FROZEN_JUDGE.json')['input_hashes']
        self.execute(1)
        self.assertEqual(len(self.http.payloads), 1)
        self.assertFalse((self.work / 'JUDGE_COMPLETED.json').exists())
        run = DevelopmentRun(self.work / 'run.sqlite')
        self.assertEqual(run.summary()['state'], 'active')
        self.assertEqual(run.summary()['request_claims'], 1)
        self.execute()
        self.assertEqual(self.http.payloads, [r['payload'] for r in self.requests])
        self.assertEqual(self.received, [True] * 8)
        self.assertEqual(self.ledger.totals()['attempts'], 8)
        summary = run.summary()
        self.assertEqual(summary['state'], 'complete')
        self.assertEqual(summary['max_requests'], 8)
        self.assertEqual(summary['cumulative_preflight_micro_cny'], 2000000)
        results = entry.lines(self.work / 'judge-results.jsonl')
        self.assertEqual([r['run_ordinal'] for r in results], list(range(1, 9)))
        self.assertTrue(all(set(r) == {'request_id', 'response', 'attempt_id', 'receipt_sha256', 'run_ordinal'} for r in results))
        complete = entry.load(self.work / 'JUDGE_COMPLETED.json')
        self.assertEqual(complete['output_sha256'], entry.sha(self.work / 'judge-results.jsonl'))
        self.assertEqual(complete['frozen_live_sha256'], entry.sha(self.work / 'FROZEN_LIVE.json'))
        self.assertEqual(entry.load(self.progress)['state'], 'judge_complete')
        before = (self.work / 'JUDGE_COMPLETED.json').read_bytes()
        self.execute()
        self.assertEqual(len(self.http.payloads), 8)
        self.assertEqual((self.work / 'JUDGE_COMPLETED.json').read_bytes(), before)
        entry.verify_hashes(self.root, old_hashes)
        self.credential.assert_not_called(); self.no_http.assert_not_called()

    def test_bad_last_payload_rejected_before_any_live_state_budget_or_credential(self):
        self.requests[-1]['payload']['messages'][0]['content'] = 'Synthetic grading instruction.'
        self.freeze()
        with patch.object(entry, 'BudgetLedger', side_effect=AssertionError('preflight reached ledger')) as ledger:
            with self.assertRaises(ValueError): self.execute()
            ledger.assert_not_called()
        self.assertFalse((self.work / 'run.sqlite').exists())
        self.assertFalse((self.work / 'FROZEN_LIVE.json').exists())
        self.assertEqual(self.http.payloads, []); self.credential.assert_not_called()

    def test_rejected_http_never_automatically_replays(self):
        self.http.status = 400
        with self.assertRaises(SystemExit): self.execute()
        self.assertEqual(self.ledger.totals()['attempts'], 1)
        with self.assertRaises(SystemExit): self.execute()
        self.assertEqual(len(self.http.payloads), 1)
        self.assertFalse((self.work / 'JUDGE_COMPLETED.json').exists())
        self.assertEqual(entry.load(self.progress)['state'], 'stopped_no_automatic_retry')
        self.assertEqual(DevelopmentRun(self.work / 'run.sqlite').summary()['request_claims'], 1)

    def test_completed_receipt_without_output_recovers_without_new_call(self):
        self.execute(1)
        (self.work / 'judge-results.jsonl').write_text('', encoding='utf-8')
        self.execute(1)
        self.assertEqual(len(self.http.payloads), 2)
        self.assertEqual(len(entry.lines(self.work / 'judge-results.jsonl')), 2)
        self.assertEqual(DevelopmentRun(self.work / 'run.sqlite').summary()['request_claims'], 2)

    def test_changed_saved_result_or_history_or_guard_blocks_continuation(self):
        self.execute(1)
        output = self.work / 'judge-results.jsonl'; before = output.read_bytes()
        rows = entry.lines(output); rows[0]['response']['model'] = 'tampered'
        output.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError): self.execute()
        output.write_bytes(before)
        with DevelopmentRun(self.work / 'run.sqlite').connect() as db:
            db.execute('UPDATE run SET max_requests=9')
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(len(self.http.payloads), 1)

    def test_frozen_tampering_duplicate_requests_and_old_open_guard_fail(self):
        path = self.work / 'judge-requests.jsonl'; original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises(ValueError): entry.verify(self.root, self.work)
        self.requests[1]['request_id'] = self.requests[0]['request_id']; self.freeze()
        with self.assertRaises(ValueError): entry.verify(self.root, self.work)
        self.requests[1]['request_id'] = 'synthetic-1'
        with DevelopmentRun(self.root / entry.OLD / 'run.sqlite').connect() as db:
            db.execute("UPDATE run SET state='active'")
        self.freeze()
        with self.assertRaises(ValueError): entry.verify(self.root, self.work)
        self.assertEqual(self.http.payloads, [])

    def test_preflight_rejects_other_parameters_media_caps_and_temperature(self):
        variants = []
        for value in (True, False, 0.0, 1, None): variants.append(dict(payload(), temperature=value))
        for key, value in (('max_completion_tokens', 1535), ('top_p', 1), ('model', 'other'),
                           ('response_format', {'type': 'text'}), ('store', True), ('n', True)):
            variants.append(dict(payload(), **{key: value}))
        image = payload(); image['messages'][1]['content'] = [{'type': 'image_url', 'image_url': {'url': 'https://invalid.example/image'}}]
        variants.append(image)
        for number, candidate in enumerate(variants):
            original = copy.deepcopy(candidate)
            with self.subTest(number=number), self.assertRaises(ValueError): entry.validate_payload(candidate)
            self.assertEqual(candidate, original)
        self.assertEqual(self.http.payloads, []); self.credential.assert_not_called()


if __name__ == '__main__': unittest.main()
