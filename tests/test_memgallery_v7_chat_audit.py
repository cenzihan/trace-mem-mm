"""Synthetic completed-run accounting, never native data, provider or real ledger."""
from contextlib import closing
from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_memory.memgallery_v7_reader import pack_memgallery_evidence, build_proxy_judge
from scripts import audit_memgallery_v7_chat as audit


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(audit.canonical(value) + '\n', encoding='utf-8')


def put_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(audit.canonical(value) + '\n' for value in values), encoding='utf-8')


def guard(path, maximum, cap, count, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript('CREATE TABLE run(max_requests INTEGER,cap INTEGER,state TEXT);'
                         'CREATE TABLE calls(ordinal INTEGER,model TEXT,held INTEGER,state TEXT,returned_model TEXT);')
        db.execute('INSERT INTO run VALUES(?,?,?)', (maximum, cap, 'complete'))
        held = 250000 if model == 'gpt-4o-mini' else 50000
        db.executemany('INSERT INTO calls VALUES(?,?,?,?,?)',
                       [(i, model, held, 'complete', model) for i in range(1, count + 1)])


def fixture(root):
    chat, encoding = root / audit.CHAT, root / audit.BASE / 'encoding'
    chat.mkdir(parents=True)
    guard(encoding / 'run.sqlite', 27, 1350000, 27, 'text-embedding-v4')
    put(root, audit.BASE / 'encoding/FROZEN_ENCODING.json', {'synthetic': 'encoding'})
    put(root, audit.BASE / 'encoding/FROZEN_LIVE.json', {'synthetic': 'encoding-live'})
    for name in ('source.sqlite', 'query.sqlite', 'receipts/source/private-embedding.sqlite', 'receipts/query/private-embedding.sqlite'):
        path = encoding / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic previously audited artifact')
    source_files = [encoding / n for n in ('source.sqlite', 'query.sqlite', 'run.sqlite',
                    'receipts/source/private-embedding.sqlite', 'receipts/query/private-embedding.sqlite')]
    source_audit_path = Path('reports/memgallery-v7-encoding-audit.json')
    put(root, source_audit_path, dict(state='passed', real_embedding_requests=27, complete_unique_run_bindings=27,
        unknown_or_inflight_operations=0, new_usage_estimate_micro_cny=12547,
        frozen_encoding_sha256=audit.sha(encoding / 'FROZEN_ENCODING.json'),
        frozen_live_sha256=audit.sha(encoding / 'FROZEN_LIVE.json'),
        artifact_sha256={p.relative_to(root).as_posix(): audit.sha(p) for p in source_files}))
    ledger = root / 'runtime/development-budget.sqlite'
    with closing(sqlite3.connect(ledger)) as db, db:
        db.executescript('CREATE TABLE attempts(id TEXT,model TEXT,status TEXT,amount_kind TEXT,conversion_basis TEXT,'
            'reserved INTEGER,accounted INTEGER,input_tokens INTEGER,output_tokens INTEGER,max_input INTEGER,'
            'max_output INTEGER,input_rate TEXT,output_rate TEXT);'
            'CREATE TABLE provider_observations(attempt_id TEXT,mode TEXT,category TEXT,http_status INTEGER,'
            'returned_model TEXT,input_tokens INTEGER,output_tokens INTEGER,estimated_usd TEXT);')
        db.executemany('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            [('historical-' + str(i), 'text-embedding-v4', 'settled', 'provider_usage_estimate', 'verified_fx',
              0, 0, 1, 0, 8192, 0, '.5', '0') for i in range(1483)])
    with closing(audit.ro(ledger)) as db:
        historical = audit.history(db, 1483)
    source = {'src': dict(text='Synthetic source.', source_session='S', source_round='R', source_date='', image_ids=[])}
    packs = []
    for i in range(16):
        for variant in audit.VARIANTS:
            pack = pack_memgallery_evidence(dict(index=i, point='FR', question='Synthetic question?'),
                                            ['src'], source, lambda s: len(s)//4 + 1)
            packs.append(dict(query_id='memgallery-v7:' + str(i), index=i, point='FR', variant=variant, pack=pack))
    put_lines(chat / 'reader-packs.jsonl', packs)
    old = root / audit.OLD_CHAT
    guard(old / 'run.sqlite', 72, 18000000, 1, 'gpt-4o-mini')
    with closing(sqlite3.connect(old / 'run.sqlite')) as db, db:
        db.execute("UPDATE run SET state='active'")
        db.execute("UPDATE calls SET state='claimed',returned_model=NULL")
    old_journal = old / 'receipts/private-embedding.sqlite'; old_journal.parent.mkdir()
    with closing(sqlite3.connect(old_journal)) as db, db:
        db.executescript('CREATE TABLE operations(state TEXT,transport_completed INTEGER,attempt_id TEXT,status INTEGER,'
                         'body BLOB,body_sha TEXT,run_path TEXT,run_ordinal INTEGER);')
        db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?)', ('unknown', 0, None, None, None, None, None, None))
    put_lines(old / 'reader-packs.jsonl', packs)
    put(root, audit.OLD_CHAT / 'FROZEN_READER.json', {'synthetic': 'old-reader'})
    put(root, audit.OLD_CHAT / 'FROZEN_LIVE.json', {'synthetic': 'old-live'})
    put(root, 'reports/memgallery-v7-chat-progress.json', dict(state='stopped_no_automatic_retry'))
    original_paths = [old / 'run.sqlite', old_journal, old / 'reader-packs.jsonl',
                      root / 'reports/memgallery-v7-chat-progress.json', old / 'FROZEN_READER.json', old / 'FROZEN_LIVE.json']
    put(root, audit.PREFLIGHT_AUDIT, dict(state='confirmed_no_provider_dispatch',
        original_provider_attempts_before=1483, original_provider_attempts_after=1483,
        original_provider_attempts_added=0, original_http_dispatches=0, original_provider_cost_added_micro_cny=0,
        original_guard_claims_preserved=1, nominal_guard_preflight_preserved_micro_cny=250000,
        original64packs_sha256=audit.sha(old / 'reader-packs.jsonl'),
        original_artifact_hashes={p.relative_to(root).as_posix(): audit.sha(p) for p in original_paths}))
    code = Path('synthetic-code.py'); (root / code).write_text('synthetic audited code')
    frozen = dict(planned_reader_answers=64, max_requests=72, max_preflight_cny='18', fixed_hold_cny='.25',
        packs_file=(audit.CHAT / 'reader-packs.jsonl').as_posix(), packs_sha256=audit.sha(chat / 'reader-packs.jsonl'),
        input_hashes={source_audit_path.as_posix(): audit.sha(root / source_audit_path),
                      (audit.CHAT / 'reader-packs.jsonl').as_posix(): audit.sha(chat / 'reader-packs.jsonl')},
        code_hashes={code.as_posix(): audit.sha(root / code)})
    put(root, audit.CHAT / 'FROZEN_READER.json', frozen)
    compatible_paths = [p for p in original_paths if p.is_relative_to(old)] + [root / audit.PREFLIGHT_AUDIT, chat / 'FROZEN_READER.json', chat / 'reader-packs.jsonl']
    put(root, audit.CHAT / 'FROZEN_COMPATIBILITY.json', dict(state='identical_reader_payloads_new_bounded_execution',
        reader_payloads_byte_identical=True, original_claims_preserved=1, provider_requests_before_recovery=1483,
        hashes={p.relative_to(root).as_posix(): audit.sha(p) for p in compatible_paths}))
    put(root, audit.CHAT / 'FROZEN_LIVE.json', dict(baseline={'attempts': 1483}, historical_sha256=historical,
                                               reader_sha256=audit.sha(chat / 'FROZEN_READER.json')))
    live_hash = audit.sha(chat / 'FROZEN_LIVE.json')
    namespace = 'memgallery-v7:' + live_hash
    guard(chat / 'run.sqlite', 72, 18000000, 72, 'gpt-4o-mini')
    journal = chat / 'receipts/private-embedding.sqlite'; journal.parent.mkdir()
    with closing(sqlite3.connect(journal)) as db, db:
        db.executescript('CREATE TABLE operations(fingerprint TEXT,scope TEXT,manifest TEXT,state TEXT,attempt_id TEXT,'
            'status INTEGER,body BLOB,body_sha TEXT,redacted INTEGER,transport_completed INTEGER,run_path TEXT,run_ordinal INTEGER);'
            'CREATE TABLE claims(scope TEXT,chunk_key TEXT,fingerprint TEXT);')
    def receipt(request_id, payload, response, ordinal):
        response['usage'] = dict(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        manifest = dict(keys=[request_id], payload_sha256=audit.digest(audit.canonical(payload).encode()),
                        run_path=str((chat / 'run.sqlite').resolve()))
        fingerprint = audit.digest(audit.canonical(dict(namespace=namespace, manifest=manifest)).encode())
        body = audit.canonical(response).encode()
        attempt, body_hash = 'synthetic-chat-' + str(ordinal), audit.digest(body)
        with closing(sqlite3.connect(journal)) as db, db:
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (fingerprint, namespace, audit.canonical(manifest), 'complete', attempt, 200, body, body_hash, 0, 1,
                 manifest['run_path'], ordinal))
            db.execute('INSERT INTO claims VALUES(?,?,?)', (namespace, request_id, fingerprint))
        with closing(sqlite3.connect(ledger)) as db, db:
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (attempt, 'gpt-4o-mini', 'uncertain', 'retained_upper_bound', 'budget_assumption_not_market_fx',
                 250000, 250000, 100, 20, 128000, payload['max_completion_tokens'], '1.5', '6'))
            db.execute('INSERT INTO provider_observations VALUES(?,?,?,?,?,?,?,?)',
                (attempt, 'live', 'success', 200, 'gpt-4o-mini', 100, 20, '0.000027'))
        return dict(request_id=request_id, attempt_id=attempt, receipt_sha256=body_hash, run_ordinal=ordinal)
    answers = []
    for ordinal, meta in enumerate(packs, 1):
        value = dict(summary='', evidence_quotes=[], answer='Synthetic answer.', cited_ids=['src'],
                     confidence='invalid' if ordinal == 1 else 'supported')
        response = dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message={'content': audit.canonical(value)})])
        request_id = audit.digest((meta['query_id'] + '\0' + meta['variant']).encode())
        saved = receipt(request_id, audit.reader_payload(meta['pack']), response, ordinal)
        saved.update({k: meta[k] for k in ('query_id', 'index', 'point', 'variant')})
        saved['answer'] = audit.reader_projection(response, meta['pack'])
        answers.append(saved)
    put_lines(chat / 'reader-answers.jsonl', answers)
    put(root, audit.CHAT / 'READER_COMPLETED.json', dict(state='complete', phase='reader', count=64,
        output_sha256=audit.sha(chat / 'reader-answers.jsonl'), frozen_live_sha256=live_hash))
    requests, results = [], []
    for i in range(8):
        pack = build_proxy_judge([dict(item_id=j, point='FR', question='Synthetic?', reference='Synthetic reference.',
                                      prediction='Synthetic answer.') for j in range(8)], lambda s: len(s)//4+1)
        request_id = 'synthetic-judge-' + str(i)
        requests.append(dict(request_id=request_id, pack=pack))
        value = dict(summary='', evidence_quotes=[], judgments=[dict(item_id=j, score=.75, reason='Synthetic.') for j in range(8)])
        response = dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message={'content': audit.canonical(value)})])
        saved = receipt(request_id, audit.proxy_judge_payload(pack), response, 65 + i)
        saved['response'] = response; results.append(saved)
    put_lines(chat / 'judge-requests.jsonl', requests); put_lines(chat / 'judge-results.jsonl', results)
    put(root, audit.CHAT / 'FROZEN_JUDGE.json', dict(judge_requests=8, reader_answers=64, grades=[0,.25,.5,.75,1],
        requests_file=(audit.CHAT / 'judge-requests.jsonl').as_posix(), requests_sha256=audit.sha(chat / 'judge-requests.jsonl'),
        input_hashes={code.as_posix(): audit.sha(root / code), 'runtime/memgallery-v7/scorer-only/gold.json': 'f' * 64},
        code_hashes={code.as_posix(): audit.sha(root / code)}))
    put(root, audit.CHAT / 'JUDGE_COMPLETED.json', dict(state='complete', phase='judge', count=8,
        output_sha256=audit.sha(chat / 'judge-results.jsonl'), frozen_live_sha256=live_hash))
    put(root, 'reports/memgallery-v7-chat-progress.json', dict(state='judge_complete', phase='judge'))


class MemGalleryV7ChatAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v7-chat-audit-synthetic-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        fixture(self.root)

    def make_reader_only_failure(self):
        """Synthetic alternate history: only the64 fixture reader calls occurred."""
        chat = self.root / audit.CHAT
        with closing(sqlite3.connect(chat / 'receipts/private-embedding.sqlite')) as db, db:
            fingerprints = [r[0] for r in db.execute('SELECT fingerprint FROM operations WHERE run_ordinal>64')]
            db.executemany('DELETE FROM claims WHERE fingerprint=?', [(v,) for v in fingerprints])
            db.execute('DELETE FROM operations WHERE run_ordinal>64')
        with closing(sqlite3.connect(chat / 'run.sqlite')) as db, db:
            db.execute('DELETE FROM calls WHERE ordinal>64')
        with closing(sqlite3.connect(self.root / 'runtime/development-budget.sqlite')) as db, db:
            ids = [('synthetic-chat-' + str(i),) for i in range(65, 73)]
            db.executemany('DELETE FROM attempts WHERE id=?', ids)
            db.executemany('DELETE FROM provider_observations WHERE attempt_id=?', ids)
        for name in ('FROZEN_JUDGE.json', 'judge-requests.jsonl', 'judge-results.jsonl', 'JUDGE_COMPLETED.json'):
            (chat / name).unlink()
        original = audit.load(chat / 'READER_COMPLETED.json')
        put(self.root, audit.CHAT / 'FAILURE_COMPLETED.json', dict(state='reader_failed_no_judge', phase='reader',
            count=64, judge_requests=0, invalid_answers=1, output_sha256=original['output_sha256'],
            frozen_live_sha256=original['frozen_live_sha256'], run={'state':'complete','request_claims':64}))
        put(self.root, 'reports/memgallery-v7-chat-progress.json', dict(state='reader_complete', phase='reader', completed=64))

    def test_completed99calls_budget_usd_readonly_invalid_preserved_no_gold(self):
        before = {p.relative_to(self.root).as_posix(): audit.sha(p) for p in self.root.rglob('*') if p.is_file()}
        result = audit.audit(self.root)
        after = {p.relative_to(self.root).as_posix(): audit.sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result['total_real_provider_requests'], 99)
        self.assertEqual(result['combined_cumulative_preflight_micro_cny'], 19350000)
        self.assertEqual(result['nominal_guard_claims_including_local_failure'], 100)
        self.assertEqual(result['nominal_guard_preflight_including_local_failure_micro_cny'], 19600000)
        self.assertEqual(result['preserved_local_preflight_provider_requests'], 0)
        self.assertEqual(Decimal(result['chat_usage_estimate_usd']), Decimal('.001944'))
        self.assertEqual(result['invalid_reader_outputs_retained'], 1)
        self.assertEqual(result['scorer_only_input_hashes_not_reopened'], 1)
        self.assertFalse(result['gold_dataset_files_opened'])
        text = audit.canonical(result)
        for forbidden in ('Synthetic answer.', 'Synthetic reference.', 'Synthetic question?'):
            self.assertNotIn(forbidden, text)

    def test_incomplete_phase_blocks_before_databases_and_never_repairs(self):
        put(self.root, 'reports/memgallery-v7-chat-progress.json', dict(state='running', phase='judge'))
        with patch.object(audit, 'ro', side_effect=AssertionError('must not open database')):
            with self.assertRaises(ValueError): audit.audit(self.root)

    def test_reader_only_closed_failure_has_no_judge_and_counts_future_plan_separately(self):
        self.make_reader_only_failure()
        before = {p.relative_to(self.root).as_posix(): audit.sha(p) for p in self.root.rglob('*') if p.is_file()}
        result = audit.audit_reader_only(self.root)
        self.assertEqual(result['total_real_provider_requests'], 91)
        self.assertEqual(result['judge_requests'], 0)
        self.assertEqual(result['invalid_reader_outputs_retained'], 1)
        self.assertEqual(result['combined_cumulative_preflight_micro_cny'], 17350000)
        self.assertEqual(result['nominal_guard_preflight_including_local_failure_micro_cny'], 17600000)
        self.assertFalse(result['judge_scores_produced'])
        self.assertEqual(result['proposed_v71']['cumulative_v7_and_v71_provider_max'], 163)
        self.assertEqual(result['proposed_v71']['cumulative_provider_preflight_cny'], '35.35')
        self.assertEqual(before, {p.relative_to(self.root).as_posix(): audit.sha(p) for p in self.root.rglob('*') if p.is_file()})
        with self.assertRaises(FileNotFoundError): audit.audit(self.root)

    def test_reader_only_requires_closed_guard_no_judge_artifact_and_exact_invalid_count(self):
        self.make_reader_only_failure()
        chat = self.root / audit.CHAT
        with closing(sqlite3.connect(chat / 'run.sqlite')) as db, db:
            db.execute("UPDATE run SET state='active'")
        with self.assertRaises(ValueError): audit.audit_reader_only(self.root)
        with closing(sqlite3.connect(chat / 'run.sqlite')) as db, db:
            db.execute("UPDATE run SET state='complete'")
        put(self.root, audit.CHAT / 'JUDGE_COMPLETED.json', {'state':'complete'})
        with self.assertRaises(ValueError): audit.audit_reader_only(self.root)
        (chat / 'JUDGE_COMPLETED.json').unlink()
        marker = audit.load(chat / 'FAILURE_COMPLETED.json'); marker['invalid_answers'] = 2
        put(self.root, audit.CHAT / 'FAILURE_COMPLETED.json', marker)
        with self.assertRaises(ValueError): audit.audit_reader_only(self.root)

    def test_provider_contract_failures_are_not_reader_schema_recovery(self):
        def response():
            return dict(choices=[dict(finish_reason='stop', message=dict(content=audit.canonical(
                dict(summary='Any string', evidence_quotes=[], answer='Substantive text'))))])
        audit.original_provider_contract(response())
        for mutate in (
            lambda r: r.update(choices=[]),
            lambda r: r['choices'][0].update(finish_reason='length'),
            lambda r: r['choices'][0]['message'].update(refusal='refused'),
            lambda r: r['choices'][0]['message'].update(tool_calls=[{}]),
            lambda r: r['choices'][0]['message'].update(content='{"summary":123,"evidence_quotes":[]}'),
            lambda r: r['choices'][0]['message'].update(content='not JSON'),
        ):
            raw = response(); mutate(raw)
            with self.assertRaises(ValueError): audit.original_provider_contract(raw)

    def test_wrong_usd_hold_or_historical_row_are_blocked(self):
        path = self.root / 'runtime/development-budget.sqlite'
        for statement in (
            "UPDATE provider_observations SET estimated_usd='9' WHERE attempt_id='synthetic-chat-1'",
            "UPDATE attempts SET accounted=0 WHERE id='synthetic-chat-1'",
            "UPDATE attempts SET input_tokens=2 WHERE id='historical-0'",
        ):
            original = path.read_bytes()
            with closing(sqlite3.connect(path)) as db, db: db.execute(statement)
            with self.subTest(statement=statement), self.assertRaises(ValueError): audit.audit(self.root)
            path.write_bytes(original)

    def test_original_failure_slot_and_payloads_cannot_be_reused_or_changed(self):
        old = self.root / audit.OLD_CHAT
        path = old / 'run.sqlite'
        original = path.read_bytes()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE calls SET state='complete',returned_model='gpt-4o-mini'")
        with self.assertRaises(ValueError): audit.audit(self.root)
        path.write_bytes(original)
        with (old / 'reader-packs.jsonl').open('a', encoding='utf-8') as stream: stream.write('\n')
        with self.assertRaises(ValueError): audit.audit(self.root)

    def test_unfinished_contractfailure_and_saved_answer_tampering_block(self):
        path = self.root / audit.CHAT / 'receipts/private-embedding.sqlite'
        original = path.read_bytes()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE operations SET state='received',transport_completed=0 WHERE run_ordinal=1")
        with self.assertRaises(ValueError): audit.audit(self.root)
        path.write_bytes(original)
        ledger = self.root / 'runtime/development-budget.sqlite'
        with closing(sqlite3.connect(ledger)) as db, db:
            db.execute("UPDATE provider_observations SET category='invalid_contract' WHERE attempt_id='synthetic-chat-1'")
        with self.assertRaises(ValueError): audit.audit(self.root)
        with closing(sqlite3.connect(ledger)) as db, db:
            db.execute("UPDATE provider_observations SET category='success' WHERE attempt_id='synthetic-chat-1'")
        answers_path = self.root / audit.CHAT / 'reader-answers.jsonl'
        answers = audit.lines(answers_path); answers[1]['answer']['answer'] = 'Changed after request.'
        put_lines(answers_path, answers)
        complete = audit.load(self.root / audit.CHAT / 'READER_COMPLETED.json')
        complete['output_sha256'] = audit.sha(answers_path)
        put(self.root, audit.CHAT / 'READER_COMPLETED.json', complete)
        with self.assertRaises(ValueError): audit.audit(self.root)


if __name__ == '__main__': unittest.main()
