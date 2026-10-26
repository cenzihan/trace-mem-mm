"""Additive accounting audit after judge completion or explicit reader-only closure.

No provider, credential, guard/ledger construction, recovery, or score changes.
SQLite connections are read-only. Scorer-only gold/mapping files are not opened;
frozen judge payloads are checked for request integrity, never used to regrade.
The only output is a new aggregate report. USD usage estimates are not invoices.
"""
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_memgallery_v7_encoding import canonical, digest, sha, load, lines, ro, history, require
from agent_memory.memgallery_v7_reader import reader_payload, proxy_judge_payload, decode_reader, decode_proxy_judge
from agent_memory.visual_recovery import _strict_json

BASE = Path('runtime/memgallery-v7')
CHAT = BASE / 'chat-temperature0'
OLD_CHAT = BASE / 'chat'
PREFLIGHT_AUDIT = Path('reports/memgallery-v7-chat-preflight-failure-audit.json')
REPORT = Path('reports/memgallery-v7-chat-accounting-audit.json')
READER_ONLY_REPORT = Path('reports/memgallery-v7-reader-failure-accounting-audit.json')
VARIANTS = ('dense', 'dense_reranked', 'hybrid', 'hybrid_reranked')


def safe_path(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'relative frozen path required')
    path = (root / name).resolve()
    require(path.is_relative_to(root), 'frozen artifact path escape')
    return path


def verify_hashes(root, hashes, *, skip_scorer_only=False):
    require(isinstance(hashes, dict) and bool(hashes), 'nonempty frozen hash inventory required')
    skipped = 0
    for name, expected in hashes.items():
        path = safe_path(root, name)
        if {'scorer-only', 'quarantine'} & set(path.parts):
            require(skip_scorer_only, 'gold cannot enter reader verification')
            skipped += 1
            continue
        require(sha(path) == expected, 'frozen public input/code hash changed')
    return skipped


def complete_guard(path, requests, cap, model, *, maximum=None):
    with closing(ro(path)) as db:
        runs = db.execute('SELECT max_requests,cap,state FROM run').fetchall()
        calls = [dict(row) for row in db.execute('SELECT * FROM calls ORDER BY ordinal')]
    require(len(runs) == 1 and tuple(runs[0]) == (maximum or requests, cap, 'complete'), 'exact completed bounded guard required')
    require(len(calls) == requests and [r['ordinal'] for r in calls] == list(range(1, requests + 1)),
            'complete sequential guard calls required')
    allowed = ('gpt-4o-mini', 'gpt-4o-mini-2024-07-18') if model == 'gpt-4o-mini' else (model,)
    held = 250000 if model == 'gpt-4o-mini' else 50000
    require(all(r['state'] == 'complete' and r['model'] == model and r['returned_model'] in allowed
                and r['held'] == held for r in calls), 'unfinished/rejected/wrong-model guard call')
    return {row['ordinal']: row for row in calls}


def reader_projection(response, packed):
    try:
        return decode_reader(response, packed)
    except (ValueError, KeyError, TypeError):
        # Exact frozen runner representation, never a repaired model answer.
        return dict(answer='', cited_ids=[], confidence='invalid', schema_status='invalid',
                    citations_valid=False, answer_ids_valid=False, model=response['model'])


def preserved_local_failure(root, chat):
    """One proven nondispatched failure remains preserved; never reopen it."""
    old = root / OLD_CHAT
    proof = load(root / PREFLIGHT_AUDIT)
    require(proof.get('state') == 'confirmed_no_provider_dispatch'
            and proof.get('original_provider_attempts_before') == proof.get('original_provider_attempts_after') == 1483
            and proof.get('original_provider_attempts_added') == proof.get('original_http_dispatches') == 0
            and proof.get('original_provider_cost_added_micro_cny') == 0
            and proof.get('original_guard_claims_preserved') == 1
            and proof.get('nominal_guard_preflight_preserved_micro_cny') == 250000,
            'confirmed zero-provider preflight failure required')
    # The original progress report is intentionally reused by the new wrapper.
    # Its hash in the failure audit is a historical snapshot, not an immutable
    # runtime artifact. Exactly the old chat directory's five files remain fixed.
    immutable = {name: value for name, value in proof['original_artifact_hashes'].items()
                 if safe_path(root, name).is_relative_to(old)}
    required = {(OLD_CHAT / name).as_posix() for name in ('FROZEN_READER.json', 'FROZEN_LIVE.json',
                'reader-packs.jsonl', 'run.sqlite', 'receipts/private-embedding.sqlite')}
    require(set(immutable) == required, 'all five original immutable chat files must remain bound')
    verify_hashes(root, immutable)
    with closing(ro(old / 'run.sqlite')) as db:
        runs = [tuple(r) for r in db.execute('SELECT max_requests,cap,state FROM run')]
        calls = [tuple(r) for r in db.execute('SELECT ordinal,model,held,state,returned_model FROM calls')]
    require(runs == [(72, 18000000, 'active')] and calls == [(1, 'gpt-4o-mini', 250000, 'claimed', None)],
            'original unbound claimed slot was modified or reused')
    with closing(ro(old / 'receipts/private-embedding.sqlite')) as db:
        rows = [dict(r) for r in db.execute('SELECT * FROM operations')]
    require(len(rows) == 1 and rows[0]['state'] == 'unknown' and rows[0]['transport_completed'] == 0
            and all(rows[0][k] is None for k in ('attempt_id','status','body','body_sha','run_path','run_ordinal')),
            'original preflight journal was altered or dispatched')
    compatibility = load(chat / 'FROZEN_COMPATIBILITY.json')
    verify_hashes(root, compatibility['hashes'])
    require(compatibility.get('state') == 'identical_reader_payloads_new_bounded_execution'
            and compatibility.get('reader_payloads_byte_identical') is True
            and compatibility.get('original_claims_preserved') == 1
            and compatibility.get('provider_requests_before_recovery') == 1483
            and compatibility['hashes'].get(PREFLIGHT_AUDIT.as_posix()) == sha(root / PREFLIGHT_AUDIT)
            and sha(old / 'reader-packs.jsonl') == sha(chat / 'reader-packs.jsonl') == proof['original64packs_sha256'],
            'new execution did not preserve exact original64 payloads and failure audit')
    return proof, immutable


def original_provider_contract(response):
    """Mirror the frozen provider admission, without repairing reader content."""
    choices = response.get('choices')
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
            'original provider choice contract invalid; audit cannot repair it')
    choice, message = choices[0], choices[0].get('message')
    require(choice.get('finish_reason') == 'stop' and isinstance(message, dict)
            and not message.get('refusal') and not message.get('tool_calls')
            and isinstance(message.get('content'), str), 'original provider response was incomplete/refused')
    # The original provider uses json.loads here. Duplicate/other answer-field
    # errors remain the strict reader decoder's invalid outcome, never repaired.
    try:
        value = json.loads(message['content'])
    except (ValueError, TypeError):
        raise ValueError('original provider JSON contract invalid') from None
    require(isinstance(value, dict) and isinstance(value.get('summary'), str)
            and isinstance(value.get('evidence_quotes'), list)
            and all(isinstance(v, str) for v in value['evidence_quotes']),
            'original provider auxiliary-field contract invalid; no recovery permitted')


def audit(root=ROOT, *, reader_only=False):
    require(type(reader_only) is bool, 'explicit boolean reader-only mode required')
    root = Path(root).resolve()
    chat = root / CHAT
    progress_path = root / 'reports/memgallery-v7-chat-progress.json'
    progress = load(progress_path)
    completion_path = chat / ('FAILURE_COMPLETED.json' if reader_only else 'JUDGE_COMPLETED.json')
    completed = load(completion_path)
    if reader_only:
        require(progress.get('state') in ('reader_complete', 'reader_failed_no_judge')
                and progress.get('phase') == 'reader' and progress.get('completed') == 64
                and completed.get('state') == 'reader_failed_no_judge' and completed.get('phase') == 'reader'
                and completed.get('count') == 64 and completed.get('judge_requests') == 0
                and completed.get('run', {}).get('state') == 'complete'
                and completed.get('run', {}).get('request_claims') == 64,
                'explicit closed64-reader failure marker required; no judge completion is inferred')
        require(not any((chat / name).exists() for name in ('FROZEN_JUDGE.json', 'judge-requests.jsonl',
                    'judge-results.jsonl', 'JUDGE_COMPLETED.json')), 'reader-only audit cannot hide judge artifacts or attempts')
        judge_count = 0
    else:
        require(progress.get('state') == 'judge_complete' and progress.get('phase') == 'judge'
                and completed.get('state') == 'complete' and completed.get('phase') == 'judge',
                'audit waits for judge_complete; no recovery or partial-completion claim')
        judge_count = completed.get('count')
        require(type(judge_count) is int and 1 <= judge_count <= 8, 'one to eight completed judge batches required')
    total_chat = 64 + judge_count
    preflight, preflight_immutable = preserved_local_failure(root, chat)
    frozen, live = load(chat / 'FROZEN_READER.json'), load(chat / 'FROZEN_LIVE.json')
    judge = None if reader_only else load(chat / 'FROZEN_JUDGE.json')
    reader_complete = load(chat / 'READER_COMPLETED.json')
    require(frozen.get('planned_reader_answers') == 64 and frozen.get('max_requests') == 72
            and frozen.get('max_preflight_cny') == '18' and frozen.get('fixed_hold_cny') == '.25',
            'fixed64 reader plus bounded judge plan changed')
    require(live['baseline']['attempts'] == 1483 and live['reader_sha256'] == sha(chat / 'FROZEN_READER.json'),
            'chat freeze must follow the original1483 attempts')
    live_hash = sha(chat / 'FROZEN_LIVE.json')
    require(reader_complete.get('state') == 'complete' and reader_complete.get('phase') == 'reader'
            and reader_complete.get('count') == 64 and reader_complete.get('frozen_live_sha256') == live_hash
            and completed.get('frozen_live_sha256') == live_hash, 'reader/judge completion bound to another live freeze')
    verify_hashes(root, frozen['input_hashes']); verify_hashes(root, frozen['code_hashes'])
    skipped = 0
    paths = dict(packs=safe_path(root, frozen['packs_file']), answers=chat / 'reader-answers.jsonl')
    expected_hashes = dict(packs=frozen['packs_sha256'], answers=reader_complete['output_sha256'])
    require(paths['packs'] == chat / 'reader-packs.jsonl', 'unexpected reader pack artifact')
    if not reader_only:
        skipped = verify_hashes(root, judge['input_hashes'], skip_scorer_only=True)
        verify_hashes(root, judge['code_hashes'])
        require(judge.get('judge_requests') == judge_count and judge.get('reader_answers') == 64
                and judge.get('grades') == [0, .25, .5, .75, 1], 'judge size or proxy grade contract changed')
        paths.update(requests=safe_path(root, judge['requests_file']), results=chat / 'judge-results.jsonl')
        require(paths['requests'] == chat / 'judge-requests.jsonl', 'unexpected judge request artifact')
        expected_hashes.update(requests=judge['requests_sha256'], results=completed['output_sha256'])
    else:
        require(completed.get('output_sha256') == reader_complete['output_sha256'], 'failure marker changed original reader output')
    for key, expected in expected_hashes.items():
        require(sha(paths[key]) == expected, 'completed request/output artifact hash mismatch')
    source_audit_path = root / 'reports/memgallery-v7-encoding-audit.json'
    require(frozen['input_hashes'].get(source_audit_path.relative_to(root).as_posix()) == sha(source_audit_path),
            'reader freeze must bind the successful encoding audit')
    source_audit = load(source_audit_path)
    require(source_audit.get('state') == 'passed' and source_audit.get('real_embedding_requests') == 27
            and source_audit.get('complete_unique_run_bindings') == 27
            and source_audit.get('unknown_or_inflight_operations') == 0,
            'complete27 encoding audit required')
    verify_hashes(root, source_audit['artifact_sha256'])
    encoding = root / BASE / 'encoding'
    require(source_audit['frozen_encoding_sha256'] == sha(encoding / 'FROZEN_ENCODING.json')
            and source_audit['frozen_live_sha256'] == sha(encoding / 'FROZEN_LIVE.json'), 'encoding freeze changed after audit')
    complete_guard(encoding / 'run.sqlite', 27, 1350000, 'text-embedding-v4')
    calls = complete_guard(chat / 'run.sqlite', total_chat, 18000000, 'gpt-4o-mini', maximum=72)
    artifact_paths = list(paths.values()) + [source_audit_path, progress_path, chat / 'FROZEN_READER.json',
        chat / 'FROZEN_LIVE.json', chat / 'READER_COMPLETED.json', completion_path,
        chat / 'run.sqlite', chat / 'receipts/private-embedding.sqlite',
        encoding / 'run.sqlite', root / PREFLIGHT_AUDIT, chat / 'FROZEN_COMPATIBILITY.json']
    if not reader_only:
        artifact_paths.append(chat / 'FROZEN_JUDGE.json')
    before = {p.relative_to(root).as_posix(): sha(p) for p in artifact_paths}
    packs, answers = lines(paths['packs']), lines(paths['answers'])
    requests, results = ([], []) if reader_only else (lines(paths['requests']), lines(paths['results']))
    require(len(packs) == len(answers) == 64 and len(requests) == len(results) == judge_count, 'incomplete reader/judge outputs')
    keys = {(p.get('index'), p.get('variant')) for p in packs}
    selected = {p.get('index') for p in packs}
    require(len(selected) == 16 and all(type(v) is int and v >= 0 for v in selected)
            and keys == {(i, variant) for i in selected for variant in VARIANTS}, 'exact16x4 frozen pack identity required')
    work = []
    for meta, saved in zip(packs, answers):
        require(meta['query_id'] == 'memgallery-v7:' + str(meta['index'])
                and meta['pack']['question_index'] == meta['index'] and meta['pack']['point'] == meta['point'],
                'reader pack native identity mismatch')
        request_id = digest((meta['query_id'] + '\0' + meta['variant']).encode('utf-8'))
        require(all(saved.get(k) == meta[k] for k in ('query_id', 'index', 'point', 'variant')),
                'saved reader output identity differs from frozen order')
        work.append(('reader', request_id, reader_payload(meta['pack']), meta['pack'], saved))
    for request, saved in zip(requests, results):
        work.append(('judge', request['request_id'], proxy_judge_payload(request['pack']), request['pack'], saved))
    require(len({row[1] for row in work}) == total_chat, 'duplicate frozen chat request')
    with closing(ro(chat / 'receipts/private-embedding.sqlite')) as db:
        operations = [dict(row) for row in db.execute('SELECT * FROM operations')]
        claims = [tuple(row) for row in db.execute('SELECT scope,chunk_key,fingerprint FROM claims')]
    require(len(operations) == total_chat, 'extra/missing/unknown journal operation')
    operations = {row['fingerprint']: row for row in operations}
    namespace = 'memgallery-v7:' + live_hash
    expected_claims, used_attempts, used_ordinals, accounting = [], [], [], []
    invalid_readers = 0
    for phase, request_id, payload, packed, saved in work:
        manifest = dict(keys=[request_id], payload_sha256=digest(canonical(payload).encode('utf-8')),
                        run_path=str((chat / 'run.sqlite').resolve()))
        fingerprint = digest(canonical(dict(namespace=namespace, manifest=manifest)).encode('utf-8'))
        row = operations.get(fingerprint)
        require(row is not None and row['scope'] == namespace and row['manifest'] == canonical(manifest)
                and row['state'] == 'complete' and row['transport_completed'] == 1
                and row['status'] == 200 and row['redacted'] == 0 and row['run_path'] == manifest['run_path'],
                'receipt/transport/accounting incomplete or changed; no automatic repair')
        require(type(row['run_ordinal']) is int and row['run_ordinal'] in calls
                and saved.get('request_id') == request_id and saved.get('attempt_id') == row['attempt_id']
                and saved.get('run_ordinal') == row['run_ordinal']
                and isinstance(row['body'], bytes) and digest(row['body']) == row['body_sha'] == saved.get('receipt_sha256'),
                'saved output is not bound to its original successful receipt')
        require((1 <= row['run_ordinal'] <= 64) if phase == 'reader' else (65 <= row['run_ordinal'] <= total_chat),
                'reader and judge guard phases overlap')
        response = _strict_json(row['body'])
        require(response.get('model') == calls[row['run_ordinal']]['returned_model'], 'receipt/guard model differs')
        original_provider_contract(response)
        if phase == 'reader':
            projected = reader_projection(response, packed)
            require(projected == saved.get('answer'), 'reader answer changed after receipt; repair prohibited')
            invalid_readers += projected['schema_status'] != 'valid'
        else:
            require(response == saved.get('response'), 'saved judge output differs from raw receipt')
            decode_proxy_judge(response, packed['item_ids'])  # Contract only; no scores returned in report.
        usage = response.get('usage', {})
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        require(type(inp) is int and type(out) is int and 0 <= inp <= 128000
                and 0 <= out <= payload['max_completion_tokens'] and type(usage.get('total_tokens')) is int
                and usage['total_tokens'] == inp + out, 'receipt usage invalid or beyond request bound')
        usd = (Decimal('.15') * inp + Decimal('.60') * out) / Decimal(1000000)
        accounting.append((row['attempt_id'], inp, out, usd, response['model'], payload['max_completion_tokens']))
        expected_claims.append((namespace, request_id, fingerprint))
        used_attempts.append(row['attempt_id']); used_ordinals.append(row['run_ordinal'])
    require(sorted(claims) == sorted(expected_claims) and len(set(used_attempts)) == total_chat
            and sorted(used_ordinals) == list(range(1, total_chat + 1)), 'chat attempts/claims/ordinals are not one-to-one')
    if reader_only:
        require(type(completed.get('invalid_answers')) is int and 0 < invalid_readers == completed['invalid_answers'],
                'failure marker invalid count differs from strict original receipt decoding')
    with closing(ro(root / 'runtime/development-budget.sqlite')) as db:
        require(history(db, 1483) == live['historical_sha256'], 'historical1483 ledger/observation rows changed')
        held = 0
        for attempt, inp, out, usd, returned_model, output_cap in accounting:
            budget = db.execute('SELECT rowid AS original_rowid,* FROM attempts WHERE id=?', (attempt,)).fetchone()
            observed = db.execute('SELECT * FROM provider_observations WHERE attempt_id=?', (attempt,)).fetchone()
            require(budget is not None and observed is not None, 'receipt ledger/observation missing')
            require(budget['original_rowid'] > 1483 and budget['model'] == 'gpt-4o-mini'
                    and budget['status'] == 'uncertain' and budget['amount_kind'] == 'retained_upper_bound'
                    and budget['conversion_basis'] == 'budget_assumption_not_market_fx'
                    and budget['reserved'] == budget['accounted'] == 250000
                    and budget['input_tokens'] == inp and budget['output_tokens'] == out
                    and budget['max_input'] == 128000 and budget['max_output'] == output_cap
                    and Decimal(budget['input_rate']) == Decimal('1.5') and Decimal(budget['output_rate']) == Decimal('6'),
                    'fixed hold/observed usage differs; do not settle or repair automatically')
            require(observed['mode'] == 'live' and observed['category'] == 'success'
                    and observed['http_status'] == 200 and observed['returned_model'] == returned_model
                    and observed['input_tokens'] == inp and observed['output_tokens'] == out
                    and observed['estimated_usd'] is not None and Decimal(observed['estimated_usd']) == usd,
                    'provider contract/model/usage/USD observation differs; audit blocked')
            held += budget['accounted']
    require(total_chat + 27 <= 99 and held + 1350000 <= 19350000, 'combined V7 request/reservation bound exceeded')
    require(before == {p.relative_to(root).as_posix(): sha(p) for p in artifact_paths}, 'chat artifact changed during read-only audit')
    verify_hashes(root, source_audit['artifact_sha256'])
    verify_hashes(root, preflight_immutable)
    usd_total = sum((row[3] for row in accounting), Decimal(0))
    result = dict(schema='memgallery-v7-chat-accounting-audit-v1', state='passed',
        audit_mode='reader_only_closed_failure' if reader_only else 'completed_reader_and_judge',
        audited_at_utc=datetime.now(timezone.utc).isoformat(), reader_requests=64, judge_requests=judge_count,
        embedding_requests=27, total_real_provider_requests=total_chat + 27, maximum_provider_requests=99,
        complete_unique_chat_run_bindings=total_chat, both_executed_guards_complete=True,
        preserved_local_preflight_guard_state='active_with_one_claimed_no_provider_attempt',
        preserved_local_preflight_claims=1, preserved_local_preflight_provider_requests=0,
        preflight_failure_audit_sha256=sha(root / PREFLIGHT_AUDIT),
        original_failed_chat_immutable_files_verified=5,
        progress_report_policy='Shared mutable progress path; original failure-progress hash retained only as a historical snapshot in its immutable failure audit.',
        unknown_or_inflight_operations=0, invalid_reader_outputs_retained=invalid_readers,
        historical1483_budget_rows_unchanged=True, reader_payloads_and_outputs_verified=64,
        judge_payloads_and_outputs_verified=judge_count, chat_usage_input_tokens=sum(r[1] for r in accounting),
        chat_usage_output_tokens=sum(r[2] for r in accounting), chat_usage_estimate_usd=str(usd_total),
        chat_usage_estimate_cny_budget_assumption=str(usd_total * 10),
        embedding_usage_estimate_cny=str(Decimal(source_audit['new_usage_estimate_micro_cny']) / 1000000),
        combined_usage_estimate_cny_budget_assumption=str(usd_total * 10 + Decimal(source_audit['new_usage_estimate_micro_cny']) / 1000000),
        chat_retained_fixed_hold_micro_cny=held, combined_cumulative_preflight_micro_cny=held + 1350000,
        maximum_combined_preflight_micro_cny=19350000,
        nominal_guard_claims_including_local_failure=total_chat + 28, maximum_nominal_guard_claims=100,
        nominal_guard_preflight_including_local_failure_micro_cny=held + 1600000,
        maximum_nominal_guard_preflight_micro_cny=19600000,
        invoice_verified=False, conversion='10 CNY/USD budget assumption, not market FX',
        ledger_status_explanation='Successful HTTP/usage verified; uncertain ledger status retains fixed CNY hold pending actual bill, not an unknown transport.',
        scorer_only_input_hashes_not_reopened=skipped, gold_dataset_files_opened=False,
        answers_or_grades_in_report=False, model_contract_repair_attempted=False, provider_requests_by_audit=0,
        existing_artifacts_modified=False, artifact_sha256=before, audit_script_sha256=sha(Path(__file__)))
    if reader_only:
        result.update(judge_not_run=True, judge_scores_produced=False, retired_unexecuted_judge_requests=8,
            failure_completed_sha256=sha(completion_path), original64_reader_outputs_repaired=False,
            proposed_v71=dict(status='planned_separate_experiment_not_executed_by_this_audit',
                directory='runtime/memgallery-v7/chat-citation-v71', new_reader_requests=64, new_judge_max_requests=8,
                new_guard_max_requests=72, new_guard_preflight_cny='18',
                cumulative_v7_and_v71_provider_max=163, cumulative_provider_preflight_cny='35.35',
                nominal_guard_claims_max_including_original_preflight=164,
                nominal_guard_preflight_cny='35.60', shared_budget_cny='500',
                change='short source aliases; explicit new representation experiment on the same16x4, never replay of the original requests'))
    return result


def audit_reader_only(root=ROOT):
    return audit(root, reader_only=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader-only', action='store_true', help='audit explicit closed reader failure, without creating judge results')
    args = parser.parse_args()
    result = audit_reader_only() if args.reader_only else audit()
    report = READER_ONLY_REPORT if args.reader_only else REPORT
    destination = ROOT / report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state=result['state'], requests=result['total_real_provider_requests'], report=report.as_posix())))
