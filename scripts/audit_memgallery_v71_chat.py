"""Final V7.1 accounting and interface audit, including the failed original64.

Read-only databases and artifacts; no retry, answer repair, scoring, credentials
or network. No report until the new judge finishes and its guard is closed.
The previous reader-only audit is rerun read-only, never overwritten.
"""
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_memgallery_v7_chat as prior
from scripts.audit_memgallery_v7_encoding import canonical, digest, sha, load, lines, ro, history, require
from agent_memory import memgallery_v71_reader as native
from agent_memory.memgallery_v7_reader import proxy_judge_payload, decode_proxy_judge
from agent_memory.visual_recovery import _strict_json

BASE = Path('runtime/memgallery-v7')
OLD = BASE / 'chat-temperature0'
CHAT = BASE / 'chat-citation-v71'
OLD_AUDIT = Path('reports/memgallery-v7-reader-failure-accounting-audit.json')
REPORT = Path('reports/memgallery-v71-chat-accounting-audit.json')


def same_inputs(previous, current):
    require(len(previous) == len(current) == 64, 'exact64 old/new reader packs required')
    aliases = set()
    for old, new in zip(previous, current):
        require(all(old[key] == new[key] for key in ('query_id','index','point','variant')),
                'V7.1 changed question/condition/order')
        left, right = old['pack'], new['pack']
        native.reader_payload(right)
        require(right['schema'] == 'memgallery-v71-reader-pack-v1' and left['schema'] == 'memgallery-v7-reader-pack-v1',
                'new explicit alias representation required')
        for field in ('question_sha256','source_memory_ids','ranked_native_ids','source_sha256','dropped_native_ids',
                      'question_index','point','image_ids','query_image_count','query_image_condition','budget'):
            require(left[field] == right[field], 'native questions, source text/order or budget changed')
        require(len(right['source_memory_ids']) == 20 and right['source_records_retained'] == 20
                and left['payload_sha256'] != right['payload_sha256'],
                'same20 complete sources but new citation payload required, never original replay')
        aliases.add(right['alias_map_sha256'])
    require(len(aliases) == 1, 'all64 requests must use the same complete-inventory alias map')
    return next(iter(aliases))


def projection(response, packed):
    try:
        return native.decode_reader(response, packed)
    except (ValueError, KeyError, TypeError):
        return dict(answer='', cited_ids=[], confidence='invalid', schema_status='invalid',
                    citations_valid=False, answer_ids_valid=False, model=response['model'])


def audit(root=ROOT):
    root = Path(root).resolve(); chat = root / CHAT
    progress_path = root / 'reports/memgallery-v71-chat-progress.json'
    progress, completion = load(progress_path), load(chat / 'JUDGE_COMPLETED.json')
    require(progress.get('state') == 'judge_complete' and progress.get('phase') == 'judge'
            and completion.get('state') == 'complete' and completion.get('phase') == 'judge',
            'V7.1 audit waits for completed judge; no automatic recovery')
    judge_count = completion.get('count')
    require(type(judge_count) is int and 1 <= judge_count <= 8, 'one to eight V7.1 judge calls required')
    new_count = 64 + judge_count
    interface = load(chat / 'FROZEN_INTERFACE.json')
    prior.verify_hashes(root, interface['hashes'])
    require(interface.get('state') == 'new_alias_interface_frozen_no_calls'
            and interface.get('same_queries_rankings_complete20sources') is True
            and interface.get('new_provider_calls_max') == 72 and interface.get('new_preflight_cny') == '18'
            and interface.get('aggregate_real_calls_max') == 163
            and interface.get('aggregate_provider_preflight_cny') == '35.35'
            and interface['hashes'].get(OLD_AUDIT.as_posix()) == sha(root / OLD_AUDIT),
            'new interface experiment must bind original failed64 and cumulative cap')
    old_report = load(root / OLD_AUDIT)
    require(old_report.get('state') == 'passed' and old_report.get('audit_mode') == 'reader_only_closed_failure'
            and old_report.get('reader_requests') == 64 and old_report.get('judge_requests') == 0
            and old_report.get('total_real_provider_requests') == 91, 'original reader failure audit missing')
    prior.verify_hashes(root, old_report['artifact_sha256'])
    old = prior.audit_reader_only(root)
    for key in ('total_real_provider_requests','invalid_reader_outputs_retained','chat_usage_estimate_usd',
                'embedding_usage_estimate_cny','combined_cumulative_preflight_micro_cny'):
        require(old[key] == old_report[key], 'original verified64 usage/outcomes changed')
    frozen, live = load(chat / 'FROZEN_READER.json'), load(chat / 'FROZEN_LIVE.json')
    judge, reader_complete = load(chat / 'FROZEN_JUDGE.json'), load(chat / 'READER_COMPLETED.json')
    require(frozen.get('planned_reader_answers') == 64 and frozen.get('max_requests') == 72
            and frozen.get('max_preflight_cny') == '18' and frozen.get('fixed_hold_cny') == '.25'
            and live['baseline']['attempts'] == 1547 and live['reader_sha256'] == sha(chat / 'FROZEN_READER.json'),
            'new reader freeze/1547-attempt baseline differs')
    live_hash = sha(chat / 'FROZEN_LIVE.json')
    namespace = 'memgallery-v7:' + live_hash
    require(live_hash != sha(root / OLD / 'FROZEN_LIVE.json'), 'new request namespace must differ from old64')
    require(reader_complete.get('state') == 'complete' and reader_complete.get('phase') == 'reader'
            and reader_complete.get('count') == 64 and reader_complete.get('frozen_live_sha256') == live_hash
            and completion.get('frozen_live_sha256') == live_hash, 'new completions are not bound to this guard')
    prior.verify_hashes(root, frozen['input_hashes']); prior.verify_hashes(root, frozen['code_hashes'])
    skipped = prior.verify_hashes(root, judge['input_hashes'], skip_scorer_only=True)
    prior.verify_hashes(root, judge['code_hashes'])
    require(judge.get('judge_requests') == judge_count and judge.get('reader_answers') == 64
            and judge.get('grades') == [0,.25,.5,.75,1], 'new judge contract differs')
    paths = dict(packs=prior.safe_path(root, frozen['packs_file']), requests=prior.safe_path(root, judge['requests_file']),
                 answers=chat / 'reader-answers.jsonl', results=chat / 'judge-results.jsonl')
    require(paths['packs'] == chat / 'reader-packs.jsonl' and paths['requests'] == chat / 'judge-requests.jsonl',
            'unexpected new request paths')
    for key, expected in (('packs',frozen['packs_sha256']),('requests',judge['requests_sha256']),
                          ('answers',reader_complete['output_sha256']),('results',completion['output_sha256'])):
        require(sha(paths[key]) == expected, 'new request/output hash changed')
    packs, answers, requests, results = (lines(paths[key]) for key in ('packs','answers','requests','results'))
    require(len(packs) == len(answers) == 64 and len(requests) == len(results) == judge_count, 'new request/output coverage incomplete')
    alias_hash = same_inputs(lines(root / OLD / 'reader-packs.jsonl'), packs)
    selected = {p['index'] for p in packs}
    require(len(selected) == 16 and {(p['index'],p['variant']) for p in packs}
            == {(i,v) for i in selected for v in prior.VARIANTS}, 'same16x4 new requests required')
    calls = prior.complete_guard(chat / 'run.sqlite', new_count, 18000000, 'gpt-4o-mini', maximum=72)
    files = list(paths.values()) + [chat / name for name in ('FROZEN_READER.json','FROZEN_LIVE.json','FROZEN_INTERFACE.json',
             'FROZEN_JUDGE.json','READER_COMPLETED.json','JUDGE_COMPLETED.json','run.sqlite','receipts/private-embedding.sqlite')]
    files += [progress_path, root / OLD_AUDIT]
    before = {p.relative_to(root).as_posix(): sha(p) for p in files}
    with closing(ro(chat / 'receipts/private-embedding.sqlite')) as db:
        operation_rows = [dict(r) for r in db.execute('SELECT * FROM operations')]
        claims = [tuple(r) for r in db.execute('SELECT scope,chunk_key,fingerprint FROM claims')]
    require(len(operation_rows) == new_count, 'extra/missing/inflight new journal operations')
    operations = {row['fingerprint']: row for row in operation_rows}
    work = []
    for meta, saved in zip(packs, answers):
        require(meta['query_id'] == 'memgallery-v7:' + str(meta['index'])
                and all(saved.get(k) == meta[k] for k in ('query_id','index','point','variant')), 'saved new reader identity/order differs')
        rid = digest((meta['query_id'] + '\0' + meta['variant']).encode('utf-8'))
        work.append(('reader', rid, native.reader_payload(meta['pack']), meta['pack'], saved))
    for request, saved in zip(requests, results):
        work.append(('judge', request['request_id'], proxy_judge_payload(request['pack']), request['pack'], saved))
    require(len({r[1] for r in work}) == new_count, 'duplicate new request IDs')
    expected_claims, attempts, ordinals, accounting, invalid = [], [], [], [], 0
    for phase, rid, payload, packed, saved in work:
        manifest = dict(keys=[rid], payload_sha256=digest(canonical(payload).encode()), run_path=str((chat / 'run.sqlite').resolve()))
        fingerprint = digest(canonical(dict(namespace=namespace,manifest=manifest)).encode())
        row = operations.get(fingerprint)
        require(row is not None and row['scope'] == namespace and row['manifest'] == canonical(manifest)
                and row['state'] == 'complete' and row['transport_completed'] == 1 and row['status'] == 200
                and row['redacted'] == 0 and row['run_path'] == manifest['run_path'], 'new journal receipt not completed/bound')
        require(type(row['run_ordinal']) is int and row['run_ordinal'] in calls and saved.get('request_id') == rid
                and saved.get('attempt_id') == row['attempt_id'] and saved.get('run_ordinal') == row['run_ordinal']
                and isinstance(row['body'],bytes) and digest(row['body']) == row['body_sha'] == saved.get('receipt_sha256'),
                'saved new outcome differs from its exact raw receipt')
        require((1 <= row['run_ordinal'] <= 64) if phase == 'reader' else (65 <= row['run_ordinal'] <= new_count),
                'new phase guard ordinals overlap')
        response = _strict_json(row['body'])
        require(response.get('model') == calls[row['run_ordinal']]['returned_model'], 'new receipt/guard model differs')
        prior.original_provider_contract(response)
        if phase == 'reader':
            decoded = projection(response, packed)
            require(decoded == saved.get('answer'), 'new alias decoder result differs from saved answer')
            invalid += decoded['schema_status'] != 'valid'
        else:
            require(saved.get('response') == response, 'new saved judge response changed')
            decode_proxy_judge(response, packed['item_ids'])
        usage = response.get('usage', {}); inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        require(type(inp) is int and type(out) is int and 0 <= inp <= 128000 and 0 <= out <= payload['max_completion_tokens']
                and type(usage.get('total_tokens')) is int and usage['total_tokens'] == inp + out, 'new receipt usage invalid')
        usd = (Decimal('.15')*inp + Decimal('.60')*out)/1000000
        accounting.append((row['attempt_id'], inp, out, usd, response['model'], payload['max_completion_tokens']))
        attempts.append(row['attempt_id']); ordinals.append(row['run_ordinal'])
        expected_claims.append((namespace, rid, fingerprint))
    require(sorted(claims) == sorted(expected_claims) and len(set(attempts)) == new_count
            and sorted(ordinals) == list(range(1,new_count+1)), 'new request/receipt/claim binding incomplete')
    with closing(ro(root / 'runtime/development-budget.sqlite')) as db:
        require(history(db, 1547) == live['historical_sha256'], 'historical1547 including original64 changed')
        held = 0
        for attempt, inp, out, usd, returned, cap in accounting:
            row = db.execute('SELECT rowid AS original_rowid,* FROM attempts WHERE id=?',(attempt,)).fetchone()
            observed = db.execute('SELECT * FROM provider_observations WHERE attempt_id=?',(attempt,)).fetchone()
            require(row is not None and observed is not None and row['original_rowid'] > 1547
                    and row['model'] == 'gpt-4o-mini' and row['status'] == 'uncertain'
                    and row['amount_kind'] == 'retained_upper_bound'
                    and row['conversion_basis'] == 'budget_assumption_not_market_fx'
                    and row['reserved'] == row['accounted'] == 250000
                    and row['input_tokens'] == inp and row['output_tokens'] == out
                    and row['max_input'] == 128000 and row['max_output'] == cap
                    and Decimal(row['input_rate']) == Decimal('1.5') and Decimal(row['output_rate']) == Decimal('6'),
                    'new fixed hold/usage/rate accounting changed')
            require(observed['mode'] == 'live' and observed['category'] == 'success' and observed['http_status'] == 200
                    and observed['returned_model'] == returned and observed['input_tokens'] == inp
                    and observed['output_tokens'] == out and observed['estimated_usd'] is not None
                    and Decimal(observed['estimated_usd']) == usd, 'new USD/provider observation differs')
            held += row['accounted']
    actual = 91 + new_count
    preflight = old_report['combined_cumulative_preflight_micro_cny'] + held
    require(actual <= 163 and preflight <= 35350000, 'cumulative V7+V7.1 budget/request limit exceeded')
    require(before == {p.relative_to(root).as_posix(): sha(p) for p in files}, 'new artifacts changed during read-only audit')
    prior.verify_hashes(root, old_report['artifact_sha256']); prior.verify_hashes(root, interface['hashes'])
    new_usd = sum((r[3] for r in accounting), Decimal(0))
    old_usd = Decimal(old_report['chat_usage_estimate_usd'])
    embedding_cny = Decimal(old_report['embedding_usage_estimate_cny'])
    return dict(schema='memgallery-v71-chat-accounting-audit-v1', state='passed',
        audited_at_utc=datetime.now(timezone.utc).isoformat(), experiment='new_alias_interface_regression_same_scene',
        new_reader_requests=64, new_judge_requests=judge_count, original_reader_requests_preserved=64,
        original_reader_invalid_outputs_preserved=old_report['invalid_reader_outputs_retained'],
        new_reader_invalid_outputs_retained=invalid, embedding_requests=27, total_real_provider_requests=actual,
        maximum_cumulative_provider_requests=163, original_preflight_claims_preserved=1,
        original_preflight_provider_requests=0, nominal_guard_claims_including_preflight=actual+1,
        maximum_nominal_guard_claims=164, cumulative_provider_preflight_micro_cny=preflight,
        maximum_cumulative_provider_preflight_micro_cny=35350000,
        nominal_guard_preflight_including_preflight_micro_cny=preflight+250000,
        maximum_nominal_guard_preflight_micro_cny=35600000, shared_budget_cny='500',
        old_reader_usage_estimate_usd=str(old_usd), new_chat_usage_estimate_usd=str(new_usd),
        combined_chat_usage_estimate_usd=str(old_usd+new_usd), embedding_usage_estimate_cny=str(embedding_cny),
        combined_usage_estimate_cny_budget_assumption=str((old_usd+new_usd)*10+embedding_cny),
        actual_invoice_verified=False, conversion='10 CNY/USD conservative budget assumption, not market FX',
        historical1547_rows_unchanged=True, old_closed64_guard_preserved=True,
        old_preflight_guard1_preserved=True, new_guard_complete=True, new_unknown_or_inflight_operations=0,
        native_questions_and_ordered20_sources_unchanged=True, new_alias_representation_payloads_changed=True,
        source_alias_map_sha256=alias_hash, original_calls_replayed=0, model_answers_repaired=False,
        scorer_only_input_hashes_not_reopened=skipped, gold_dataset_files_opened=False,
        question_answer_or_grades_in_report=False, provider_requests_by_audit=0, existing_artifacts_modified=False,
        previous_reader_failure_audit_sha256=sha(root / OLD_AUDIT), artifact_sha256=before,
        audit_script_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    result = audit()
    destination = ROOT / REPORT
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(state=result['state'], requests=result['total_real_provider_requests'], report=REPORT.as_posix())))
