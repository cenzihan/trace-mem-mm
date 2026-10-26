"""Additive JSON-format-only correction of the frozen V7.1 blind judge.

No new selection, reference lookup, reader inference, answer repair or network.
The previous closed65 and every frozen original batch remain independently bound.
"""
import argparse
from contextlib import closing, contextmanager
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_memgallery_v71 as prior
from scripts import score_memgallery_v7 as base
from scripts.prepare_memgallery_v7 import save_new

WORK = 'runtime/memgallery-v7/judge-json-v71'
FROZEN = WORK + '/FROZEN_JUDGE.json'
REQUESTS = WORK + '/judge-requests.jsonl'
MAPPING = WORK + '/scorer-only/judge-mapping.json'
LIVE = WORK + '/FROZEN_LIVE.json'
GUARD = WORK + '/run.sqlite'
JOURNAL = WORK + '/receipts/private-embedding.sqlite'
RESULTS = WORK + '/judge-results.jsonl'
COMPLETE = WORK + '/JUDGE_COMPLETED.json'
PROGRESS = 'reports/memgallery-v71-judge-json-progress.json'
REJECTION = 'reports/memgallery-v71-judge-json-rejection.json'
PREP_REPORT = 'reports/memgallery-v71-judge-json-preparation.json'
SCORES = 'reports/memgallery-v71-reader-scores.json'
DETAILS = WORK + '/scorer-only/reader-scores.json'
SUFFIX = '\nReturn JSON.'
CODE = tuple(dict.fromkeys((*prior.ORIGINAL_CODE, *prior.EXTRA_CODE,
    'scripts/score_memgallery_v71_judge_json.py')))
PROVENANCE = dict(**prior.PROVENANCE, judge_format_correction='system suffix: \\nReturn JSON.',
    new_reader_inferences_in_correction=0, reader_answers_changed=False,
    original_blind_batches_and_mapping_preserved=True, original_native_gold_reopened=False,
    new_judge_inferences=8, rejected_judge_requests_preserved=1)


@contextmanager
def bindings(**values):
    old = {name: getattr(base, name) for name in values}
    try:
        for name, value in values.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def old_guard_calls(root, *, final=False):
    """Accept only an explicitly closed64-success plus acknowledged400 run."""
    with closing(base.ro(root, prior.PATHS['GUARD'])) as db:
        runs = db.execute('SELECT max_requests,cap,state FROM run').fetchall()
        rows = [dict(r) for r in db.execute(
            'SELECT ordinal,model,held,state,returned_model FROM calls ORDER BY ordinal')]
        rejects = [dict(r) for r in db.execute('SELECT * FROM known_json400_rejections')]
    if (len(runs) != 1 or tuple(runs[0]) != (72, 18000000, 'complete')
            or len(rows) != 65 or [r['ordinal'] for r in rows] != list(range(1, 66))
            or any(r['model'] != 'gpt-4o-mini' or r['held'] != 250000 for r in rows)
            or any(r['state'] != 'complete' or r['returned_model'] not in
                ('gpt-4o-mini', 'gpt-4o-mini-2024-07-18') for r in rows[:64])
            or rows[64]['state'] != 'rejected_response' or len(rejects) != 1
            or rejects[0]['ordinal'] != 65 or rejects[0]['http_status'] != 400
            or rejects[0]['category'] != 'client_error' or rows[64]['returned_model'] is not None
            or rejects[0]['receipt_sha256'] != 'c259bb7f479b9faee0845ca7c0c7d8fca618e090ab8f4270edd4de252b2488f3'):
        raise ValueError('closed64 success plus acknowledged400 guard required')
    return {r['ordinal']: r for r in rows[:64]}


def verify_reader(root):
    names = (*prior.PATHS, 'reader_payload', 'decode_reader', 'CODE',
             'verify_reader_complete', 'save_new', 'guard_calls')
    with bindings(**{name: getattr(base, name) for name in names}):
        prior.install()
        base.guard_calls = old_guard_calls
        return prior.verify_reader_complete(root, final=True)


def verify_rejection(root):
    report = base.load(root, REJECTION)
    if report.get('state') != 'passed':
        raise ValueError('independent explicit400 acknowledgement audit required')
    hashes = report.get('artifact_sha256')
    required = {prior.PATHS[name] for name in ('GUARD', 'JOURNAL', 'ANSWERS',
                                              'JUDGE_FROZEN', 'JUDGE_REQUESTS')}
    if not isinstance(hashes, dict) or not required <= set(hashes):
        raise ValueError('rejection audit must bind closed run and original artifacts')
    base.check_hashes(root, hashes, gold_allowed=True)
    old_guard_calls(root)
    return report


def corrected_payload(pack, count_tokens):
    original = base.proxy_judge_payload(pack)
    payload = copy.deepcopy(original)
    payload['messages'][0]['content'] += SUFFIX
    verify_correction(original, payload)
    system = payload['messages'][0]['content']
    user = payload['messages'][1]['content'][0]['text']
    counts = [count_tokens(system), count_tokens(user)]
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('nonnegative integer tokenizer counts required')
    tokens, size = sum(counts) + 32, len(system.encode('utf-8')) + len(user.encode('utf-8'))
    if tokens > 8192 or size > 24000:
        raise ValueError('JSON suffix exceeds original judge input cap; no truncation')
    return payload, dict(text_token_count=tokens, utf8_text_bytes=size)


def verify_correction(original, corrected):
    expected = copy.deepcopy(original)
    expected['messages'][0]['content'] += SUFFIX
    if corrected != expected:
        raise ValueError('only the exact system JSON suffix may change')


def corrected_batches(requests, mapping, reader, count_tokens):
    if len(requests) != 8 or len(mapping) != 8:
        raise ValueError('all eight original blind batches required')
    private = {r['request_id']: r['items'] for r in mapping}
    if len(private) != 8 or len({r['request_id'] for r in requests}) != 8:
        raise ValueError('unique original batch IDs required')
    answers = {(r['index'], r['variant']): r for r in reader['answers']}
    out, remapped, stats, seen = [], [], [], set()
    for request in requests:
        old_id, pack = request['request_id'], request['pack']
        if old_id not in private:
            raise ValueError('original blind mapping differs')
        payload, budget = corrected_payload(pack, count_tokens)
        items = private[old_id]
        if len(items) != 8 or {v['item_id'] for v in items} != set(pack['item_ids']):
            raise ValueError('exact eight-item original blind mapping required')
        public = base.strict_load(payload['messages'][1]['content'][0]['text'])
        public_by_id = {v['item_id']: v for v in public}
        if len(public) != 8 or set(public_by_id) != set(pack['item_ids']):
            raise ValueError('original blind request item coverage changed')
        for item in items:
            key = item['index'], item['variant']
            if key not in answers or key in seen:
                raise ValueError('duplicate or extra mapped reader answer')
            answer = answers[key]
            invalid = answer['answer']['schema_status'] != 'valid'
            if (type(item['invalid_schema']) is not bool or item['invalid_schema'] != invalid
                    or item['point'] != answer['point']
                    or public_by_id[item['item_id']]['point'] != item['point']
                    or public_by_id[item['item_id']]['prediction'] !=
                        ('INVALID_READER_RESPONSE' if invalid else answer['answer']['answer'])):
                raise ValueError('frozen blind prediction differs from original reader')
            seen.add(key)
        identity = base.digest(base.canonical(['judge-json-v71', old_id, payload]).encode('utf-8'))
        out.append(dict(request_id=identity, payload=payload))
        remapped.append(dict(request_id=identity, original_request_id=old_id, items=copy.deepcopy(items)))
        stats.append(budget)
    if seen != set(answers) or len(seen) != 64:
        raise ValueError('same64 reader outcomes must remain covered')
    return out, remapped, stats


def old_artifacts(root, reader):
    frozen = base.load(root, prior.PATHS['JUDGE_FROZEN'])
    if (frozen.get('judge_requests') != 8 or frozen.get('reader_answers') != 64
            or frozen.get('requests_file') != prior.PATHS['JUDGE_REQUESTS']
            or frozen.get('mapping_file') != prior.PATHS['JUDGE_MAPPING']
            or frozen.get('grades') != [0, .25, .5, .75, 1]):
        raise ValueError('original five-grade blind freeze required')
    base.check_hashes(root, frozen['code_hashes'])
    # The already frozen requests contain references. Never reopen native gold.
    inputs = {name: value for name, value in frozen['input_hashes'].items() if name != base.GOLD}
    base.check_hashes(root, inputs, gold_allowed=True)
    for name, value in reader['input_hashes'].items():
        if frozen['input_hashes'].get(name) != value:
            raise ValueError('original judge no longer binds all reader inputs')
    requests = base.lines(base.verified(root, prior.PATHS['JUDGE_REQUESTS'], frozen['requests_sha256']))
    mapping = base.strict_load(base.verified(root, prior.PATHS['JUDGE_MAPPING'], frozen['mapping_sha256']))
    return frozen, requests, mapping, inputs


def prepare(root=ROOT, *, count_tokens=None):
    for name in (FROZEN, REQUESTS, MAPPING, PREP_REPORT):
        if base.path_at(root, name).exists():
            raise FileExistsError('JSON correction already prepared; preserve frozen artifacts')
    verify_rejection(root)
    reader = verify_reader(root)
    old, requests, mapping, inputs = old_artifacts(root, reader)
    requests, mapping, stats = corrected_batches(requests, mapping, reader,
                                                 count_tokens or base.local_tokenizer(root))
    base.check_hashes(root, inputs, gold_allowed=True)
    raw = b''.join(base.canonical(row).encode('utf-8') + b'\n' for row in requests)
    request_hash = save_new(base.path_at(root, REQUESTS), raw, raw=True)
    mapping_hash = save_new(base.path_at(root, MAPPING), mapping)
    bound = (*reader['input_hashes'], prior.PATHS['JUDGE_FROZEN'], prior.PATHS['JUDGE_REQUESTS'],
             prior.PATHS['JUDGE_MAPPING'], prior.PATHS['GUARD'], prior.PATHS['JOURNAL'], REJECTION)
    inputs.update({name: base.sha(root, name) for name in bound})
    inputs[MAPPING] = mapping_hash
    frozen = dict(state='json_format_correction_frozen_after_reader64_and_rejected400',
        requests_file=REQUESTS, requests_sha256=request_hash, request_count=8, judge_requests=8,
        planned_new_requests=8, max_requests=8, max_preflight_cny='2', fixed_hold_cny='.25',
        input_hashes=inputs, code_hashes={name: base.sha(root, name) for name in CODE},
        old_judge_frozen_file=prior.PATHS['JUDGE_FROZEN'],
        old_judge_frozen_sha256=base.sha(root, prior.PATHS['JUDGE_FROZEN']),
        old_failed_complete_file=REJECTION, old_failed_complete_sha256=base.sha(root, REJECTION),
        mapping_file=MAPPING, mapping_sha256=mapping_hash, model='gpt-4o-mini',
        questions=16, reader_answers=64, grades=[0, .25, .5, .75, 1], seed=base.SEED,
        invalid_reader_outcomes_score_zero=True, method_names_sent_to_judge=False,
        output_tokens=1536, max_input_tokens=8192, max_utf8_bytes=24000,
        original_native_gold_sha256_metadata=old['input_hashes'][base.GOLD],
        batch_budget_stats=stats, **PROVENANCE)
    frozen_hash = save_new(base.path_at(root, FROZEN), frozen)
    report = dict(state=frozen['state'], frozen_file=FROZEN, frozen_sha256=frozen_hash,
        requests_file=REQUESTS, requests_sha256=request_hash, questions=16, reader_answers=64,
        invalid_reader_answers=sum(r['answer']['schema_status'] != 'valid' for r in reader['answers']),
        judge_requests=8, provider_requests=0, max_input_tokens_observed=max(s['text_token_count'] for s in stats),
        max_utf8_bytes_observed=max(s['utf8_text_bytes'] for s in stats), official_score=False, **PROVENANCE)
    save_new(base.path_at(root, PREP_REPORT), report)
    return report


def judge_guard(root):
    with closing(base.ro(root, GUARD)) as db:
        runs = db.execute('SELECT max_requests,cap,state FROM run').fetchall()
        rows = [dict(r) for r in db.execute(
            'SELECT ordinal,model,held,state,returned_model FROM calls ORDER BY ordinal')]
    if (len(runs) != 1 or tuple(runs[0]) != (8, 2000000, 'complete') or len(rows) != 8
            or [r['ordinal'] for r in rows] != list(range(1, 9))
            or any(r['model'] != 'gpt-4o-mini' or r['held'] != 250000 or r['state'] != 'complete'
                   or r['returned_model'] not in ('gpt-4o-mini', 'gpt-4o-mini-2024-07-18') for r in rows)):
        raise ValueError('fresh eight-call completed judge guard required')
    return {r['ordinal']: r for r in rows}


def finalize(root=ROOT, *, bootstrap_samples=5000):
    if any(base.path_at(root, name).exists() for name in (SCORES, DETAILS)):
        raise FileExistsError('final scores already exist; preserve them')
    completed, progress = base.load(root, COMPLETE), base.load(root, PROGRESS)
    live_raw = base.verified(root, LIVE, completed['frozen_live_sha256'])
    live, frozen = base.strict_load(live_raw), base.load(root, FROZEN)
    if (completed.get('state') != 'complete' or completed.get('phase') != 'judge' or completed.get('count') != 8
            or progress.get('state') != 'judge_complete' or progress.get('phase') != 'judge'
            or live.get('judge_sha256') != base.sha(root, FROZEN)
            or frozen.get('requests_file') != REQUESTS or frozen.get('request_count') != 8):
        raise ValueError('completed bound JSON correction judge required')
    base.check_hashes(root, live['code_hashes'])
    base.check_hashes(root, frozen['code_hashes'])
    base.check_hashes(root, frozen['input_hashes'], gold_allowed=True)
    verify_rejection(root)
    reader = verify_reader(root)
    _, old_requests, old_mapping, _ = old_artifacts(root, reader)
    expected, expected_mapping, _ = corrected_batches(old_requests, old_mapping, reader, base.local_tokenizer(root))
    requests = base.lines(base.verified(root, REQUESTS, frozen['requests_sha256']))
    mapping = base.strict_load(base.verified(root, MAPPING, frozen['mapping_sha256']))
    if requests != expected or mapping != expected_mapping:
        raise ValueError('original batches or JSON-only correction changed')
    results = base.lines(base.verified(root, RESULTS, completed['output_sha256']))
    calls = judge_guard(root)
    by_id = {r['request_id']: r for r in requests}
    private = {r['request_id']: r['items'] for r in mapping}
    namespace = 'memgallery-v71-judge-json:' + base.digest(live_raw)
    seen, attempts, ordinals, details = set(), set(reader['reader_attempts']), set(), []
    with bindings(GUARD=GUARD, JOURNAL=JOURNAL):
        for row in results:
            identity = row.get('request_id')
            if (identity not in by_id or identity in seen or row.get('attempt_id') in attempts
                    or type(row.get('run_ordinal')) is not int or row['run_ordinal'] not in calls
                    or row['run_ordinal'] in ordinals):
                raise ValueError('one distinct new receipt per original blind batch required')
            response = base.receipt_response(root, namespace, identity, by_id[identity]['payload'], row, calls)
            if response != row.get('response'):
                raise ValueError('saved judge output differs from durable receipt')
            items = private[identity]
            grades = base.decode_proxy_judge(response, [item['item_id'] for item in items])
            for item in items:
                details.append(dict(index=item['index'], point=item['point'], variant=item['variant'],
                    invalid_schema=item['invalid_schema'], score=0. if item['invalid_schema'] else grades[item['item_id']]))
            seen.add(identity); attempts.add(row['attempt_id']); ordinals.add(row['run_ordinal'])
    if seen != set(by_id) or ordinals != set(range(1, 9)) or len(details) != 64:
        raise ValueError('complete eight-batch sixty-four-item judge required')
    with closing(base.ro(root, JOURNAL)) as db:
        if (db.execute('SELECT count(*) FROM operations').fetchone()[0] != 8
                or db.execute('SELECT count(*) FROM claims').fetchone()[0] != 8):
            raise ValueError('extra journal operations or claims')
    groups = base.aggregate_reader(details, samples=bootstrap_samples)
    base.check_hashes(root, frozen['input_hashes'], gold_allowed=True)
    report = dict(state='complete', questions=16, reader_answers=64, judge_requests=8, groups=groups,
        metric='gpt-4o-mini five-grade blinded text-derived proxy; not official Mem-Gallery score',
        grades=[0, .25, .5, .75, 1], invalid_reader_outcomes_score_zero=True,
        invalid_reader_answers=sum(r['invalid_schema'] for r in details),
        upstream_provenance=base.UPSTREAM_PROVENANCE, official_score=False, official_smoke=0, official_full=0,
        bootstrap=dict(seed=base.SEED, samples=bootstrap_samples, unit='paired original question',
            caveat='descriptive small public-scene pilot; no multiplicity or shared-evidence dependence adjustment'),
        judge_frozen_sha256=base.sha(root, FROZEN), reader_answers_sha256=base.sha(root, prior.PATHS['ANSWERS']),
        judge_results_sha256=base.sha(root, RESULTS), judge_live_sha256=base.digest(live_raw), provider_requests=0,
        raw_model_answers_changed=False, old_experiment_results_overwritten=False, **PROVENANCE)
    save_new(base.path_at(root, DETAILS), details)
    save_new(base.path_at(root, SCORES), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'final'))
    args = parser.parse_args(argv)
    try:
        result = prepare() if args.phase == 'prepare' else finalize()
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    except Exception as exc:
        raise SystemExit('Isolated JSON correction scorer stopped: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
