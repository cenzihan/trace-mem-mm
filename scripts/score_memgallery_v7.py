"""Isolated V7 scoring; gold is opened only after frozen complete predictions.

No provider, network, credential or model construction. Per-question material is
written only under the private runtime directory; reports contain aggregates.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.memgallery_v7 import native_id, VARIANTS
from agent_memory.memgallery_v7_reader import (
    build_proxy_judge, decode_proxy_judge, decode_reader, proxy_judge_payload,
    question_text, reader_payload, UPSTREAM_PROVENANCE, POINTS,
)
from scripts.prepare_memgallery_v7 import strict_load, save_new

BASE = 'runtime/memgallery-v7'
CHAT = BASE + '/chat-temperature0'
GOLD = BASE + '/scorer-only/gold.json'
QUERIES = BASE + '/questions-public.json'
SELECTION = BASE + '/READER_SELECTION.json'
PREPARATION = BASE + '/PREPARATION.json'
MAPPING = BASE + '/encoding/source-mapping.json'
ENCODING = BASE + '/encoding/FROZEN_ENCODING.json'
RANKINGS = BASE + '/rerank/retrieval-results.jsonl'
PACKS = CHAT + '/reader-packs.jsonl'
ANSWERS = CHAT + '/reader-answers.jsonl'
READER_FROZEN = CHAT + '/FROZEN_READER.json'
LIVE_FROZEN = CHAT + '/FROZEN_LIVE.json'
JUDGE_FROZEN = CHAT + '/FROZEN_JUDGE.json'
JUDGE_REQUESTS = CHAT + '/judge-requests.jsonl'
JUDGE_MAPPING = CHAT + '/scorer-only/judge-mapping.json'
JUDGE_RESULTS = CHAT + '/judge-results.jsonl'
CHAT_PROGRESS = 'reports/memgallery-v7-chat-progress.json'
GUARD = CHAT + '/run.sqlite'
JOURNAL = CHAT + '/receipts/private-embedding.sqlite'
SEED = 20261026
CODE = ('scripts/score_memgallery_v7.py', 'agent_memory/memgallery_v7_reader.py',
        'agent_memory/memgallery_v7.py', 'scripts/prepare_memgallery_v7.py')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def path_at(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('relative artifact path required')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact path escape')
    return path


def sha(root, name):
    return digest(path_at(root, name).read_bytes())


def load(root, name):
    return strict_load(path_at(root, name).read_bytes())


def verified(root, name, expected):
    raw = path_at(root, name).read_bytes()
    if digest(raw) != expected:
        raise ValueError('frozen artifact changed')
    return raw


def lines(raw):
    if raw and not raw.endswith(b'\n'):
        raise ValueError('incomplete JSONL tail')
    rows = raw.splitlines()
    if any(not row.strip() for row in rows):
        raise ValueError('blank JSONL record')
    return [strict_load(row) for row in rows]


def check_hashes(root, hashes, *, gold_allowed=False):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('nonempty frozen hash inventory required')
    for name, expected in hashes.items():
        if not gold_allowed and ('scorer-only' in Path(name).parts or 'quarantine' in Path(name).parts):
            raise ValueError('gold or raw container cannot enter pre-completion verification')
        verified(root, name, expected)


def write_lines(root, name, rows):
    body = b''.join(canonical(row).encode('utf-8') + b'\n' for row in rows)
    return save_new(path_at(root, name), body, raw=True)


def validated_mapping(mapping):
    if not isinstance(mapping, dict) or not isinstance(mapping.get('by_native_id'), dict):
        raise ValueError('complete composite native mapping required')
    result = mapping['by_native_id']
    if not result:
        raise ValueError('empty source mapping')
    for identity, row in result.items():
        if (not isinstance(row, dict) or identity != native_id(row['source_session'], row['source_round'])):
            raise ValueError('source composite identity mismatch')
    return result


def resolve_evidence(gold, mapping):
    """Resolve clues within annotated sessions; never zip unrelated label lists."""
    sources = validated_mapping(mapping)
    clues, sessions = gold.get('clue'), gold.get('session_id')
    if any(not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
           for values in (clues, sessions)):
        raise ValueError('native clue and session arrays required')
    known_sessions = {row['source_session'] for row in sources.values()}
    if not set(sessions) <= known_sessions:
        raise ValueError('reference session is absent from complete source corpus')
    if clues and not sessions:
        raise ValueError('nonempty clues require their native sessions')
    resolved = set()
    for clue in set(clues):
        candidates = [identity for identity, row in sources.items()
                      if row['source_round'] == clue and row['source_session'] in sessions]
        if len(candidates) != 1:
            raise ValueError('clue/session mapping is unresolved or ambiguous')
        resolved.add(candidates[0])
    return resolved


def query_map(rows):
    out = {}
    for row in rows:
        index = row.get('index')
        if type(index) is not int or index < 0 or index in out or row.get('point') not in POINTS:
            raise ValueError('unique original question identities required')
        out[index] = row
    if not out:
        raise ValueError('complete nonempty query collection required')
    return out


def gold_map(rows, questions):
    out = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'index', 'answer', 'clue', 'session_id'}
                or type(row['index']) is not int or row['index'] not in questions or row['index'] in out
                or not isinstance(row['answer'], str)):
            raise ValueError('one original reference per complete query required')
        out[row['index']] = row
    if set(out) != set(questions):
        raise ValueError('gold coverage differs from complete queries')
    return out


def validate_rankings(rows, questions, mapping):
    known = set(validated_mapping(mapping))
    expected_k = min(100, len(known))
    result = {}
    for row in rows:
        index = row.get('index')
        if (type(index) is not int or index not in questions or index in result
                or row.get('query_id') != 'memgallery-v7:' + str(index)
                or row.get('point') != questions[index]['point']
                or not isinstance(row.get('methods'), dict) or set(row['methods']) != set(VARIANTS)):
            raise ValueError('complete four-method native ranking identity required')
        for variant, ranking in row['methods'].items():
            ids, scores = ranking.get('ranked_ids'), ranking.get('scores')
            if (not isinstance(ids, list) or len(ids) != expected_k or len(set(ids)) != len(ids)
                    or not set(ids) <= known or not isinstance(scores, list) or len(scores) != len(ids)
                    or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
                raise ValueError('complete unique finite native top100 ranking required')
        for base in ('dense', 'hybrid'):
            if set(row['methods'][base]['ranked_ids']) != set(row['methods'][base+'_reranked']['ranked_ids']):
                raise ValueError('reranker changed its frozen top100 candidate pool')
        result[index] = row
    if set(result) != set(questions):
        raise ValueError('all original questions must be ranked before any gold access')
    return result


def recall(ids, evidence, k):
    if not evidence:
        return None
    found = len(set(ids[:k]) & evidence)
    return dict(fractional_recall=found / len(evidence), any_hit=float(found > 0),
                all_evidence_coverage=float(found == len(evidence)))


def retrieval_statistics(query_rows, gold_rows, mapping, ranking_rows):
    questions = query_map(query_rows)
    rankings = validate_rankings(ranking_rows, questions, mapping)
    gold = gold_map(gold_rows, questions)
    details = []
    for index in sorted(questions):
        evidence = resolve_evidence(gold[index], mapping)
        details.append(dict(index=index, point=questions[index]['point'], gold_count=len(evidence),
            methods={variant: {str(k): recall(rankings[index]['methods'][variant]['ranked_ids'], evidence, k)
                               for k in (20, 100)} for variant in VARIANTS}))
    groups = {}
    for point in ['all', *sorted({row['point'] for row in details})]:
        group = [row for row in details if point == 'all' or row['point'] == point]
        scored = [row for row in group if row['gold_count']]
        groups[point] = dict(questions=len(group), recall_denominator=len(scored),
            empty_gold_questions=len(group)-len(scored),
            variants={variant: {str(k): {metric: sum(row['methods'][variant][str(k)][metric] for row in scored)/len(scored)
                if scored else None for metric in ('fractional_recall', 'any_hit', 'all_evidence_coverage')}
                for k in (20, 100)} for variant in VARIANTS})
    return groups, details


def paired(values, samples=5000):
    if (not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or type(samples) is not int or samples < 1):
        raise ValueError('finite paired observations and positive resample count required')
    rng = random.Random(SEED)
    draws = sorted(sum(values[rng.randrange(len(values))] for _ in values)/len(values) for _ in range(samples))
    def quantile(p):
        pos = (len(draws)-1)*p
        lo, hi = math.floor(pos), math.ceil(pos)
        return draws[lo] + (draws[hi]-draws[lo])*(pos-lo)
    return dict(questions=len(values), mean_delta=sum(values)/len(values),
                wins=sum(v > 0 for v in values), ties=sum(v == 0 for v in values), losses=sum(v < 0 for v in values),
                ci95=[quantile(.025), quantile(.975)])


def aggregate_reader(details, samples=5000):
    by_key = {}
    for row in details:
        key = (row.get('index'), row.get('variant'))
        if (type(key[0]) is not int or key[1] not in VARIANTS or key in by_key
                or not isinstance(row.get('point'), str) or type(row.get('invalid_schema')) is not bool
                or type(row.get('score')) not in (int, float) or row['score'] not in (0, .25, .5, .75, 1)
                or row['invalid_schema'] and row['score'] != 0):
            raise ValueError('unique exact five-grade outcomes; invalid reader is always zero')
        by_key[key] = row
    indices = {key[0] for key in by_key}
    if not indices or set(by_key) != {(index, variant) for index in indices for variant in VARIANTS}:
        raise ValueError('paired four-variant reader coverage required')
    if any(len({by_key[index, variant]['point'] for variant in VARIANTS}) != 1 for index in indices):
        raise ValueError('point differs between conditions')
    groups = {}
    for point in ['all', *sorted({row['point'] for row in details})]:
        selected = sorted(index for index in indices if point == 'all' or by_key[index, VARIANTS[0]]['point'] == point)
        groups[point] = dict(questions=len(selected), variants={variant: dict(
            questions=len(selected), proxy_mean=sum(by_key[i, variant]['score'] for i in selected)/len(selected),
            invalid_answers=sum(by_key[i, variant]['invalid_schema'] for i in selected),
            full_credit_rate=sum(by_key[i, variant]['score'] == 1 for i in selected)/len(selected)) for variant in VARIANTS},
            paired={base+'_reranked_minus_'+base: paired([by_key[i, base+'_reranked']['score'] - by_key[i, base]['score']
                    for i in selected], samples=samples) for base in ('dense', 'hybrid')})
    return groups


def judge_batches(query_rows, selected, answer_rows, gold_rows, count_tokens):
    questions = query_map(query_rows)
    references = gold_map(gold_rows, questions)
    if (not isinstance(selected, list) or len(selected) != 16 or any(type(i) is not int for i in selected)
            or len(set(selected)) != 16 or not set(selected) <= set(questions)):
        raise ValueError('the frozen sixteen-question selection is required')
    answers = {}
    for row in answer_rows:
        key = (row.get('index'), row.get('variant'))
        if key in answers or key[0] not in selected or key[1] not in VARIANTS:
            raise ValueError('duplicate or extra reader outcome')
        answer = row.get('answer')
        if (not isinstance(answer, dict) or answer.get('schema_status') not in ('valid', 'invalid')
                or not isinstance(answer.get('answer'), str)
                or (answer['schema_status'] == 'invalid' and answer['answer'] != '')):
            raise ValueError('explicit valid or failed reader outcome required')
        answers[key] = row
    if set(answers) != {(index, variant) for index in selected for variant in VARIANTS}:
        raise ValueError('all sixty-four completed reader outcomes required')
    rng = random.Random(SEED)
    requests, mapping = [], []
    for offset in range(0, 16, 2):
        identities = [(index, variant) for index in selected[offset:offset+2] for variant in VARIANTS]
        rng.shuffle(identities)
        items, private = [], []
        for item_id, (index, variant) in enumerate(identities):
            answer = answers[index, variant]['answer']
            invalid = answer['schema_status'] == 'invalid'
            items.append(dict(item_id=item_id, point=questions[index]['point'], question=question_text(questions[index]),
                              reference=references[index]['answer'], prediction='INVALID_READER_RESPONSE' if invalid else answer['answer']))
            private.append(dict(item_id=item_id, index=index, point=questions[index]['point'],
                                variant=variant, invalid_schema=invalid))
        pack = build_proxy_judge(items, count_tokens)
        request_id = 'memgallery-v7-judge-' + digest(canonical([offset//2, pack['pack_sha256']]).encode('utf-8'))
        requests.append(dict(request_id=request_id, pack=pack))
        mapping.append(dict(request_id=request_id, items=private))
    return requests, mapping


RERANK_RUN = 'reports/memgallery-v7-rerank-run.json'
RERANK_FROZEN = BASE + '/rerank/FROZEN_CONFIG.json'
READER_COMPLETE = CHAT + '/READER_COMPLETED.json'
JUDGE_COMPLETE = CHAT + '/JUDGE_COMPLETED.json'
RETRIEVAL_SCORES = 'reports/memgallery-v7-retrieval-scores.json'
RETRIEVAL_DETAILS = BASE + '/scorer-only/retrieval-scores.json'
READER_SCORES = 'reports/memgallery-v7-reader-scores.json'
READER_DETAILS = CHAT + '/scorer-only/reader-scores.json'
JUDGE_PREPARATION = 'reports/memgallery-v7-judge-preparation.json'


def verify_retrieval_complete(root=ROOT):
    """Only public projections/complete predictions; never opens a gold path."""
    report = load(root, RERANK_RUN)
    if (report.get('state') != 'retrieval_complete_gold_not_loaded' or report.get('gold_loaded') is not False
            or report.get('frozen_file') != RERANK_FROZEN or report.get('ranking_file') != RANKINGS
            or report.get('methods') != list(VARIANTS)):
        raise ValueError('complete frozen four-method ranking report required before gold')
    frozen = strict_load(verified(root, RERANK_FROZEN, report['frozen_sha256']))
    for group in ('input_hashes', 'artifact_hashes', 'code_hashes'):
        check_hashes(root, frozen[group])
    if ENCODING not in frozen['input_hashes']:
        raise ValueError('reranking must bind the complete source encoding freeze')
    encoding = strict_load(verified(root, ENCODING, frozen['input_hashes'][ENCODING]))
    check_hashes(root, encoding['input_hashes'])
    check_hashes(root, encoding['code_hashes'])
    prep = load(root, PREPARATION)
    check_hashes(root, prep['public_artifact_hashes'])
    if prep.get('state') != 'prepared_complete_scene_no_model_calls':
        raise ValueError('complete native preparation required')
    questions = strict_load(verified(root, QUERIES, prep['public_artifact_hashes'][QUERIES]))
    mapping = strict_load(verified(root, MAPPING, encoding['input_hashes'][MAPPING]))
    ranking_rows = lines(verified(root, RANKINGS, report['ranking_sha256']))
    rankings = validate_rankings(ranking_rows, query_map(questions), mapping)
    remote = strict_load(verified(root, report['remote_run_file'], report['remote_run_sha256']))
    if (remote.get('state') != 'complete' or remote.get('frozen_config_sha256') != report['frozen_sha256']
            or remote.get('completed_pairs') != report.get('pair_count')
            or report.get('questions') != len(rankings)
            or report.get('source_count') != len(validated_mapping(mapping))):
        raise ValueError('complete native corpus/query count and GPU run required')
    bound = (RERANK_RUN, RERANK_FROZEN, ENCODING, PREPARATION, QUERIES, MAPPING, RANKINGS, report['remote_run_file'])
    return dict(questions=questions, mapping=mapping, rankings=ranking_rows, preparation=prep,
                input_hashes={name: sha(root, name) for name in bound})


def read_gold_after_completion(root, preparation):
    return strict_load(verified(root, GOLD, preparation['scorer_only_gold_sha256']))


def score_retrieval(root=ROOT):
    for name in (RETRIEVAL_SCORES, RETRIEVAL_DETAILS):
        if path_at(root, name).exists():
            raise FileExistsError('retrieval scoring already exists; preserve it')
    complete = verify_retrieval_complete(root)
    gold = read_gold_after_completion(root, complete['preparation'])
    groups, details = retrieval_statistics(complete['questions'], gold, complete['mapping'], complete['rankings'])
    check_hashes(root, complete['input_hashes'])
    report = dict(state='complete', groups=groups, methods=list(VARIANTS),
        metric='macro native composite-source recall', empty_gold_recall=None,
        empty_gold_policy='excluded from recall denominator and reported separately for every point',
        native_identity='lossless session_id plus round label; no ambiguous remapping',
        input_hashes={**complete['input_hashes'], GOLD: sha(root, GOLD)},
        code_hashes={name: sha(root, name) for name in CODE}, provider_requests=0,
        official_score=False, official_smoke=0, official_full=0)
    save_new(path_at(root, RETRIEVAL_DETAILS), details)
    save_new(path_at(root, RETRIEVAL_SCORES), report)
    return report


def ro(root, name):
    db = sqlite3.connect(path_at(root, name).as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


def guard_calls(root, *, final=False):
    with closing(ro(root, GUARD)) as db:
        runs = db.execute('SELECT max_requests,cap,state FROM run').fetchall()
        rows = db.execute('SELECT ordinal,model,held,state,returned_model FROM calls ORDER BY ordinal').fetchall()
    expected_count = 72 if final else 64
    expected_state = 'complete' if final else 'active'
    if (len(runs) != 1 or tuple(runs[0]) != (72, 18000000, expected_state)
            or len(rows) != expected_count or [r['ordinal'] for r in rows] != list(range(1, expected_count+1))
            or any(r['model'] != 'gpt-4o-mini' or r['held'] != 250000 or r['state'] != 'complete'
                   or r['returned_model'] not in ('gpt-4o-mini', 'gpt-4o-mini-2024-07-18') for r in rows)):
        raise ValueError('exact bounded chat guard and fully completed calls required')
    return {row['ordinal']: dict(row) for row in rows}


def receipt_response(root, namespace, request_id, payload, saved, calls):
    if (saved.get('request_id') != request_id or not isinstance(saved.get('attempt_id'), str)
            or not saved['attempt_id'] or type(saved.get('run_ordinal')) is not int
            or saved['run_ordinal'] not in calls or not isinstance(saved.get('receipt_sha256'), str)):
        raise ValueError('complete persisted receipt identity required')
    manifest = dict(keys=[request_id], payload_sha256=digest(canonical(payload).encode('utf-8')),
                    run_path=str(path_at(root, GUARD)))
    fingerprint = digest(canonical(dict(namespace=namespace, manifest=manifest)).encode('utf-8'))
    with closing(ro(root, JOURNAL)) as db:
        rows = db.execute('SELECT * FROM operations WHERE fingerprint=?', (fingerprint,)).fetchall()
        claims = db.execute('SELECT fingerprint FROM claims WHERE scope=? AND chunk_key=?', (namespace, request_id)).fetchall()
    if len(rows) != 1 or len(claims) != 1 or claims[0]['fingerprint'] != fingerprint:
        raise ValueError('one bound journal operation and request claim required')
    row = dict(rows[0])
    if (row['manifest'] != canonical(manifest) or row['scope'] != namespace or row['state'] != 'complete'
            or row['transport_completed'] != 1 or row['status'] != 200 or row['redacted'] != 0
            or row['attempt_id'] != saved['attempt_id'] or row['run_ordinal'] != saved['run_ordinal']
            or row['run_path'] != manifest['run_path'] or not isinstance(row['body'], bytes)
            or digest(row['body']) != row['body_sha'] or row['body_sha'] != saved['receipt_sha256']):
        raise ValueError('receipt payload, scope, accounting, or saved body changed')
    response = strict_load(row['body'])
    if response.get('model') != calls[row['run_ordinal']]['returned_model']:
        raise ValueError('receipt returned model differs from completed run claim')
    return response


def verify_reader_complete(root=ROOT, *, final=False):
    completion = load(root, READER_COMPLETE)
    if completion.get('state') != 'complete' or completion.get('phase') != 'reader' or completion.get('count') != 64:
        raise ValueError('all sixty-four reader outcomes must complete before gold')
    if not final:
        progress = load(root, CHAT_PROGRESS)
        if progress.get('phase') != 'reader' or progress.get('state') != 'reader_complete':
            raise ValueError('reader live phase has not completed')
    live_raw = verified(root, LIVE_FROZEN, completion['frozen_live_sha256'])
    live = strict_load(live_raw)
    frozen = strict_load(verified(root, READER_FROZEN, live['reader_sha256']))
    if (frozen.get('state') != 'reader_inputs_frozen_no_calls' or frozen.get('planned_reader_answers') != 64
            or frozen.get('max_requests') != 72 or frozen.get('max_preflight_cny') != '18'
            or frozen.get('packs_file') != PACKS):
        raise ValueError('same bounded frozen sixteen-question reader required')
    check_hashes(root, frozen['input_hashes'])
    check_hashes(root, frozen['code_hashes'])
    complete = verify_retrieval_complete(root)
    questions = query_map(complete['questions'])
    selection = load(root, SELECTION)
    selected = selection['selected_indices']
    if len(selected) != 16 or len(set(selected)) != 16 or not set(selected) <= set(questions):
        raise ValueError('frozen sixteen original question identities required')
    packs = {}
    for row in lines(verified(root, PACKS, frozen['packs_sha256'])):
        key = (row.get('index'), row.get('variant'))
        if (key in packs or key[0] not in selected or key[1] not in VARIANTS
                or row.get('query_id') != 'memgallery-v7:'+str(key[0])
                or row.get('point') != questions[key[0]]['point']):
            raise ValueError('unique original reader pack identity required')
        pack = row['pack']
        reader_payload(pack)
        if pack['question_index'] != key[0] or pack['point'] != row['point']:
            raise ValueError('pack does not bind the original question')
        packs[key] = row
    expected = {(index, variant) for index in selected for variant in VARIANTS}
    if set(packs) != expected:
        raise ValueError('all sixty-four frozen reader packs required')
    calls = guard_calls(root, final=final)
    answers, attempts, ordinals = {}, set(), set()
    namespace = 'memgallery-v7:' + digest(live_raw)
    for row in lines(verified(root, ANSWERS, completion['output_sha256'])):
        key = (row.get('index'), row.get('variant'))
        if (key not in expected or key in answers or row.get('query_id') != packs[key]['query_id']
                or row.get('point') != questions[key[0]]['point']
                or row.get('attempt_id') in attempts or row.get('run_ordinal') in ordinals
                or type(row.get('run_ordinal')) is not int or not 1 <= row['run_ordinal'] <= 64):
            raise ValueError('one unique complete reader receipt per condition required')
        request_id = digest((row['query_id']+'\0'+row['variant']).encode('utf-8'))
        response = receipt_response(root, namespace, request_id, reader_payload(packs[key]['pack']), row, calls)
        try:
            decoded = decode_reader(response, packs[key]['pack'])
        except (ValueError, KeyError, TypeError):
            decoded = dict(answer='', cited_ids=[], confidence='invalid', schema_status='invalid',
                           citations_valid=False, answer_ids_valid=False, model=response['model'])
        if decoded != row['answer']:
            raise ValueError('saved reader output differs from its original raw receipt')
        answers[key] = row
        attempts.add(row['attempt_id']); ordinals.add(row['run_ordinal'])
    if set(answers) != expected or ordinals != set(range(1, 65)):
        raise ValueError('complete exact sixty-four original reader outcomes required')
    inputs = {**complete['input_hashes'], **{name: sha(root, name) for name in
              (READER_COMPLETE, READER_FROZEN, LIVE_FROZEN, PACKS, ANSWERS, SELECTION)}}
    return dict(**{k: complete[k] for k in ('questions', 'preparation')}, selected=selected,
                answers=list(answers.values()), namespace=namespace, calls=calls,
                input_hashes=inputs, reader_attempts=attempts)


def local_tokenizer(root):
    """Explicit cached tokenizer; network fallback disabled before construction."""
    import os
    report = load(root, 'reports/reader-tokenizer.json')
    sys.path.insert(0, str(path_at(root, 'runtime/reader-deps')))
    os.environ['TIKTOKEN_CACHE_DIR'] = str(path_at(root, 'runtime/tokenizer-cache'))
    import tiktoken
    import tiktoken.load
    def no_network(_):
        raise RuntimeError('tokenizer network fallback forbidden')
    tiktoken.load.read_file = no_network
    if tiktoken.__version__ != report['version']:
        raise ValueError('frozen local tokenizer version mismatch')
    encoder = tiktoken.get_encoding('o200k_base')
    return lambda text: len(encoder.encode(text, disallowed_special=()))


def prepare_judge(root=ROOT, *, count_tokens=None):
    for name in (JUDGE_FROZEN, JUDGE_REQUESTS, JUDGE_MAPPING, JUDGE_PREPARATION):
        if path_at(root, name).exists():
            raise FileExistsError('judge preparation already exists; preserve it')
    complete = verify_reader_complete(root)
    gold = read_gold_after_completion(root, complete['preparation'])
    requests, mapping = judge_batches(complete['questions'], complete['selected'], complete['answers'],
                                     gold, count_tokens or local_tokenizer(root))
    check_hashes(root, complete['input_hashes'])
    request_hash = write_lines(root, JUDGE_REQUESTS, requests)
    mapping_hash = save_new(path_at(root, JUDGE_MAPPING), mapping)
    inputs = {**complete['input_hashes'], GOLD: sha(root, GOLD), JUDGE_MAPPING: mapping_hash}
    frozen = dict(state='blind_judge_frozen_after_all_reader_receipts', requests_file=JUDGE_REQUESTS,
        requests_sha256=request_hash, input_hashes=inputs, code_hashes={name: sha(root, name) for name in CODE},
        mapping_file=JUDGE_MAPPING, mapping_sha256=mapping_hash, questions=16, reader_answers=64,
        judge_requests=8, planned_new_requests=8, model='gpt-4o-mini', seed=SEED,
        metric='five-grade blinded proxy, not official', grades=[0, .25, .5, .75, 1],
        invalid_reader_outcomes_score_zero=True, method_names_sent_to_judge=False,
        no_prediction_repair=True, output_tokens=1536, max_input_tokens=8192, max_utf8_bytes=24000)
    freeze_hash = save_new(path_at(root, JUDGE_FROZEN), frozen)
    report = dict(state=frozen['state'], questions=16, reader_answers=64, judge_requests=8,
        invalid_reader_answers=sum(row['answer']['schema_status'] == 'invalid' for row in complete['answers']),
        frozen_file=JUDGE_FROZEN, frozen_sha256=freeze_hash, requests_file=JUDGE_REQUESTS,
        requests_sha256=request_hash, provider_requests=0, official_score=False)
    save_new(path_at(root, JUDGE_PREPARATION), report)
    return report


def finalize(root=ROOT, *, bootstrap_samples=5000):
    for name in (READER_SCORES, READER_DETAILS):
        if path_at(root, name).exists():
            raise FileExistsError('final reader scores already exist; preserve them')
    completed = load(root, JUDGE_COMPLETE)
    progress = load(root, CHAT_PROGRESS)
    if (completed.get('state') != 'complete' or completed.get('phase') != 'judge' or completed.get('count') != 8
            or completed.get('frozen_live_sha256') != sha(root, LIVE_FROZEN)
            or progress.get('phase') != 'judge' or progress.get('state') != 'judge_complete'):
        raise ValueError('all eight blind judge requests and closed guard required')
    frozen = load(root, JUDGE_FROZEN)
    if frozen.get('judge_requests') != 8 or frozen.get('requests_file') != JUDGE_REQUESTS:
        raise ValueError('exact eight frozen blind judge requests required')
    reader = verify_reader_complete(root, final=True)
    check_hashes(root, frozen['input_hashes'], gold_allowed=True)
    check_hashes(root, frozen['code_hashes'])
    requests = lines(verified(root, JUDGE_REQUESTS, frozen['requests_sha256']))
    results = lines(verified(root, JUDGE_RESULTS, completed['output_sha256']))
    mapping = strict_load(verified(root, JUDGE_MAPPING, frozen['mapping_sha256']))
    by_request = {row['request_id']: row for row in requests}
    private = {row['request_id']: row['items'] for row in mapping}
    if len(requests) != 8 or len(by_request) != 8 or len(mapping) != 8 or set(private) != set(by_request):
        raise ValueError('exact unique eight-batch blind mapping required')
    seen, attempts, ordinals, details = set(), set(reader['reader_attempts']), set(), []
    for row in results:
        identity = row.get('request_id')
        if (identity not in by_request or identity in seen or row.get('attempt_id') in attempts
                or type(row.get('run_ordinal')) is not int or not 65 <= row['run_ordinal'] <= 72
                or row['run_ordinal'] in ordinals):
            raise ValueError('one unique completed receipt per blind judge request required')
        pack = by_request[identity]['pack']
        response = receipt_response(root, reader['namespace'], identity, proxy_judge_payload(pack), row, reader['calls'])
        if response != row.get('response'):
            raise ValueError('saved judge response differs from its receipt')
        grades = decode_proxy_judge(response, pack['item_ids'])
        if {item['item_id'] for item in private[identity]} != set(grades):
            raise ValueError('blind mapping differs from decoded item identities')
        for item in private[identity]:
            details.append(dict(index=item['index'], point=item['point'], variant=item['variant'],
                                invalid_schema=item['invalid_schema'],
                                score=0. if item['invalid_schema'] else grades[item['item_id']]))
        seen.add(identity); attempts.add(row['attempt_id']); ordinals.add(row['run_ordinal'])
    if seen != set(by_request) or ordinals != set(range(65, 73)) or len(details) != 64:
        raise ValueError('complete eight-batch sixty-four-item judge coverage required')
    if {row['index'] for row in details} != set(reader['selected']):
        raise ValueError('blind scores changed the frozen selection')
    groups = aggregate_reader(details, samples=bootstrap_samples)
    check_hashes(root, frozen['input_hashes'], gold_allowed=True)
    report = dict(state='complete', questions=16, reader_answers=64, judge_requests=8, groups=groups,
        metric='gpt-4o-mini five-grade blinded text-derived proxy; not official Mem-Gallery score',
        grades=[0, .25, .5, .75, 1], invalid_reader_outcomes_score_zero=True,
        upstream_provenance=UPSTREAM_PROVENANCE, official_score=False, official_smoke=0, official_full=0,
        bootstrap=dict(seed=SEED, samples=bootstrap_samples, unit='paired original question',
            caveat='descriptive small public-scene pilot; no multiplicity or shared-evidence dependence adjustment'),
        source_condition='complete original dialogue and author source/query captions; zero raw image inference',
        judge_frozen_sha256=sha(root, JUDGE_FROZEN), reader_answers_sha256=sha(root, ANSWERS),
        judge_results_sha256=sha(root, JUDGE_RESULTS), provider_requests=0,
        raw_model_answers_changed=False, old_experiment_results_overwritten=False)
    save_new(path_at(root, READER_DETAILS), details)
    save_new(path_at(root, READER_SCORES), report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('retrieval', 'judge-prepare', 'final'))
    args = parser.parse_args()
    try:
        result = {'retrieval': score_retrieval, 'judge-prepare': prepare_judge, 'final': finalize}[args.phase]()
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    except Exception as exc:
        raise SystemExit('Isolated V7 scorer stopped: ' + type(exc).__name__) from None
