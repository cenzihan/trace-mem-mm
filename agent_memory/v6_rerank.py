"""Offline V6 candidate union, deterministic reranking, and isolated scoring.

This module contains no provider, remote execution, credential, or reader code.
The source representation and both V5 Top100 candidate sets remain immutable.
"""
import json
import math
import random
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.native_metrics import retrieval_metrics
from scripts.prepare_atm_v5_queries import (
    digest, load_json, path_at, query_id, read_questions, save_new, sha, verified_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
WORK = 'runtime/atm-v6-rerank'
FROZEN = WORK + '/FROZEN_CONFIG.json'
QUERIES = WORK + '/queries.jsonl'
SOURCES = WORK + '/sources.jsonl'
CANDIDATES = WORK + '/candidates.jsonl'
PAIRS = WORK + '/pairs.jsonl'
MODEL_SCORES = WORK + '/reranker-scores.jsonl'
MODEL_RUN = WORK + '/RERANK_RUN.json'
RANKINGS = WORK + '/retrieval-results.jsonl'
DETAILS = WORK + '/isolated-per-question-scores.json'
PREPARATION = 'reports/atm-v6-rerank-preparation.json'
RUN_REPORT = 'reports/atm-v6-rerank-run.json'
SCORE_REPORT = 'reports/atm-v6-rerank-scores.json'
V5_RUN = 'reports/atm-v5-dense-run.json'
V5_SCORES = 'reports/atm-v5-dense-scores.json'
SOURCE_INDEX = 'runtime/atm-n0/source-index.sqlite'
REMOTE_RUNNER = 'scripts/run_atm_v6_remote.py'
CODE = ('agent_memory/v6_rerank.py', 'scripts/prepare_atm_v6_rerank.py',
        'scripts/score_atm_v6_rerank.py', REMOTE_RUNNER)
BASES = ('dense', 'hybrid')
METHODS = ('dense', 'dense_reranked', 'hybrid', 'hybrid_reranked')
KS = (20, 100)
METRICS = ('fractional_recall', 'any_hit', 'all_evidence_coverage')


def now():
    return datetime.now(timezone.utc).isoformat()


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        if 'scorer-only' in path_at(root, name).parts:
            raise ValueError('gold must not enter a pre-scoring manifest')
        verified_bytes(root, name, expected)


def json_lines(raw):
    return [load_json(line) for line in raw.splitlines() if line.strip()]


def write_lines(root, relative, rows):
    path = path_at(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def pair_id(identity, source_id):
    return digest(json.dumps([identity, source_id], ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def validate_ranking(ranking, known):
    ids, scores = ranking.get('ranked_ids'), ranking.get('scores')
    if (not isinstance(ids, list) or not isinstance(scores, list) or len(ids) != len(scores)
            or len(ids) > 100 or any(not isinstance(identity, str) or not identity for identity in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= known
            or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
        raise ValueError('unique known native sources and finite aligned scores required')


def stable_rerank(ids, scores_by_id):
    """Ties retain the retriever's original rank, not a new lexical tie-break."""
    if len(ids) != len(set(ids)) or any(identity not in scores_by_id for identity in ids):
        raise ValueError('complete unique candidate scores required')
    if any(type(scores_by_id[identity]) not in (int, float) or not math.isfinite(scores_by_id[identity]) for identity in ids):
        raise ValueError('finite relevance score required')
    return sorted(ids, key=lambda identity: -scores_by_id[identity])


def source_records(root, expected_hash, expected_count):
    verified_bytes(root, SOURCE_INDEX, expected_hash)
    with closing(sqlite3.connect(path_at(root, SOURCE_INDEX).as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        records = [dict(source_id=identity, text=text) for identity, text in db.execute('SELECT id,text FROM sources ORDER BY id')]
    if (len(records) != expected_count or len({row['source_id'] for row in records}) != expected_count
            or any(not isinstance(row['source_id'], str) or not row['source_id']
                   or not isinstance(row['text'], str) or not row['text'].strip() for row in records)):
        raise ValueError('complete source-only native corpus required')
    return records


def validate_spec(spec):
    if (not isinstance(spec, dict)
            or any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in ('model', 'revision'))
            or type(spec.get('max_length')) is not int or spec['max_length'] < 1
            or spec.get('score_direction') != 'higher_is_better'
            or type(spec.get('workers')) is not int or spec['workers'] != 4):
        raise ValueError('explicit model, revision, max_length, higher_is_better and four workers required')
    # Preserve every extra execution setting, including preprocessing, as data.
    return load_json(json.dumps(spec, allow_nan=False))


def prepare_project(root=ROOT, *, reranker_spec, expected_sources=11034, expected_questions=1044,
                    remote_runner_sha256=None):
    root = Path(root).resolve()
    if path_at(root, WORK).exists() or path_at(root, PREPARATION).exists():
        raise FileExistsError('V6 preparation already claimed; do not overwrite')
    spec = validate_spec(reranker_spec)
    run_raw = path_at(root, V5_RUN).read_bytes()
    run = load_json(run_raw)
    if (run.get('state') != 'retrieval_complete_gold_not_loaded'
            or run.get('source_count') != expected_sources or run.get('questions') != expected_questions):
        raise ValueError('completed V5 full-corpus retrieval required')
    v5_raw = verified_bytes(root, run['frozen_file'], run['frozen_sha256'])
    v5 = load_json(v5_raw)
    inputs = {**v5['input_hashes'], **v5['code_hashes'], V5_RUN: digest(run_raw),
              run['frozen_file']: digest(v5_raw), run['ranking_file']: run['ranking_sha256'],
              V5_SCORES: sha(root, V5_SCORES)}
    check_hashes(root, inputs)
    if v5['source_count'] != expected_sources or v5['query_count'] != expected_questions:
        raise ValueError('V5 frozen/run count mismatch')
    public = read_questions(root, v5['datasets'])
    questions = {row['cache_id']: row for row in public}
    if len(questions) != expected_questions:
        raise ValueError('complete original public questions required')
    sources = source_records(root, v5['source_index_sha256'], expected_sources)
    known = {row['source_id'] for row in sources}
    rankings = json_lines(verified_bytes(root, run['ranking_file'], run['ranking_sha256']))
    candidates, pairs, seen = [], [], set()
    for row in rankings:
        identity = query_id(row['dataset'], row['id'])
        if identity not in questions or identity in seen or row['qtype'] != questions[identity]['qtype']:
            raise ValueError('V5 ranking query identity mismatch')
        seen.add(identity)
        methods = {}
        for name in BASES:
            validate_ranking(row['methods'][name], known)
            methods[name] = {field: row['methods'][name][field] for field in ('ranked_ids', 'scores')}
        candidates.append(dict(query_id=identity, dataset=row['dataset'], id=row['id'], qtype=row['qtype'], methods=methods))
        for source_id in sorted(set(methods['dense']['ranked_ids']) | set(methods['hybrid']['ranked_ids'])):
            pairs.append(dict(pair_id=pair_id(identity, source_id), query_id=identity, source_id=source_id))
    if seen != set(questions):
        raise ValueError('V5 rankings must cover the original question set exactly')
    codes = {name: sha(root, name) for name in CODE}
    if remote_runner_sha256 is not None and remote_runner_sha256 != codes[REMOTE_RUNNER]:
        raise ValueError('explicit remote runner hash differs from the reviewed source file')
    check_hashes(root, {**inputs, **codes})
    write_lines(root, QUERIES, (dict(query_id=row['cache_id'], dataset=row['dataset'], id=row['id'],
        qtype=row['qtype'], question=row['question']) for row in public))
    write_lines(root, SOURCES, sources)
    write_lines(root, CANDIDATES, candidates)
    write_lines(root, PAIRS, pairs)
    artifacts = {name: sha(root, name) for name in (QUERIES, SOURCES, CANDIDATES, PAIRS)}
    check_hashes(root, {**inputs, **codes})
    frozen = dict(version='V6-frozen-V5-candidate-reranking-v1', prepared_at_utc=now(),
        input_hashes=inputs, code_hashes=codes, artifact_hashes=artifacts, datasets=v5['datasets'],
        source_count=expected_sources, source_index_sha256=v5['source_index_sha256'],
        query_count=expected_questions, counts=dict(Counter(row['dataset'] for row in public)),
        pair_count=len(pairs), reranker_spec=spec, methods=list(METHODS), top_k=100,
        remote_runner_sha256=codes[REMOTE_RUNNER], workers=spec['workers'],
        source_representation='exact frozen N0 formatted text; no rechunking',
        candidate_policy='each retriever retains its own frozen V5 Top100; union pairs scored once',
        tie_policy='stable original retriever rank', condition=v5['condition'], exposure=v5['exposure'],
        gold_loaded=False, provider_requests=0, reader_results=None)
    save_new(root, FROZEN, frozen)
    report = dict(state='prepared_gold_not_loaded', source_count=expected_sources, questions=expected_questions,
        counts=frozen['counts'], pair_count=len(pairs), frozen_file=FROZEN, frozen_sha256=sha(root, FROZEN),
        artifact_hashes=artifacts, reranker_spec=spec, provider_requests=0, gold_loaded=False)
    save_new(root, PREPARATION, report)
    return report


def verified_preparation(root):
    prep_raw = path_at(root, PREPARATION).read_bytes()
    prep = load_json(prep_raw)
    if prep.get('state') != 'prepared_gold_not_loaded' or prep.get('frozen_file') != FROZEN:
        raise ValueError('completed V6 preparation required')
    frozen_raw = verified_bytes(root, FROZEN, prep['frozen_sha256'])
    frozen = load_json(frozen_raw)
    if (frozen['artifact_hashes'] != prep['artifact_hashes'] or frozen['pair_count'] != prep['pair_count']
            or frozen['query_count'] != prep['questions'] or frozen['source_count'] != prep['source_count']
            or frozen['reranker_spec'] != prep['reranker_spec']):
        raise ValueError('preparation/freeze identity mismatch')
    check_hashes(root, {**frozen['input_hashes'], **frozen['code_hashes'], **frozen['artifact_hashes']})
    validate_spec(frozen['reranker_spec'])
    if frozen.get('remote_runner_sha256') != frozen['code_hashes'].get(REMOTE_RUNNER) or frozen.get('workers') != 4:
        raise ValueError('frozen remote runner identity or worker count mismatch')
    return frozen, digest(frozen_raw), digest(prep_raw)


def verified_model_scores(root, frozen, frozen_hash):
    raw = path_at(root, MODEL_RUN).read_bytes()
    run = load_json(raw)
    if (run.get('state') != 'complete' or run.get('frozen_config_sha256') != frozen_hash
            or run.get('pairs_sha256') != frozen['artifact_hashes'][PAIRS]
            or run.get('reranker_spec') != frozen['reranker_spec']
            or run.get('completed_pairs') != frozen['pair_count']
            or run.get('scores_file') != MODEL_SCORES):
        raise ValueError('complete exact frozen reranker execution required')
    expected = json_lines(verified_bytes(root, PAIRS, frozen['artifact_hashes'][PAIRS]))
    pair_map = {}
    for row in expected:
        if (set(row) != {'pair_id', 'query_id', 'source_id'}
                or row['pair_id'] != pair_id(row['query_id'], row['source_id']) or row['pair_id'] in pair_map):
            raise ValueError('invalid or duplicate prepared pair')
        pair_map[row['pair_id']] = row
    scores = {}
    for row in json_lines(verified_bytes(root, MODEL_SCORES, run['scores_sha256'])):
        if (set(row) != {'pair_id', 'score'} or row['pair_id'] not in pair_map or row['pair_id'] in scores
                or type(row['score']) not in (int, float) or not math.isfinite(row['score'])):
            raise ValueError('unknown, duplicate, or nonfinite model pair score')
        scores[row['pair_id']] = row['score']
    if len(pair_map) != frozen['pair_count'] or set(scores) != set(pair_map):
        raise ValueError('every union pair must be scored exactly once')
    return scores, run, digest(raw)


def rank_project(root=ROOT):
    root = Path(root).resolve()
    if path_at(root, RANKINGS).exists() or path_at(root, RUN_REPORT).exists():
        raise FileExistsError('V6 rankings already claimed; preserve outputs')
    frozen, frozen_hash, prep_hash = verified_preparation(root)
    scores, model_run, model_hash = verified_model_scores(root, frozen, frozen_hash)
    candidates = json_lines(verified_bytes(root, CANDIDATES, frozen['artifact_hashes'][CANDIDATES]))
    known = {row['source_id'] for row in json_lines(verified_bytes(root, SOURCES, frozen['artifact_hashes'][SOURCES]))}
    public = {row['query_id']: row for row in json_lines(verified_bytes(root, QUERIES, frozen['artifact_hashes'][QUERIES]))}
    rows, seen, used = [], set(), set()
    for candidate in candidates:
        identity = candidate['query_id']
        if (identity not in public or identity in seen or identity != query_id(candidate['dataset'], candidate['id'])
                or candidate['qtype'] != public[identity]['qtype'] or set(candidate['methods']) != set(BASES)):
            raise ValueError('prepared candidate identity mismatch')
        seen.add(identity)
        methods = {}
        for base in BASES:
            ranking = candidate['methods'][base]
            validate_ranking(ranking, known)
            mapping = {source_id: scores[pair_id(identity, source_id)] for source_id in ranking['ranked_ids']}
            used.update(pair_id(identity, source_id) for source_id in ranking['ranked_ids'])
            order = stable_rerank(ranking['ranked_ids'], mapping)
            methods[base] = dict(ranked_ids=ranking['ranked_ids'], scores=ranking['scores'])
            methods[base + '_reranked'] = dict(ranked_ids=order, scores=[mapping[source_id] for source_id in order])
        rows.append(dict(dataset=candidate['dataset'], id=candidate['id'], qtype=candidate['qtype'], methods=methods))
    if (seen != set(public) or len(seen) != frozen['query_count'] or len(known) != frozen['source_count']
            or used != set(scores)):
        raise ValueError('complete exact candidate pair union required')
    verified_preparation(root)
    verified_bytes(root, MODEL_RUN, model_hash)
    verified_bytes(root, MODEL_SCORES, model_run['scores_sha256'])
    write_lines(root, RANKINGS, rows)
    report = dict(state='retrieval_complete_gold_not_loaded', finished_at_utc=now(),
        source_count=frozen['source_count'], questions=len(rows), pair_count=len(scores),
        frozen_file=FROZEN, frozen_sha256=frozen_hash, preparation_sha256=prep_hash,
        ranking_file=RANKINGS, ranking_sha256=sha(root, RANKINGS),
        model_run_sha256=model_hash, model_scores_sha256=model_run['scores_sha256'],
        candidates_unchanged=True, provider_requests=0, gold_loaded=False)
    save_new(root, RUN_REPORT, report)
    return report


def aggregate(rows, method):
    return {str(k): {metric: sum(float(row['methods'][method][str(k)][metric]) for row in rows) / len(rows)
                    for metric in METRICS} for k in KS}


def paired_summary(rows, left, right, *, samples, seed):
    changes = {(str(k), metric): [float(row['methods'][left][str(k)][metric]) -
        float(row['methods'][right][str(k)][metric]) for row in rows] for k in KS for metric in METRICS}
    variable = {key: values for key, values in changes.items() if any(value != values[0] for value in values)}
    draws = {key: [] for key in variable}
    rng, n = random.Random(seed), len(rows)
    for _ in range(samples if variable else 0):
        indices = [rng.randrange(n) for _ in range(n)]
        for key, values in variable.items():
            draws[key].append(sum(values[index] for index in indices) / n)
    def percentile(values, p):
        position = (len(values)-1)*p
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper]-values[lower])*(position-lower)
    output = {str(k): {} for k in KS}
    for (k, metric), values in changes.items():
        distribution = sorted(draws[(k, metric)]) if (k, metric) in draws else [values[0]]
        output[k][metric] = dict(mean_delta=sum(values)/n, wins=sum(v > 1e-12 for v in values),
            ties=sum(abs(v) <= 1e-12 for v in values), losses=sum(v < -1e-12 for v in values),
            ci95=[percentile(distribution, .025), percentile(distribution, .975)])
    return output


def score_project(root=ROOT, *, bootstrap_samples=5000, bootstrap_seed=20261025):
    root = Path(root).resolve()
    if path_at(root, SCORE_REPORT).exists() or path_at(root, DETAILS).exists():
        raise FileExistsError('V6 scores already claimed; preserve outputs')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('positive paired bootstrap replicate count required')
    frozen, frozen_hash, prep_hash = verified_preparation(root)
    model_scores, model_run, model_hash = verified_model_scores(root, frozen, frozen_hash)
    run_raw = path_at(root, RUN_REPORT).read_bytes()
    run = load_json(run_raw)
    if (run.get('state') != 'retrieval_complete_gold_not_loaded' or run.get('frozen_file') != FROZEN
            or run.get('frozen_sha256') != frozen_hash or run.get('preparation_sha256') != prep_hash
            or run.get('model_run_sha256') != model_hash or run.get('model_scores_sha256') != model_run['scores_sha256']
            or run.get('source_count') != frozen['source_count'] or run.get('questions') != frozen['query_count']
            or run.get('pair_count') != frozen['pair_count'] or run.get('ranking_file') != RANKINGS):
        raise ValueError('complete frozen V6 ranking identity required')
    public = {row['cache_id']: row for row in read_questions(root, frozen['datasets'])}
    known = {row['source_id'] for row in json_lines(verified_bytes(root, SOURCES, frozen['artifact_hashes'][SOURCES]))}
    controls = {row['query_id']: row for row in json_lines(verified_bytes(root, CANDIDATES, frozen['artifact_hashes'][CANDIDATES]))}
    rankings = json_lines(verified_bytes(root, RANKINGS, run['ranking_sha256']))
    seen = set()
    for row in rankings:
        identity = query_id(row['dataset'], row['id'])
        if (identity in seen or identity not in public or identity not in controls
                or row['qtype'] != public[identity]['qtype'] or set(row['methods']) != set(METHODS)):
            raise ValueError('duplicate/unknown ranking query or method')
        seen.add(identity)
        for method in METHODS:
            validate_ranking(row['methods'][method], known)
        for base in BASES:
            control = controls[identity]['methods'][base]
            if row['methods'][base] != control:
                raise ValueError('frozen V5 baseline ranking changed')
            mapping = {source_id: model_scores[pair_id(identity, source_id)] for source_id in control['ranked_ids']}
            order = stable_rerank(control['ranked_ids'], mapping)
            if row['methods'][base+'_reranked'] != dict(ranked_ids=order, scores=[mapping[s] for s in order]):
                raise ValueError('reranking changed candidates, model scores, or stable ties')
    if seen != set(public) or seen != set(controls) or len(rankings) != frozen['query_count']:
        raise ValueError('rankings must cover the complete query set')
    # This is the first and only point where native gold can be opened.
    gold, gold_hashes = {}, {}
    for spec in frozen['datasets']:
        native = load_json(verified_bytes(root, spec['gold_file'], spec['gold_sha256']))
        gold_hashes[spec['gold_file']] = spec['gold_sha256']
        if not isinstance(native, list) or len(native) != spec['count']:
            raise ValueError('native gold count mismatch')
        for row in native:
            identity = query_id(spec['name'], row['id'])
            labels = row.get('evidence_ids')
            if (identity not in public or identity in gold or row.get('question') != public[identity]['question']
                    or row.get('qtype') != public[identity]['qtype'] or not isinstance(labels, list) or not labels
                    or any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels)
                    or not set(labels) <= known):
                raise ValueError('exact native gold/public/source identity required')
            gold[identity] = labels
    if set(gold) != set(public):
        raise ValueError('gold query identity set mismatch')
    details = [dict(dataset=row['dataset'], id=row['id'], qtype=row['qtype'],
        methods={method: {str(k): retrieval_metrics(row['methods'][method]['ranked_ids'],
            gold[query_id(row['dataset'], row['id'])], k) for k in KS} for method in METHODS}) for row in rankings]
    v5 = load_json(verified_bytes(root, V5_SCORES, frozen['input_hashes'][V5_SCORES]))
    groups = {}
    for spec in frozen['datasets']:
        selected = [row for row in details if row['dataset'] == spec['name']]
        methods = {method: aggregate(selected, method) for method in METHODS}
        for base in BASES:
            for k in KS:
                for metric in METRICS:
                    if not math.isclose(methods[base][str(k)][metric], v5['datasets'][spec['name']]['methods'][base][str(k)][metric], abs_tol=1e-12):
                        raise ValueError('V6 unchanged baseline does not reproduce V5 scores')
                    if methods[base][str(100)][metric] != methods[base+'_reranked'][str(100)][metric]:
                        raise ValueError('Top100 coverage must be unchanged by candidate-only reranking')
        groups[spec['name']] = dict(questions=len(selected), methods=methods,
            paired={base+'_reranked_minus_'+base: paired_summary(selected, base+'_reranked', base,
                samples=bootstrap_samples, seed=bootstrap_seed) for base in BASES})
    verified_preparation(root)
    verified_bytes(root, RUN_REPORT, digest(run_raw))
    verified_bytes(root, MODEL_RUN, model_hash)
    verified_bytes(root, MODEL_SCORES, model_run['scores_sha256'])
    verified_bytes(root, RANKINGS, run['ranking_sha256'])
    for name, expected in gold_hashes.items():
        verified_bytes(root, name, expected)
    report = dict(state='complete', scored_at_utc=now(), questions=len(details), source_count=frozen['source_count'],
        pair_count=frozen['pair_count'], datasets=groups, reranker_spec=frozen['reranker_spec'],
        condition=frozen['condition'], exposure=frozen['exposure'],
        comparison='Dense/Hybrid x same candidate-only reranker; each original Top100 remains fixed',
        metrics='macro fractional recall, any-hit, all-evidence coverage; native sources at K20/K100',
        bootstrap=dict(samples=bootstrap_samples, seed=bootstrap_seed, unit='paired question',
            interpretation='descriptive public-development intervals; no multiplicity adjustment; shared evidence dependence not modeled'),
        frozen_sha256=frozen_hash, verified_run_sha256=digest(run_raw), ranking_sha256=run['ranking_sha256'],
        model_run_sha256=model_hash, model_scores_sha256=model_run['scores_sha256'], gold_sha256=gold_hashes,
        top100_coverage_unchanged=True, original_v5_scores_reproduced=True,
        gold_only_loaded_after_ranking_validation=True, provider_requests=0,
        reader_answer_accuracy=None, no_new_heldout_claim=True, official_smoke=0, official_full=0)
    save_new(root, DETAILS, details)
    save_new(root, SCORE_REPORT, report)
    return report
