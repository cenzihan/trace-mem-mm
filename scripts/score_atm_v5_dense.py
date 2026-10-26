"""Two separate offline phases: frozen retrieval, then isolated native scoring.

Usage: python -B scripts/score_atm_v5_dense.py retrieve
       python -B scripts/score_atm_v5_dense.py score
Neither phase creates embeddings, constructs a provider, or changes old runs.
"""
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache
from agent_memory.shared_dense import SharedDenseIndex, _unit
from agent_memory.native_metrics import retrieval_metrics
from scripts.prepare_atm_v5_queries import (
    CODE, QUERY_CACHE, QUERY_FROZEN, SOURCE_CACHE, SOURCE_FROZEN,
    digest, load_json, path_at, query_id, read_questions, save_new, sha, verified_bytes,
)

WORK = 'runtime/atm-v5-dense'
FROZEN = WORK + '/FROZEN_CONFIG.json'
RANKINGS = WORK + '/retrieval-results.jsonl'
DETAILS = WORK + '/isolated-per-question-scores.json'
RUN_REPORT = 'reports/atm-v5-dense-run.json'
SCORE_REPORT = 'reports/atm-v5-dense-scores.json'
SOURCE_INDEX = 'runtime/atm-n0/source-index.sqlite'
METHODS = ('flat', 'dense', 'hybrid')
KS = (1, 5, 10, 20, 50, 100)
METRICS = ('fractional_recall', 'any_hit', 'all_evidence_coverage')


def _hashes(root, values):
    for relative, expected in values.items():
        if 'scorer-only' in path_at(root, relative).parts:
            raise ValueError('gold cannot be included in a pre-scoring input manifest')
        verified_bytes(root, relative, expected)


def _baseline(root, datasets):
    output = {}
    for spec in datasets:
        raw = verified_bytes(root, spec['baseline_file'], spec['baseline_sha256'])
        rows = [load_json(line) for line in raw.splitlines() if line.strip()]
        if len(rows) != spec['count']:
            raise ValueError('frozen flat control question count mismatch')
        for row in rows:
            key = query_id(spec['name'], row['id'])
            if key in output:
                raise ValueError('duplicate question in flat control')
            result = row if spec['baseline_method'] is None else row['methods'][spec['baseline_method']]
            output[key] = dict(qtype=row['qtype'], ranked_ids=result['ranked_ids'], scores=result['scores'])
    return output


def _source_ids(root, expected_hash, expected_count):
    verified_bytes(root, SOURCE_INDEX, expected_hash)
    with closing(sqlite3.connect(path_at(root, SOURCE_INDEX).as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        ids = [row[0] for row in db.execute('SELECT id FROM sources')]
    if (len(ids) != expected_count or len(set(ids)) != expected_count
            or any(not isinstance(identity, str) or not identity for identity in ids)):
        raise ValueError('complete unique native source IDs required')
    return set(ids)


def _validate_ranking(row, known):
    ids, scores = row.get('ranked_ids'), row.get('scores')
    if (not isinstance(ids, list) or not isinstance(scores, list) or len(ids) != len(scores)
            or len(ids) > 100 or any(not isinstance(identity, str) for identity in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= known
            or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
        raise ValueError('invalid native ranking, score, or evidence budget')


def _dense_lane_scores(index, query_vector, identities):
    """Recover cosine values only for the selected <=100 parents, not a second full search."""
    unit = _unit(query_vector, index.dimensions)
    return [max(max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(unit, index.vectors[key]))))
                for key in index.source_chunks[identity]) for identity in identities]


def retrieve(root=ROOT, *, offline_test=False, expected_sources=11034, timeout_seconds=3600):
    root = Path(root).resolve()
    if path_at(root, WORK).exists() or path_at(root, RUN_REPORT).exists():
        raise FileExistsError('V5 retrieval already claimed; preserve frozen outputs')
    start = time.monotonic()
    query_raw = path_at(root, QUERY_FROZEN).read_bytes()
    query_frozen = load_json(query_raw)
    _hashes(root, query_frozen['input_hashes'])
    _hashes(root, query_frozen['code_hashes'])
    source_raw = verified_bytes(root, SOURCE_FROZEN, query_frozen['source_frozen_sha256'])
    source_frozen = load_json(source_raw)
    _hashes(root, source_frozen['code_hashes'])
    if (source_frozen['scope'] != query_frozen['source_scope']
            or source_frozen['source_count'] != expected_sources):
        raise ValueError('query/source frozen scope or complete source count mismatch')
    source_scope = CacheScope(**source_frozen['scope'])
    query_scope = CacheScope(**query_frozen['scope'])
    if (query_scope.namespace == source_scope.namespace
            or query_scope.dataset_revision != source_scope.dataset_revision
            or query_scope.model != source_scope.model or query_scope.dimensions != source_scope.dimensions):
        raise ValueError('query/source cache model, dimension, or namespace mismatch')
    inputs = dict(query_frozen['input_hashes'])
    inputs.update({QUERY_FROZEN: digest(query_raw), SOURCE_FROZEN: digest(source_raw),
                   SOURCE_INDEX: source_frozen['source_index_sha256']})
    # Never create an empty cache as a side effect of retrieval.
    for relative in (SOURCE_CACHE, QUERY_CACHE):
        if not path_at(root, relative).is_file():
            raise ValueError('completed published caches required before retrieval')
        inputs[relative] = sha(root, relative)
    source_cache = SourceEmbeddingCache(path_at(root, SOURCE_CACHE), source_scope, offline_test=offline_test)
    query_cache = SourceEmbeddingCache(path_at(root, QUERY_CACHE), query_scope, offline_test=offline_test)
    source_snapshot = source_cache.read_published()
    query_snapshot = query_cache.read_published()
    if (source_snapshot['manifest_sha256'] != source_frozen['manifest_sha256']
            or query_snapshot['manifest_sha256'] != query_frozen['manifest_sha256']):
        raise ValueError('published cache differs from frozen manifest')
    index = SharedDenseIndex(source_cache)
    known = _source_ids(root, source_frozen['source_index_sha256'], expected_sources)
    if set(index.documents) != known:
        raise ValueError('published dense cache does not cover every native source')
    with closing(sqlite3.connect(path_at(root, SOURCE_INDEX).as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        if any(index.documents[identity] != text for identity, text in db.execute('SELECT id,text FROM sources')):
            raise ValueError('shared representation must reconstruct frozen flat source text exactly')
    code_hashes = {name: sha(root, name) for name in CODE}
    frozen = dict(version='V5-shared-dense', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        input_hashes=inputs, code_hashes=code_hashes, datasets=query_frozen['datasets'],
        source_count=expected_sources, source_index_sha256=source_frozen['source_index_sha256'],
        source_published_sha256=source_snapshot['published_sha256'],
        query_published_sha256=query_snapshot['published_sha256'],
        source_scope=source_frozen['scope'], query_scope=query_frozen['scope'],
        query_count=query_frozen['query_count'], methods=list(METHODS), top_k=100,
        dense='maximum chunk cosine per native source over the entire corpus',
        hybrid='equal RRF60 over flat and dense native Top100 lanes',
        flat='exact saved N0 regular / V4 Hard flat control',
        condition='external_Qwen_SGM_shared_text_embedding_v4_not_strict_from_raw',
        exposure='Both regular and Hard were already evaluated; public development comparison, not new heldout data',
        gold_loaded=False, provider_requests=0)
    questions = read_questions(root, frozen['datasets'])
    public = {row['cache_id']: row for row in questions}
    if len(public) != frozen['query_count'] or set(query_snapshot['source_chunks']) != set(public):
        raise ValueError('complete query cache identity set required')
    query_chunks = {row['key']: row for row in query_snapshot['chunks']}
    query_vectors = {}
    for key, row in public.items():
        keys = query_snapshot['source_chunks'][key]
        if len(keys) != 1 or query_chunks[keys[0]]['text'] != row['question']:
            raise ValueError('query cache must contain the exact original question in one chunk')
        query_vectors[key] = query_chunks[keys[0]]['vector']
    controls = _baseline(root, frozen['datasets'])
    if set(controls) != set(public):
        raise ValueError('flat controls must cover every original query')
    for key, row in controls.items():
        if row['qtype'] != public[key]['qtype']:
            raise ValueError('flat control qtype mismatch')
        _validate_ranking(row, known)
    _hashes(root, {**inputs, **code_hashes})
    save_new(root, FROZEN, frozen)
    with path_at(root, RANKINGS).open('x', encoding='utf-8') as target:
        for question in questions:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError('bounded offline retrieval deadline reached')
            key = question['cache_id']
            tick = time.perf_counter()
            hybrid = index.hybrid(question['question'], query_vectors[key], namespace=source_scope.namespace, top_k=100)
            hybrid_ms = (time.perf_counter() - tick) * 1000
            dense_ids = hybrid['lanes']['dense']
            dense_scores = _dense_lane_scores(index, query_vectors[key], dense_ids)
            if hybrid['lanes']['flat'] != controls[key]['ranked_ids']:
                raise ValueError('current flat lane differs from its fixed control')
            methods = dict(flat=dict(ranked_ids=controls[key]['ranked_ids'], scores=controls[key]['scores'], reused=True),
                dense=dict(ranked_ids=dense_ids, scores=dense_scores, reused_hybrid_dense_lane=True),
                hybrid=dict(ranked_ids=hybrid['ranked_ids'], scores=hybrid['scores'], elapsed_ms=hybrid_ms))
            for row in methods.values():
                _validate_ranking(row, known)
            target.write(json.dumps(dict(dataset=question['dataset'], id=question['id'], qtype=question['qtype'], methods=methods),
                                    allow_nan=False) + '\n')
    _hashes(root, {**inputs, **code_hashes})
    report = dict(state='retrieval_complete_gold_not_loaded', finished_at_utc=datetime.now(timezone.utc).isoformat(),
        source_count=expected_sources, questions=len(questions), counts=dict(Counter(row['dataset'] for row in questions)),
        frozen_file=FROZEN, frozen_sha256=sha(root, FROZEN), ranking_file=RANKINGS,
        ranking_sha256=sha(root, RANKINGS), elapsed_seconds=time.monotonic()-start,
        provider_requests=0, gold_loaded=False, flat_controls_reproduced=True,
        source_and_query_caches_unchanged=True, prior_artifacts_unchanged=True)
    save_new(root, RUN_REPORT, report)
    return report


def _aggregate(rows, method):
    return {str(k): {metric: sum(float(row['methods'][method][str(k)][metric]) for row in rows)/len(rows)
                    for metric in METRICS} for k in KS}


def _paired(rows, method):
    result = {}
    for k in KS:
        result[str(k)] = {}
        for metric in METRICS:
            changes = [float(row['methods'][method][str(k)][metric])-float(row['methods']['flat'][str(k)][metric]) for row in rows]
            result[str(k)][metric] = dict(mean_delta=sum(changes)/len(rows),
                wins=sum(value > 1e-12 for value in changes), ties=sum(abs(value) <= 1e-12 for value in changes),
                losses=sum(value < -1e-12 for value in changes))
    return result


def score(root=ROOT, *, expected_sources=11034):
    root = Path(root).resolve()
    if path_at(root, SCORE_REPORT).exists() or path_at(root, DETAILS).exists():
        raise FileExistsError('V5 scores already exist; preserve completed outputs')
    run_raw = path_at(root, RUN_REPORT).read_bytes()
    run = load_json(run_raw)
    if run.get('state') != 'retrieval_complete_gold_not_loaded' or run.get('source_count') != expected_sources:
        raise ValueError('complete frozen retrieval required before gold is opened')
    frozen = load_json(verified_bytes(root, run['frozen_file'], run['frozen_sha256']))
    _hashes(root, {**frozen['input_hashes'], **frozen['code_hashes']})
    if frozen['source_count'] != expected_sources or frozen['query_count'] != run['questions']:
        raise ValueError('frozen/run count mismatch')
    known = _source_ids(root, frozen['source_index_sha256'], expected_sources)
    questions = read_questions(root, frozen['datasets'])
    public = {row['cache_id']: row for row in questions}
    controls = _baseline(root, frozen['datasets'])
    ranking_raw = verified_bytes(root, run['ranking_file'], run['ranking_sha256'])
    rankings = [load_json(line) for line in ranking_raw.splitlines() if line.strip()]
    if len(rankings) != run['questions'] or len(public) != run['questions'] or set(controls) != set(public):
        raise ValueError('complete original question rankings required')
    seen = set()
    for row in rankings:
        key = query_id(row['dataset'], row['id'])
        if key in seen or key not in public or row['qtype'] != public[key]['qtype'] or set(row['methods']) != set(METHODS):
            raise ValueError('duplicate/unknown query or method mismatch')
        seen.add(key)
        for ranking in row['methods'].values():
            _validate_ranking(ranking, known)
        if any(row['methods']['flat'][field] != controls[key][field] for field in ('ranked_ids', 'scores')):
            raise ValueError('frozen flat control changed')
    if seen != set(public):
        raise ValueError('rankings do not cover exact query identity set')
    # Only now may this separate scoring process open native gold files.
    gold = {}
    gold_hashes = {}
    for spec in frozen['datasets']:
        raw = verified_bytes(root, spec['gold_file'], spec['gold_sha256'])
        gold_hashes[spec['gold_file']] = spec['gold_sha256']
        rows = load_json(raw)
        if not isinstance(rows, list) or len(rows) != spec['count']:
            raise ValueError('native gold count mismatch')
        for row in rows:
            key = query_id(spec['name'], row['id'])
            labels = row.get('evidence_ids')
            if (key in gold or key not in public or row.get('qtype') != public[key]['qtype']
                    or row.get('question') != public[key]['question']
                    or not isinstance(labels, list) or not labels
                    or any(not isinstance(value, str) or not value for value in labels)
                    or len(set(labels)) != len(labels) or not set(labels) <= known):
                raise ValueError('native gold/question/source identity mismatch')
            gold[key] = labels
    if set(gold) != set(public):
        raise ValueError('gold and original question sets differ')
    details = []
    for row in rankings:
        labels = gold[query_id(row['dataset'], row['id'])]
        details.append(dict(dataset=row['dataset'], id=row['id'], qtype=row['qtype'], gold_count=len(labels),
            methods={method: {str(k): retrieval_metrics(row['methods'][method]['ranked_ids'], labels, k) for k in KS}
                     for method in METHODS}))
    groups = {}
    for spec in frozen['datasets']:
        selected = [row for row in details if row['dataset'] == spec['name']]
        groups[spec['name']] = dict(questions=len(selected),
            methods={method: _aggregate(selected, method) for method in METHODS},
            paired_vs_flat={method: _paired(selected, method) for method in ('dense', 'hybrid')},
            by_qtype={qtype: dict(questions=len(subset), methods={method: _aggregate(subset, method) for method in METHODS})
                      for qtype in sorted({row['qtype'] for row in selected})
                      if (subset := [row for row in selected if row['qtype'] == qtype])})
    _hashes(root, {**frozen['input_hashes'], **frozen['code_hashes']})
    verified_bytes(root, RUN_REPORT, digest(run_raw))
    verified_bytes(root, run['ranking_file'], run['ranking_sha256'])
    for name, expected in gold_hashes.items():
        verified_bytes(root, name, expected)
    report = dict(state='complete', scored_at_utc=datetime.now(timezone.utc).isoformat(),
        source_count=expected_sources, questions=len(details), datasets=groups,
        condition=frozen['condition'], exposure=frozen['exposure'],
        metric_formula=dict(fractional_recall='macro mean |unique(topK) intersect gold| / |gold|',
            any_hit='macro mean indicator(nonempty intersection)', all_evidence_coverage='macro mean indicator(gold subset unique(topK))',
            unit='native source artifact, never chunk/view/frame'),
        flat_controls_unchanged=True, source_and_query_caches_unchanged=True,
        gold_only_loaded_after_ranking_validation=True, provider_requests=0,
        reader_answer_accuracy=None, no_new_heldout_claim=True, official_smoke=0, official_full=0,
        verified_run_sha256=digest(run_raw), ranking_sha256=run['ranking_sha256'],
        frozen_sha256=run['frozen_sha256'], gold_sha256=gold_hashes)
    save_new(root, DETAILS, details)
    save_new(root, SCORE_REPORT, report)
    return report


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('retrieve', 'score'):
        raise SystemExit('Use retrieve after both caches are published, then score in a separate process.')
    result = retrieve() if sys.argv[1] == 'retrieve' else score()
    print(json.dumps(result, indent=2))
