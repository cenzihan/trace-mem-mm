"""Freeze V7 candidates or finalize verified GPU scores, never load gold or APIs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The existing encoding module's credential-reader import is inert; no reader is called.
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import encode_memgallery_v7 as encoding
from scripts import control_memgallery_v7_remote as control
from scripts import run_memgallery_v7_remote as wrapper
from agent_memory.memgallery_v7 import VARIANTS, base_rankings, union_pairs, rerank_variants
from agent_memory.shared_dense import SharedDenseIndex

WORK = ROOT / 'runtime/memgallery-v7/rerank'
FROZEN = 'runtime/memgallery-v7/rerank/FROZEN_CONFIG.json'
RANKING = 'runtime/memgallery-v7/rerank/retrieval-results.jsonl'
RUN = 'reports/memgallery-v7-rerank-run.json'
PREP = 'reports/memgallery-v7-rerank-preparation.json'
CODE = (*control.CODE, 'scripts/prepare_memgallery_v7_rerank.py',
        'agent_memory/memgallery_v7.py', 'agent_memory/shared_dense.py', 'agent_memory/atm_n0.py',
        'agent_memory/retrieval.py', 'agent_memory/v6_rerank.py')


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    if WORK.exists() or (ROOT / PREP).exists():
        raise FileExistsError('preserve existing V7 ranking preparation')
    frozen_encoding = encoding.verify()
    progress = encoding.load(encoding.PROGRESS)
    if (progress.get('state') != 'encoding_complete' or progress.get('run', {}).get('state') != 'complete'):
        raise ValueError('completed encoding and closed run required')
    source_cache, query_cache = (encoding.cache(stage, frozen_encoding) for stage in ('source', 'query'))
    source_snapshot, query_snapshot = source_cache.read_published(), query_cache.read_published()
    for stage, snapshot in (('source', source_snapshot), ('query', query_snapshot)):
        if (snapshot['scope'] != frozen_encoding['scopes'][stage]
                or snapshot['manifest_sha256'] != frozen_encoding['counts'][stage]['manifest_sha256']):
            raise ValueError('published embedding identity differs from frozen encoding')
    if source_snapshot['scope']['namespace'] == query_snapshot['scope']['namespace']:
        raise ValueError('source and query embedding namespaces must be separate')
    docs = encoding.lines(encoding.WORK / 'source-documents.jsonl')
    query_docs = encoding.lines(encoding.WORK / 'query-documents.jsonl')
    if len(docs) != 186 or len(query_docs) != 74:
        raise ValueError('complete fixed independent scene required')
    index = SharedDenseIndex(source_cache)
    documents = {row['id']: row['text'] for row in docs}
    if len(documents) != len(docs) or documents != index.documents:
        raise ValueError('complete published sources must preserve exact frozen document text')
    query_chunks = {row['key']: row for row in query_snapshot['chunks']}
    query_sources = query_snapshot['source_chunks']
    if set(query_sources) != {row['query_id'] for row in query_docs} or len(query_sources) != len(query_docs):
        raise ValueError('all frozen native questions require their exact published query vectors')
    queries, candidates, pairs = [], [], []
    for row in query_docs:
        identity, question = row['query_id'], row['query']
        if (identity != 'memgallery-v7:' + str(question['index'])
                or row['text'] != encoding.query_text(question) or len(query_sources[identity]) != 1):
            raise ValueError('unchanged query/caption representation and one query chunk required')
        chunk = query_chunks[query_sources[identity][0]]
        if chunk['text'] != row['text']:
            raise ValueError('query vector text differs from frozen original representation')
        bases = base_rankings(index, row['text'], chunk['vector'], namespace=index.namespace)
        methods = {name: {key: value[key] for key in ('ranked_ids', 'scores')} for name, value in bases.items()}
        queries.append(dict(query_id=identity, index=question['index'], point=question['point'], question=row['text']))
        candidates.append(dict(query_id=identity, index=question['index'], point=question['point'], methods=methods))
        pairs.extend(union_pairs(identity, methods))
    # Recheck frozen inputs before publishing any new phase artifact.
    encoding.verify()
    inputs = dict(frozen_encoding['input_hashes'])
    for path in (encoding.WORK / 'FROZEN_ENCODING.json', encoding.PROGRESS,
                 encoding.WORK / 'source.sqlite', encoding.WORK / 'query.sqlite'):
        inputs[relative(path)] = encoding.sha(path)
    codes = {name: encoding.sha(ROOT / name) for name in CODE}
    WORK.mkdir()
    for name, rows in (('queries.jsonl', queries),
                       ('sources.jsonl', [dict(source_id=row['id'], text=row['text']) for row in docs]),
                       ('candidates.jsonl', candidates), ('pairs.jsonl', pairs)):
        encoding.write_lines(WORK / name, rows)
    artifacts = {relative(WORK / name): encoding.sha(WORK / name) for name in wrapper.worker.FILES}
    frozen = dict(experiment='memgallery-v7', created_at_utc=now(), state='prepared_gold_not_loaded',
        source_count=len(docs), query_count=len(queries), pair_count=len(pairs), workers=4,
        reranker_spec=wrapper.SPEC, remote_runner_sha256=codes[control.CODE[0]],
        remote_wrapper_sha256=codes[control.CODE[1]], code_hashes=codes, input_hashes=inputs,
        artifact_hashes=artifacts, methods=list(VARIANTS), top_k=100,
        source_published_sha256=source_snapshot['published_sha256'],
        query_published_sha256=query_snapshot['published_sha256'],
        encoding_frozen_file=relative(encoding.WORK / 'FROZEN_ENCODING.json'),
        encoding_frozen_sha256=encoding.sha(encoding.WORK / 'FROZEN_ENCODING.json'),
        source_representation='original full dialogue and author-provided captions; no raw images',
        query_representation=frozen_encoding['query_text_rule'],
        candidate_policy='own frozen Dense/Hybrid Top100; score union pairs once',
        tie_policy='original retriever rank', gold_loaded=False, provider_requests=0)
    encoding.save(ROOT / FROZEN, frozen, exclusive=True)
    control.frozen()
    report = dict(state='prepared_gold_not_loaded', experiment='memgallery-v7', questions=len(queries),
                  source_count=len(docs), pair_count=len(pairs), methods=list(VARIANTS),
                  frozen_file=FROZEN, frozen_sha256=encoding.sha(ROOT / FROZEN),
                  artifact_hashes=artifacts, gold_loaded=False, provider_requests=0)
    encoding.save(ROOT / PREP, report, exclusive=True)
    return {key: report[key] for key in ('state', 'questions', 'source_count', 'pair_count', 'gold_loaded', 'provider_requests')}


def finalize():
    if (ROOT / RANKING).exists() or (ROOT / RUN).exists() or (WORK / 'COMPLETED.json').exists():
        raise FileExistsError('preserve completed V7 rankings and report')
    data = control.frozen()
    frozen_hash = encoding.sha(ROOT / FROZEN)
    remote_run_path = WORK / 'RERANK_RUN.json'
    run = wrapper.worker.strict_json(remote_run_path.read_bytes())
    score_path = WORK / 'reranker-scores.jsonl'
    if (run.get('state') != 'complete' or run.get('experiment') != 'memgallery-v7'
            or run.get('frozen_config_sha256') != frozen_hash
            or run.get('remote_runner_sha256') != encoding.sha(control.RUNNER)
            or run.get('remote_wrapper_sha256') != encoding.sha(control.WRAPPER)
            or run.get('reranker_spec') != data['frozen']['reranker_spec']
            or run.get('pairs_sha256') != encoding.sha(WORK / 'pairs.jsonl')
            or run.get('scores_file') != relative(score_path)
            or run.get('scores_sha256') != encoding.sha(score_path)
            or run.get('completed_pairs') != len(data['pairs'])):
        raise ValueError('exact complete frozen GPU result required before finalization')
    reports = run.get('workers')
    if not isinstance(reports, list) or len(reports) != 4:
        raise ValueError('all four exact worker reports required')
    for i, report in enumerate(reports):
        control.validate_report(report, i, data['frozen'])
        if report.get('state') != 'complete' or report.get('completed_pairs') != len(data['pairs'][i::4]):
            raise ValueError('each frozen GPU shard must be complete')
    if wrapper.worker.committed_prefix(score_path, data['pairs']) != len(data['pairs']):
        raise ValueError('complete ordered pair scores required')
    scores = {row['pair_id']: row['score'] for row in wrapper.worker.json_lines(score_path)}
    output = []
    for row in wrapper.worker.json_lines(WORK / 'candidates.jsonl'):
        union = union_pairs(row['query_id'], row['methods'])
        methods = rerank_variants(row['methods'], {pair['source_id']: scores[pair['pair_id']] for pair in union})
        output.append(dict(query_id=row['query_id'], index=row['index'], point=row['point'], methods=methods))
    if len(output) != data['frozen']['query_count']:
        raise ValueError('all original questions must finish ranking')
    control.frozen()
    encoding.write_lines(ROOT / RANKING, output)
    report = dict(state='retrieval_complete_gold_not_loaded', experiment='memgallery-v7',
                  completed_at_utc=now(), questions=len(output), source_count=len(data['sources']),
                  pair_count=len(data['pairs']), methods=list(VARIANTS),
                  frozen_file=FROZEN, frozen_sha256=frozen_hash, ranking_file=RANKING,
                  ranking_sha256=encoding.sha(ROOT / RANKING), remote_run_file=relative(remote_run_path),
                  remote_run_sha256=encoding.sha(remote_run_path), gold_loaded=False,
                  provider_requests=0, raw_media=False, top100_pools_preserved=True)
    encoding.save(ROOT / RUN, report, exclusive=True)
    encoding.save(WORK / 'COMPLETED.json', dict(state='complete', experiment='memgallery-v7',
        retrieval_results_sha256=report['ranking_sha256'], questions=len(output),
        source_count=len(data['sources']), pair_count=len(data['pairs']),
        frozen_sha256=frozen_hash, report_file=RUN, report_sha256=encoding.sha(ROOT / RUN),
        remote_run_sha256=report['remote_run_sha256'], gold_loaded=False), exclusive=True)
    return {key: report[key] for key in ('state', 'questions', 'source_count', 'pair_count', 'gold_loaded', 'provider_requests')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'finalize'))
    args = parser.parse_args()
    try:
        print(json.dumps(prepare() if args.phase == 'prepare' else finalize()))
    except Exception as exc:
        raise SystemExit('V7 ranking stopped: ' + type(exc).__name__) from None
