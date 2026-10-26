"""One-shot, zero-provider V4 vs flat on full ATM sources and Hard31 questions.

Only source files and the question-only projection are read here. The scorer is a
different executable. No regular questions, answers, providers or credentials.
"""
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.atm_n0 import NativeBM25
from agent_memory.parent_sparse import build_sgm_views, ParentSparseIndex

WORK = ROOT / 'runtime/atm-v4-hard'
REPORT = ROOT / 'reports/atm-v4-hard-run.json'
NAMESPACE = 'atm-hard:v4-sparse:78e826dc'
REVISION = '78e826dc07e97466b2f54443831ef9a83ab8b27c'
GOLD_SHA256 = 'acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a'
CODE_FILES = (
    'agent_memory/parent_sparse.py', 'agent_memory/atm_n0.py',
    'agent_memory/retrieval.py', 'agent_memory/native_metrics.py',
    'scripts/run_atm_v4_hard_local.py', 'scripts/score_atm_v4_hard_isolated.py',
)
PRESERVED = (
    'reports/native-mm-v2.json', 'reports/native-v3-completed.json',
    'reports/native-v3-reader.json', 'reports/native-v3-scores.json',
    'reports/atm-n0-run.json', 'reports/atm-n0-scores.json',
    'runtime/atm-n0/FROZEN_CONFIG.json', 'runtime/atm-n0/source-index.sqlite',
    'runtime/atm-n0/retrieval-results.jsonl',
    'runtime/development-budget.sqlite', 'runtime/evaluation-ledger.sqlite',
    'runtime/native-mm-v3/run.sqlite',
    'data/atm-schema/corpus-inventory.json', 'reports/atm-n0-download.json',
    'docs/v4-sparse-plan.md',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    if WORK.exists() or REPORT.exists():
        raise SystemExit('V4 Hard run already claimed; preserve the attempt and inspect it.')
    started = time.monotonic()
    before = {name: sha(ROOT / name) for name in PRESERVED}
    code_hashes = {name: sha(ROOT / name) for name in CODE_FILES}
    download = json.loads((ROOT / 'reports/atm-n0-download.json').read_text(encoding='utf-8'))
    if download.get('complete') is not True:
        raise ValueError('complete verified N0 source download required')
    sources, source_files = {}, []
    for item in download['files']:
        path = ROOT / 'data/atm-n0-source' / item['file']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('source changed since N0')
        sources[item['file']] = json.loads(raw)
        source_files.append(dict(file=path.relative_to(ROOT).as_posix(), sha256=item['sha256']))
    inventory = json.loads((ROOT / 'data/atm-schema/corpus-inventory.json').read_text(encoding='utf-8'))
    expected = {modality: {row['evidence_id'] for row in inventory['records']
                          if row['modality'] == modality} for modality in ('image', 'video')}
    records, views, coverage = build_sgm_views(
        sources['image_batch_results.json'], sources['video_batch_results.json'], sources['emails.json'],
        expected_images=expected['image'], expected_videos=expected['video'], expected_emails=6742,
        namespace=NAMESPACE,
    )
    flat_documents = {row['id']: row['text'] for row in records}
    if len(flat_documents) != 11034 or len(records) != 11034:
        raise ValueError('full 11034 native sources required')
    original_index = ROOT / 'runtime/atm-n0/source-index.sqlite'
    with closing(sqlite3.connect(original_index.as_uri() + '?mode=ro', uri=True)) as db:
        previous_documents = dict(db.execute('SELECT id,text FROM sources'))
    if flat_documents != previous_documents:
        raise ValueError('flat control must equal frozen N0 source text exactly')
    flat = NativeBM25(flat_documents)
    candidate = ParentSparseIndex(records, views, namespace=NAMESPACE)
    source_ids_hash = hashlib.sha256(json.dumps(sorted(flat_documents), separators=(',', ':')).encode()).hexdigest()
    query_path = ROOT / 'data/atm-schema/atm-bench-hard.json.questions.json'
    # Hash bytes for identity only; parse no question before method/source freeze.
    query_hash = sha(query_path)
    frozen = dict(
        version='V4-Sparse', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_before_question_parsing=True, namespace=NAMESPACE, dataset_revision=REVISION,
        condition='external_Qwen_SGM_three_view_sparse_not_strict_from_raw',
        evaluation_scope='same_corpus_new_question_first_performance_comparison',
        prior_exposure='Hard QA previously machine-parsed for schema/overlap; no prior Hard performance run found',
        correlated_views=True, no_cross_corpus_generalization_claim=True,
        methods=['flat', 'v4'], bm25=dict(k1=1.2, b=.75), rrf=dict(k=60, weights=[1, 1, 1]),
        detail_chunks=dict(token_window=512, token_stride=384, native_collapse='max score before lane top100'),
        lane_fields=dict(flat='exact frozen N0 formatter',
                         summary='media short_caption else caption; email short_summary',
                         detail='media caption OCR tags timestamp location_name; email detail timestamp'),
        max_native_candidates_per_lane=100, top_k=100, metric_ks=[1, 5, 10, 20, 50, 100],
        no_question_specific_weights=True, all_distractors=True, source_count=11034,
        source_files=source_files, source_ids_sha256=source_ids_hash,
        source_index_file=original_index.relative_to(ROOT).as_posix(),
        source_index_sha256=sha(original_index),
        query_file=query_path.relative_to(ROOT).as_posix(), query_sha256=query_hash,
        gold_file='data/atm-schema/scorer-only/atm-bench-hard.json', gold_sha256=GOLD_SHA256,
        code_hashes=code_hashes, preserved_hashes=before, coverage=coverage,
        provider_requests=0, model_answering=False, raw_media_loaded=False,
    )
    WORK.mkdir()
    save_new(WORK / 'FROZEN_CONFIG.json', frozen)
    save_new(WORK / 'preserved-before.json', before)
    print(json.dumps(dict(stage='source_and_methods_frozen',source_count=11034,provider_requests=0)), flush=True)
    query_raw = query_path.read_bytes()
    if hashlib.sha256(query_raw).hexdigest() != query_hash:
        raise ValueError('questions changed after freeze')
    questions = json.loads(query_raw)
    if (not isinstance(questions, list) or len(questions) != 31
            or any(set(row) != {'id', 'question', 'qtype'} for row in questions)
            or len({row['id'] for row in questions}) != 31):
        raise ValueError('exact question-only Hard31 projection required')
    ranking_path = WORK / 'retrieval-results.jsonl'
    latencies = {'flat': [], 'v4': []}
    with ranking_path.open('x', encoding='utf-8') as target:
        for question in questions:
            if time.monotonic() - started > 600:
                raise TimeoutError('bounded local evaluation deadline')
            tick = time.perf_counter()
            baseline = flat.search(question['question'], 100)
            flat_elapsed = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            result = candidate.search(question['question'], top_k=100)
            v4_elapsed = (time.perf_counter() - tick) * 1000
            ids = result['ranked_ids']
            score_values = result['scores']
            if isinstance(score_values, dict):
                score_values = [score_values[identity] for identity in ids]
            methods = dict(
                flat=dict(ranked_ids=[identity for identity, _ in baseline],
                          scores=[score for _, score in baseline], elapsed_ms=flat_elapsed),
                v4=dict(ranked_ids=ids, scores=score_values, elapsed_ms=v4_elapsed,
                        lanes=result['lanes']),
            )
            for method, row in methods.items():
                if len(row['ranked_ids']) > 100 or len(set(row['ranked_ids'])) != len(row['ranked_ids']):
                    raise ValueError('native source candidate budget violated')
                if not set(row['ranked_ids']) <= set(flat_documents):
                    raise ValueError('candidate not in full source inventory')
                latencies[method].append(row['elapsed_ms'])
            target.write(json.dumps(dict(id=question['id'], qtype=question['qtype'], methods=methods)) + '\n')
    for name, digest in code_hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError('method changed during frozen run')
    after = {name: sha(ROOT / name) for name in PRESERVED}
    if before != after:
        raise ValueError('prior experiment or ledger changed')
    report = dict(
        state='retrieval_complete_gold_not_loaded', finished_at_utc=datetime.now(timezone.utc).isoformat(),
        questions=31, source_count=11034, source_ids_sha256=source_ids_hash,
        ranking_file=ranking_path.relative_to(ROOT).as_posix(), ranking_sha256=sha(ranking_path),
        frozen_file='runtime/atm-v4-hard/FROZEN_CONFIG.json', frozen_sha256=sha(WORK / 'FROZEN_CONFIG.json'),
        source_index_sha256=sha(original_index), code_hashes=code_hashes,
        question_sha256=query_hash, qtypes=dict(Counter(row['qtype'] for row in questions)),
        latencies_ms=latencies, elapsed_seconds=time.monotonic() - started,
        provider_requests=0, price_cny='0', gold_loaded=False, model_answering=False,
        previous_artifacts_and_ledgers_unchanged=True,
    )
    save_new(REPORT, report)
    print(json.dumps({key: report[key] for key in ('state', 'questions', 'source_count', 'elapsed_seconds', 'provider_requests')}), flush=True)


if __name__ == '__main__':
    main()
