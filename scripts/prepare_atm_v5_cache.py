"""Prepare the full source manifest and empty V5 cache locally, never encode.

No questions/gold, provider, credential, budget/run construction or network.
Missing cache entries are a planning count, never authorization to dispatch.
"""
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache

WORK=ROOT/'runtime/atm-v5-source-cache'
REPORT=ROOT/'reports/atm-v5-cache-preparation.json'
PRESERVED=(
    'runtime/atm-n0/source-index.sqlite','runtime/atm-n0/retrieval-results.jsonl',
    'reports/atm-n0-run.json','reports/atm-n0-scores.json',
    'reports/atm-v4-hard-run.json','reports/atm-v4-hard-scores.json',
    'runtime/atm-v4-hard/FROZEN_CONFIG.json','runtime/atm-v4-hard/retrieval-results.jsonl',
    'reports/native-v3-completed.json','runtime/development-budget.sqlite',
    'runtime/native-mm-v3/run.sqlite','runtime/evaluation-ledger.sqlite',
)
CODE=('agent_memory/embedding_cache.py','agent_memory/shared_dense.py',
      'scripts/prepare_atm_v5_cache.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)


def main():
    if WORK.exists() or REPORT.exists():
        raise SystemExit('V5 source preparation already exists; inspect it, do not reset the cache.')
    before={name:sha(ROOT/name) for name in PRESERVED}
    source=ROOT/'runtime/atm-n0/source-index.sqlite'
    n0=json.loads((ROOT/'reports/atm-n0-run.json').read_text(encoding='utf-8'))
    if sha(source)!=n0['source_index_sha256']:
        raise ValueError('frozen source index hash mismatch')
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        columns={row[1] for row in db.execute('PRAGMA table_info(sources)')}
        if columns!={'namespace','id','modality','text','timestamp','description_present'}:
            raise ValueError('expected source-only N0 table required')
        records=[dict(id=identity,text=text) for identity,text in db.execute('SELECT id,text FROM sources ORDER BY id')]
        counts=dict(db.execute('SELECT modality,COUNT(*) FROM sources GROUP BY modality'))
    if len(records)!=11034 or len({row['id'] for row in records})!=11034:
        raise ValueError('complete unique native source set required')
    scope=CacheScope(namespace='atm-78e826dc-sgm-shared-v5',
        dataset_revision='78e826dc07e97466b2f54443831ef9a83ab8b27c',
        representation='n0-flat-utf8-8192-v1',dimensions=256)
    cache=SourceEmbeddingCache(WORK/'source-cache.sqlite',scope)
    prepared=cache.prepare_sources(records)
    missing=cache.missing_batches()
    missing_count=sum(len(batch) for batch in missing)
    if missing_count!=prepared['chunk_count']:
        raise ValueError('new source-only cache unexpectedly contains vectors')
    try:
        cache.publish()
    except ValueError:
        incomplete_publication_rejected=True
    else:
        raise ValueError('empty source cache must not publish a dense index')
    after={name:sha(ROOT/name) for name in PRESERVED}
    if before!=after:
        raise ValueError('frozen evaluation or real ledger changed')
    frozen=dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        condition='external_Qwen_SGM_source_cache_not_strict_raw_media',
        source_index_sha256=before['runtime/atm-n0/source-index.sqlite'],
        manifest_sha256=prepared['manifest_sha256'],source_count=len(records),source_counts=counts,
        unique_chunk_count=prepared['chunk_count'],source_utf8_bytes=sum(len(row['text'].encode()) for row in records),
        code_hashes={name:sha(ROOT/name) for name in CODE},preserved_hashes=before,
        scope=dict(namespace=scope.namespace,dataset_revision=scope.dataset_revision,
                   representation=scope.representation,model=scope.model,dimensions=scope.dimensions))
    save_new(WORK/'FROZEN_SOURCE.json',frozen)
    report=dict(state='source_cache_prepared_no_vectors',prepared_at_utc=frozen['prepared_at_utc'],
        source_count=len(records),source_counts=counts,unique_chunk_count=prepared['chunk_count'],
        missing_chunks=missing_count,source_embedding_batches=len(missing),batch_limit=10,
        proposed_source_only_hold_cny=str(Decimal('.05')*len(missing)),
        hold_is_planning_not_actual_fee_or_approval=True,
        vectors_created=0,published_dense_index=False,empty_publication_rejected=incomplete_publication_rejected,
        source_text_preserved=True,condition=frozen['condition'],manifest_sha256=prepared['manifest_sha256'],
        frozen_source_sha256=sha(WORK/'FROZEN_SOURCE.json'),
        provider_requests=0,new_guard=False,credentials_read=False,questions_loaded=False,gold_loaded=False,
        old_evaluation_and_ledgers_unchanged=True,
        required_before_dispatch=['reviewed source transfer scope','single explicit bounded run scope covering all source batches',
            'per-batch receipt journal and attempt/run binding; unknown operations never inferred retryable from cache misses'],
        next_comparisons=['dense-only full corpus','flat BM25 plus shared dense equal RRF60'],
        new_benchmark_scores=False)
    save_new(REPORT,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
