"""Fresh independent-scene source/query encoding; receipt first, no retry.

Preparation cannot read scorer-only artifacts. Paid work needs --live and its
own exact frozen guard. Earlier experiments and completed guards stay untouched.
"""
import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.development_run import DevelopmentRun, ScopedTransport
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache
from agent_memory.embedding_executor import DurableEmbeddingExecutor
from agent_memory.memgallery_v7 import source_documents, source_mapping, project_queries, REPRESENTATION
from agent_memory.provider_http import BudgetedJSONTransport, NativeHTTPS, BAILIAN_BEIJING_ENDPOINT, SHARED_LEDGER
from bailian_save_and_check import read_bailian

BASE = ROOT / 'runtime/memgallery-v7'
WORK = BASE / 'encoding'
PROGRESS = ROOT / 'reports/memgallery-v7-encoding-progress.json'
PRICE = 'https://help.aliyun.com/zh/model-studio/text-embedding-v4'
CHECKED = '2026-10-26T04:21:00+00:00'
REVISION = 'af912daba984e896e253016b7c7e334ef92c2a6f'
CODE = ('scripts/encode_memgallery_v7.py', 'agent_memory/memgallery_v7.py',
        'agent_memory/native_memgallery.py', 'agent_memory/embedding_cache.py',
        'agent_memory/embedding_executor.py', 'agent_memory/provider_http.py',
        'agent_memory/development_run.py', 'agent_memory/budget.py', 'agent_memory/visual_recovery.py')


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    target = path if exclusive else path.with_suffix('.tmp')
    with target.open('x' if exclusive else 'w', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n'); out.flush(); os.fsync(out.fileno())
    if not exclusive: os.replace(target, path)


def write_lines(path, rows):
    with Path(path).open('x', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        out.flush(); os.fsync(out.fileno())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text(encoding='utf-8').splitlines() if row.strip()]


def query_text(row):
    """One fixed representation shared by embedding, BM25 and reranking."""
    images = row.get('question_image') or row.get('query_image')
    captions = row.get('image_caption')
    if not images:
        if captions: raise ValueError('query captions without an image')
        return row['question']
    images = images if isinstance(images, list) else [images]
    captions = captions if isinstance(captions, list) else [captions]
    if (len(images) != len(captions) or
            any(not isinstance(v, str) or not v.strip() for v in images + captions)):
        raise ValueError('each query image needs its original nonempty author caption')
    return row['question'] + '\nQUERY_IMAGE_CAPTIONS:\n' + '\n'.join(captions)


def history(limit):
    with closing(sqlite3.connect(SHARED_LEDGER.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT * FROM attempts WHERE rowid<=? ORDER BY rowid', (limit,)).fetchall()
        observations = db.execute('SELECT o.* FROM provider_observations o JOIN attempts a ON a.id=o.attempt_id WHERE a.rowid<=? ORDER BY a.rowid', (limit,)).fetchall()
    return hashlib.sha256(json.dumps([rows, observations], separators=(',', ':')).encode()).hexdigest()


@contextmanager
def singleton(directory=WORK):
    import msvcrt
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a+b') as lock:
        if lock.tell() == 0: lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally: lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def cache(stage, frozen):
    return SourceEmbeddingCache(WORK / (stage + '.sqlite'), CacheScope(**frozen['scopes'][stage]))


def verify():
    frozen = load(WORK / 'FROZEN_ENCODING.json')
    for name, expected in {**frozen['input_hashes'], **frozen['code_hashes']}.items():
        if 'scorer-only' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError('encoding frozen inputs or code changed')
    return frozen


def prepare():
    if (WORK / 'FROZEN_ENCODING.json').exists(): raise FileExistsError('encoding already frozen')
    prep = load(BASE / 'PREPARATION.json')
    inputs = prep['public_artifact_hashes']
    for name, expected in inputs.items():
        if 'scorer-only' in Path(name).parts or sha(ROOT / name) != expected:
            raise ValueError('source-only preparation changed')
    docs = source_documents(load(BASE / 'memory-source.json'))
    queries = project_queries(load(BASE / 'questions-public.json'))
    qdocs = [dict(query_id='memgallery-v7:' + str(q['index']), query=q, text=query_text(q)) for q in queries]
    if any(len(q['text'].encode('utf-8')) > 8192 for q in qdocs):
        raise ValueError('a query exceeds the frozen one-vector bound')
    WORK.mkdir(parents=True, exist_ok=True)
    write_lines(WORK / 'source-documents.jsonl', docs)
    write_lines(WORK / 'query-documents.jsonl', qdocs)
    save(WORK / 'source-mapping.json', source_mapping(docs), exclusive=True)
    scopes = {stage: asdict(CacheScope(namespace='memgallery-v7-architecture-' + stage,
              dataset_revision=REVISION, representation=REPRESENTATION if stage == 'source' else 'native-question-author-caption-v1'))
              for stage in ('source', 'query')}
    frozen = dict(scopes=scopes)
    counts = {}
    for stage, records in (('source', [dict(id=d['id'], text=d['text']) for d in docs]),
                           ('query', [dict(id=q['query_id'], text=q['text']) for q in qdocs])):
        item = cache(stage, frozen)
        prepared = item.prepare_sources(records)
        counts[stage] = dict(records=prepared['source_count'], chunks=prepared['chunk_count'],
                             batches=len(item.missing_batches()), manifest_sha256=prepared['manifest_sha256'])
    planned = sum(c['batches'] for c in counts.values())
    if not 1 <= planned <= 100: raise ValueError('encoding beyond independent bounded plan')
    extra = [WORK / name for name in ('source-documents.jsonl', 'query-documents.jsonl', 'source-mapping.json')]
    inputs = {**inputs, **{p.relative_to(ROOT).as_posix(): sha(p) for p in extra},
              (BASE / 'PREPARATION.json').relative_to(ROOT).as_posix(): sha(BASE / 'PREPARATION.json')}
    frozen.update(created_at_utc=now(), state='encoding_frozen_no_calls', counts=counts,
        max_requests=planned, max_preflight_cny=str(Decimal(planned) * Decimal('.05')),
        input_hashes=inputs, code_hashes={name: sha(ROOT / name) for name in CODE},
        endpoint=BAILIAN_BEIJING_ENDPOINT, model='text-embedding-v4', dimensions=256,
        price_source=PRICE, price_checked_at=CHECKED, fixed_hold_cny='.05',
        query_text_rule='original question plus all original query image captions; no generated captions',
        raw_image_calls=0, gold_loaded=False, max_reader_calls=64, max_judge_calls=8,
        planned_chat_preflight_cny='18', variants=['dense','dense_reranked','hybrid','hybrid_reranked'],
        authorization='User directed autonomous real experiment continuation, with Docker deferred, inside existing CNY500 budget.')
    save(WORK / 'FROZEN_ENCODING.json', frozen, exclusive=True)
    return dict(state=frozen['state'], counts=counts, planned_requests=planned,
                maximum_encoding_hold_cny=frozen['max_preflight_cny'], raw_image_calls=0, gold_loaded=False)


def execute(max_batches=None):
    with singleton():
        frozen = verify(); ledger = BudgetLedger(SHARED_LEDGER)
        live = WORK / 'FROZEN_LIVE.json'
        if not live.exists():
            if (WORK / 'run.sqlite').exists(): raise ValueError('run without live freeze')
            baseline = ledger.totals()
            save(live, dict(encoding_sha256=sha(WORK / 'FROZEN_ENCODING.json'), baseline=baseline,
                historical_sha256=history(baseline['attempts']), created_at_utc=now()), exclusive=True)
        started = load(live)
        if started['encoding_sha256'] != sha(WORK / 'FROZEN_ENCODING.json') or history(started['baseline']['attempts']) != started['historical_sha256']:
            raise ValueError('frozen encoding or historical accounting changed')
        runpath = WORK / 'run.sqlite'
        run = DevelopmentRun(runpath) if runpath.exists() else DevelopmentRun(runpath, create=True,
            max_requests=frozen['max_requests'], max_reservation_cny=frozen['max_preflight_cny'], approval_scope='native_100_20')
        if run.summary()['state'] != 'active': raise ValueError('finished encoding guard never reopens')
        rate = VerifiedRate('text-embedding-v4', '.5', '0', PRICE, CHECKED)
        def factory(http, observer):
            base = BudgetedJSONTransport(model='text-embedding-v4', endpoint=BAILIAN_BEIJING_ENDPOINT,
                ledger=ledger, rate=rate, credential_reader=read_bailian, http=http, mode='live', enabled=True,
                prices_confirmed=True, fixed_reserve_cny='.05', timeout_seconds=60, attempt_observer=observer)
            return ScopedTransport(base, run, '.05')
        completed = 0; stage = None; start = time.monotonic()
        def progress(state, error=None):
            value = dict(state=state, stage=stage, updated_at_utc=now(), run=run.summary(), budget=ledger.totals())
            if error: value['error_type'] = error
            save(PROGRESS, value); print(json.dumps(value), flush=True)
        try:
            for stage in ('source','query'):
                item = cache(stage, frozen)
                executor = DurableEmbeddingExecutor(item, WORK / 'receipts' / stage, NativeHTTPS(), factory)
                for batch in item.missing_batches():
                    if time.monotonic() - start > 7200: raise ValueError('invocation deadline')
                    result = executor.execute(batch)
                    if result['run_accounting_check_required']: raise ValueError('receipt accounting review required')
                    completed += 1; progress('running')
                    if max_batches and completed >= max_batches:
                        progress('bounded_invocation_finished_guard_active'); return
                item.publish()
            if history(started['baseline']['attempts']) != started['historical_sha256']: raise ValueError('history changed')
            run.close(); progress('encoding_complete')
        except BaseException as exc:
            category = getattr(exc, 'category', type(exc).__name__)
            progress('stopped_no_automatic_retry', category)
            raise SystemExit('Encoding stopped: ' + category + '; inspect durable state.') from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare','encode'))
    parser.add_argument('--live', action='store_true'); parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.phase == 'prepare': print(json.dumps(prepare()))
    elif not args.live: raise SystemExit('Explicit --live required')
    else: execute(args.max_batches)
