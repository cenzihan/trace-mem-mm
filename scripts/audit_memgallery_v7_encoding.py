"""Additive, read-only audit of completed V7 encoding; no provider or gold access.

Only the new aggregate report is written. Existing cache/journal/run/ledger
connections use SQLite mode=ro and query_only. Cache construction is bypassed
because its ordinary constructor initializes tables. No repair or recovery runs.
"""
import hashlib
import json
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache
from agent_memory.memgallery_v7_reader import question_text
from agent_memory.visual_recovery import _strict_json

WORK = Path('runtime/memgallery-v7/encoding')
REPORT = Path('reports/memgallery-v7-encoding-audit.json')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def load(path):
    return _strict_json(path.read_bytes())


def lines(path):
    return [_strict_json(line) for line in path.read_bytes().splitlines() if line.strip()]


def ro(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


def require(condition, message):
    if not condition:
        raise ValueError(message)


def history(db, limit):
    rows = [tuple(r) for r in db.execute('SELECT * FROM attempts WHERE rowid<=? ORDER BY rowid', (limit,))]
    observations = [tuple(r) for r in db.execute('SELECT o.* FROM provider_observations o JOIN attempts a '
                    'ON a.id=o.attempt_id WHERE a.rowid<=? ORDER BY a.rowid', (limit,))]
    return digest(json.dumps([rows, observations], separators=(',', ':')).encode())


def published_readonly(path, scope):
    # Reuse exact integrity validators without the mutating constructor.
    item = SourceEmbeddingCache.__new__(SourceEmbeddingCache)
    item.path, item.scope = path.resolve(), CacheScope(**scope)
    item.scope_json = canonical(asdict(item.scope))
    item.scope_sha256 = digest(item.scope_json.encode('utf-8'))
    with closing(ro(path)) as db:
        return item._snapshot(db, publish=False)


def audit(root=ROOT):
    root = Path(root).resolve()
    work = root / WORK
    frozen, live = load(work / 'FROZEN_ENCODING.json'), load(work / 'FROZEN_LIVE.json')
    require(live['encoding_sha256'] == sha(work / 'FROZEN_ENCODING.json'), 'live/encoding freeze mismatch')
    require(frozen['max_requests'] == 27 and frozen['max_preflight_cny'] == '1.35', 'expected completed27 plan changed')
    preserved = dict(frozen['input_hashes'], **frozen['code_hashes'])
    for relative, expected in preserved.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and 'scorer-only' not in path.parts, 'audit input outside public scope')
        require(sha(path) == expected, 'frozen source/input/code hash changed')
    mutable_paths = [work / 'source.sqlite', work / 'query.sqlite', work / 'run.sqlite',
                     work / 'receipts/source/private-embedding.sqlite', work / 'receipts/query/private-embedding.sqlite']
    before = {p.relative_to(root).as_posix(): sha(p) for p in mutable_paths}
    with closing(ro(work / 'run.sqlite')) as db:
        runs = db.execute('SELECT max_requests,cap,state FROM run').fetchall()
        calls = {r['ordinal']: dict(r) for r in db.execute('SELECT * FROM calls ORDER BY ordinal')}
    require(len(runs) == 1 and tuple(runs[0]) == (27, 1350000, 'complete'), 'bounded guard not complete')
    require(set(calls) == set(range(1, 28)), '27 unique sequential guard ordinals required')
    require(all(c['state'] == 'complete' and c['model'] == 'text-embedding-v4'
                and c['returned_model'] == 'text-embedding-v4' and c['held'] == 50000
                for c in calls.values()), 'guard has unfinished or wrong-model calls')
    stages, all_attempts, all_ordinals, query_image_count = {}, [], [], 0
    accounting = []
    for stage in ('source', 'query'):
        snapshot = published_readonly(work / (stage + '.sqlite'), frozen['scopes'][stage])
        manifest, expected = snapshot['manifest'], frozen['counts'][stage]
        require(snapshot['manifest_sha256'] == expected['manifest_sha256']
                and manifest['source_count'] == expected['records']
                and manifest['chunk_count'] == expected['chunks'], 'cache differs from frozen manifest/counts')
        documents = lines(work / (stage + '-documents.jsonl'))
        key = 'id' if stage == 'source' else 'query_id'
        texts = {row[key]: row['text'] for row in documents}
        require(len(texts) == len(documents) == manifest['source_count'], 'source document identity mismatch')
        entries = {row['key']: row for row in snapshot['chunks']}
        restored = {identity: ''.join(entries[chunk]['text'] for chunk in keys)
                    for identity, keys in snapshot['source_chunks'].items()}
        require(restored == texts, 'cache chunks do not reconstruct exact frozen document text')
        if stage == 'query':
            for row in documents:
                query = row['query']
                question_text(query)  # Schema/aligned-caption check only; never selects by meaning.
                require(not (query.get('question_image') and query.get('query_image')
                             and query['question_image'] != query['query_image']), 'query aliases conflict')
                images = query.get('question_image') or query.get('query_image')
                captions = query.get('image_caption')
                if images:
                    query_image_count += len(images) if isinstance(images, list) else 1
                    captions = captions if isinstance(captions, list) else [captions]
                    expected_text = query['question'] + '\nQUERY_IMAGE_CAPTIONS:\n' + '\n'.join(captions)
                else:
                    expected_text = query['question']
                require(row['text'] == expected_text, 'frozen query representation differs')
        with closing(ro(work / ('receipts/' + stage + '/private-embedding.sqlite'))) as db:
            operations = [dict(r) for r in db.execute('SELECT * FROM operations')]
            claims = [tuple(r) for r in db.execute('SELECT scope,chunk_key,fingerprint FROM claims')]
        require(len(operations) == expected['batches'], 'journal batch count differs from freeze')
        covered, expected_claims = [], []
        for row in operations:
            require(row['scope'] == snapshot['scope_sha256'] and row['state'] == 'complete'
                    and row['transport_completed'] == 1 and row['status'] == 200 and row['redacted'] == 0,
                    'unfinished, redacted or invalid journal operation')
            identity = _strict_json(row['manifest'])
            keys = identity['keys']
            require(1 <= len(keys) <= 10 and len(set(keys)) == len(keys)
                    and set(keys) <= set(entries), 'journal chunk keys are not an aligned cache batch')
            payload = dict(model='text-embedding-v4', input=[entries[k]['text'] for k in keys],
                           dimensions=256, encoding_format='float')
            expected_identity = dict(scope_sha256=snapshot['scope_sha256'],
                manifest_sha256=snapshot['manifest_sha256'], keys=keys,
                text_sha256s=[entries[k]['text_sha256'] for k in keys],
                payload_sha256=digest(canonical(payload).encode('utf-8')))
            require(identity == expected_identity and row['manifest'] == canonical(expected_identity)
                    and row['fingerprint'] == digest(canonical(expected_identity).encode('utf-8')),
                    'journal identity does not match exact frozen source payload')
            require(row['run_path'] == str((work / 'run.sqlite').resolve())
                    and row['run_ordinal'] in calls and isinstance(row['attempt_id'], str), 'journal run binding mismatch')
            require(isinstance(row['body'], bytes) and digest(row['body']) == row['body_sha'], 'receipt body hash changed')
            response = _strict_json(row['body'])
            require(response.get('model') == 'text-embedding-v4', 'receipt returned wrong model')
            vectors = response.get('data')
            require(isinstance(vectors, list) and len(vectors) == len(keys)
                    and all(type(v.get('index')) is int for v in vectors)
                    and {v['index'] for v in vectors} == set(range(len(keys))), 'receipt vector alignment differs')
            vectors = sorted(vectors, key=lambda v: v['index'])
            for chunk, vector in zip(keys, vectors):
                require(entries[chunk]['vector'] == vector['embedding']
                        and entries[chunk]['provenance'] == dict(returned_model='text-embedding-v4',
                            receipt_sha256=row['body_sha'], budget_attempt_id=row['attempt_id']),
                        'cache vector/provenance differs from its exact successful receipt')
            usage = response['usage']['total_tokens']
            require(type(usage) is int and 0 <= usage <= 8192 * len(keys), 'receipt usage outside bounds')
            accounting.append((row['attempt_id'], usage))
            all_attempts.append(row['attempt_id']); all_ordinals.append(row['run_ordinal'])
            covered.extend(keys)
            expected_claims.extend((row['scope'], k, row['fingerprint']) for k in keys)
        require(len(covered) == len(set(covered)) and set(covered) == set(entries), 'chunks missing or encoded more than once')
        require(sorted(claims) == sorted(expected_claims), 'journal claims do not bind every exact chunk')
        stages[stage] = dict(records=len(texts), chunks=len(entries), complete_batches=len(operations),
                            manifest_sha256=snapshot['manifest_sha256'], published_sha256=snapshot['published_sha256'],
                            exact_source_reconstruction=True, exact_receipt_vector_match=True)
    require(len(all_attempts) == len(set(all_attempts)) == 27
            and sorted(all_ordinals) == list(range(1, 28)), 'requests do not bind one-to-one to all27 guard slots')
    ledger_path = root / 'runtime/development-budget.sqlite'
    with closing(ro(ledger_path)) as db:
        require(history(db, live['baseline']['attempts']) == live['historical_sha256'], 'historical budget rows changed')
        accounted = 0
        for attempt, tokens in accounting:
            budget = db.execute('SELECT rowid AS original_rowid,* FROM attempts WHERE id=?', (attempt,)).fetchone()
            observed = db.execute('SELECT * FROM provider_observations WHERE attempt_id=?', (attempt,)).fetchone()
            require(budget is not None and observed is not None, 'journal budget receipt binding missing')
            require(budget['original_rowid'] > live['baseline']['attempts'] and budget['status'] == 'settled'
                    and budget['model'] == 'text-embedding-v4' and budget['reserved'] == 50000
                    and budget['input_tokens'] == tokens and budget['output_tokens'] == 0
                    and budget['amount_kind'] == 'provider_usage_estimate', 'embedding budget is not exactly settled')
            require(observed['mode'] == 'live' and observed['category'] == 'success'
                    and observed['http_status'] == 200 and observed['returned_model'] == 'text-embedding-v4'
                    and observed['input_tokens'] == tokens and observed['output_tokens'] == 0,
                    'successful provider observation does not match receipt')
            accounted += budget['accounted']
    require(before == {p.relative_to(root).as_posix(): sha(p) for p in mutable_paths}, 'audit changed/found concurrent encoding artifacts')
    for relative, expected in preserved.items():
        require(sha(root / relative) == expected, 'frozen input/code changed during audit')
    return dict(schema='memgallery-v7-encoding-additive-audit-v1', state='passed',
        audited_at_utc=datetime.now(timezone.utc).isoformat(), stages=stages, real_embedding_requests=27,
        guard_complete=True, complete_unique_run_bindings=27, unknown_or_inflight_operations=0,
        query_image_count=query_image_count, all_query_images_have_aligned_original_captions=True,
        query_alias_conflicts=0, historical_budget_rows_unchanged=True,
        new_usage_estimate_micro_cny=accounted, cumulative_preflight_micro_cny=1350000,
        frozen_encoding_sha256=sha(work / 'FROZEN_ENCODING.json'),
        frozen_live_sha256=sha(work / 'FROZEN_LIVE.json'), artifact_sha256=before,
        audit_script_sha256=sha(Path(__file__)), source_text_or_answers_in_report=False,
        provider_requests_by_audit=0, gold_files_opened=False, existing_artifacts_modified=False,
        future_version_improvements=[
            'Before dispatch, compare live cache manifest/counts with frozen counts; this additive audit verifies the completed run.',
            'Reject zero/negative/noninteger max_batches instead of interpreting zero as an unlimited invocation.',
            'Check query image aliases for consistency and all existing journal accounting flags before any next dispatch.'],
        condition='original dialogue plus author image captions; no raw image inference; no benchmark performance claim')


if __name__ == '__main__':
    result = audit()
    destination = ROOT / REPORT
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state=result['state'], requests=27, source_records=result['stages']['source']['records'],
                         query_records=result['stages']['query']['records'], report=REPORT.as_posix())))
