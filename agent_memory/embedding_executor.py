"""Durable one-attempt embedding batches; explicit transport/guard wiring only.

HTTP receipts are committed before transport parsing. Recovery may import a valid
saved vector but NEVER settles a budget ledger or finishes an interrupted run
claim. Callers must check ``run_accounting_check_required`` and reconcile the
existing journal/guard before continuing. Unknown requests are never replayed,
including through a different partition of the same missing chunk keys.
"""
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path

from .development_run import ScopedTransport
from .embedding_cache import SourceEmbeddingCache, RUNTIME
from .provider_http import BudgetedJSONTransport, HTTPReply, validate_endpoint
from .visual_recovery import _safe_body, _strict_json


BODY_LIMIT = 8 * 1024 * 1024


class EmbeddingExecutorBlocked(RuntimeError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class _Journal:
    extra_tables = frozenset()

    def __init__(self, root, offline_test):
        self.root = Path(root).resolve()
        allowed = Path(tempfile.gettempdir()).resolve() if offline_test else RUNTIME.resolve()
        if (self.root == allowed or not self.root.is_relative_to(allowed)
                or (offline_test and self.root.is_relative_to(RUNTIME.resolve()))):
            raise ValueError('embedding journal must be inside runtime or isolated test temp')
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'private-embedding.sqlite'
        if self.path.is_symlink():
            raise ValueError('journal symlinks are not supported')
        with self.connect() as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and (not {'operations','claims'}.issubset(tables) or tables-{'operations','claims'}-self.extra_tables):
                raise ValueError('wrong embedding journal database')
            db.execute('''CREATE TABLE IF NOT EXISTS operations(
                fingerprint TEXT PRIMARY KEY, scope TEXT NOT NULL, manifest TEXT NOT NULL,
                state TEXT NOT NULL, attempt_id TEXT, status INTEGER, body BLOB,
                body_sha TEXT, redacted INTEGER, transport_completed INTEGER NOT NULL DEFAULT 0,
                run_path TEXT, run_ordinal INTEGER)''')
            db.execute('''CREATE TABLE IF NOT EXISTS claims(
                scope TEXT NOT NULL, chunk_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                PRIMARY KEY(scope,chunk_key))''')
            columns = {row[1] for row in db.execute('PRAGMA table_info(operations)')}
            if 'run_path' not in columns:
                db.execute('ALTER TABLE operations ADD COLUMN run_path TEXT')
            if 'run_ordinal' not in columns:
                db.execute('ALTER TABLE operations ADD COLUMN run_ordinal INTEGER')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def row(self, fingerprint):
        with self.connect() as db:
            row = db.execute('SELECT * FROM operations WHERE fingerprint=?', (fingerprint,)).fetchone()
        return dict(row) if row is not None else None

    def install(self, fingerprint, scope, manifest):
        encoded = _json(manifest)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for key in manifest['keys']:
                row = db.execute('SELECT fingerprint FROM claims WHERE scope=? AND chunk_key=?', (scope, key)).fetchone()
                if row is not None and row[0] != fingerprint:
                    raise EmbeddingExecutorBlocked('chunk already belongs to another batch; no repartition or replay')
            db.execute('INSERT OR IGNORE INTO operations(fingerprint,scope,manifest,state) VALUES(?,?,?,?)',
                       (fingerprint, scope, encoded, 'ready'))
            row = db.execute('SELECT scope,manifest FROM operations WHERE fingerprint=?', (fingerprint,)).fetchone()
            if row['scope'] != scope or row['manifest'] != encoded:
                raise EmbeddingExecutorBlocked('stored embedding request identity changed')
            db.executemany('INSERT OR IGNORE INTO claims VALUES(?,?,?)',
                           ((scope, key, fingerprint) for key in manifest['keys']))

    def start(self, fingerprint):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT scope,state FROM operations WHERE fingerprint=?', (fingerprint,)).fetchone()
            if row is None or row['state'] != 'ready':
                raise EmbeddingExecutorBlocked('operation already attempted; no automatic replay')
            if db.execute("SELECT 1 FROM operations WHERE scope=? AND state NOT IN ('ready','complete') LIMIT 1",
                          (row['scope'],)).fetchone():
                raise EmbeddingExecutorBlocked('unresolved embedding batch blocks further dispatch')
            db.execute("UPDATE operations SET state='inflight' WHERE fingerprint=?", (fingerprint,))

    def attach(self, fingerprint, attempt, run_path=None, run_ordinal=None):
        if not isinstance(attempt, str) or not attempt or len(attempt) > 128:
            raise EmbeddingExecutorBlocked('budget attempt identity required')
        if (run_path is None) != (run_ordinal is None):
            raise EmbeddingExecutorBlocked('complete run path/ordinal binding required')
        if run_path is not None and (not isinstance(run_path, str) or not Path(run_path).is_absolute()
                                     or type(run_ordinal) is not int or run_ordinal < 1):
            raise EmbeddingExecutorBlocked('invalid run binding')
        with self.connect() as db:
            changed = db.execute("UPDATE operations SET attempt_id=?,run_path=?,run_ordinal=? "
                                 "WHERE fingerprint=? AND state='inflight' AND attempt_id IS NULL",
                                 (attempt, run_path, run_ordinal, fingerprint))
            if changed.rowcount != 1:
                raise EmbeddingExecutorBlocked('budget attempt cannot be rebound')

    def received(self, fingerprint, reply, headers):
        if (not isinstance(reply, HTTPReply) or type(reply.status) is not int or not 100 <= reply.status <= 599
                or not isinstance(reply.body, bytes) or len(reply.body) > BODY_LIMIT):
            raise EmbeddingExecutorBlocked('receipt size/type invalid')
        safe, redacted = _safe_body(reply.body, headers)
        if len(safe) > BODY_LIMIT:
            raise EmbeddingExecutorBlocked('sanitized receipt exceeds local limit')
        with self.connect() as db:
            changed = db.execute("""UPDATE operations SET state='received',status=?,body=?,body_sha=?,redacted=?
                WHERE fingerprint=? AND state='inflight' AND attempt_id IS NOT NULL""",
                (reply.status, safe, _sha(safe), int(redacted), fingerprint))
            if changed.rowcount != 1:
                raise EmbeddingExecutorBlocked('receipt requires one bound in-flight budget attempt')

    def unknown(self, fingerprint):
        with self.connect() as db:
            db.execute("UPDATE operations SET state='unknown' WHERE fingerprint=? AND state='inflight'", (fingerprint,))


class _ReceiptHTTP:
    def __init__(self, inner, journal, fingerprint, payload, after_received):
        self.inner, self.journal, self.fingerprint = inner, journal, fingerprint
        self.payload, self.after_received = payload, after_received
        self.endpoint = None
        self.is_fake_http = getattr(inner, 'is_fake_http', False)

    def exchange(self, endpoint, body, headers, timeout, max_response_bytes):
        if endpoint != self.endpoint or _json(_strict_json(body)) != _json(self.payload):
            raise EmbeddingExecutorBlocked('request changed after durable preparation')
        row = self.journal.row(self.fingerprint)
        if row['state'] != 'inflight' or not row['attempt_id']:
            raise EmbeddingExecutorBlocked('budget attempt must be bound before network dispatch')
        try:reply = self.inner.exchange(endpoint, body, headers, timeout, max_response_bytes)
        except BaseException as exc:
            self.failure_phase=getattr(exc,'phase',None)
            self.failure_kind=getattr(exc,'transport_kind',None)
            raise
        self.journal.received(self.fingerprint, reply, headers)
        if self.after_received is not None:
            self.after_received()
        return reply


class DurableEmbeddingExecutor:
    """Factory(wrapped_http, attempt_observer) returns a budgeted transport.

    Live execution additionally requires ScopedTransport (approved DevelopmentRun)
    and exactly one transport attempt. Successful cache imports preserve the
    receipt hash and budget attempt reference. This class never creates a guard,
    approves a transfer, reads credentials, settles a ledger, or retries a request.
    """
    def __init__(self, cache, journal_root, http, transport_factory, *, offline_test=False, after_received=None):
        if not isinstance(cache, SourceEmbeddingCache) or not callable(transport_factory):
            raise ValueError('source cache and explicit budgeted transport factory required')
        if type(offline_test) is not bool or (after_received is not None and not callable(after_received)):
            raise ValueError('explicit boolean mode and optional callable fault hook required')
        if offline_test and getattr(http, 'is_fake_http', False) is not True:
            raise ValueError('offline executor requires explicitly fake HTTP')
        if not offline_test and not cache.path.is_relative_to(RUNTIME.resolve()):
            raise ValueError('live executor cache must be in project runtime')
        self.cache, self.http, self.factory = cache, http, transport_factory
        self.offline_test, self.after_received = offline_test, after_received
        self.journal = _Journal(journal_root, offline_test)

    def _prepare(self, batch):
        if not isinstance(batch, (list, tuple)) or not 1 <= len(batch) <= 10:
            raise ValueError('complete 1..10 source cache batch required')
        if any(not isinstance(row, dict) or set(row) != {'key', 'text'} for row in batch):
            raise ValueError('batch entries must contain cache key/text only')
        keys = [row['key'] for row in batch]
        if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError('unique source chunk keys required')
        entries = []
        with closing(sqlite3.connect(self.cache.path)) as db:
            db.execute('PRAGMA query_only=ON')
            manifest, manifest_hash, _ = self.cache._state(db)
            known = {key for source in manifest['sources'] for key in source['chunk_keys']}
            if not set(keys) <= known:
                raise ValueError('batch not in this frozen cache scope')
            for row in batch:
                stored = db.execute('SELECT key,text,text_hash,vector,vector_hash,provenance,entry_hash '
                    'FROM source_cache_chunks WHERE scope=? AND key=?', (self.cache.scope_sha256, row['key'])).fetchone()
                if stored is None:
                    raise ValueError('source chunk missing')
                entry, _ = self.cache._entry(stored)
                if row['text'] != entry['text']:
                    raise ValueError('batch text differs from original cache chunk')
                entries.append(entry)
        payload = dict(model=self.cache.scope.model, input=[row['text'] for row in batch],
                       dimensions=self.cache.scope.dimensions, encoding_format='float')
        identity = dict(scope_sha256=self.cache.scope_sha256, manifest_sha256=manifest_hash,
                        keys=keys, text_sha256s=[entry['text_sha256'] for entry in entries],
                        payload_sha256=_sha(_json(payload).encode('utf-8')))
        fingerprint = _sha(_json(identity).encode('utf-8'))
        return fingerprint, identity, payload, entries

    def _recover(self, fingerprint, identity, payload, recovered):
        row = self.journal.row(fingerprint)
        if (row is None or row['state'] not in ('received', 'invalid_response', 'complete')
                or row['scope'] != self.cache.scope_sha256 or row['manifest'] != _json(identity)
                or row['body'] is None or _sha(row['body']) != row['body_sha'] or not row['attempt_id']):
            raise EmbeddingExecutorBlocked('saved successful receipt unavailable or altered; no automatic replay')
        try:
            if row['status'] != 200:
                raise ValueError('saved HTTP response is not successful')
            data = _strict_json(row['body'])
            if not isinstance(data, dict) or data.get('model') != self.cache.scope.model:
                raise ValueError('saved embedding model mismatch')
            usage = data.get('usage')
            if (not isinstance(usage, dict) or type(usage.get('total_tokens')) is not int
                    or not 0 <= usage['total_tokens'] <= 8192 * len(identity['keys'])
                    or ('prompt_tokens' in usage and (type(usage['prompt_tokens']) is not int
                        or usage['prompt_tokens'] != usage['total_tokens']))):
                raise ValueError('saved embedding usage invalid')
            results = data.get('data')
            if (not isinstance(results, list) or len(results) != len(identity['keys'])
                    or any(not isinstance(item, dict) or type(item.get('index')) is not int for item in results)
                    or {item['index'] for item in results} != set(range(len(results)))):
                raise ValueError('saved embedding batch is incomplete or misaligned')
            vectors = [self.cache._vector(item.get('embedding')) for item in sorted(results, key=lambda item: item['index'])]
            self.cache.import_batch(identity['keys'], vectors, dict(returned_model=data['model'],
                receipt_sha256=row['body_sha'], budget_attempt_id=row['attempt_id']))
        except (ValueError, UnicodeError, TypeError, KeyError):
            with self.journal.connect() as db:
                db.execute("UPDATE operations SET state='invalid_response' WHERE fingerprint=?", (fingerprint,))
            raise EmbeddingExecutorBlocked('saved embedding receipt failed validation; no automatic replay') from None
        with self.journal.connect() as db:
            db.execute("UPDATE operations SET state='complete' WHERE fingerprint=?", (fingerprint,))
        return dict(fingerprint=fingerprint, state='complete', keys=list(identity['keys']),
                    attempt_id=row['attempt_id'], receipt_sha256=row['body_sha'], recovered=recovered,
                    run_path=row['run_path'], run_ordinal=row['run_ordinal'],
                    run_accounting_check_required=not bool(row['transport_completed']))

    def execute(self, batch):
        fingerprint, identity, payload, entries = self._prepare(batch)
        existing = self.journal.row(fingerprint)
        if existing is None and all(entry['vector'] is not None for entry in entries):
            return dict(fingerprint=fingerprint, state='cache_hit', keys=identity['keys'],
                        recovered=True, run_accounting_check_required=True)
        self.journal.install(fingerprint, self.cache.scope_sha256, identity)
        state = self.journal.row(fingerprint)['state']
        if state in ('received', 'invalid_response', 'complete'):
            return self._recover(fingerprint, identity, payload, recovered=True)
        if state != 'ready':
            raise EmbeddingExecutorBlocked('unknown/inflight outcome retained; no automatic replay')
        if any(entry['vector'] is not None for entry in entries):
            raise EmbeddingExecutorBlocked('mixed cached/missing batch cannot be dispatched')
        self.journal.start(fingerprint)
        wrapped = _ReceiptHTTP(self.http, self.journal, fingerprint, payload, self.after_received)
        binding = {}
        def observer(attempt):
            run = binding.get('run')
            if run is None:
                self.journal.attach(fingerprint, attempt)
                return
            with run.connect() as db:
                claims = db.execute("SELECT ordinal,model FROM calls WHERE state='claimed'").fetchall()
            if len(claims) != 1 or claims[0][1] != self.cache.scope.model:
                raise EmbeddingExecutorBlocked('exactly one matching claimed run ordinal required')
            self.journal.attach(fingerprint, attempt, str(Path(run.path).resolve()), claims[0][0])
        try:
            transport = self.factory(wrapped, observer)
            base = transport.transport if isinstance(transport, ScopedTransport) else transport
            if (not isinstance(base, BudgetedJSONTransport) or base.model != self.cache.scope.model
                    or base.http is not wrapped or base.attempt_observer is not observer
                    or base.retry.max_attempts != 1 or base.retry.retry_429
                    or (base.mode == 'live' and not isinstance(transport, ScopedTransport))
                    or (self.offline_test and base.mode != 'offline_fake')):
                raise EmbeddingExecutorBlocked('one-attempt budgeted transport and approved live run guard required')
            validate_endpoint(self.cache.scope.model, base.endpoint)
            if isinstance(transport, ScopedTransport):
                run_path = Path(transport.run.path).resolve()
                if base.mode == 'live' and not run_path.is_relative_to(RUNTIME.resolve()):
                    raise EmbeddingExecutorBlocked('live run guard must be in project runtime')
                binding['run'] = transport.run
            wrapped.endpoint = base.endpoint
            transport(base.endpoint, payload)
            with self.journal.connect() as db:
                db.execute('UPDATE operations SET transport_completed=1 WHERE fingerprint=?', (fingerprint,))
            return self._recover(fingerprint, identity, payload, recovered=False)
        except BaseException:
            self.journal.unknown(fingerprint)
            raise
