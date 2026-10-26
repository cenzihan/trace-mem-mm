"""Local cache of successfully obtained source embeddings, with frozen provenance.

This module has no provider, credential, request journal, or network integration.
``missing_batches`` is a cache inventory, NOT permission to send or retry a request.
An external durable request journal/budget gate must resolve uncertain attempts
before any caller dispatches work. Importing a receipt reference does not settle
that journal or ledger. Source texts remain sensitive local runtime artifacts.
"""
import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
DIMENSIONS = (64, 128, 256, 512, 768, 1024, 1536, 2048)
CHUNK_BYTES = 8192
TABLES = {'source_cache_scopes', 'source_cache_chunks'}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('canonical SHA256 required')
    return value


@dataclass(frozen=True)
class CacheScope:
    namespace: str
    dataset_revision: str
    representation: str
    model: str = 'text-embedding-v4'
    dimensions: int = 256

    def __post_init__(self):
        if not all(_nonempty(value) for value in
                   (self.namespace, self.dataset_revision, self.representation)):
            raise ValueError('explicit nonempty namespace, revision and representation required')
        if self.model != 'text-embedding-v4':
            raise ValueError('source cache only supports text-embedding-v4')
        if type(self.dimensions) is not int or self.dimensions not in DIMENSIONS:
            raise ValueError('unsupported text-embedding-v4 dimensions')


def _split(text):
    """Split on Unicode boundaries without adding/removing any byte or character."""
    pieces, current, size = [], [], 0
    for character in text:
        width = len(character.encode('utf-8'))
        if size + width > CHUNK_BYTES:
            pieces.append(''.join(current))
            current, size = [], 0
        current.append(character)
        size += width
    if current:
        pieces.append(''.join(current))
    return pieces


class SourceEmbeddingCache:
    """One frozen source manifest per scope; scopes never share vector identities.

    Production paths must be inside this project's runtime directory. Explicit
    offline tests must use the system temporary directory, outside project runtime.
    Public methods open/close their own connections; no background work exists.
    """
    def __init__(self, path, scope, offline_test=False):
        if not isinstance(scope, CacheScope) or type(offline_test) is not bool:
            raise ValueError('CacheScope and explicit boolean offline_test required')
        self.path = Path(path).resolve()
        runtime, temporary = RUNTIME.resolve(), Path(tempfile.gettempdir()).resolve()
        valid = (self.path.is_relative_to(temporary) and not self.path.is_relative_to(runtime)
                 if offline_test else self.path.is_relative_to(runtime))
        if not valid or self.path in (runtime, temporary) or self.path.suffix != '.sqlite':
            raise ValueError('cache must be a .sqlite file within runtime or an explicit temporary test directory')
        self.scope = scope
        self.scope_json = _json(asdict(scope))
        self.scope_sha256 = _sha(self.scope_json)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables and tables != TABLES:
                raise ValueError('refusing an unrelated or incomplete SQLite database')
            db.execute('BEGIN IMMEDIATE')
            db.execute('''CREATE TABLE IF NOT EXISTS source_cache_scopes (
                scope TEXT PRIMARY KEY, scope_json TEXT NOT NULL,
                manifest TEXT, manifest_hash TEXT, published TEXT, published_hash TEXT)''')
            db.execute('''CREATE TABLE IF NOT EXISTS source_cache_chunks (
                scope TEXT NOT NULL, key TEXT NOT NULL, text TEXT NOT NULL,
                text_hash TEXT NOT NULL, vector TEXT, vector_hash TEXT,
                provenance TEXT, entry_hash TEXT, PRIMARY KEY(scope,key))''')
            db.execute('INSERT OR IGNORE INTO source_cache_scopes(scope,scope_json) VALUES (?,?)',
                       (self.scope_sha256, self.scope_json))
            self._state(db, prepared=False)

    def _key(self, text_hash):
        return _sha(_json([self.scope_sha256, text_hash]))

    def _state(self, db, prepared=True):
        row = db.execute('SELECT scope_json,manifest,manifest_hash,published,published_hash '
                         'FROM source_cache_scopes WHERE scope=?', (self.scope_sha256,)).fetchone()
        if row is None or row[0] != self.scope_json or _sha(row[0]) != self.scope_sha256:
            raise ValueError('cache scope integrity failure')
        if row[1] is None:
            if prepared:
                raise ValueError('prepare the complete source manifest first')
            if any(value is not None for value in row[2:]):
                raise ValueError('unprepared scope contains inconsistent state')
            return None, None, None
        if _sha(row[1]) != _digest(row[2]):
            raise ValueError('source manifest hash mismatch')
        manifest = json.loads(row[1])
        if manifest.get('scope_sha256') != self.scope_sha256 or manifest.get('scope') != asdict(self.scope):
            raise ValueError('source manifest belongs to another scope')
        published = None
        if row[3] is not None:
            if _sha(row[3]) != _digest(row[4]):
                raise ValueError('published snapshot hash mismatch')
            published = json.loads(row[3])
        elif row[4] is not None:
            raise ValueError('incomplete publication record')
        return manifest, row[2], published

    def _vector(self, vector):
        if not isinstance(vector, (list, tuple)) or len(vector) != self.scope.dimensions:
            raise ValueError('vector dimension mismatch')
        if any(type(value) not in (int, float) for value in vector):
            raise ValueError('vector values must be finite numbers, not booleans')
        try:
            values = [float(value) for value in vector]
            norm = math.hypot(*values)
        except (OverflowError, ValueError):
            raise ValueError('vector numerical overflow') from None
        if any(not math.isfinite(value) for value in values) or not math.isfinite(norm) or norm <= 0:
            raise ValueError('vector must have finite positive norm')
        return values

    def _provenance(self, value):
        if (not isinstance(value, dict)
                or set(value) != {'returned_model', 'receipt_sha256', 'budget_attempt_id'}
                or value['returned_model'] != self.scope.model
                or not _nonempty(value['budget_attempt_id'])):
            raise ValueError('exact model, receipt SHA256 and nonempty budget attempt reference required')
        _digest(value['receipt_sha256'])
        return dict(value)

    def _entry(self, row):
        key, text, text_hash, vector_json, vector_hash, provenance_json, entry_hash = row
        if (not _nonempty(text) or len(text.encode('utf-8')) > CHUNK_BYTES
                or _sha(text) != _digest(text_hash) or self._key(text_hash) != key):
            raise ValueError('source chunk integrity failure')
        result = dict(key=key, text=text, text_sha256=text_hash, vector=None, provenance=None)
        if vector_json is None:
            if any(value is not None for value in (vector_hash, provenance_json, entry_hash)):
                raise ValueError('incomplete vector record')
            return result, None
        vector = self._vector(json.loads(vector_json))
        provenance = self._provenance(json.loads(provenance_json))
        if _sha(vector_json) != _digest(vector_hash) or vector_json != _json(vector):
            raise ValueError('cached vector hash mismatch')
        expected_entry = _sha(_json(dict(key=key, text_sha256=text_hash,
                                       vector_sha256=vector_hash, provenance=provenance)))
        if expected_entry != _digest(entry_hash):
            raise ValueError('embedding provenance integrity failure')
        result.update(vector=vector, provenance=provenance)
        return result, entry_hash

    def _all(self, db, manifest):
        entries, hashes = {}, {}
        rows = db.execute('SELECT key,text,text_hash,vector,vector_hash,provenance,entry_hash '
                          'FROM source_cache_chunks WHERE scope=? ORDER BY key', (self.scope_sha256,))
        for row in rows:
            entry, digest = self._entry(row)
            entries[entry['key']] = entry
            hashes[entry['key']] = digest
        expected = {key for source in manifest['sources'] for key in source['chunk_keys']}
        if set(entries) != expected or len(entries) != manifest['chunk_count']:
            raise ValueError('missing or unreferenced source chunks')
        if len(manifest['sources']) != manifest['source_count']:
            raise ValueError('source manifest count mismatch')
        source_ids = set()
        for source in manifest['sources']:
            if source['id'] in source_ids or not source['chunk_keys']:
                raise ValueError('invalid source identity or missing source chunks')
            source_ids.add(source['id'])
            restored = ''.join(entries[key]['text'] for key in source['chunk_keys'])
            if _sha(restored) != source['text_sha256'] or len(restored.encode('utf-8')) != source['utf8_bytes']:
                raise ValueError('native source reconstruction mismatch')
        return entries, hashes

    def prepare_sources(self, records):
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError('complete nonempty list of source records required')
        sources, chunks, identities = [], {}, set()
        for record in records:
            if (not isinstance(record, dict) or set(record) != {'id', 'text'}
                    or not _nonempty(record['id']) or not _nonempty(record['text'])):
                raise ValueError('source-only id/text fields and nonempty original text required')
            if record['id'] in identities:
                raise ValueError('duplicate native source ID')
            identities.add(record['id'])
            keys = []
            for piece in _split(record['text']):
                if not _nonempty(piece):
                    raise ValueError('source produces an unsupported whitespace-only chunk; no trimming or source omission allowed')
                digest = _sha(piece)
                key = self._key(digest)
                chunks[key] = (piece, digest)
                keys.append(key)
            sources.append(dict(id=record['id'], text_sha256=_sha(record['text']),
                utf8_bytes=len(record['text'].encode('utf-8')), chunk_keys=keys))
        manifest = dict(version=1, scope=asdict(self.scope), scope_sha256=self.scope_sha256,
            chunk_limit_utf8_bytes=CHUNK_BYTES, source_count=len(sources), chunk_count=len(chunks),
            sources=sorted(sources, key=lambda row: row['id']))
        encoded, digest = _json(manifest), _sha(_json(manifest))
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            existing, previous_digest, _ = self._state(db, prepared=False)
            if existing is not None:
                if previous_digest != digest or existing != manifest:
                    raise ValueError('scope source manifest already frozen; use a new representation scope')
                self._all(db, existing)
            else:
                if db.execute('SELECT 1 FROM source_cache_chunks WHERE scope=? LIMIT 1',
                              (self.scope_sha256,)).fetchone():
                    raise ValueError('unprepared scope contains orphan chunks')
                db.executemany('INSERT INTO source_cache_chunks(scope,key,text,text_hash) VALUES (?,?,?,?)',
                    ((self.scope_sha256, key, text, text_hash) for key, (text, text_hash) in chunks.items()))
                db.execute('UPDATE source_cache_scopes SET manifest=?,manifest_hash=? WHERE scope=?',
                           (encoded, digest, self.scope_sha256))
        return dict(manifest=manifest, manifest_sha256=digest,
                    source_count=len(sources), chunk_count=len(chunks))

    def missing_batches(self, batch_size=10):
        """Return missing cache values only; external journal gates still apply."""
        if type(batch_size) is not int or not 1 <= batch_size <= 10:
            raise ValueError('embedding batches contain at most 10 inputs')
        with closing(sqlite3.connect(self.path)) as db:
            manifest, _, _ = self._state(db)
            entries, _ = self._all(db, manifest)
        missing = [dict(key=key, text=entry['text']) for key, entry in entries.items() if entry['vector'] is None]
        return [missing[start:start + batch_size] for start in range(0, len(missing), batch_size)]

    def import_batch(self, keys, vectors, provenance):
        if (not isinstance(keys, (list, tuple)) or not 1 <= len(keys) <= 10
                or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys)
                or not isinstance(vectors, (list, tuple)) or len(keys) != len(vectors)):
            raise ValueError('one complete aligned unique batch of at most 10 vectors required')
        proof = self._provenance(provenance)
        values = [self._vector(vector) for vector in vectors]
        inserted = 0
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            manifest, _, published = self._state(db)
            known = {key for source in manifest['sources'] for key in source['chunk_keys']}
            if not set(keys) <= known:
                raise ValueError('chunk belongs to another scope or is not in frozen sources')
            for key, vector in zip(keys, values):
                row = db.execute('SELECT key,text,text_hash,vector,vector_hash,provenance,entry_hash '
                    'FROM source_cache_chunks WHERE scope=? AND key=?', (self.scope_sha256, key)).fetchone()
                if row is None:
                    raise ValueError('frozen source chunk missing')
                entry, _ = self._entry(row)
                encoded = _json(vector)
                if entry['vector'] is not None:
                    if entry['vector'] != vector:
                        raise ValueError('successful cached vector cannot be overwritten')
                    continue  # Retain the first successful receipt; never replace its provenance.
                if published is not None:
                    raise ValueError('published snapshot unexpectedly missing a vector')
                vector_hash = _sha(encoded)
                entry_hash = _sha(_json(dict(key=key, text_sha256=entry['text_sha256'],
                    vector_sha256=vector_hash, provenance=proof)))
                db.execute('UPDATE source_cache_chunks SET vector=?,vector_hash=?,provenance=?,entry_hash=? '
                    'WHERE scope=? AND key=?', (encoded, vector_hash, _json(proof), entry_hash, self.scope_sha256, key))
                inserted += 1
        return dict(imported=inserted, reused=len(keys) - inserted)

    def _snapshot(self, db, publish):
        manifest, manifest_hash, previous = self._state(db)
        entries, hashes = self._all(db, manifest)
        if any(value is None for value in hashes.values()):
            raise ValueError('all source chunk vectors must be complete before publication')
        core = dict(scope_sha256=self.scope_sha256, manifest_sha256=manifest_hash,
                    chunk_sha256s=hashes)
        encoded, digest = _json(core), _sha(_json(core))
        if previous is not None and previous != core:
            raise ValueError('published snapshot no longer matches cache contents')
        if previous is None:
            if not publish:
                raise ValueError('source embedding cache is not published')
            db.execute('UPDATE source_cache_scopes SET published=?,published_hash=? WHERE scope=?',
                       (encoded, digest, self.scope_sha256))
        return dict(scope=asdict(self.scope), scope_sha256=self.scope_sha256,
            manifest=manifest, manifest_sha256=manifest_hash, published_sha256=digest,
            chunks=list(entries.values()),
            source_chunks={source['id']: list(source['chunk_keys']) for source in manifest['sources']})

    def publish(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            return self._snapshot(db, publish=True)

    def read_published(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('PRAGMA query_only=ON')
            return self._snapshot(db, publish=False)
