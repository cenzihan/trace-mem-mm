import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .adapters import ConfiguredModelBoundary, IdentityMockReranker, ModelConfig
from .protocol import (ContractError, MEDIA_LIMIT, canonical, canonical_digest, content_parts, image_bytes,
                       image_hashes, media_size, text_of, validate_add, validate_search)


def tokens(text):
    # Unicode Latin words/numbers + individual CJK characters; no fitted vocabulary.
    return set(re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.casefold()))


class MemoryStore:
    def __init__(self, path, config=None, reranker=None):
        self.config = config or ModelConfig()
        ConfiguredModelBoundary(self.config)
        self.reranker = reranker or IdentityMockReranker()
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            kind=db.execute('PRAGMA application_id').fetchone()[0]
            version=db.execute('PRAGMA user_version').fetchone()[0]
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if kind not in (0,0x414D4D32) or version>3 or tables & {'budget','attempts','smoke_attempts'}:
                raise ValueError('unsupported or wrong database type; no destructive migration')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS requests (
                    user_id TEXT NOT NULL, request_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    digest TEXT NOT NULL, PRIMARY KEY(user_id, request_id));
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS memory_scope ON memories(user_id, session_id);
                CREATE TABLE IF NOT EXISTS evidence (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    part_ordinal INTEGER NOT NULL, sha256 TEXT NOT NULL, mime TEXT NOT NULL,
                    original BLOB NOT NULL, PRIMARY KEY(memory_id, part_ordinal));
            ''')
            db.execute('PRAGMA application_id=1095585074')
            if version<1:db.execute('PRAGMA user_version=1')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, payload):
        validate_add(payload)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            response = self._persist_add(db, payload)
        return response

    def _guard_operation(self,db,uid,req,digest,operation_token=None):
        operation=None
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='add_operations'").fetchone():
            operation=db.execute('SELECT digest,state,owner_token FROM add_operations WHERE user_id=? AND request_id=?',(uid,req)).fetchone()
            if operation:
                if operation[0]!=digest:raise ContractError('request_id reused with a different payload',409)
                if operation[1]!='complete' and (operation[1]!='preparing' or operation_token is None or operation_token!=operation[2]):
                    raise ContractError('model Add outcome unresolved; do not automatically retry',503)
        if operation_token is not None and (not operation or operation[1]!='preparing' or operation[2]!=operation_token):
            raise ContractError('model Add ownership changed; inspect outcome',503)

    def _persist_add(self, db, payload, operation_token=None):
        """Shared raw persistence inside the caller's transaction; no independent commit."""
        digest = canonical_digest(payload)
        uid, req, session = (payload[k] for k in ("user_id", "request_id", "session_id"))
        self._guard_operation(db,uid,req,digest,operation_token)
        response = dict(success=True, request_id=req, user_id=uid, session_id=session)
        old = db.execute("SELECT digest FROM requests WHERE user_id=? AND request_id=?", (uid, req)).fetchone()
        if old:
            if old[0] != digest:
                raise ContractError("request_id reused with a different payload", 409)
            return response
        db.execute("INSERT INTO requests VALUES (?,?,?,?)", (uid, req, session, digest))
        persisted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for i, msg in enumerate(payload["messages"]):
            parts = content_parts(msg["content"])
            mid = "mem_" + hashlib.sha256(canonical([uid, session, req, i]).encode()).hexdigest()
            created = (datetime.fromtimestamp(msg["timestamp"] / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
                       if "timestamp" in msg else persisted_at)
            db.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?)",
                       (mid, uid, session, req, i, msg["role"], canonical(parts), created))
            for ordinal, part in enumerate(parts):
                if part["type"] == "image_url":
                    raw, mime = image_bytes(part["image_url"]["url"])
                    db.execute("INSERT INTO evidence VALUES (?,?,?,?,?)",
                               (mid, ordinal, hashlib.sha256(raw).hexdigest(), mime, raw))
        return response

    def search(self, payload):
        query = validate_search(payload)
        qt, qi = tokens(text_of(query)), image_hashes(query)
        with self.connect() as db:
            rows = db.execute("SELECT id,content,created_at FROM memories WHERE user_id=? ORDER BY id",
                              (payload["user_id"],)).fetchall()
        candidates, scores = [], []
        for mid, raw, created in rows:
            parts = json.loads(raw)
            mt, mi = tokens(text_of(parts)), image_hashes(parts)
            lexical = len(qt & mt) / math.sqrt(len(qt) * len(mt)) if qt and mt else 0.0
            exact_image = float(bool(qi & mi))
            candidates.append(dict(id=mid, content=parts, created_at=created))
            scores.append(lexical + exact_image)
        # Options are intentionally unused by this baseline and are never persisted.
        ranked_scores = self.reranker.score(query, candidates, scores)
        if len(ranked_scores) != len(candidates) or any(type(s) not in (int, float) or not math.isfinite(s) for s in ranked_scores):
            raise RuntimeError("reranker returned invalid scores")
        ranked = sorted(zip(candidates, ranked_scores), key=lambda pair: (-pair[1], pair[0]["id"]))
        result, used = [], 0
        for candidate, score in ranked:
            if len(result) >= payload["top_k"]:
                break
            size = media_size(candidate["content"])
            if used + size > MEDIA_LIMIT:
                # Clear capacity error, no silent evidence deletion or truncation.
                raise ContractError("ranked evidence exceeds 30 MiB Search response; reduce top_k", 422)
            used += size
            result.append(dict(candidate, score=score))
        return {"data": result}

    def purge_user(self, user_id):
        """Explicit lifecycle cleanup, not exposed as a public HTTP endpoint."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            db.execute("DELETE FROM requests WHERE user_id=?", (user_id,))
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='add_operations'").fetchone():
                db.execute('DELETE FROM add_operations WHERE user_id=?',(user_id,))
        with self.connect() as db:
            db.execute("VACUUM")
