"""Native facts plus original multimodal evidence; no extra provider requests."""
from itertools import zip_longest
import json

from .mem0_bridge import Mem0System
from .protocol import ContractError, MEDIA_LIMIT, content_parts, image_hashes, media_size, validate_search
from .retrieval import bm25, ranked

VERSION = 'tracemem-mm-mem0-evidence-v3.1.1'


def pack_mixed(originals, facts, top_k, *, limit=MEDIA_LIMIT):
    """Alternate two evidence channels, skipping only whole over-budget records."""
    selected, used = [], 0
    for pair in zip_longest(originals, facts):
        for row in pair:
            if row is None:
                continue
            if len(selected) >= top_k:
                return selected
            size = media_size(content_parts(row['content']))
            if used + size > limit:
                continue
            selected.append(dict(row, score=1.0 / (len(selected) + 1)))
            used += size
    return selected


class Mem0EvidenceSystem(Mem0System):
    context_anchor_limit = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS mem0_source_index (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                indexed_text TEXT NOT NULL)''')

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    search_evidence='original-parts-and-labelled-derived-facts',
                    retrieval='native-mem0+BM25-source-caption+exact-image+same-add-context',
                    source_query_persisted=False, additional_source_model_calls=0,
                    score_semantics='reciprocal-final-rank-not-confidence',
                    calibrated_abstention=False)

    def _persist_descriptions(self, db, payload, descriptions):
        rows = db.execute('''SELECT id,ordinal FROM memories
            WHERE user_id=? AND request_id=? ORDER BY ordinal''',
            (payload['user_id'], payload['request_id'])).fetchall()
        if len(rows) != len(descriptions):
            raise ValueError('one description per original message required')
        db.executemany('INSERT INTO mem0_source_index VALUES (?,?)',
                       [(mid, descriptions[ordinal]) for mid, ordinal in rows])

    def search(self, payload):
        parts = validate_search(payload)
        uid = payload['user_id']
        with self.lock:
            # Validate the archive before any paid query call. A missing index is
            # not permission to silently backfill or process earlier failed Adds.
            with self.connect() as db:
                rows = db.execute('''SELECT m.id,m.created_at,m.session_id,m.request_id,
                    m.ordinal,i.indexed_text FROM memories m
                    LEFT JOIN mem0_source_index i ON i.memory_id=m.id
                    WHERE m.user_id=? ORDER BY m.id''', (uid,)).fetchall()
                if any(row[5] is None for row in rows):
                    raise ContractError('original source index incomplete', 503)
                identities = set()
                for digest in image_hashes(parts):
                    identities.update(r[0] for r in db.execute('''SELECT e.memory_id
                        FROM evidence e JOIN memories m ON m.id=e.memory_id
                        WHERE m.user_id=? AND e.sha256=?''', (uid, digest)))
            query, facts = self._native_results(payload, parts)
            if not rows:
                return {'data': []}
            scores = bm25(query, {row[0]: row[5] for row in rows})
            order = list(dict.fromkeys([*sorted(identities),
                         *ranked({mid: score for mid, score in scores.items() if score > 0})]))
            by_id = {row[0]: row for row in rows}
            positions = {(r[2], r[3], r[4]): r[0] for r in rows}
            expanded, seen = [], set()
            for rank, mid in enumerate(order):
                row = by_id[mid]
                group = [mid]
                if rank < self.context_anchor_limit:
                    group += [positions.get((row[2], row[3], row[4]+offset)) for offset in (-1, 1)]
                for candidate in group:
                    if candidate is not None and candidate not in seen:
                        seen.add(candidate)
                        expanded.append(candidate)

            def originals():
                # Fetch only the originals needed to fill the bounded response.
                # Keep each complete ordered message, including original images.
                with self.connect() as db:
                    for mid in expanded:
                        value = db.execute('''SELECT content,created_at FROM memories
                            WHERE user_id=? AND id=?''', (uid, mid)).fetchone()
                        if value is None:
                            raise ContractError('source disappeared during search', 503)
                        yield dict(id=mid, content=json.loads(value[0]), created_at=value[1])
            return {'data': pack_mixed(originals(), facts, min(payload['top_k'], 100))}
