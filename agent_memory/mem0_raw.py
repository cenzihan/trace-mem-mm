"""Explicit Mem0 infer=False ablation: original evidence, no generated facts.

Text embeddings use the same academic adapter and journal. Images are archived
unchanged and match by identity only; this mode has no image semantic model.
"""
import hashlib
import json
import math

from .mem0_bridge import native_scope
from .mem0_evidence import Mem0EvidenceSystem, pack_mixed
from .mem0_transport import Mem0ServiceTransport
from .protocol import ContractError, canonical, image_hashes, text_of, content_parts, validate_search
from .retrieval import bm25, ranked, rrf

VERSION = 'tracemem-mm-mem0-raw-v4.0.0'


class Mem0RawTransport(Mem0ServiceTransport):
    # One durable scope for this ablation family, including independent pilots.
    # It shares all money, receipt and permanent-quarantine history.
    receipt_scope = 'native-mem0-raw-v4'

    def __call__(self, endpoint, payload):
        if payload.get('model') != 'text-embedding-v4':
            raise ValueError('raw Mem0 mode forbids LLM and caption dispatch')
        return super().__call__(endpoint, payload)


class Mem0RawSystem(Mem0EvidenceSystem):
    def capabilities(self):
        return dict(super().capabilities(), version=VERSION, mode='native-mem0-infer-false',
                    models=['text-embedding-v4'], llm_calls_enabled=False,
                    vision='exact-image-identity-only', caption_preview=None,
                    search_evidence='complete-original-parts-only',
                    retrieval='native-mem0-raw+BM25+exact-image+same-add-context',
                    native_infer=False, generated_facts=False)

    def _describe(self, parts):
        # Defense in depth: inherited helpers cannot introduce a caption call.
        return text_of(parts)

    def _order_sources(self, payload, parts, rows, order, identities):
        """Explicit candidate extension point; raw v4 ordering is unchanged."""
        return order

    def _write_native(self, payload):
        descriptions = []
        for ordinal, message in enumerate(payload['messages']):
            text = text_of(content_parts(message['content']))
            descriptions.append(text)
            if not text.strip():
                # Image-only source is still committed atomically to the original
                # archive/identity index. No fabricated text enters the vector DB.
                continue
            mid = 'mem_' + hashlib.sha256(canonical([
                payload['user_id'], payload['session_id'], payload['request_id'], ordinal
            ]).encode()).hexdigest()
            result = self.native.add([dict(role=message['role'], content=text)],
                                     user_id=native_scope(payload['user_id']), infer=False,
                                     metadata={'tracemem_source_id': mid})
            self.ready()
            rows = result.get('results') if isinstance(result, dict) else None
            if (not isinstance(rows, list) or len(rows) != 1 or
                    rows[0].get('event') != 'ADD' or rows[0].get('memory') != text or
                    not isinstance(rows[0].get('id'), str) or not rows[0]['id']):
                raise ValueError('native raw write did not acknowledge the complete source')
        return descriptions

    def search(self, payload):
        parts = validate_search(payload)
        uid, query = payload['user_id'], text_of(parts)
        with self.lock:
            self.ready()
            with self.connect() as db:
                if db.execute("SELECT 1 FROM add_operations WHERE user_id=? AND state!='complete'", (uid,)).fetchone():
                    raise ContractError('user has unresolved Add', 503)
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
            if not rows:
                return {'data': []}
            by_id = {row[0]: row for row in rows}
            native_ids = []
            if query.strip() and any(row[5].strip() for row in rows):
                result = self.native.search(query, filters={'user_id': native_scope(uid)},
                                            top_k=100, rerank=False)
                self.ready()  # SDK may swallow a failed optional entity embedding.
                for value in result['results']:
                    score = value.get('score')
                    mid = value.get('metadata', {}).get('tracemem_source_id')
                    if (value.get('user_id') != native_scope(uid) or mid not in by_id or
                            value.get('memory') != by_id[mid][5] or
                            type(score) not in (float, int) or not math.isfinite(score)):
                        raise ContractError('native result/source mapping mismatch', 503)
                    if mid not in native_ids:
                        native_ids.append(mid)
            lexical = bm25(query, {r[0]: r[5] for r in rows})
            lexical_ids = ranked({mid: score for mid, score in lexical.items() if score > 0})
            fused = ranked(rrf([native_ids, lexical_ids]))
            order = list(dict.fromkeys([*sorted(identities), *fused]))
            order = self._order_sources(payload, parts, rows, order, identities)
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
                with self.connect() as db:
                    for mid in expanded:
                        value = db.execute('''SELECT content,created_at FROM memories
                            WHERE user_id=? AND id=?''', (uid, mid)).fetchone()
                        if value is None:
                            raise ContractError('source disappeared during search', 503)
                        yield dict(id=mid, content=json.loads(value[0]), created_at=value[1])
            return {'data': pack_mixed(originals(), [], min(payload['top_k'], 100))}
