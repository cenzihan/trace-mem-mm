"""Explicit dynamic Dense Top100 -> candidate rerank, preserving original evidence.

This is a new online candidate, not the frozen V7 benchmark implementation.
Models and budgeted transports must be supplied explicitly by the caller.
"""
import json
import math

from .protocol import ContractError, MEDIA_LIMIT, image_hashes, media_size, text_of, validate_search
from .retrieval import cosine, ranked
from .system import AgentMemorySystem, RetrievalConfig


class OnlineDenseRerankSystem(AgentMemorySystem):
    candidate_limit = 100
    reranker_model = 'Qwen/Qwen3-VL-Reranker-2B'

    def __init__(self, path, *, embedding, card_adapter, scoring_adapter):
        if (getattr(embedding, 'model', None) != 'text-embedding-v4'
                or getattr(embedding, 'dimensions', None) != 256
                or getattr(card_adapter, 'model', None) != 'gpt-4o-mini'
                or getattr(scoring_adapter, 'model', None) != self.reranker_model):
            raise ValueError('explicit fixed academic model adapters required')
        super().__init__(path, RetrievalConfig(cards=True, temporal=False, links=False,
                         image_identity=True, dense=True, rerank=True, candidate_limit=100),
                         embedding=embedding, card_adapter=card_adapter, scoring_adapter=scoring_adapter)

    def capabilities(self):
        return dict(super().capabilities(), retrieval='dense-top100+candidate-rerank',
                    deployment_profile='online_dense_rerank_candidate',
                    release_stage='candidate_pending_native_acceptance',
                    reranker=self.scoring_adapter.model, embedding_dimensions=256,
                    source_index='original-text+source-grounded-gpt-summary',
                    image_query_mode='caption-text-or-exact-source-image-identity',
                    candidate_limit=100, platform_answer_generated=False)

    def retrieve(self, payload):
        parts = validate_search(payload)
        query = text_of(parts)
        uid = payload['user_id']
        with self.connect() as db:
            db.execute('BEGIN')
            missing = db.execute('''SELECT count(*) FROM memories m LEFT JOIN cards c
                ON m.id=c.memory_id WHERE m.user_id=? AND
                (c.id IS NULL OR c.vector IS NULL OR c.vector_model IS NULL
                 OR c.vector_model!='text-embedding-v4')''', (uid,)).fetchone()[0]
            if missing:
                raise ContractError('dense index incomplete; no lexical fallback', 503)
            # Never load the entire user's original image bodies into a ranking snapshot.
            rows = db.execute('''SELECT m.id,m.created_at,r.digest,c.indexed_text,c.vector
                FROM memories m JOIN cards c ON m.id=c.memory_id
                JOIN requests r ON r.user_id=m.user_id AND r.request_id=m.request_id
                WHERE m.user_id=? ORDER BY m.id''', (uid,)).fetchall()
            identity_matches = []
            hashes = image_hashes(parts) if not query.strip() else []
            if hashes:
                for digest in hashes:
                    identity_matches.extend(row[0] for row in db.execute('''SELECT DISTINCT e.memory_id
                        FROM evidence e JOIN memories m ON m.id=e.memory_id
                        WHERE m.user_id=? AND e.sha256=?''', (uid, digest)))
        if not rows:
            return {'data': []}, {'candidate_count': 0, 'model_calls_required': False}
        rowmap = {row[0]: row for row in rows}
        if query.strip():
            try:
                vectors = {row[0]: json.loads(row[4]) for row in rows}
                for value in vectors.values():
                    if (not isinstance(value, list) or len(value) != 256
                            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
                        raise ValueError('invalid source vector')
                    cosine(value, value)
            except (TypeError, ValueError):
                raise ContractError('invalid dense source vectors', 503) from None
            vector = self.embedding.embed([query])[0]
            dense = {mid: cosine(vector, value) for mid, value in vectors.items()}
            order = ranked(dense)[:self.candidate_limit]
        else:
            # No invented query caption and no unsupported image embedding model.
            dense = {mid: 1.0 for mid in set(identity_matches) if mid in rowmap}
            order = ranked(dense)[:self.candidate_limit]

        originals = {}

        def original(mid):
            if mid not in originals:
                with self.connect() as db:
                    value = db.execute('''SELECT m.content,m.created_at,r.digest
                        FROM memories m JOIN requests r ON r.user_id=m.user_id AND r.request_id=m.request_id
                        WHERE m.user_id=? AND m.id=?''', (uid, mid)).fetchone()
                if value is None or value[1] != rowmap[mid][1] or value[2] != rowmap[mid][2]:
                    raise ContractError('source changed during retrieval', 503)
                originals[mid] = dict(id=mid, content=json.loads(value[0]), created_at=value[1])
            return originals[mid]

        scores = dense
        if order and query.strip():
            # Generated source descriptions are ranking metadata, never returned evidence.
            candidates = [dict(original(mid), retrieval_text=rowmap[mid][3]) for mid in order]
            values = self.scoring_adapter.score(parts, candidates, [dense[mid] for mid in order])
            if (not isinstance(values, (list, tuple)) or len(values) != len(order)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
                raise ValueError('one finite candidate score per source required')
            scores = dict(zip(order, values))
            order = ranked(scores)
        selected, used = [], 0
        for mid in order[:payload['top_k']]:
            memory = original(mid)
            size = media_size(memory['content'])
            if used + size > MEDIA_LIMIT:
                break
            used += size
            selected.append(dict(memory, score=scores[mid]))
        return {'data': selected}, dict(source_count=len(rows), candidate_count=len(order),
                                        dense_enabled=True, rerank_executed=bool(query.strip() and order),
                                        original_parts_preserved=True)
