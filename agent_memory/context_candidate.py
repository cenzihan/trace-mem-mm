"""Opt-in retrieval candidate; never selected by the deployed service factory.

Ranks original evidence using dense, lexical and exact-image channels. Adjacent
messages are context only, not independent relevance judgments. No evaluation
questions, learned thresholds, new models or additional provider calls are used.
"""
import json
import math

from .bailian_app import BailianServingSystem
from .protocol import ContractError, MEDIA_LIMIT, image_hashes, media_size, text_of, validate_search
from .retrieval import bm25, cosine, ranked, rrf

VERSION = 'tracemem-mm-context-v2.1.0-rc1'


def pack_originals(order, top_k, original, *, limit=MEDIA_LIMIT):
    """Skip a whole over-budget record and continue to smaller ranked records."""
    selected, used, skipped = [], 0, 0
    for mid in order:
        if len(selected) == top_k:
            break
        memory = original(mid)
        size = media_size(memory['content'])
        if used + size > limit:
            skipped += 1
            continue
        used += size
        # This is a rank score, not a calibrated probability or provider score.
        selected.append(dict(memory, score=1.0 / (len(selected) + 1)))
    return selected, used, skipped


class EvidenceContextCandidate(BailianServingSystem):
    """Reuse the approved academic adapters and Add implementation unchanged.

    Context expansion is limited to immediate neighbours inside the same Add
    request. Cross-request chronology is deliberately not inferred from IDs or
    arrival time. All sessions inside the exact user scope remain searchable.
    """
    context_anchor_limit = 12

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    retrieval='dense+BM25+exact-image-RRF+rerank+adjacent-source-context',
                    deployment_profile='library-only-context-candidate',
                    release_stage='offline-validated-candidate-not-deployed',
                    image_query_mode='exact-source-image-identity-including-mixed-queries',
                    reranker_source_text_only=True,
                    context_anchor_limit=self.context_anchor_limit,
                    context_scope='same-user-same-session-same-add-request',
                    score_semantics='reciprocal-final-rank-not-confidence',
                    calibrated_abstention=False)

    def retrieve(self, payload):
        parts = validate_search(payload)
        query, uid = text_of(parts), payload['user_id']
        with self.connect() as db:
            db.execute('BEGIN')
            missing = db.execute('''SELECT count(*) FROM memories m LEFT JOIN cards c
                ON m.id=c.memory_id WHERE m.user_id=? AND
                (c.id IS NULL OR c.vector IS NULL OR c.vector_model IS NULL
                 OR c.vector_model!='text-embedding-v4')''', (uid,)).fetchone()[0]
            if missing:
                raise ContractError('dense index incomplete; no lexical fallback', 503)
            # No original image bodies are loaded into this ranking snapshot.
            rows = db.execute('''SELECT m.id,m.created_at,r.digest,c.indexed_text,c.vector,
                m.session_id,m.request_id,m.ordinal
                FROM memories m JOIN cards c ON m.id=c.memory_id
                JOIN requests r ON r.user_id=m.user_id AND r.request_id=m.request_id
                WHERE m.user_id=? ORDER BY m.id''', (uid,)).fetchall()
            identities = set()
            for digest in image_hashes(parts):
                identities.update(row[0] for row in db.execute('''SELECT DISTINCT e.memory_id
                    FROM evidence e JOIN memories m ON m.id=e.memory_id
                    WHERE m.user_id=? AND e.sha256=?''', (uid, digest)))
        if not rows:
            return {'data': []}, {'candidate_count': 0, 'model_calls_required': False}
        rowmap = {row[0]: row for row in rows}
        identities.intersection_update(rowmap)
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
            lexical = bm25(query, {row[0]: row[3] for row in rows})
            fused = rrf([ranked(dense), ranked({m: s for m, s in lexical.items() if s > 0}),
                         sorted(identities)])
            # Preserve exact image matches in a bounded candidate pool even when
            # text embeddings miss them. A match is not itself a final answer.
            order = list(dict.fromkeys([*sorted(identities), *ranked(fused)]))[:self.candidate_limit]
        else:
            dense = {mid: 1.0 for mid in identities}
            order = sorted(identities)[:self.candidate_limit]

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

        reranked = bool(order and query.strip())
        if reranked:
            candidates = [dict(original(mid), retrieval_text=rowmap[mid][3]) for mid in order]
            values = self.scoring_adapter.score(parts, candidates, [dense[mid] for mid in order])
            if (not isinstance(values, (list, tuple)) or len(values) != len(order)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError('one finite candidate score per source required')
            order = ranked(dict(zip(order, values)))
            if identities:
                # The text-only hosted adapter cannot see the query image. Keep
                # exact identity as a separate final ranking signal, not prose
                # injected into either the question or the original evidence.
                order = ranked(rrf([order, sorted(identities & set(order))]))

        # Structural context, never another user or a fabricated joined message.
        positions = {(row[5], row[6], row[7]): row[0] for row in rows}
        expanded, seen = [], set()
        for rank, mid in enumerate(order):
            row = rowmap[mid]
            group = [mid]
            if rank < self.context_anchor_limit:
                group += [positions.get((row[5], row[6], row[7] + offset)) for offset in (-1, 1)]
            for candidate in group:
                if candidate is not None and candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
        selected, used, skipped = pack_originals(expanded, min(payload['top_k'], self.candidate_limit), original)
        return {'data': selected}, dict(source_count=len(rows), candidate_count=len(order),
            exact_image_candidates=len(identities & set(order)),
            context_candidates=len(set(expanded) - set(order)),
            response_media_bytes=used, media_records_skipped=skipped,
            rerank_executed=reranked, original_parts_preserved=True,
            query_persisted=False, calibrated_abstention=False)
