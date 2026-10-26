"""Offline full-corpus dense and flat/dense RRF over a published source cache.

Queries and query vectors remain transient. This module never encodes text,
loads credentials, dispatches providers, or selects questions for evaluation.
"""
import heapq
import math

from .atm_n0 import NativeBM25


def _unit(vector, dimensions):
    if (not isinstance(vector, (list, tuple)) or len(vector) != dimensions
            or any(type(value) not in (int, float) for value in vector)):
        raise ValueError('finite query vector with the published dimensions required')
    try:
        values = tuple(float(value) for value in vector)
        norm = math.hypot(*values)
    except (OverflowError, ValueError):
        raise ValueError('query vector numerical overflow') from None
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('finite nonzero vector norm required')
    return tuple(value / norm for value in values)


class SharedDenseIndex:
    """All chunks are searched; native parents receive their best chunk cosine.

    Both methods share exactly the same completed source vectors. Hybrid uses
    top100 native dense and flat BM25 lanes with equal RRF60, then returns at
    most100 native sources. This is a fixed candidate, not a tuned improvement.
    """
    def __init__(self, cache):
        snapshot = cache.read_published()  # Validates complete manifest and hashes.
        self.scope = dict(snapshot['scope'])
        self.namespace = self.scope['namespace']
        self.dimensions = self.scope['dimensions']
        chunks = {row['key']: row for row in snapshot['chunks']}
        self.source_chunks = {identity: tuple(keys) for identity, keys in snapshot['source_chunks'].items()}
        self.vectors = {key: _unit(row['vector'], self.dimensions) for key, row in chunks.items()}
        self.documents = {identity: ''.join(chunks[key]['text'] for key in keys)
                          for identity, keys in self.source_chunks.items()}
        self.sparse = NativeBM25(self.documents)

    def _validate(self, namespace, top_k):
        if namespace != self.namespace:
            raise ValueError('query namespace does not match published sources')
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('native evidence limit must be within 1..100')

    def dense(self, query_vector, *, namespace, top_k=100):
        self._validate(namespace, top_k)
        query = _unit(query_vector, self.dimensions)
        chunk_scores = {key: max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(query, vector))))
                        for key, vector in self.vectors.items()}
        scores = {identity: max(chunk_scores[key] for key in keys)
                  for identity, keys in self.source_chunks.items()}
        order = heapq.nsmallest(top_k, scores, key=lambda identity: (-scores[identity], identity))
        return dict(ranked_ids=order, scores=[scores[identity] for identity in order],
                    source_count=len(self.source_chunks), method='shared_source_dense_max_chunk')

    def hybrid(self, question, query_vector, *, namespace, top_k=100):
        self._validate(namespace, top_k)
        if not isinstance(question, str) or not question.strip():
            raise ValueError('nonempty transient original query required')
        dense = self.dense(query_vector, namespace=namespace, top_k=100)
        sparse = self.sparse.search(question, 100)
        lanes = dict(flat=[identity for identity, _ in sparse], dense=dense['ranked_ids'])
        scores = {}
        for lane in lanes.values():
            for rank, identity in enumerate(lane, 1):
                scores[identity] = scores.get(identity, 0.0) + 1.0 / (60 + rank)
        order = heapq.nsmallest(top_k, scores, key=lambda identity: (-scores[identity], identity))
        return dict(ranked_ids=order, scores=[scores[identity] for identity in order], lanes=lanes,
                    source_count=len(self.source_chunks), method='shared_source_flat_dense_rrf60',
                    query_persisted=False)
