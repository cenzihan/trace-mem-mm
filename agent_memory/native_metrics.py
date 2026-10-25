"""Harness-only evidence metrics; no service import or benchmark loading/training."""


def retrieval_metrics(retrieved_occurrence_ids,gold_occurrence_ids,k):
    """Score canonical source occurrences, never the number of generated cards.

    Fractional recall = |unique(top-k) intersect GT| / |GT|.
    Any-hit and all-evidence coverage are separately named. No-GT questions need
    a separate abstention/answer metric and receive no fabricated recall score.
    """
    if type(k) is not int or k<1:raise ValueError('positive integer k required')
    for sequence in (retrieved_occurrence_ids,gold_occurrence_ids):
        if not isinstance(sequence,(list,tuple)) or any(not isinstance(item,str) or not item for item in sequence):
            raise ValueError('canonical occurrence ID sequences required')
    retrieved=set(retrieved_occurrence_ids[:k]);gold=set(gold_occurrence_ids);hits=len(retrieved & gold)
    return dict(k=k,gold_evidence_count=len(gold),retrieved_unique_count=len(retrieved),hits=hits,
                fractional_recall=hits/len(gold) if gold else None,
                any_hit=bool(hits) if gold else None,
                all_evidence_coverage=gold<=retrieved if gold else None)
