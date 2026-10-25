"""Original deterministic sparse retrieval and rank fusion; no fitted benchmark data."""
import math
import re
from collections import Counter


def tokenize(text):
    text = text.casefold()
    # Whole Latin words + individual CJK characters and adjacent CJK bigrams.
    result = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text)
    for run in re.findall(r"[\u3400-\u9fff]{2,}", text):
        result.extend(run[i:i+2] for i in range(len(run)-1))
    return result


def bm25(query, documents, k1=1.2, b=0.75):
    if not documents:
        return {}
    counts = {key: Counter(tokenize(text)) for key, text in documents.items()}
    lengths = {key: sum(counter.values()) for key, counter in counts.items()}
    average = sum(lengths.values()) / max(1, len(lengths)) or 1
    df = Counter(token for counter in counts.values() for token in counter)
    terms = set(tokenize(query))
    result = {}
    for key, counter in counts.items():
        score = 0.0
        for term in terms:
            frequency = counter[term]
            if frequency:
                idf = math.log(1 + (len(documents) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * frequency * (k1 + 1) / (frequency + k1 * (1-b+b*lengths[key]/average))
        result[key] = score
    return result


def ranked(scores):
    return sorted(scores, key=lambda key: (-scores[key], key))


def rrf(rankings, weights=None, constant=60):
    weights = weights or [1.0] * len(rankings)
    result = Counter()
    for ranking, weight in zip(rankings, weights):
        for position, key in enumerate(ranking, 1):
            result[key] += weight / (constant + position)
    return dict(result)


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimension mismatch")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("non-finite embedding")
    ln = math.sqrt(sum(v*v for v in left)); rn = math.sqrt(sum(v*v for v in right))
    return sum(a*b for a,b in zip(left,right))/(ln*rn) if ln and rn else 0.0
