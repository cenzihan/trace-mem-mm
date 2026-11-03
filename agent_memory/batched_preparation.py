"""Bounded preparation within one Add; never mix owners or store raw inputs."""


def prepare_embeddings(adapter, texts, batch_size=1, byte_limit=80000):
    if type(batch_size) is not int or not 1 <= batch_size <= 10:
        raise ValueError('preparation batches must contain one to ten messages')
    pending = []
    size = 0

    def flush():
        if len(pending) == 1:
            vectors = [adapter.embed(pending[0], 'add')]
        else:
            # Existing adapter retains per-message UTF-8 chunking, ordered
            # response validation, weighted pooling and 256-d normalization.
            vectors = adapter.embed_batch(pending, 'add')
        if len(vectors) != len(pending):
            raise ValueError('embedding preparation result count mismatch')
        return list(zip(pending, vectors))

    for text in texts:
        if not text.strip():
            continue
        length = len(text.encode('utf-8'))
        if pending and (len(pending) >= batch_size or size + length > byte_limit):
            yield from flush()
            pending = []
            size = 0
        pending.append(text)
        size += length
        # Oversized single messages use exactly the old chunked path and do
        # not accumulate with other messages. This is not input truncation.
        if len(pending) >= batch_size or size >= byte_limit:
            yield from flush()
            pending = []
            size = 0
    if pending:
        yield from flush()
