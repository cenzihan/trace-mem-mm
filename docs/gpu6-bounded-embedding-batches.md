# GPU6 bounded embedding preparation

The GPU6 rc4 profile groups up to ten messages from one Add into embedding
preparation batches capped at 80,000 UTF-8 bytes. A larger individual message
still follows the previous chunking path. Other profiles retain batch size one.
No text is shortened or shared across users or Add requests. The existing
text-embedding-v4 adapter keeps its 8,000-byte chunks, ten-input provider limit,
ordered response validation, weighted per-message pooling and 256-dimensional
normalization. This changes provider request grouping, not the model or inputs.

The original receipt journal and ledger account for each actual batch request;
historical calls and unknown records remain intact. All embeddings must finish
before the existing serialized atomic memory commit. A failed batch does not
commit partial memory, and completed Add requests remain idempotent.

Synthetic tests compare ordered vectors with duplicates, Unicode, long inputs,
request-local error flags, native user isolation, duplicate requests and failure
after a successful first batch. These tests establish implementation behavior,
not independent official compatibility, score quality or sustained Full speed.
Deployment measurements must distinguish successful Add completions from
provider calls and exclude transition errors from steady-state throughput.
