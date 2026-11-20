# GPU6 request-local media reuse

The rc5 GPU6 Add entry point scopes a bounded cache around its existing call
chain. Successful inline-image decoding and JPEG preview generation are reused
only for exactly equal immutable Data URI strings in that Add. Returned preview
parts are fresh dictionaries; cached bytes and preview strings are immutable.
Other profiles and Search do not enable this scope.

The cache retains at most 32 MiB of decoded bytes and preview strings and at most
128 entries per request. Larger results still use the original validating path.
References to URI strings already held by the request do not copy those strings.
All retained entries are cleared on normal return or exception. Nested scopes
and concurrent threads do not share a cache. There is no global or disk cache.

Every message/part field check, total media-size accounting, source digest and
original persistence still runs. Identical repeated images count repeatedly
toward the 30 MiB request limit. Invalid results are never cached. A changed URI
is validated independently. This does not modify originals, embedding inputs,
models, receipt fingerprints, ledger history or source selection.

Validation covers original bytes and identical preview output, mutated caller
copies, invalid Base64/MIME/pixels, duplicate-image size limits, bounded retention,
concurrent/nested scopes and exceptional cleanup. An isolated native benchmark
compares persisted originals, evidence and vectors with reuse disabled/enabled.
Synthetic speedups are not guarantees of official Full throughput or quality.
