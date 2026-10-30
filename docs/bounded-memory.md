# Bounded temporary memory, v4.3.13-rc1

Large multimodal Add bodies previously built a whole canonical JSON string and
then a whole UTF-8 copy three times along the ingestion path. A non-BMP character
makes the intermediate Python string use four bytes per character, including its
ASCII base64 images. Incremental hashing emits the exact same bytes without
joining the entire request. Request fingerprints, duplicate recognition, original
stored evidence, model selection and retrieval behavior remain unchanged.

Readiness previously materialized every receipt as a Python dictionary on every
health probe and model call. It now validates rows from a streaming SQLite cursor,
closing the cursor/connection even when a late invalid row raises. All receipt,
recovery and accounting checks remain in place. This still scans history in O(N)
time and holds a read cursor during validation; it does not claim constant-time
health checks. Parallel transport tests exercise readiness during owned calls.

Local synthetic CPython 3.11 tracemalloc measurements (not total process RSS):
- 32 MiB ASCII image strings plus non-BMP text: whole-request hash peak
  268,436,315 bytes; incremental hash 12,587,006 bytes; digest identical.
- 25,000 synthetic receipts: old materialization alone 28,511,840 bytes;
  new full readiness 10,239 bytes. Timing is not compared: workloads differ.

The pinned Qdrant local implementation also retains payloads and dense vectors in
RAM even with persistent storage. This patch does not migrate that index, reduce
stored memories, change retrieval, or prove the complete Full workload fits into
900 MiB. Image decoding, SQLite persistence and allocator retention remain other
contributors to peak/RSS. Long-run production improvement requires observation
after an explicitly coordinated deployment; no hot injection into a running worker.
