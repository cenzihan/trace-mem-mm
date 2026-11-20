# Final GPU6 service profile

The final profile is `tracemem-mm-gpu6-v4.3.15-rc7`, implemented by
`agent_memory.gpu6_app:create_gpu6_app`. It combines the raw Mem0 index,
choice-aware multimodal evidence selection, disk-backed vectors and bounded
parallel preparation. The withdrawn rc8 readiness implementation is excluded.

## Execution and storage

- One process owns the receipt journal and production state. The final profile
  admits 16 active requests with a bounded admission queue and a 120-second
  queue wait. A validated control file supports 2, 4, 8 or 16 active requests.
- Sixteen provider clients retain independent attempt receipts; timed-out
  workers keep their lanes until the underlying network operation finishes.
- Text embeddings use `text-embedding-v4`; evidence selection uses
  `gpt-4o-mini`. No hosted reranker is used. Bounded provider retries retain
  every attempt and its accounting reservation.
- Embedding preparation batches at most 10 messages within one Add. Immutable
  media results are reused only within that request; native writes remain
  serialized. Original text and image content is preserved.
- SQLite vectors are scanned in bounded batches. Readiness uses partial indexes
  for exceptional receipts and reuses one read-only ledger connection per
  locked validation, without caching validation results across calls.
- Search accepts complete query/choice text without the earlier arbitrary
  query-byte and choice-count limits. Request-body and evidence-view bounds,
  schema checks and provider response validation remain in place.

## Deployment boundaries

This is a retained deployment profile, not a portable default. It requires
explicit migration/accounting activation, verified disk migration, the original
state and receipt directories, token files, a concurrency control file and an
unexpired service lease. OpenAI traffic uses its deployment-specific local
proxy tunnel with end-to-end TLS verification. `deploy/docker/Dockerfile.mem0` builds the
native dependencies; its default command starts the earlier base factory, so
using the final profile requires an explicit entrypoint override.

The profile records historical-rate estimates without enforcing the original
aggregate cap or price-age shutdown. This policy is explicit and specific to
this deployment; estimates are not invoices. Other factories retain their own
checks. Unknown outcomes and frozen recovery evidence remain preserved.

Any state migration requires a stopped sole writer, exact backups and verified
source/vector/payload preservation. Do not run two writers or restore an old
snapshot over newer writes. Credentials, deployment files and runtime evidence
are kept outside the public repository. Source publication does not alter the
running service or extend its lease.
