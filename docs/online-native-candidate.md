# Online native-data candidate

Status: integration candidate under private real-data acceptance, not a formal
release or an official AML score. The deterministic Docker profile is unchanged.

The opt-in `agent_memory.online_app:create_app()` composes GPT-4o-mini source
cards, 256-dimensional text-embedding-v4 retrieval, and Dense Top100 followed by
the original Qwen/Qwen3-VL-Reranker-2B scoring model. The private scorer response
must identify revision `4bd860ac4f15ad1897a214615cccc700f8f71818` and weights SHA256
`466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`.
Hosted aliases are not considered interchangeable with these weights.
The [Bailian API reference](https://help.aliyun.com/zh/model-studio/text-rerank-api)
lists `qwen3-vl-rerank`, but the checked reference does not establish that it is
this exact 2B checkpoint. The user's requirement is exact model identity, so the
hosted alternative is excluded and no hosted reranker requests were made.

Original source text and ordered image parts are preserved as returned evidence.
GPT summaries are derived ranking metadata. Only generated text quotations found
verbatim in the original text are retained as quotations; other proposed quotes
are omitted and counted. The original provider receipt and summary are unchanged.
This fixes a local validation failure observed on the third native Add: seven
provider responses were already durable, but that Add had not committed. Recovery
used the exact saved response, without another request for that source.
That recovery completed once. A later acceptance attempt stopped during the
private scorer connectivity check, before dispatching any new model request.
The current continuation retains the same inputs, request IDs, grant and receipts.

Qwen scores source text plus the generated source summary; it does not receive
image bytes in this candidate. Source images are sent to GPT with low image
detail. Queries use original question text and dataset-provided captions where
present. Image-only queries support exact source-image identity. These limits
must accompany any resulting metrics; this is not full semantic image-query
support and is not the frozen V7 benchmark implementation.

## Accounting and failure behavior

The online transport requires an explicit activation flag, a hash-pinned,
expiring grant, and one existing parent-ledger reservation per allowed model
request. Its SQLite journal is a receipt outbox, not another budget ledger.
Receipts are saved before parsing, and unresolved requests stop further dispatch.
There are no automatic provider retries. Remote receipts must be reconciled with
the original parent budget; reserved slots are not evidence of actual requests.
The existing finite embedding-only smoke grant is not reused.

## Native acceptance scope

Acceptance uses the earlier complete Mem-Gallery architecture/art/culture scene
at revision `af912daba984e896e253016b7c7e334ef92c2a6f`: 14 sessions, 186 rounds,
372 messages, 74 original questions, and all distractors. All 36 referenced
original images passed repository content-hash verification. Of these, 34 retain
their original bytes in requests; two exceed the 10 MiB interface limit and use
explicit same-dimension JPEG quality85 variants after lossless conversion could
not meet the limit. The original files remain preserved. No image is cropped.

The acceptance checks durable Add, idempotence, conflicting retries, user scope,
container recreation, and original evidence hashes in Search responses. Rankings
must be frozen before gold scoring. This previously used scene is an engineering
regression, not independent unseen-scene evidence. Input files, media, questions,
model receipts, credentials, and budget databases remain private and are not
included in this repository. Official Smoke has not run.

Validation of the source change: 598 offline checks passed. Full native acceptance
was still running when this candidate checkpoint was prepared.
The pinned GPU scorer is currently reached through an authenticated private
tunnel. Its bounded temporary process still needs a persistent deployment before
an official endpoint can be offered; private connectivity alone is not production
acceptance.
