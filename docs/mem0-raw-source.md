# Native Mem0 raw-source ablation

`tracemem-mm-mem0-raw-v4.0.0` uses the pinned Mem0 2.2.1 SDK with
`infer=False` and its local Qdrant backend. It stores original message text as
memory instead of generating facts. Text embedding remains `text-embedding-v4`
at 256 dimensions. LLM/caption requests are explicitly rejected before dispatch.
This is a separate ablation, not the default Mem0 fact-extraction pipeline.

The hypothesis is that removing synchronous GPT extraction from Add can make
original evidence available without that network dependency. This does not
establish that the underlying GPT connection problem is fixed. Embedding failures
still stop processing; unknown calls and their full budget holds are preserved.

Each text memory carries a stable source mapping. Search combines native Mem0
results, BM25 over original text, exact image identity and immediate neighbours
within the same Add. It searches all sessions of the exact user. Only complete
original ordered content parts are returned; generated facts and query text are
never inserted as sources. Whole records exceeding the remaining 30 MiB media
budget are skipped. Rank scores are not calibrated confidence values.

Image-only messages are archived and searchable by image identity, with no
fabricated embedding text. This version cannot recognize semantically similar
images or describe previously unseen query images. It has no learned abstention
policy. Quality and reliability require independent validation and evaluation;
offline tests do not establish an official score.

An Add is acknowledged only after all its native writes and the atomic original
archive/index commit complete. A partial native write marks the Add uncertain;
the user remains blocked rather than exposing a partial batch or replaying it.
Restarts preserve idempotence, the shared provider journal, quarantines and the
original budget. Prior deployment states and counters are not reset.

Activation requires `TRACEMEM_MEM0_RAW_ENABLED=approved-native-mem0-raw` plus
the existing explicit native deployment settings, with factory
`agent_memory.mem0_app:create_raw_app()`. The durable receipt scope is
`native-mem0-raw-v4`. It uses the existing validated embedding transport, not a
new provider or alternate region.

The [official API guide](https://agentmemoryleaderboard.ai/api-guide) permits
internal architecture choices and requires synchronous Add results to be
persisted and immediately searchable. The academic model restriction is retained:
this ablation dispatches only the designated embedding model and has no LLM
component. The website's Full checklist also describes GPT use during Add; this
no-LLM ablation must be disclosed explicitly, not represented as a GPT-based
entry or as approved for Full.

Mem0 attribution and pinned dependency licenses remain in
[the baseline documentation](native-mem0-baseline.md). Tests exercise the actual
SDK with synthetic vectors, original synthetic images and no external traffic.
