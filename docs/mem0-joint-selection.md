# Mem0 raw storage with joint evidence selection

The previous visual candidate made many sequential hosted reranker requests per
Search. A transport failure prevented a complete evaluation. This candidate
tests a different retrieval architecture: one bounded `gpt-4o-mini` selection
request per Search, while Add retains Mem0 2.2.1 `infer=False` original storage
and `text-embedding-v4` (256 dimensions). It does not fix the historical GPT
transport failures and has no new official quality result yet.

Native retrieval, BM25 and exact image identity produce a pool of at most 32
sources: up to 16 existing ranked sources and 16 image sources, filled from the
existing order. A single joint view includes a 1,500 UTF-8 byte prefix per source,
source timestamps and local session group numbers. The query is limited to
4,000 UTF-8 bytes. At most eight image previews, including the query image, use
JPEG at 512 pixels and quality 85, each at most 256 KiB. Missing previews and
truncated text are explicitly labelled. The request is capped at 3 MiB.

The model returns only up to eight unique known candidate indices. Exact image
matches remain first, followed by selected sources and adjacent messages from
the same Add. Returned evidence contains complete original text and original
image bytes, never model-generated answers, facts or resized preview bytes.
An empty valid selection can yield no evidence; it is not calibrated abstention.
No response failure becomes an empty successful result or a hidden fallback.

The response must be a complete usage-bearing stream with a stop finish, stable
response ID and model, and the exact observed model
`gpt-4o-mini-2024-07-18`. Invalid indices, duplicate JSON keys, partial streams
and provider errors fail closed. POST requests are not retried.

This mode retains the original money ledger, base receipt journal and
`native-mem0-raw-v4` scope, including all preserved failures and quarantines.
It neither invokes nor unlocks the blocked hosted reranker journal. Existing
limits, uncertain-call reservations and replay protections remain intact.
Activation requires `TRACEMEM_MEM0_JOINT_ENABLED=approved-native-mem0-joint`
and `agent_memory.mem0_app:create_joint_app()`.

Offline checks use synthetic evidence, synthetic provider responses and the
real pinned Mem0 SDK with no network. Independent legal-input validation and a
new fixed release are required before an official evaluation. Official
protected sources, questions and answers are not development inputs. Fewer
requests and joint context are hypotheses, not demonstrated reliability or
quality gains.
