# Hosted reranker candidate

Current patch release: `tracemem-mm-bailian-v2.0.3`. The final format audit adds
strict GPT source-card JSON Schema,20-message admission and empty-options
compatibility. See the [itemized audit](official-format-audit.md) for wire checks,
capacity limits and remaining verification boundaries. The first official Smoke
exposed a well-formed GPT response whose optional evidence_quotes field was an
object instead of an array. The service stopped on that schema mismatch. This
patch discards invalid auxiliary containers instead of interpreting their values
as evidence. Model/prompt, generated summary and original source remain unchanged;
invalid substantive summaries, refusals, truncation and unknown calls still fail.
An explicit operator-only acknowledgement can retire that known HTTP200 failure
without any replay. It preserves the failed operation and original ledger rows,
appends hash/usage-linked proof and removes the private raw response copy. It does
not resume the failed Add or convert the failure into a successful observation.

The original concurrency patch in v2.0.1 is retained. Official external-track
settings require Add concurrency16–64 and Search concurrency16–256. Set both
to16. The service admits up to16 requests, serializes model/index work to
preserve receipt-first accounting, and waits at most30 seconds for the active
request. Full admission or queue expiry returns429 with Retry-After5 before
reading the body or starting model calls. Eighteen HTTP threads leave room for
health checks. This is bounded queuing, not16 simultaneous model calls or a
throughput guarantee. The deployment lease is checked again after queueing.
The [official guide](https://agentmemoryleaderboard.ai/docs) documents bounded
retry handling for429 and states that concurrency is an upper bound.

Registered patch endpoints are `/v2.1/add`, `/v2.1/search` and `/v2.1/health`.
The registry rejected a URL change after the first Smoke, then accepted a
separate `tracemem-mm-bailian-v2.0.2` version using the existing URLs. The next
patch keeps that registered address and is separately versioned as v2.0.3.
`/v2.2` was the previous patch alias; current deployment identity is reported
by health and must match the selected official version before submission.
The old v2.0.1 job is terminal failed and must not be resumed against new code.
Its source, tag, image and failure records remain preserved. The unused v2.0.0
binding is superseded and `/v2` returns410. The patch retains
the existing hosted state, receipt journals, original shared money ledger,
immutable512-call ceiling and deletion deadline. No counters or model identity
are reset. The earlier v1 remains separate. Six focused offline tests cover
16 simultaneous admissions, health/auth isolation, capacity overflow, queue
timeout, failure recovery and expiry while waiting. The real-data retrieval
results below describe the unchanged retrieval implementation in v2.0.0.

The user approved a separate hosted reranker version on2026-10-27 after recurring
private GPU tunnel interruptions. `tracemem-mm-bailian-v2.0.0` uses Bailian's
`qwen3-vl-rerank`; this is a different model identity from the submitted v1's
`Qwen/Qwen3-VL-Reranker-2B` snapshot. The hosted response does not prove a weight
revision. The v1 tag, API routes, frozen results and application are preserved.

The [AML guide](https://agentmemoryleaderboard.ai/api-guide) leaves rerankers
unrestricted in Academic. Embeddings remain text-embedding-v4 and all LLM work
remains gpt-4o-mini. Search returns original evidence, without generating answers.

Only the scorer changes. GPT still processes original images during Add, and
the hosted scorer receives the same source text and source-grounded summaries
as v1. Images are not newly sent to the hosted reranker. Query-image handling
retains the existing caption-text or exact-source-identity limitation.

## API, accounting and operational limits

- [DashScope rerank contract](https://help.aliyun.com/zh/model-studio/text-rerank-api):
  preserve each whole document, return all candidate scores, validate the index
  permutation and finite0..1 scores. No output document copies are requested.
- [Model page](https://help.aliyun.com/zh/model-studio/qwen3-vl-rerank): Beijing
  input price checked2026-10-27 is CNY0.5/million tokens; output is not billed.
  The published quota is600RPM/9millionTPM, not a guarantee for an individual account.
- At most100 candidate documents. Admission is conservatively limited to4000
  UTF-8 query bytes and8000 bytes/document. Split whole-document batches using
  repeated query/instruction bytes and a framing margin, within120000 per batch.
  Reserve the full120000-token maximum for each HTTP request, then record usage.
- The existing CNY500 server ledger is the only money budget. The new hosted
  journal has an immutable lifetime call ceiling; it does not replenish budget.
  Timeout/unknown/failure records block further requests and are never auto-replayed.
  Successful bodies are removed after validation; retain hashes and token counts.
- The hosted service has no private GPU dependency. Health checks validate local
  configuration, budget and journals without paid upstream probes. A200 health
  response therefore does not prove current provider reachability.
- `compose.bailian.yaml` runs a distinct loopback8769 service, with the original
  lease end, authentication and shared ledger. The512 hosted-request ceiling is
  a server guard, not permission to run Full or restart completed experiments.

## Validation status

The v2.0.2 patch passed646 offline checks and Docker CI. The first official
Smoke made11 real provider requests (9 GPT,2 embedding) before Add stopped;
it did not reach hosted reranking or produce an official score. The worker's
terminal410 followed intentional retirement of the failed deployment; the
underlying contract failure was invalid optional quote metadata. The new
version is bound but has not yet run its own official Smoke. Task-wide usage
is1/5; the platform's hourly gate still applies. No Full run is authorized.

All632 offline checks passed, including16 hosted transport tests; two subsequent
source-preservation and old-version-isolation checks also passed. The real
comparison completed on the previous74-question Mem-Gallery scene, preserving
exactly the same100 candidates per question and the original indexed text.
All79 hosted requests succeeded, with no new GPT or embedding calls. All7400
scores were frozen before the separate scorer accessed gold. Among60 questions
with evidence, API-message Recall@20 increased from90.52% to92.52%; Recall@100
remained99.23%. Five questions improved and two regressed. The other14 questions
have empty gold and are excluded from recall averages.

This is engineering regression on a previously examined scene, not an independent
benchmark, statistical superiority claim or official score. Rerank-only median
time was1.811s; this excludes query embedding and cannot be compared directly with
v1's complete Search latency. See the [aggregate report](../reports/bailian-v2-regression.json).
One report-serialization error after the first successful call was repaired using
saved evidence, with no model replay. All79 receipts are complete and minimized.

## Deployed native-data smoke

The hosted [patch health endpoint](https://45.192.104.173/v2.1/health) uses the
existing hosted state. The original flat routes still serve the submitted v1.
Before the admission patch, a fresh real Mem-Gallery image-containing Add and two Searches
(one after container restart) passed. The run made2 real GPT calls,3 embedding
calls and2 hosted rerank calls. Original image/text hashes, persistence, ranked
source identities, idempotence, conflict rejection, authorization and user isolation
passed. This is development smoke, not official AML Smoke or admission approval.

The original application is approved and its Eval Key is configured privately.
Bind patch endpoints as a distinct version; do not resubmit the application or
silently change v1. Official Smoke is separate from the development checks above.
The new service lease and content-deletion deadline retain the original bounds.
