# Hosted reranker candidate

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

The distinct [v2 health endpoint](https://45.192.104.173/v2/health) is deployed.
Authenticated endpoints are `/v2/add` and `/v2/search`; the old routes still serve
the submitted v1. A fresh real Mem-Gallery image-containing Add and two Searches
(one after container restart) passed. The run made2 real GPT calls,3 embedding
calls and2 hosted rerank calls. Original image/text hashes, persistence, ranked
source identities, idempotence, conflict rejection, authorization and user isolation
passed. This is development smoke, not official AML Smoke or admission approval.

Version binding still requires the issued AML Eval Key. Bind these distinct
endpoints to that key; do not resubmit the application or silently change v1.
The new service lease and content-deletion deadline retain the original bounds.
