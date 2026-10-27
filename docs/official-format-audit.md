# Official interface audit — v2.0.4

Checked 2026-10-27 against the [AML API Guide](https://agentmemoryleaderboard.ai/api-guide)
and [error/quota documentation](https://agentmemoryleaderboard.ai/docs). This is a
format and lifecycle audit, not a claim that the official Smoke has passed.

The preceding v2.0.3 passed657 offline checks including11 wire-format regressions.
Its official run revealed a separate dispatch recovery defect: the health gate
accepted an audited historical failure while the journal dispatch gate still
blocked it. v2.0.4 applies the same proof validation to both gates and adds four
recovery regressions including a fresh dispatch after restart. A pristine prepared
operation with no attempt or receipt may be explicitly retired without replay;
failed/unknown calls and financial history remain intact.
No provider calls were made by these tests. Docker CI and the next official
Smoke remain separate deployment/availability checks.

| Check | Confirmation and regression evidence |
|---|---|
| Add envelope | Required request/user/session IDs and ordered messages; exact ID echo after commit. `test_contract`, `test_official_formats`. |
| Roles and time | User/assistant, optional integer Unix milliseconds, UTC source time retained; invalid input rejected before model work. |
| Plain and multimodal content | Nonempty text or ordered text/image parts; original part order and media survive retrieval/restart. |
| Images | Inline Base64 JPEG/PNG/WebP; MIME signature, 10 MiB per image and 30 MiB per Add checked. Remote image URLs rejected. `test_media_limits`. |
| Synchronous Add | No 202 or task polling; HTTP200 only after source and index persistence. |
| Retry identity | Same request/body returns the original result without models; changed body returns409; unresolved prior work is never replayed automatically. |
| Message count | New hosted profile accepts20 messages, covering AML's documented ordinary-text chunk boundary; legacy default remains16. |
| Search envelope | Query, user_id and positive integer top_k; optional top-level string options. Empty options now accepted as no choices. |
| Scope | Same user may search multiple source sessions; no cross-user retrieval. Search has no session filter. |
| Search response | JSON object with ordered data array; stable nonempty IDs, original content, finite scores, at most requested top_k. Empty results use data:[]. |
| Query privacy | Search queries/options are not persisted; generated summaries remain index metadata, never returned as original evidence. |
| Image-only query | Valid envelope and exact source-image matching; no invented caption. General unseen-image semantic retrieval remains a declared limitation. |
| Authentication | Registered Bearer auth; unauthorized Add/Search returns401 without provider dispatch; GET health is public. |
| HTTP and errors | JSON UTF-8, accurate Content-Length, no-store; malformed JSON400, invalid fields422, declared capacity413, overload429 with Retry-After5. |
| Concurrency | Sixteen admissions, one active model operation,30s bounded queue; lease rechecked after waiting. `test_service_queue`. |
| GPT request | gpt-4o-mini, two ordered messages, original images,512 output-token cap, n=1, store=false. New profile requests strict source-card JSON Schema. |
| GPT response | Correct model, one completed choice, valid usage and nonempty summary required. Refusals/truncation remain failures; malformed optional quotes are omitted without interpreting them as evidence. |
| Embedding request | Fixed text-embedding-v4,256 dimensions,float encoding, ordered batches of at most10 texts. |
| Embedding response | Original index permutation restored; exact vector dimension, finite nonzero values and usage checked. |
| Rerank request | Native DashScope endpoint; qwen3-vl-rerank with query/documents inside input, top_n/return_documents inside parameters. Text-string form is explicitly supported. |
| Rerank response | output.results must cover each original document once; indices and finite0..1 relevance scores checked. Native usage is audited; no observed weight revision is invented. |
| Failure accounting | Receipt before interpretation; original budget history preserved; failed/unknown calls block. The previous one-off known-response acknowledgement is not rerun. |
| Version and deployment | New fixed version/tag; original v1 preserved. Registry URL lock requires the registered /v2.1 address; the previous failed job must not resume against new code. |

## Why the previous JSON mode was insufficient

[OpenAI's Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
distinguishes valid JSON from schema adherence and lists gpt-4o-mini as supported.
The hosted patch now declares summary:string and evidence_quotes:array[string],
both required, with strict:true and additionalProperties:false. The prompt and
original inputs are preserved. Defensive parsing remains necessary for refusals,
truncation, transport failures and legacy responses; no automatic billable retry
or downgrade to another model is introduced.

Provider references: [GPT-4o-mini](https://developers.openai.com/api/docs/models/gpt-4o-mini),
[Bailian embedding](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai/),
[Bailian rerank](https://help.aliyun.com/zh/model-studio/text-rerank-api).

## Capacity and interpretation limits

The deployed profile still declares local limits:64 MiB HTTP body,20 messages,
8 images/message,6000 UTF-8 bytes/message including deterministic index facts,
4000 UTF-8 query bytes,256-dimensional embeddings and100 rerank candidates.
These are service capacity limits, not additional AML format requirements.
Excess input returns413 without silent source truncation. Embedding and rerank
adapters also enforce their reviewed byte/token bounds. The API accepts image
queries, but new-image semantic matching and cross-batch rerank score calibration
are not established by this audit. Passing offline wire checks does not establish
provider availability, throughput under the official workload, or retrieval quality.

Smoke quota checked in the official docs: one per hour and30 per track this cycle.
This task's stricter authorization remains five total attempts across tracks;
the first failed attempt is retained and counts. Full is outside the authorized scope.
