# Protocol baseline — read 2026-10-24 UTC

Authority: [API Guide](https://agentmemoryleaderboard.ai/api-guide), [Docs](https://agentmemoryleaderboard.ai/docs), [Rules](https://agentmemoryleaderboard.ai/rules), [Competition](https://agentmemoryleaderboard.ai/competition/). 原始来源、读取时刻和哈希在 `sources/manifest.json`。官网多个路径返回同一个包含多个栏目与双语内容的前端页面；这不是猜测的独立 API schema，也没有声称取得机器可读 OpenAPI。

| Interface | Contract implemented |
|---|---|
| Add POST | `request_id`, `user_id`, `session_id`, ordered `messages`; every message has `role` (`user`/`assistant`) and non-empty `content`; optional integer Unix-ms `timestamp` |
| Add success | HTTP 200 only after durable commit and immediate searchability; `success: true`; exact echo of all three IDs |
| Search POST | original `query`, exact `user_id`, positive integer `top_k`; optional multiple-choice `options` is a separate top-level string array |
| Search success | `data` in relevance order; stable non-empty `id` and ordered `content`; optional numeric `score`, `created_at`; empty result is `data: []`; never more than `top_k`; formal K=100 |
| Health GET | no authentication; any 2xx is healthy; this project's extra JSON reports mock state |
| Auth | Token / Bearer Authorization or X-Api-Key; `none` only for public Smoke under official rules |

Multimodal content/query accepts plain text or ordered parts. `text` parts require non-empty `text`; `image_url` parts contain `image_url.url`. Images must be inline Base64 Data URIs in JPEG / PNG / WebP, no remote URLs. Each decoded image is at most 10 MiB; one Add and one Search response each allow at most 30 MiB decoded media. All text and image part order is preserved. See original synthetic `fixtures/add.json` and `fixtures/search.json` for runnable examples.

This service normalizes plain strings to one text part for multimodal output. It uses one message as one memory occurrence and retains session/request/message/part provenance. The source `user_id` is the sole retrieval boundary; Search does not contain `session_id`. Session provenance must not become an extra filter. Different users never share retrieval candidates or derived indexes.

Synchronous Add/Search requests may take up to 30 minutes in the platform contract; no Add-status endpoint or asynchronous polling protocol is invented here. This local WSGI application does not itself configure a production gateway timeout or 30-minute worker SLA. Add retries retain exactly the same logical request ID and body. Identical retry succeeds without duplicate memories; conflicting retry gets 409. Add retryable statuses include 408/409/425/429/500/502/503/504/524; 409 alone never means success. Search 409 is permanent; other retryable Search statuses exclude 409/524. Retry policy belongs to the caller/platform; this service does not launch retries or evaluation jobs.

Malformed payloads get explicit 4xx. No global character cap is specified by AML; this project's default 64 MiB HTTP-body and 30 MiB query-media capacities are local operational limits, with clear 413/422 errors. B0 reports 422 on response media overflow. B2 selects a relevance-ranked prefix of complete memory records within the 30 MiB response budget; top_k is a maximum, not a required count. No text/image part is cropped or removed inside a selected memory. In-memory trace records the media-budget selection count. Auth failure gets 401; forbidden scopes would require an external authorization registry before production. Current shared-key mode authenticates the trusted platform caller and does not implement per-key user-scope ACLs.

The code validates MIME signatures, not full image decoding. B1/B2 adds real BM25/RRF, source cards, explicit temporal facts and bounded same-namespace links; live model interfaces remain disabled. Dense retrieval, model-generated cards and learned reranking have not been run. Tests validate this local interpretation only; no official offline harness, real HTTP deployment, online Smoke, Full, or organizer certification is claimed. Per-track Multimodal timeout/concurrency still needs its frozen contract; do not treat a Textual 1800-second implementation value as verified Multimodal settings.
