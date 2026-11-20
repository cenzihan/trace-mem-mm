# Complete participant-interface inventory

Checked against the [AML API Guide](https://agentmemoryleaderboard.ai/api-guide)
on 2026-10-27. Applies to the synchronous Multimodal context candidate. The
documented participant interfaces and conditions are enumerated below; platform
job management and scoring are not participant-hosted endpoints.

| Interface | Contract | Implementation / check |
|---|---|---|
| POST Add | Ordered messages, request/user/session IDs; HTTP200 only after durable storage and immediate searchability; echo all three IDs and boolean success. | Registered `/v2.1/add`, inherited durable Add/journal implementation; contract, idempotence, conflict and restart regressions. |
| POST Search | Same user scope, original question, optional options, top_k; ordered data array of stable IDs and unchanged source content; at most top_k. | Registered `/v2.1/search`, context candidate; empty user scope returns data:[], no model call. |
| GET Health | Public unauthenticated GET, any2xx healthy. | `/v2.1/health`; checks local configuration, budget/journals, lease and readiness. Returns exact version. It does not secretly invoke a model. |
| Conditional Add-status | Required only for an approved asynchronous HTTP202 flow; URL contains `{task_id}`, response includes task_id. | Not applicable: this version is synchronous, never returns202, and does not register a status URL. |

Authentication uses the registered Bearer scheme for Add/Search. Token and
X-Api-Key are alternative schemes allowed by AML, not three simultaneously
required credentials. Unauthenticated requests return401 before model dispatch.

Messages allow the documented user/assistant roles and optional integer Unix-ms
timestamps. Multimodal strings or ordered text/image parts are accepted. Images
must be inline JPEG/PNG/WebP, at most10MiB decoded each and30MiB per Add or Search
response. Original parts retain order and content. The context candidate skips
an entire result that cannot fit, rather than truncating an original record.

Invalid JSON returns400; invalid fields422; wrong credentials401; declared local
capacity413; overload429 with Retry-After5; unresolved dependencies503 or500.
Unknown/failed provider work is not automatically replayed. Same logical Add
request/body is idempotent, while conflicting bodies return409. Responses use
JSON UTF-8 with accurate Content-Length and no-store.

Local limits are disclosed, not claimed as universal AML limits:20 messages/Add,
8 images/message,6000 source UTF-8 bytes including deterministic indexed facts,
4000 query UTF-8 bytes,64MiB HTTP body,100 candidates,16 admissions with30s queue
waiting and one active model request. Gunicorn's liveness timeout is1900s; the
gateway's request timeouts must also be verified on deployment. This is not a
promise that every official payload fits these capacities.

The guide does not require participant Answer, Eval, job-status, public scoring,
or deletion HTTP endpoints. The platform owns Answer/Eval and job management.
Data deletion is a participant retention obligation: the existing scheduled
purge remains2026-11-07 UTC, earlier than the guide's30-day maximum. The lease
remains2026-11-05T16:30Z. Private logs contain minimal HTTP metadata only.

The deployed public wire check verifies health/version, missing-auth401,
malformed/invalid request handling, empty-scope Search200 and the response
schema without additional paid model work. The separately authorized official
Smoke provides the real positive Add/Search/Answer/Eval integration check.
Offline tests are never represented as that official result.

Official Smoke quota: one per hour, at most30 per track this edition. The task's
separate existing five-attempt authorization remains a stricter local control.
