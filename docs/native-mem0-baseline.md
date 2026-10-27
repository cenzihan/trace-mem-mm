# Native Mem0 multimodal baseline

This opt-in profile runs the real `mem0ai==2.2.1` package. Its wheel SHA256 is
`fe91bb91ac8926231993a4aa58df00a60c6c74c709e6a338fe399500776eef4d`.
The Linux CPython3.11 dependency lock includes wheel hashes. This is a distinct
experimental profile, now `tracemem-mm-mem0-v3.0.3`, with no claimed evaluation result.
Mem0 is Apache-2.0 licensed; its license remains in the installed distribution.

Mem0 performs native additive extraction, deduplication, entity linking and hybrid
search against embedded, persistent Qdrant. Native prompts and scoring are kept.
Provider adapters use the required GPT4o-mini and text-embedding-v4 (256 dimensions)
through the original receipt-first budget transport. They do not instantiate an
unaccounted provider SDK. Telemetry is disabled before importing Mem0. SDK logs
that could include source content are suppressed.

The transport may repeat up to three unauthenticated TLS handshakes before sending
exactly one HTTP POST. It never retries a sent request, response read, or certificate
verification failure. New failures record only their transport phase. A legacy
failure with no phase remains unknown; the phase must never be inferred afterward.

An operator can permanently retire an unresolved input scope, retaining the exact
unknown receipt and its full money hold. A separate, reviewed receipt scope can
then process independent inputs after validating an append-only quarantine proof.
The retired user scope and known model-payload hashes remain blocked. The original
scope stays blocked, and an abandonment is never presented as success or permission
to resume/replay that operation. There is no HTTP route to create such proofs.

The multimodal bridge captions each source/query image using GPT4o-mini, low image
detail. Source text and captions are sent to native Mem0 in ordered, lossless
batches of at most4000UTF-8 bytes. Long individual text is split, not truncated.
Embedding input exceeding8000bytes is chunked and pooled with byte-length weights;
this adaptation is explicitly part of this baseline, not Mem0's default embedding.
Search returns native derived memory facts, labelled as such, and never generates
a final answer. The original text/images remain in a separate user-scoped archive.
This baseline does not return source-image attachments or use the hosted reranker.
That makes it a complete-system comparison, not a single-factor reranker ablation.

API identities are hashed before passing to Mem0 to prevent its whitespace
normalization from merging different users. Add has a durable preparing/complete/
uncertain journal. Unknown or partially failed operations block the affected scope;
automatic SDK embedding fallback cannot replay a failed provider call. All model
traffic shares the existing service receipts and cumulative money ledger.

An explicit operator consent can append a cumulative budget increase in that same
ledger. Original budget and accounting rows remain unchanged. New code reads the
validated consent chain; frozen old versions continue enforcing their smaller
original ceiling, including after restart. No HTTP route can increase budgets.

Build with `docker build -f Dockerfile.mem0 .`. The service factory requires an
explicit activation, private state and secrets, and the original deployment lease.
Local synthetic SDK tests establish protocol and persistence behavior only. They
are neither real model calls nor benchmark scores. Each deployed version must be
frozen and independently verified before its official submission.

When PyPI connectivity is unreliable, download the locked Linux wheels with
`pip download --only-binary=:all: --require-hashes` first, then expose that wheel
directory on a private build network. Set build arguments `PIP_NO_INDEX=1` and
`PIP_FIND_LINKS` to that package cache. Installation still verifies every original
wheel hash; the cache must contain no application data or credentials.

Upstream: [Mem0 source](https://github.com/mem0ai/mem0) and
[OSS documentation](https://docs.mem0.ai/open-source/overview).
# Streaming transport candidate (v3.0.2)

The native memory algorithm and fixed academic models are unchanged. Chat Completions
requests use streaming with `include_usage` so the connection can deliver events while
the completion is generated. This is a reliability hypothesis after a connection
failed while awaiting response headers, not a proven diagnosis or score improvement.
See the [OpenAI streaming contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

The full wire response is durably received before parsing. A valid stream must finish
with `[DONE]`, one stopped choice, consistent completion/model identities, and final
usage. Partial output, refusals, tool calls, model substitutions and missing usage
cannot become stored memory. Successful wire bodies are minimized using the existing
receipt policy; failed responses and accounting history are retained. No sent POST is
retried. Future transport diagnostics add only a whitelisted exception kind, byte
count and streaming flag; they cannot explain an earlier unrecorded exception.

## Bounded caption images (v3.0.3)

A separate real-image connectivity check also disconnected before HTTP response
headers with streaming enabled. Streaming alone therefore did not establish a fix.
This profile additionally sends an in-memory JPEG derivative, fitting within512pixels
per side without enlargement, quality85, and at most256KiB, to the existing low-detail
caption call. It applies EXIF orientation, composites transparency on white, and
strips metadata. Source images in the archive remain byte-for-byte unchanged.
Animated or undecodable images are rejected; the local decode limit is24million pixels.

This is a bounded transport hypothesis, not a claim of equivalent visual detail or
better benchmark performance. Fine text and small objects may be lost in the view.
The fixed models, native Mem0 prompts, and no-retry policy remain unchanged.
See [OpenAI's image input and detail guidance](https://developers.openai.com/api/docs/guides/images-vision).
