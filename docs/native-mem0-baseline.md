# Native Mem0 multimodal baseline

This opt-in profile runs the real `mem0ai==2.2.1` package. Its wheel SHA256 is
`fe91bb91ac8926231993a4aa58df00a60c6c74c709e6a338fe399500776eef4d`.
The Linux CPython3.11 dependency lock includes wheel hashes. This is a distinct
experimental version, `tracemem-mm-mem0-v3.0.0`, with no claimed evaluation result.
Mem0 is Apache-2.0 licensed; its license remains in the installed distribution.

Mem0 performs native additive extraction, deduplication, entity linking and hybrid
search against embedded, persistent Qdrant. Native prompts and scoring are kept.
Provider adapters use the required GPT4o-mini and text-embedding-v4 (256 dimensions)
through the original receipt-first budget transport. They do not instantiate an
unaccounted provider SDK. Telemetry is disabled before importing Mem0. SDK logs
that could include source content are suppressed.

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

Upstream: [Mem0 source](https://github.com/mem0ai/mem0) and
[OSS documentation](https://docs.mem0.ai/open-source/overview).
