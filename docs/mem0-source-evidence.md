# Native Mem0 with original evidence

`tracemem-mm-mem0-evidence-v3.1.1` is a separate candidate. The completed
facts-only baseline remains fixed as `tracemem-mm-mem0-v3.0.3`. This candidate
has no official result yet.

The baseline archives original messages but returns only extracted facts. The
candidate adds a source channel without changing Mem0's prompts, Qdrant search,
provider models or caption representation. It indexes original text together
with image descriptions already produced during Add, in the same transaction as
the original archive. Queries and options are never persisted in that index.

Search computes native Mem0 results once. In parallel it ranks originals with
BM25 over that existing source index and exact image-byte identity. The first
twelve source anchors can include immediately adjacent messages from the same
Add and session. It alternates original evidence and clearly labelled derived
facts, retaining each channel's order. Scores are reciprocal output positions,
not confidence probabilities. Whole original messages keep every ordered text
and image part. A record exceeding the remaining decoded-media allowance is
skipped intact; smaller later records can still fit. Output obeys top_k and the
30MiB media limit.

This is a controlled test of an added source channel, not a claim that Mem0
provides image provenance or that any score will improve. Source ranking has no
independent dense vectors or visual reranker. Exact identity cannot recognize a
different photograph of the same object. The512px caption view still may lose
small details, and the system has no calibrated abstention. Derived facts can
compete with original evidence for context capacity.

Activate only with `agent_memory.mem0_app:create_evidence_app()` and both the
existing native activation flag and
`TRACEMEM_MEM0_EVIDENCE_ENABLED=approved-native-mem0-evidence`. Use a separate
original archive for the new fixed version. Existing archives without the
source index fail before a paid query; they are not silently backfilled.

The original cumulative budget and receipt journal remain unchanged. The new
`native-mem0-v3.1.1` execution scope still checks the entire original journal:
unresolved calls block it unless an operator has independently audited and
permanently retired their old scope. Retired user and payload identities remain
blocked, their records and full reservations remain intact, and the old scope
cannot resume. No grant, historical unknown, lifetime counter, lease or cleanup
deadline is reset. Provider traffic
uses only GPT4o-mini and text-embedding-v4. The new source channel itself makes
no extra model request. Independent synthetic tests verify image identity,
source/context preservation, scope isolation, atomic failure, restart,
idempotence, query non-persistence and whole-record media packing.

The evidence factory also enables strict application-stream completion. After
receiving a full SSE DONE event, it validates the entire response, a single
approved model and response identity, finish reason, and consistent usage
totals. It then closes its dedicated connection without waiting for HTTP EOF.
The generic transport and facts-only factory keep the old read-to-EOF behavior.
This avoids discarding an already complete semantic response solely because
its HTTP envelope lacks a terminating chunk. It cannot repair a stream that
ends before the model response is complete. Such failures remain unknown,
retain their full reservation, and never trigger a POST retry.

Future transport failures record only bounded scalar diagnostics: received byte
count, status, and whether a complete SSE marker and valid response were seen.
They do not log chunks, prompts, keys or exception messages. An independent
HTTP/1.1 chunk-decoder test reproduces the missing-final-chunk case, including
partial streams, malformed usage, Unicode/event splits and no-replay behavior.
This test establishes handling of that failure class, not the cause of a past
unrecorded response or a general guarantee of network reliability.
