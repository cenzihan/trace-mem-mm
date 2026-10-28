# Deadline phase diagnostics candidate

The caller's overall deadline previously returned `deadline_unknown` even when
the HTTP worker had reached a known phase. This made one independent timeout
impossible to distinguish from DNS/TLS, POST, header or body waiting.

The candidate keeps a thread-safe snapshot of the last observed phase, received
byte count, HTTP status when available, and SSE completeness flags. It contains
no URL, credential, input, output body or exception message. A caller deadline
still produces a timeout failure, preserves the full reservation, cancels the
worker and keeps the transport busy until that worker exits. It cannot accept
partial output, replay the request, infer delivery or reclassify old failures.

Four deterministic blocked-worker tests cover TLS, POST, headers and body. They
verify that the snapshot contains scalars only, no POST occurs after a TLS-stage
deadline, and no concurrent request reuses the occupied worker slot.

This is observability work, not a connection repair or an algorithm change. It
is not deployed and has no official evaluation result. Old timeout phases remain
unknown; future phase snapshots do not establish the root cause of past events.
