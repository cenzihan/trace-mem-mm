# Hosted rerank failure diagnostics candidate

This candidate preserves scalar transport diagnostics for future hosted-rerank
failures. The hosted receipt wrapper already captured transport phase and response
byte counts, but its caller did not persist them. A `tls_error` category alone
cannot establish whether the request had started sending.

Both service transports now use the same bounded append-only diagnostic writer.
It records a local receipt fingerprint, UTC time, serialized request size, streaming
flag, whitelisted phase/error class, and validated response scalars. It excludes
headers, credentials, endpoint URLs, exception messages, source text, images,
queries, and response bodies. Missing phases remain null; historical attempts
are not backfilled. The private log stops growing at 256 KiB.

Diagnostic writes happen after durable failure classification. A full or
unwritable log cannot turn an error into success, release a reservation, or permit
another dispatch. The original journal, lifetime limits, unknown-outcome blocks,
and full uncertain-call reservations remain authoritative. No POST retries,
quarantine acknowledgements, fallback retrieval, model changes, or protocol changes
are added.

This is an observability change, not a demonstrated reliability or score
improvement. It has not been deployed or evaluated with a real model. Socket
failure injection checks both pre-POST and post-start TLS failures without network
or provider calls. A later reliability change requires separate diagnosis and
validation on new independent inputs.
