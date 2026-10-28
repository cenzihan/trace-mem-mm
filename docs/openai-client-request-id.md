# OpenAI client request correlation

Each new `gpt-4o-mini` Chat Completions attempt uses the existing random,
durably reserved ledger attempt ID as `X-Client-Request-Id`. The ID has the
fixed ASCII form `attempt_` followed by 32 lowercase hexadecimal characters.
It is independent of user IDs, messages, media, and credentials. The header is
not sent to the embedding provider or to any other endpoint.
An authorized prepaid transport uses its already-reserved parent attempt ID;
this does not create a new reservation or reopen a closed grant. That existing
outbox trusts its frozen grant and does not independently query the parent
ledger when it sends; the header does not strengthen that assumption.

The existing reservation is written before the header is constructed. In the
receipt-backed service, the existing observer binds that attempt to the journal
before the HTTP exchange; the receipt wrapper checks that the bound attempt
matches the outgoing ID. If binding, comparison, or ID validation fails, no
request is sent and the existing conservative failure accounting applies. The
ID is retained in the private ledger as before. Do not copy it, receipts, or
private input into public materials.

OpenAI documents this header as a way for its support team to correlate some
timeouts or network failures even when no response request ID is available:
[OpenAI API reference](https://developers.openai.com/api/reference/overview#debugging-requests).
The header is **not an idempotency key**, proof of dispatch, proof of completion,
or permission to retry. A recorded `dispatched` state can precede a failed
observer and does not by itself establish that an HTTP request was sent. This
change does not enable a POST retry, infer an outcome, or release a reservation.
The comparison is not an atomic transaction across the budget, journal, and
network; it does not prove user/source ownership or prevent a hostile or
non-cooperating writer from changing state after the check.

Earlier requests did not include this header and cannot be retroactively given
one. In particular, it does not reconcile the existing unknown attempt or make
the protected service ready. Provider support correspondence remains subject to
the user's separate messaging authorization.
