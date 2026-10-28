# Explicit recovery from complete rejected responses

The service formerly allowed operator-reviewed retirement of a frozen unknown
outcome, while a complete HTTP200 response rejected by an experimental output
contract had no equivalent recovery path. Because experimental and serving
requests retain the same original journal, such a failure also blocked serving.

The v4.3.3 recovery candidate keeps the successful v4.3.1 retrieval algorithm and
adds an operator-only schema2 proof for a frozen, complete SSE application-contract
failure. A proof checks the exact original row, body hash, request-payload hash,
dated model, complete stream/stop/usage, original observation and full retained
money. The failed response remains failed and retained; no bill settlement or
request replay occurs. Retired execution users and exact payloads stay blocked.
Any new unknown or known failure is still blocking until separately reviewed.

This is not automatic exception suppression or permission to accept invalid
model output. No request handler creates recovery proofs. Missing/corrupt bodies,
incomplete streams, changed observations, changed accounting and removed user
retirement invalidate the proof. Existing schema1 unknown proofs remain supported.
The same scope, journal, original ledger, model restrictions and limits remain.

Independent regression coverage includes receipt preservation, fresh-request
admission, old-user/payload rejection, new-failure blocking and transaction-level
revalidation. This candidate has no new official quality score. Production
installation requires the exact image and the reviewed private frozen proof.
