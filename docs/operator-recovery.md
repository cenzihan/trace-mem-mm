# Operator recovery v4.3.2

An unresolved provider outcome must keep its receipt state and full budget hold.
The prior global stop also blocked unrelated new work indefinitely after an
independent execution had been stopped. This candidate permits explicit operator
acceptance of a frozen unknown without calling it successful, unreceived or free.

The private operator record must preserve the exact unknown row and attempt,
request hash and original held amount. It must retire the users in the affected
execution's admission snapshot. This is a conservative exclusion, not a claim
of retrospectively proven request attribution. Retired users are blocked at Add
and Search; exact old model payloads are also blocked before a new reservation.

Both readiness and SQLite dispatch admission validate the accepted record. A new
unknown, in-flight request, changed receipt or released old hold still stops work.
No journal or scope is replaced; no old experiment allowance is reopened. The
operator record is private, manually reviewed, and never written by an API route.
Its authorization must cover accepting that the old outcome remains uncertain.

This patch keeps v4.3.1 retrieval/model behavior. It is not a reliability claim
about large-image inference and has not produced a new official score.

The patch was deployed after reviewed admission snapshots, image verification,
focused guard tests and a new independent end-to-end preflight, while all official
tracks were idle. All eight provider calls completed successfully; image retrieval
and cross-session controls passed. The unrelated-query control returned evidence
and failed. That quality failure is preserved, not relabelled as a pass. The
release restores service availability only; it is not a new quality-qualified
Smoke candidate. Twelve public health, authorization and format checks passed
without additional model calls. Existing memory state, unknown receipt, full
reserve, model identity, original lease and cleanup schedule remain unchanged.
