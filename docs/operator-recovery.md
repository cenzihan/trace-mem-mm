# Operator recovery candidate v4.3.2

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
about large-image inference and has not yet produced a new official score.
Deployment requires reviewed admission snapshots, image verification and a new
independent preflight, with no active official evaluation during the cutover.
