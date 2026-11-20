# GPU6 validation connection reuse

The GPU6 profile revalidates retained failures before readiness and provider
dispatch. With thousands of historical failures, opening a fresh SQLite ledger
connection for each row makes the shared journal lock expensive.

The rc6 profile reuses one read-only ledger connection within each readiness or
journal-start validation. Every row and accounting condition is still checked.
There is no cached verdict, no explicit long-lived read transaction, and no
connection shared between threads or retained across calls. The original write
connection, reservations, observations, retries and single-writer lease remain
unchanged. Other serving profiles retain their existing connection policy.

An isolated network-disabled benchmark with 150,000 completed synthetic receipts
and 2,006 retained synthetic failures measured 16 fake-provider calls at concurrency
16 in 15.85 seconds before and 3.04 seconds after. This measures validation overhead,
not a promise of equivalent Full throughput or provider latency improvement.

Tests cover read-only enforcement, closure on exceptions, thread isolation,
immediate visibility of released accounting and missing observations, existing
unknown guards, parallel native Add and committed-memory behavior.
