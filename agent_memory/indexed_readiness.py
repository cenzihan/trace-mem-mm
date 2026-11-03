"""SQLite-maintained exception indexes, without trusting a cached validation.

Only already-complete minimized receipts satisfying the existing ready() checks
are omitted. Every active, failed, unknown, or malformed receipt still passes
through the original Python validation. SQLite updates membership atomically on
INSERT/UPDATE, including changes made by another connection. No history is
deleted and no paid request identity, receipt, or accounting row is rewritten.
"""

# COALESCE is essential: SQL NULL must not hide an incomplete/corrupt receipt.
NEEDS_CHECK = """NOT COALESCE((state='complete'
    AND receipt_policy='usage_only' AND body IS NULL AND redacted=1
    AND transport_completed IS NOT NULL AND transport_completed!=0
    AND typeof(body_sha)='text' AND length(body_sha)=64), 0)"""


def install_readiness_indexes(db):
    db.execute('CREATE INDEX IF NOT EXISTS gpu6_readiness_exceptions_v1 '
               'ON operations(fingerprint) WHERE ' + NEEDS_CHECK)
    db.execute("CREATE INDEX IF NOT EXISTS gpu6_unfinished_scope_v1 "
               "ON operations(scope) WHERE state NOT IN ('ready','complete')")


def readiness_rows(db):
    # Force the reviewed partial index; fail closed if it is missing, rather
    # than silently returning to full-history scans under the shared live lock.
    return db.execute('SELECT * FROM operations INDEXED BY '
                      'gpu6_readiness_exceptions_v1 WHERE ' + NEEDS_CHECK)
