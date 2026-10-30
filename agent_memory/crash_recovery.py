"""Exclusive-worker restart recovery; no receipt, money or memory is erased.

The process lease is held for the entire worker lifetime. A replacement worker
may acknowledge incomplete journal entries only before accepting any request.
No provider response is invented or replayed: the old attempt and reservation
remain unchanged and a subsequent authorized retry gets a fresh budget attempt.
"""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from .operator_recovery import row_digest
from .service_transport import never_dispatched


class ProcessLease:
    def __init__(self, path):
        self.file = open(path, 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                if not self.file.read(1):self.file.write(b'0');self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise RuntimeError('another memory worker owns this receipt journal') from None

    def close(self):
        if not self.file.closed:self.file.close()


class RestartAcknowledgements:
    def __init__(self, journal, ledger, scope):
        self.journal, self.ledger, self.scope = journal, ledger, scope
        self.path = journal.root / 'worker-restart-acknowledgements.json'
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else {}
        # Stream rows so old failure bodies cannot all be loaded simultaneously.
        with journal.connect() as db:
            for row in db.execute("SELECT * FROM operations WHERE scope=? AND state IN ('ready','inflight','received')", (scope,)):
                row = dict(row)
                key = row['fingerprint']
                if key in self.entries:
                    if not self.permits(row):raise ValueError('restart acknowledgement changed')
                    continue
                accounting = self.accounting(row)
                self.entries[key] = dict(row_sha256=row_digest(row), accounting=accounting,
                    original_state=row['state'], acknowledged_at=datetime.now(timezone.utc).isoformat(),
                    reason='exclusive replacement worker; prior result unavailable; billed retry authorized')
        self.save()

    def accounting(self, row):
        if row['scope'] != self.scope or row['state'] not in ('ready','inflight','received'):
            raise ValueError('not an interrupted operation in this scope')
        if row['attempt_id'] is None:
            if not never_dispatched(row):raise ValueError('unbound operation is not pristine')
            return None
        with self.ledger.connect() as db:
            amount = db.execute('SELECT model,status,reserved,accounted FROM attempts WHERE id=?',
                                (row['attempt_id'],)).fetchone()
            observed = db.execute('SELECT category FROM provider_observations WHERE attempt_id=?',
                                  (row['attempt_id'],)).fetchone()
        if not amount or amount[0] not in ('gpt-4o-mini','text-embedding-v4'):
            raise ValueError('interrupted attempt lacks original accounting')
        if observed is None and (amount[1] not in ('reserved','dispatched') or amount[3] < amount[2]):
            raise ValueError('unknown attempt must retain its full reservation')
        if amount[1] == 'needs_reconciliation':raise ValueError('budget anomaly requires reconciliation')
        return dict(attempt_id=row['attempt_id'], row=list(amount), observation=list(observed) if observed else None)

    def permits(self, row):
        proof = self.entries.get(row['fingerprint'])
        return bool(proof and proof['row_sha256'] == row_digest(row)
                    and proof['accounting'] == self.accounting(row))

    def save(self):
        temp = self.path.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as f:
            json.dump(self.entries, f, sort_keys=True, indent=2)
            f.flush();os.fsync(f.fileno())
        os.replace(temp, self.path)
        if os.name != 'nt':
            fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
