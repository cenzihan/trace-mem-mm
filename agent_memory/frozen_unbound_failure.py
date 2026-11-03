"""Explicit, hash-bound acknowledgement; original unknowns and charges stay.

This never attributes an unlinked ledger attempt to a particular journal row.
The operator freezes the conservative set of uncertain charges after draining
the sole writer. New unknowns are not covered by the acknowledgement.
"""
import json
from .operator_recovery import row_digest
from .service_transport import never_dispatched


class FrozenUnboundFailures:
    def __init__(self, path, ledger, scope):
        self.path, self.ledger, self.scope = path, ledger, scope

    def permits(self, row):
        if row['state'] != 'unknown' or row['attempt_id'] is not None:return False
        if not self.path.exists():return False
        value = json.loads(self.path.read_text())
        if value.get('schema') != 1 or not value.get('authorization'):
            raise ValueError('explicit frozen failure authorization required')
        proof = value.get('rows', {}).get(row['fingerprint'])
        if proof is None:return False
        if (row['scope'] != self.scope or row['state'] != 'unknown'
                or not never_dispatched(row) or row_digest(row) != proof):
            raise ValueError('frozen unbound failure changed')
        charges = value.get('uncertain_charges', [])
        if not charges:raise ValueError('preserved uncertain charges required')
        with self.ledger.connect() as db:
            for old in charges:
                current = db.execute('SELECT a.id,a.model,a.status,a.reserved,a.accounted,o.category,o.http_status '
                    'FROM attempts a JOIN provider_observations o ON a.id=o.attempt_id WHERE a.id=?', (old[0],)).fetchone()
                if (current is None or list(current) != old or old[1] not in ('gpt-4o-mini','text-embedding-v4')
                        or old[2] != 'uncertain' or old[3] <= 0 or old[4] < old[3]
                        or old[5] not in ('network_error','timeout','tls_error') or old[6] is not None):
                    raise ValueError('uncertain charge changed or was released')
        return True
