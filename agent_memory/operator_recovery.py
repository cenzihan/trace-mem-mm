"""Explicit operator acceptance of frozen unknowns; never settles or retries them.

Recovery retires every user in the affected execution's admission snapshot.
It does not claim that an unknown was uncharged, unreceived or attributable to
one of those users. New unknowns still stop admission and dispatch.
"""
import hashlib
import json

def row_digest(row):
    return hashlib.sha256(json.dumps(dict(row),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def hash_string(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

class OperatorRecovery:
    def __init__(self,path,journal,ledger):
        self.path,self.journal,self.ledger=path,journal,ledger

    def proofs(self):
        if not self.path.exists():return {}
        value=json.loads(self.path.read_text())
        if set(value)!={'schema','authorization','retired_user_sha256','unknowns'} or value['schema']!=1:
            raise ValueError('invalid operator recovery schema')
        users=value['retired_user_sha256']
        if not value['authorization'] or not isinstance(users,list) or not users or any(not hash_string(x) for x in users):
            raise ValueError('explicit authorization and conservative user retirement required')
        result={}
        for p in value['unknowns']:
            if set(p)!={'fingerprint','row_sha256','attempt_id','reserved_micro_cny','payload_sha256'}:
                raise ValueError('invalid frozen unknown proof')
            row=self.journal.row(p['fingerprint'])
            if (not row or row['state']!='unknown' or row['status'] is not None or row['body'] is not None
                    or row['transport_completed'] or row['attempt_id']!=p['attempt_id']
                    or row_digest(row)!=p['row_sha256'] or not hash_string(p['payload_sha256'])
                    or json.loads(row['manifest']).get('payload_sha256')!=p['payload_sha256']):
                raise ValueError('frozen unknown changed')
            with self.ledger.connect() as db:
                amount=db.execute('SELECT accounted,reserved FROM attempts WHERE id=?',(p['attempt_id'],)).fetchone()
                outcome=db.execute('SELECT category,http_status FROM provider_observations WHERE attempt_id=?',(p['attempt_id'],)).fetchone()
            if (not amount or type(p['reserved_micro_cny']) is not int or p['reserved_micro_cny']<=0
                    or amount!=(p['reserved_micro_cny'],p['reserved_micro_cny'])
                    or outcome not in [('timeout',None),('network_error',None),('tls_error',None)]):
                raise ValueError('full unknown reserve and failure must remain')
            if p['fingerprint'] in result:raise ValueError('duplicate operator proof')
            result[p['fingerprint']]=dict(p,retired_user_sha256=users)
        if not result:raise ValueError('empty recovery record')
        return result

    def permits(self,row):
        p=self.proofs().get(row['fingerprint'])
        return bool(p and row_digest(row)==p['row_sha256'])

    def assert_user(self,user_id):
        digest=hashlib.sha256(user_id.encode()).hexdigest()
        if any(digest in p['retired_user_sha256'] for p in self.proofs().values()):
            raise ValueError('retired execution user cannot be replayed')

    def assert_payload(self,payload_sha):
        if any(p['payload_sha256']==payload_sha for p in self.proofs().values()):
            raise ValueError('unknown model payload cannot be replayed')
