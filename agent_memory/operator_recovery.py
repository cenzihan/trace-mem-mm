"""Explicit operator acceptance of frozen unknowns; never settles or retries them.

Recovery retires every user in the affected execution's admission snapshot.
It does not claim that an unknown was uncharged, unreceived or attributable to
one of those users. New unknowns still stop admission and dispatch.
"""
import hashlib
import json

def row_digest(row):
    value=dict(row)
    if isinstance(value.get('body'),bytes):
        value['body']={'sha256':hashlib.sha256(value['body']).hexdigest(),'bytes':len(value['body'])}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def hash_string(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

class OperatorRecovery:
    def __init__(self,path,journal,ledger):
        self.path,self.journal,self.ledger=path,journal,ledger

    def proofs(self):
        if not self.path.exists():return {}
        value=json.loads(self.path.read_text())
        keys={'schema','authorization','retired_user_sha256','unknowns'}
        if value.get('schema')==2:keys.add('known_failures')
        if set(value)!=keys or value['schema'] not in (1,2):
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
        for p in value.get('known_failures',[]):
            if set(p)!={'fingerprint','row_sha256','attempt_id','reserved_micro_cny','payload_sha256','body_sha256'}:
                raise ValueError('invalid frozen known-failure proof')
            row=self.journal.row(p['fingerprint'])
            if (not row or row['state']!='failed' or row['status']!=200
                    or not isinstance(row['body'],bytes) or not row['body']
                    or row['transport_completed'] or row['redacted'] or row['receipt_policy']!='retain'
                    or row['attempt_id']!=p['attempt_id'] or row_digest(row)!=p['row_sha256']
                    or not hash_string(p['body_sha256']) or row['body_sha']!=p['body_sha256']
                    or hashlib.sha256(row['body']).hexdigest()!=p['body_sha256']
                    or not hash_string(p['payload_sha256'])
                    or json.loads(row['manifest']).get('payload_sha256')!=p['payload_sha256']):
                raise ValueError('frozen known failure changed')
            # Receipt completeness is established without accepting its rejected
            # application content or replaying the HTTP request. Keep the body.
            from .chat_stream import assemble_chat_stream
            response=assemble_chat_stream(row['body'])
            if response['model']!='gpt-4o-mini-2024-07-18':
                raise ValueError('known failure model mismatch')
            with self.ledger.connect() as db:
                amount=db.execute('SELECT accounted,reserved FROM attempts WHERE id=?',(p['attempt_id'],)).fetchone()
                outcome=db.execute('SELECT category,http_status,returned_model,input_tokens,output_tokens FROM provider_observations WHERE attempt_id=?',(p['attempt_id'],)).fetchone()
            expected=('invalid_contract',200,response['model'],response['usage']['prompt_tokens'],response['usage']['completion_tokens'])
            if (type(p['reserved_micro_cny']) is not int or p['reserved_micro_cny']<=0
                    or amount!=(p['reserved_micro_cny'],p['reserved_micro_cny']) or outcome!=expected):
                raise ValueError('known failure accounting or observation changed')
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
            raise ValueError('retired model payload cannot be replayed')
