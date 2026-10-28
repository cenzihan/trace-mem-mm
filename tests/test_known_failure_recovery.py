"""Operator-only recovery from a preserved complete, rejected model response."""
import hashlib
import json
import unittest

import test_service_transport as fixtures
from test_chat_stream import wire, events
from agent_memory.mem0_transport import Mem0ServiceTransport
from agent_memory.provider_http import HTTPReply, ProviderFailure, OPENAI_ENDPOINT
from agent_memory.operator_recovery import row_digest


class KnownFailureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.ServiceTransportTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        class Http:
            is_fake_http=True
            def __init__(self):self.calls=0;self.good=False;self.unknown=False
            def exchange(self,*args):
                self.calls+=1
                if self.unknown:raise ProviderFailure('network_error')
                text='{"memory":[{"text":"fresh synthetic fact"}]}' if self.good else '{"unexpected":[]}'
                return HTTPReply(200,wire(events(text)),{'content-type':'text/event-stream'})
        self.http=Http();f=self.f
        self.t=Mem0ServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,f.p/'known-receipts',
            {'gpt-4o-mini':lambda:'SYNTHETIC'},http=self.http,offline_test=True)
        self.payload={'model':'gpt-4o-mini','messages':[{'role':'system','content':'Return synthetic source memory JSON.'},{'role':'user','content':[{'type':'text','text':'old independent input'}]}],
                      'max_completion_tokens':128,'store':False,'n':1,'response_format':{'type':'json_object'}}
        with self.assertRaises(ProviderFailure):self.t(OPENAI_ENDPOINT,self.payload)
        with self.t.journal.connect() as db:self.row=dict(db.execute('select * from operations').fetchone())
        with self.t.ledger.connect() as db:reserve=db.execute('select reserved from attempts where id=?',(self.row['attempt_id'],)).fetchone()[0]
        self.proof={'fingerprint':self.row['fingerprint'],'row_sha256':row_digest(self.row),
            'attempt_id':self.row['attempt_id'],'reserved_micro_cny':reserve,
            'payload_sha256':json.loads(self.row['manifest'])['payload_sha256'],'body_sha256':self.row['body_sha']}
        self.record={'schema':2,'authorization':'independent operator fixture',
            'retired_user_sha256':[hashlib.sha256(b'old-user').hexdigest()],
            'unknowns':[],'known_failures':[self.proof]}
        self.save()
    def save(self):self.t.operator_recovery.path.write_text(json.dumps(self.record))
    def test_new_request_allowed_without_altering_failed_receipt_or_reserve(self):
        self.t.ready();self.http.good=True
        fresh=dict(self.payload,messages=[self.payload['messages'][0],{'role':'user','content':[{'type':'text','text':'new independent input'}]}])
        self.t(OPENAI_ENDPOINT,fresh)
        self.assertEqual(self.t.journal.row(self.row['fingerprint']),self.row)
        with self.t.ledger.connect() as db:money=db.execute('select accounted,reserved from attempts where id=?',(self.row['attempt_id'],)).fetchone()
        self.assertEqual(money,(250000,250000));self.assertEqual(self.http.calls,2)
    def test_old_payload_still_blocked_before_dispatch(self):
        with self.assertRaises(ValueError):self.t(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(self.http.calls,1)
    def test_old_user_blocked_new_user_allowed(self):
        with self.assertRaises(ValueError):self.t.assert_source_allowed({'user_id':'old-user'})
        self.t.assert_source_allowed({'user_id':'fresh-user'})
    def test_new_unknown_still_blocks(self):
        self.http.unknown=True
        with self.assertRaises(ProviderFailure):self.t(OPENAI_ENDPOINT,dict(self.payload,messages=[self.payload['messages'][0],{'role':'user','content':[{'type':'text','text':'fresh'}]}]))
        with self.assertRaises(ValueError):self.t.ready()
    def test_new_known_failure_still_blocks(self):
        with self.assertRaises(ProviderFailure):self.t(OPENAI_ENDPOINT,dict(self.payload,messages=[self.payload['messages'][0],{'role':'user','content':[{'type':'text','text':'fresh'}]}]))
        with self.assertRaises(ValueError):self.t.ready()
    def test_removed_or_changed_body_blocks(self):
        for body in (None,b'changed'):
            with self.t.journal.connect() as db:db.execute('update operations set body=? where fingerprint=?',(body,self.row['fingerprint']))
            with self.assertRaises(ValueError):self.t.ready()
    def test_changed_status_or_marked_complete_blocks(self):
        with self.t.journal.connect() as db:db.execute("update operations set state='complete' where fingerprint=?",(self.row['fingerprint'],))
        with self.assertRaises(ValueError):self.t.ready()
    def test_accounting_release_blocks(self):
        with self.t.ledger.connect() as db:db.execute('update attempts set accounted=0 where id=?',(self.row['attempt_id'],))
        with self.assertRaises(ValueError):self.t.ready()
    def test_changed_observed_usage_blocks(self):
        with self.t.ledger.connect() as db:db.execute('update provider_observations set input_tokens=999 where attempt_id=?',(self.row['attempt_id'],))
        with self.assertRaises(ValueError):self.t.ready()
    def test_changed_category_blocks(self):
        with self.t.ledger.connect() as db:db.execute("update provider_observations set category='success' where attempt_id=?",(self.row['attempt_id'],))
        with self.assertRaises(ValueError):self.t.ready()
    def test_no_user_retirement_or_empty_authorization_blocks(self):
        for k,v in [('retired_user_sha256',[]),('authorization','')]:
            old=self.record[k];self.record[k]=v;self.save()
            with self.assertRaises(ValueError):self.t.ready()
            self.record[k]=old
    def test_duplicate_proof_blocks(self):
        self.record['known_failures'].append(self.proof);self.save()
        with self.assertRaises(ValueError):self.t.ready()
    def test_incomplete_body_cannot_be_accepted_even_with_new_hash(self):
        body=wire(events('{"unexpected":[]}'),False)
        with self.t.journal.connect() as db:db.execute('update operations set body=?,body_sha=? where fingerprint=?',
            (body,hashlib.sha256(body).hexdigest(),self.row['fingerprint']))
        row=self.t.journal.row(self.row['fingerprint']);self.proof.update(row_sha256=row_digest(row),body_sha256=row['body_sha']);self.save()
        with self.assertRaises(ProviderFailure):self.t.ready()
    def test_transaction_start_rechecks_recovery_proof(self):
        self.t.journal.install('prepared',self.t.receipt_scope,{'keys':['prepared']})
        self.t.operator_recovery.path.unlink()
        with self.assertRaises(Exception):self.t.journal.start('prepared')
        self.assertEqual(self.t.journal.row('prepared')['state'],'ready')
