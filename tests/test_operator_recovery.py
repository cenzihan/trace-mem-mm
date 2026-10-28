import hashlib,json,unittest
from tests import test_service_transport as fixtures
from agent_memory.provider_http import ProviderFailure
from agent_memory.operator_recovery import row_digest

class OperatorRecoveryTests(unittest.TestCase):
    setUp=fixtures.ServiceTransportTests.setUp
    make=fixtures.ServiceTransportTests.make
    call=fixtures.ServiceTransportTests.call
    def prepare(self):
        t=self.make();self.http.failure=True
        with self.assertRaises(ProviderFailure):self.call(t)
        self.http.failure=False
        with t.journal.connect() as db:r=dict(db.execute('SELECT * FROM operations').fetchone())
        with self.ledger.connect() as db:amount=db.execute('SELECT reserved FROM attempts WHERE id=?',(r['attempt_id'],)).fetchone()[0]
        record={'schema':1,'authorization':'Synthetic test explicit operator acceptance',
            'retired_user_sha256':[hashlib.sha256(b'old-user').hexdigest()],
            'unknowns':[dict(fingerprint=r['fingerprint'],row_sha256=row_digest(r),attempt_id=r['attempt_id'],
                reserved_micro_cny=amount,payload_sha256=json.loads(r['manifest'])['payload_sha256'])]}
        t.operator_recovery.path.write_text(json.dumps(record))
        return t,r,record
    def test_fresh_call_succeeds_unknown_and_money_preserved(self):
        t,r,_=self.prepare();self.payload['input']=['Fresh unrelated synthetic input']
        self.call(t)
        self.assertEqual(t.journal.row(r['fingerprint']),r)
        with self.ledger.connect() as db:a=db.execute('SELECT accounted,reserved FROM attempts WHERE id=?',(r['attempt_id'],)).fetchone()
        self.assertEqual(a[0],a[1]);self.make().ready()
    def test_exact_unknown_replay_blocked_before_call(self):
        t,r,_=self.prepare()
        with self.assertRaises(ValueError):self.call(t)
        self.assertEqual(self.http.calls,1)
    def test_old_user_block_new_user_allowed(self):
        t,r,_=self.prepare()
        with self.assertRaises(ValueError):t.assert_source_allowed({'user_id':'old-user'})
        t.assert_source_allowed({'user_id':'new-user'})
    def test_new_unknown_reblocks(self):
        t,r,_=self.prepare();self.payload['input']=['New fixture'];self.http.failure=True
        with self.assertRaises(ProviderFailure):self.call(t)
        with self.assertRaises(ValueError):t.ready()
    def test_corrupted_old_row_blocks(self):
        t,r,_=self.prepare()
        with t.journal.connect() as db:db.execute("UPDATE operations SET state='complete' WHERE fingerprint=?",(r['fingerprint'],))
        with self.assertRaises(ValueError):t.ready()
    def test_released_reserve_blocks(self):
        t,r,_=self.prepare()
        with self.ledger.connect() as db:db.execute('UPDATE attempts SET accounted=0 WHERE id=?',(r['attempt_id'],))
        with self.assertRaises(ValueError):t.ready()
    def test_empty_user_retirement_fails(self):
        t,r,v=self.prepare();v['retired_user_sha256']=[];t.operator_recovery.path.write_text(json.dumps(v))
        with self.assertRaises(ValueError):t.ready()
    def test_new_journal_start_checks_unknown(self):
        t,r,_=self.prepare();t.journal.install('test-ready',t.receipt_scope,{'keys':['test-ready']})
        t.operator_recovery.path.unlink()
        with self.assertRaises(Exception):t.journal.start('test-ready')
    def test_transaction_rejects_second_inflight_operation(self):
        t,_,_=self.prepare()
        for key in ('first','second'):t.journal.install(key,t.receipt_scope,{'keys':[key]})
        t.journal.start('first')
        with self.assertRaises(Exception):t.journal.start('second')
        with self.assertRaises(ValueError):t.ready()
        self.assertEqual(t.journal.row('second')['state'],'ready')
    def test_different_unknown_is_not_covered_by_old_proof(self):
        t,_,_=self.prepare();t.journal.install('other',t.receipt_scope,{'keys':['other']})
        with t.journal.connect() as db:db.execute("UPDATE operations SET state='unknown' WHERE fingerprint='other'")
        with self.assertRaises(ValueError):t.ready()
        t.journal.install('new',t.receipt_scope,{'keys':['new']})
        with self.assertRaises(Exception):t.journal.start('new')
