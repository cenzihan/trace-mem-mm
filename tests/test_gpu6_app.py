import json
import tempfile
import threading
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import test_service_transport as fixture
from agent_memory.gpu6_app import GPU6Transport, RecordingLedger, FileConcurrency, GPU6HTTPS
from agent_memory.two_lane_options import TwoLaneRetryTransport
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, BudgetedJSONTransport, ProviderBlocked
from agent_memory.scoped_unknown import request_owner


class GPU6Tests(unittest.TestCase):
    setUp = fixture.ServiceTransportTests.setUp

    def make(self, cls=GPU6Transport, factory=fixture.FakeHTTP):
        t=cls(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},
              http=self.http,http_factory=factory,offline_test=True)
        self.addCleanup(t.close)
        return t

    def test_expired_timestamp_retained_only_in_explicit_profile(self):
        v=json.loads(self.prices.read_text());v['checked_at']=(datetime.now(timezone.utc)-timedelta(days=10)).isoformat()
        self.prices.write_text(json.dumps(v));old=self.prices.read_bytes()
        t=self.make();t.ready();self.assertEqual(t.prices()['checked_at'],v['checked_at'])
        with request_owner('synthetic'):t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        self.assertEqual(self.prices.read_bytes(),old)
        with t.ledger.connect() as db:
            self.assertEqual(db.execute('SELECT price_checked_at FROM attempts').fetchone()[0],v['checked_at'])
        t.close();other=self.make(TwoLaneRetryTransport)
        with self.assertRaises(ValueError):other.ready()

    def test_uncapped_recording_retains_original_budget_and_history(self):
        t=self.make();rate=VerifiedRate(model='text-embedding-v4',checked_at=self.config['checked_at'],**self.config['models']['text-embedding-v4'])
        first=t.ledger.reserve(rate,1,fixed_reserve_cny='501')
        second=t.ledger.reserve(rate,1)
        with t.ledger.connect() as db:
            self.assertEqual(db.execute('SELECT cap FROM budget').fetchone()[0],500000000)
            self.assertEqual(db.execute('SELECT count(*) FROM attempts').fetchone()[0],2)
        with self.assertRaises(ValueError):BudgetLedger(t.ledger.path).reserve(rate,1)

    def test_sixteen_provider_lanes_own_distinct_receipts(self):
        gate=threading.Barrier(16)
        class HTTP(fixture.FakeHTTP):
            def exchange(self,*args):gate.wait(20);return super().exchange(*args)
        t=self.make(factory=HTTP)
        def call(i):
            with request_owner(str(i)):return t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for f in [pool.submit(call,i) for i in range(16)]:f.result(40)
        self.assertEqual(t.ledger.totals()['attempts'],16)
        self.assertEqual(t._lanes.qsize(),16)
        self.assertFalse(t._live)

    def test_hot_limit_can_grow_and_shrink_without_aborting_work(self):
        p=self.p/'concurrency.json';p.write_text('{"concurrency":2}')
        gate=FileConcurrency(p);self.assertTrue(gate.acquire(0));self.assertTrue(gate.acquire(0));self.assertFalse(gate.acquire(0))
        p.write_text('{"concurrency":4}');self.assertTrue(gate.acquire(0));self.assertTrue(gate.acquire(0))
        p.write_text('{"concurrency":2}');self.assertFalse(gate.acquire(0));gate.release();gate.release();self.assertFalse(gate.acquire(0))
        gate.release();self.assertTrue(gate.acquire(0));gate.release();gate.release()
        p.write_text('{"concurrency":true}')
        with self.assertRaises(ValueError):gate.acquire(0)

    def test_journal_transaction_and_readiness_share_lock_order(self):
        t=self.make();entered=threading.Event();release=threading.Event();observed=[]
        def writer():
            with t.journal.connect() as db:
                db.execute('BEGIN IMMEDIATE');entered.set();release.wait(5)
                # A journal writer must already own the readiness lock, not
                # acquire it after SQLite while a reader owns it in reverse.
                observed.append(t._live_lock._is_owned())
                t._runtime_active({'scope':t.receipt_scope,'state':'ready','fingerprint':'synthetic'})
        with ThreadPoolExecutor(max_workers=2) as pool:
            w=pool.submit(writer);self.assertTrue(entered.wait(3))
            r=pool.submit(t.ready);release.set();w.result(5);r.result(5)
        self.assertEqual(observed,[True])

    def test_large_history_parallel_writes_with_health_polling(self):
        t=self.make()
        with t.journal.connect() as db:
            db.executemany("INSERT INTO operations(fingerprint,scope,manifest,state,body_sha,redacted,transport_completed,receipt_policy) VALUES(?,?,?,'complete',?,1,1,'usage_only')",
                ((f'history-{i}','synthetic','{}','a'*64) for i in range(90000)))
        def call(i):
            with request_owner(f'synthetic-{i}'):return t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs=[pool.submit(call,i) for i in range(8)]
            health=pool.submit(lambda:[t.ready() for _ in range(8)])
            for job in jobs:job.result(60)
            health.result(60)
        t.ready();self.assertEqual(t.ledger.totals()['attempts'],8)
        with t.journal.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM operations WHERE state!='complete'").fetchone()[0],0)

    def test_indexed_readiness_checks_changed_history_and_nulls(self):
        import sqlite3
        from contextlib import closing
        t=self.make()
        with request_owner('synthetic'):t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        with t.journal.connect() as db:
            original=dict(db.execute('SELECT * FROM operations').fetchone())
            self.assertEqual(list(t._readiness_rows(db)), [])
        # A separate connection simulates operator edits: no process-local
        # validation cache can conceal a modified historical receipt.
        changes=[('state','unknown'),('receipt_policy','retain'),
                 ('body',b'synthetic'),('redacted',0),
                 ('redacted',None),('transport_completed',0),
                 ('body_sha',None),('body_sha',b'a'*64),('body_sha','short')]
        for column,value in changes:
            with self.subTest(column=column,value=value):
                with closing(sqlite3.connect(t.journal.path)) as db, db:
                    db.execute('UPDATE operations SET '+column+'=?',(value,))
                with t.journal.connect() as db:
                    self.assertEqual(len(list(t._readiness_rows(db))),1)
                with self.assertRaises(ValueError):t.ready()
                with closing(sqlite3.connect(t.journal.path)) as db, db:
                    db.execute('UPDATE operations SET '+column+'=?',(original[column],))
                t.ready()

    def test_indexed_readiness_work_is_independent_of_completed_history(self):
        from agent_memory.indexed_readiness import NEEDS_CHECK
        t=self.make()
        with t.journal.connect() as db:
            db.executemany("INSERT INTO operations(fingerprint,scope,manifest,state,body_sha,redacted,transport_completed,receipt_policy) VALUES(?,?,?,'complete',?,1,1,'usage_only')",
                ((f'history-{i}','synthetic','{}','a'*64) for i in range(90000)))
            steps=[]
            db.set_progress_handler(lambda:steps.append(1) or 0,100)
            self.assertEqual(list(t._readiness_rows(db)),[])
            db.set_progress_handler(None,0)
            self.assertLess(len(steps),10,'must not visit the 90k completed rows')
            plan=db.execute('EXPLAIN QUERY PLAN SELECT * FROM operations INDEXED BY gpu6_readiness_exceptions_v1 WHERE '+NEEDS_CHECK).fetchall()
            self.assertIn('gpu6_readiness_exceptions_v1',str([tuple(r) for r in plan]))
            db.execute("INSERT INTO operations(fingerprint,scope,manifest,state) VALUES('late-unknown','synthetic','{}','unknown')")
        with self.assertRaises(ValueError):t.ready()

    def test_indexed_readiness_failure_does_not_open_network_lane(self):
        t=self.make()
        with t.journal.connect() as db:db.execute('DROP INDEX gpu6_readiness_exceptions_v1')
        import sqlite3
        with self.assertRaises(sqlite3.OperationalError):
            with request_owner('synthetic'):t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        self.assertEqual(t.ledger.totals()['attempts'],0)
        self.assertEqual(t._lanes.qsize(),16)

    def test_frozen_unbound_unknown_preserves_charges_and_rejects_changes(self):
        from agent_memory.operator_recovery import row_digest
        t=self.make();rate=VerifiedRate(model='text-embedding-v4',checked_at=self.config['checked_at'],**self.config['models']['text-embedding-v4'])
        a=t.ledger.reserve(rate,100);t.ledger.mark_dispatched(a)
        t.ledger.observe_provider(a,mode='offline_fake',category='network_error')
        with t.journal.connect() as db:
            db.execute("INSERT INTO operations(fingerprint,scope,manifest,state) VALUES(?,?,?,'unknown')",('unbound',t.receipt_scope,'{}'))
        row=t.journal.row('unbound')
        with self.assertRaises(ValueError):t.ready()
        with t.ledger.connect() as db:
            charge=list(db.execute('SELECT a.id,a.model,a.status,a.reserved,a.accounted,o.category,o.http_status FROM attempts a JOIN provider_observations o ON a.id=o.attempt_id WHERE a.id=?',(a,)).fetchone())
        proof={'schema':1,'authorization':'synthetic bounded retry; retain unknown and all charges','rows':{'unbound':row_digest(row)},'uncertain_charges':[charge]}
        t.frozen_unbound_failures.path.write_text(json.dumps(proof));t.ready()
        self.assertEqual(t.journal.row('unbound'),row)
        with t.journal.connect() as db:
            db.execute("INSERT INTO operations(fingerprint,scope,manifest,state) VALUES(?,?,?,'unknown')",('not-approved',t.receipt_scope,'{}'))
        with self.assertRaises(ValueError):t.ready()
        changed=dict(row,state='complete')
        self.assertFalse(t.frozen_unbound_failures.permits(changed))
        changed=dict(row,manifest='changed')
        with self.assertRaises(ValueError):t.frozen_unbound_failures.permits(changed)
        with t.ledger.connect() as db:db.execute('UPDATE attempts SET accounted=0 WHERE id=?',(a,))
        with self.assertRaises(ValueError):t.frozen_unbound_failures.permits(row)

    def test_openai_only_tunnel_preserves_hostname_and_tls_verification(self):
        h=GPU6HTTPS();c=h.connection('api.openai.com',5)
        self.assertEqual((c.host,c.port,c._tunnel_host),('127.0.0.1',28979,'api.openai.com'))
        self.assertTrue(c._context.check_hostname)
        b=h.connection('dashscope.aliyuncs.com',5);self.assertEqual(b.host,'dashscope.aliyuncs.com');self.assertIsNone(b._tunnel_host)
