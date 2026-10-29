import json
import unittest
from unittest.mock import patch
import test_service_transport as fixture
from agent_memory.recoverable_options import RetrySelectionTransport
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, ProviderFailure

class RetryTransportTests(unittest.TestCase):
    setUp = fixture.ServiceTransportTests.setUp

    def make(self):
        t=RetrySelectionTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},http=self.http,offline_test=True)
        t.retry_delays=(0,0,0,0,0)
        return t

    def call(self,t):
        from agent_memory.scoped_unknown import request_owner
        with request_owner('synthetic-owner'):return t(BAILIAN_BEIJING_ENDPOINT,self.payload)

    def test_timeout_then_success_has_two_receipts_and_full_unknown_hold(self):
        t=self.make(); original=self.http.exchange; n=0
        def exchange(*a):
            nonlocal n
            n+=1
            if n==1:raise ProviderFailure('timeout')
            return original(*a)
        with patch.object(self.http,'exchange',side_effect=exchange):self.call(t)
        with t.journal.connect() as db:self.assertEqual([r[0] for r in db.execute('SELECT state FROM operations ORDER BY rowid')],['unknown','complete'])
        with self.ledger.connect() as db:
            a=db.execute('SELECT accounted,reserved FROM attempts ORDER BY rowid').fetchall()
        self.assertEqual(len(a),2);self.assertEqual(a[0][0],a[0][1])

    def test_exhaustion_is_six_attempts_and_later_request_can_succeed(self):
        t=self.make();self.http.failure=True
        with self.assertRaises(ProviderFailure):self.call(t)
        self.assertEqual(self.http.calls,6)
        self.http.failure=False;self.call(self.make())
        self.assertEqual(self.http.calls,7)
        self.assertEqual(self.ledger.totals()['attempts'],7)

    def test_auth_failure_is_not_retried(self):
        t=self.make()
        with patch.object(self.http,'exchange',side_effect=ProviderFailure('authentication')) as http:
            with self.assertRaises(ProviderFailure):self.call(t)
        self.assertEqual(http.call_count,1)

    def test_inflight_is_not_silently_ignored(self):
        t=self.make()
        with t.journal.connect() as db:
            t.journal.install('synthetic',t.receipt_scope,{'keys':['synthetic']})
        with self.assertRaises(ValueError):t.ready()

import tempfile
import importlib.util
from pathlib import Path
from test_mem0_native import FakeEmbedding, FakeTransport, FakeCards
from agent_memory.recoverable_options import RecoverableOptionsSystem

@unittest.skipUnless(importlib.util.find_spec('mem0'), 'pinned Mem0 required')
class RetryAddTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.e=FakeEmbedding();self.t=FakeTransport()
        self.store=RecoverableOptionsSystem(Path(self.temp.name)/'memory.sqlite',embedding=self.e,transport=self.t,card_adapter=FakeCards())
        self.payload=dict(user_id='synthetic',request_id='one',session_id='s',messages=[dict(role='user',content='first synthetic'),dict(role='user',content='second synthetic')])

    def tearDown(self):
        self.store.close()
        conn=getattr(self.store.native.db,'connection',None)
        if conn:conn.close()
        self.temp.cleanup()

    def test_partial_failure_cleans_vectors_then_same_add_succeeds_once(self):
        original=self.e.embed; n=0
        def fail(texts):
            nonlocal n
            n+=1
            if n==2:raise RuntimeError('synthetic failure')
            return original(texts)
        with patch.object(self.e,'embed',side_effect=fail):
            with self.assertRaises(Exception):self.store.add(self.payload)
        with self.store.connect() as db:
            self.assertEqual(db.execute('select count(*) from memories').fetchone()[0],0)
            self.assertEqual(db.execute('select count(*) from add_operations').fetchone()[0],0)
        vs=self.store.native.vector_store
        self.assertEqual(vs.client.count(collection_name=vs.collection_name).count,0)
        self.store.ready();self.store.add(self.payload)
        before=len(self.e.calls);self.store.add(self.payload)
        self.assertEqual(len(self.e.calls),before)
        self.assertEqual(vs.client.count(collection_name=vs.collection_name).count,2)

    def test_archive_failure_rolls_back_and_preserves_other_success(self):
        other=dict(self.payload,request_id='committed')
        self.store.add(other)
        with patch.object(self.store,'_persist_descriptions',side_effect=RuntimeError('synthetic archive failure')):
            with self.assertRaises(RuntimeError):self.store.add(self.payload)
        self.store.ready()
        with self.store.connect() as db:
            self.assertEqual(db.execute('select count(*) from memories').fetchone()[0],2)
            self.assertEqual(db.execute('select count(*) from requests').fetchone()[0],1)
        vs=self.store.native.vector_store
        self.assertEqual(vs.client.count(collection_name=vs.collection_name).count,2)
        self.store.add(self.payload)
        self.assertEqual(vs.client.count(collection_name=vs.collection_name).count,4)
