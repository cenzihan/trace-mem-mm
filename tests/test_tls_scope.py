import hashlib
import json
import ssl
import unittest
from unittest.mock import patch

from agent_memory.provider_http import NativeHTTPS,ProviderFailure,OPENAI_ENDPOINT
from agent_memory.service_transport import ServiceTransport
import test_service_transport as fixtures


class TLSSetupTests(unittest.TestCase):
    def test_only_setup_repeats_one_post(self):
        class Sock:
            def settimeout(self,value):pass
        class Response:
            status=200
            def read1(self,n):return b''
            def getheader(self,*args):return ''
        class Connection:
            setups=0;posts=0
            def __init__(self,*args,**kwargs):self.sock=Sock()
            def connect(self):
                Connection.setups+=1
                if Connection.setups<3:raise ssl.SSLError('synthetic handshake')
            def request(self,*args,**kwargs):Connection.posts+=1
            def getresponse(self):return Response()
            def close(self):pass
        with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
            result=NativeHTTPS(handshake_attempts=3).exchange(OPENAI_ENDPOINT,b'{}',{},2,1024)
        self.assertEqual(result.status,200);self.assertEqual(Connection.posts,1);self.assertEqual(Connection.setups,3)
    def test_response_failure_and_certificate_failure_never_repeat(self):
        for error_phase in ['response','certificate']:
            class Sock:
                def settimeout(self,value):pass
            class Connection:
                setups=0;posts=0
                def __init__(self,*args,**kwargs):self.sock=Sock()
                def connect(self):
                    Connection.setups+=1
                    if error_phase=='certificate':raise ssl.SSLCertVerificationError('synthetic cert')
                def request(self,*args,**kwargs):Connection.posts+=1
                def getresponse(self):raise ssl.SSLError('synthetic response')
                def close(self):pass
            with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
                with self.assertRaises(ProviderFailure) as caught:NativeHTTPS(handshake_attempts=3).exchange(OPENAI_ENDPOINT,b'{}',{},2,1024)
            self.assertEqual(Connection.setups,1)
            self.assertEqual(Connection.posts,1 if error_phase=='response' else 0)
            self.assertEqual(caught.exception.phase,'response_headers' if error_phase=='response' else 'tls_setup')


class QuarantineTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ServiceTransportTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.transport=self.fixture.make();self.fixture.http.failure=True
        with self.assertRaises(ProviderFailure):self.fixture.call(self.transport)
        self.fixture.http.failure=False
        with self.transport.journal.connect() as db:self.row=dict(db.execute("SELECT * FROM operations WHERE state='unknown'").fetchone())
    def proof(self):
        from agent_memory.embedding_executor import _json
        return {'fingerprint':self.row['fingerprint'],'attempt_id':self.row['attempt_id'],
            'scope':self.row['scope'],'row_sha256':hashlib.sha256(_json(self.row).encode()).hexdigest(),
            'blocked_user_sha256':hashlib.sha256(b'retired-user').hexdigest(),'reason':'explicit permanent retirement of synthetic failed scope'}
    def test_unknown_not_ignored_and_retirement_never_replays(self):
        original_used=self.transport.ledger.totals()['accounted_micro_cny']
        self.transport.receipt_scope='independent-reviewed-version'
        with self.assertRaises(ValueError):self.transport.ready()
        self.transport.quarantine_path.write_text(json.dumps(self.proof())+'\n')
        self.transport.ready()
        with self.assertRaises(ValueError):self.transport.assert_source_allowed({'user_id':'retired-user'})
        with self.assertRaises(ValueError):self.fixture.call(self.transport)
        self.assertEqual(self.fixture.http.calls,1)
        self.fixture.payload['input']=['A different independent synthetic source.']
        self.fixture.call(self.transport)
        self.assertEqual(self.fixture.http.calls,2)
        self.assertEqual(self.transport.journal.row(self.row['fingerprint'])['state'],'unknown')
        self.assertGreaterEqual(self.transport.ledger.totals()['accounted_micro_cny'],original_used)
        self.transport.receipt_scope=self.row['scope']
        with self.assertRaises(ValueError):self.transport.ready()
    def test_tamper_blocks_independent_traffic(self):
        proof=self.proof();proof['row_sha256']='0'*64
        self.transport.quarantine_path.write_text(json.dumps(proof)+'\n');self.transport.receipt_scope='new'
        with self.assertRaises(ValueError):self.transport.ready()
