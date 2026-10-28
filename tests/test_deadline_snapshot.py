"""Deterministic worker barriers; no network, secrets or real waiting deadline."""
import json
import queue
import threading
import unittest
from unittest.mock import patch

from agent_memory.provider_http import NativeHTTPS,ProviderFailure,OPENAI_ENDPOINT


class DeadlineSnapshotTests(unittest.TestCase):
    def check_timeout(self,stop_at):
        entered=threading.Event();release=threading.Event();closed=threading.Event()
        posts=[];real_queue=queue.Queue
        def barrier(stage):
            if stop_at==stage:
                entered.set()
                if not release.wait(3):raise AssertionError('test worker was not released')
        class DeadlineQueue(real_queue):
            def get(self,timeout=None):
                if not entered.wait(2):raise AssertionError('expected worker stage not reached')
                raise queue.Empty()
        class Sock:
            def settimeout(self,value):pass
        class Response:
            status=200
            reads=0
            def getheader(self,*args):return 'application/json'
            def read1(self,n):
                self.reads+=1
                if self.reads==1:return b'PRIVATE_BODY_FRAGMENT'
                barrier('response_body');return b''
        class Connection:
            def __init__(self,*args,**kwargs):self.sock=Sock()
            def connect(self):barrier('tls_setup')
            def request(self,*args,**kwargs):posts.append(1);barrier('post_started')
            def getresponse(self):barrier('response_headers');return Response()
            def close(self):closed.set()
        transport=NativeHTTPS()
        with patch('agent_memory.provider_http.queue.Queue',DeadlineQueue),patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
            try:
                with self.assertRaises(ProviderFailure) as caught:
                    transport.exchange(OPENAI_ENDPOINT,b'PRIVATE_INPUT',{'Authorization':'PRIVATE_TOKEN'},1,1024)
                error=caught.exception
                self.assertEqual(error.category,'timeout');self.assertEqual(error.phase,stop_at)
                self.assertEqual(error.transport_kind,'TimeoutError')
                expected=21 if stop_at=='response_body' else 0
                self.assertEqual(error.response_diagnostics,{'received_bytes':expected,'response_status':200 if expected else None,'sse_done_frame':False,'sse_complete':False})
                self.assertNotIn('PRIVATE',str(error)+json.dumps(error.response_diagnostics))
                # The old worker still owns its slot. A later call cannot race it.
                with self.assertRaises(ProviderFailure) as busy:transport.exchange(OPENAI_ENDPOINT,b'new',{},1,1024)
                self.assertEqual(busy.exception.category,'network_error')
            finally:
                release.set();self.assertTrue(closed.wait(2))
        self.assertEqual(len(posts),0 if stop_at=='tls_setup' else 1)

    def test_dns_or_tls_deadline_preserves_stage_and_prevents_late_post(self):self.check_timeout('tls_setup')
    def test_post_deadline_records_no_delivery_claim_or_replay(self):self.check_timeout('post_started')
    def test_header_deadline_records_stage_without_status_or_body(self):self.check_timeout('response_headers')
    def test_body_deadline_keeps_only_scalar_progress_and_still_fails(self):self.check_timeout('response_body')


if __name__=='__main__':unittest.main()
