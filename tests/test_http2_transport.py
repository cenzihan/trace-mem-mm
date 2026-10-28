import importlib.util
import json
import ssl
import threading
import time
import unittest
from unittest.mock import patch

from agent_memory.http2_transport import HTTP2AcademicHTTPS
from agent_memory.provider_http import OPENAI_ENDPOINT,BAILIAN_BEIJING_ENDPOINT,ProviderFailure,HTTPReply
from agent_memory.chat_stream import assemble_chat_stream
from test_chat_stream import events,wire


@unittest.skipUnless(importlib.util.find_spec('httpx') and importlib.util.find_spec('h2'),'optional pinned HTTP2 dependencies')
class HTTP2TransportTests(unittest.TestCase):
    body=json.dumps({'model':'gpt-4o-mini','stream':True,'stream_options':{'include_usage':True}}).encode()

    def network(self,mode='ok',raw=None,status=200):
        import h2.connection,h2.config,h2.events
        raw=wire(events()) if raw is None else raw
        class Network:
            def __init__(self):
                self.h2=h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=False))
                self.h2.initiate_connection();self.pending=self.h2.data_to_send()
                self.posts=0;self.closed=False;self.tls_verified=False;self.connects=0;self.request_headers=[]
            def write(self,data,timeout=None):
                for event in self.h2.receive_data(data):
                    if isinstance(event,h2.events.RequestReceived):
                        self.posts+=1;self.request_headers.append(dict(event.headers))
                    if isinstance(event,h2.events.DataReceived):
                        self.h2.acknowledge_received_data(event.flow_controlled_length,event.stream_id)
                    if isinstance(event,h2.events.StreamEnded):
                        if mode=='headers_gone':continue
                        if mode=='goaway':
                            self.h2.close_connection(error_code=0,last_stream_id=0);continue
                        self.h2.send_headers(event.stream_id,[(b':status',str(status).encode()),
                            (b'content-type',b'text/event-stream')])
                        content=raw[:50] if mode=='partial' else raw
                        for i in range(0,len(content),16000):
                            self.h2.send_data(event.stream_id,content[i:i+16000],
                                end_stream=(mode=='ok' and i+16000>=len(content)))
                self.pending+=self.h2.data_to_send()
            def read(self,max_bytes,timeout=None):
                data=self.pending[:max_bytes];self.pending=self.pending[max_bytes:];return data
            def close(self):self.closed=True
            def start_tls(self,ssl_context,server_hostname=None,timeout=None):
                self.tls_verified=(ssl_context.verify_mode==ssl.CERT_REQUIRED and ssl_context.check_hostname
                                   and server_hostname=='api.openai.com')
                return self
            def get_extra_info(self,info):
                if info=='ssl_object':return self
                if info=='is_readable':return bool(self.pending)
                return None
            def selected_alpn_protocol(self):return 'h2'
        net=Network()
        def connect(*args,**kwargs):net.connects+=1;return net
        mock=patch('httpcore._backends.sync.SyncBackend.connect_tcp',side_effect=connect)
        mock.start();self.addCleanup(mock.stop)
        client=HTTP2AcademicHTTPS();self.addCleanup(client.close)
        return client,net

    def exchange(self,client,**kwargs):
        return client.exchange(OPENAI_ENDPOINT,self.body,{'Authorization':'Bearer SYNTHETIC_ONLY'},
                               kwargs.get('timeout',2),kwargs.get('limit',8192))

    def test_real_http2_frames_reuse_one_connection_and_validate_models(self):
        client,net=self.network()
        for _ in range(2):
            reply=self.exchange(client)
            self.assertEqual(assemble_chat_stream(reply.body)['model'],'gpt-4o-mini-2024-07-18')
        self.assertTrue(net.tls_verified)
        self.assertEqual((net.posts,net.connects),(2,1))
        self.assertEqual(client.counters(),dict(gpt_posts_started=2,http2_responses=2,
                                               complete_sse_responses=2,new_connections=1))

    def test_client_request_id_survives_http2_frame_encoding(self):
        client,net=self.network()
        request_id='attempt_'+'a'*32
        client.exchange(OPENAI_ENDPOINT,self.body,
            {'Authorization':'Bearer SYNTHETIC_ONLY','X-Client-Request-Id':request_id},2,8192)
        self.assertEqual(net.posts,1)
        self.assertEqual(net.request_headers[0][b'x-client-request-id'],request_id.encode())

    def test_header_disconnect_stops_without_post_retry_or_sensitive_error(self):
        client,net=self.network('headers_gone')
        with self.assertRaises(ProviderFailure) as caught:self.exchange(client)
        self.assertEqual(caught.exception.phase,'response_headers')
        self.assertEqual(caught.exception.transport_kind,'HTTPXRemoteProtocolError')
        self.assertEqual(caught.exception.response_diagnostics['received_bytes'],0)
        with self.assertRaises(ProviderFailure):self.exchange(client)
        self.assertEqual(net.posts,1);self.assertTrue(net.closed)
        self.assertNotIn('SYNTHETIC',str(caught.exception))

    def test_partial_body_cannot_be_accepted(self):
        client,net=self.network('partial')
        with self.assertRaises(ProviderFailure) as caught:self.exchange(client)
        self.assertEqual(caught.exception.phase,'response_body')
        self.assertFalse(caught.exception.response_diagnostics['sse_complete'])
        self.assertEqual(net.posts,1)

    def test_semantically_complete_but_http2_interrupted_is_still_unknown(self):
        client,net=self.network('no_end')
        with self.assertRaises(ProviderFailure) as caught:self.exchange(client)
        self.assertTrue(caught.exception.response_diagnostics['sse_complete'])
        self.assertEqual(caught.exception.response_diagnostics['response_status'],200)
        self.assertEqual(net.posts,1)

    def test_goaway_cannot_cause_a_second_post(self):
        client,net=self.network('goaway')
        with self.assertRaises(ProviderFailure):self.exchange(client)
        self.assertEqual(net.posts,1)

    def test_response_byte_limit_is_enforced_before_decode(self):
        client,net=self.network(raw=wire(events('x'*5000)))
        with self.assertRaises(ProviderFailure) as caught:self.exchange(client,limit=1024)
        self.assertEqual(caught.exception.category,'response_limit')
        self.assertEqual(net.posts,1)

    def test_incomplete_sse_with_clean_http2_end_fails_model_contract(self):
        client,net=self.network(raw=wire(events(),False))
        reply=self.exchange(client)
        with self.assertRaises(ProviderFailure):assemble_chat_stream(reply.body)
        self.assertEqual(client.counters()['complete_sse_responses'],0)

    def test_redirect_returns_once_and_is_never_followed(self):
        client,net=self.network(status=307)
        self.assertEqual(self.exchange(client).status,307)
        self.assertEqual(net.posts,1)

    def test_embedding_uses_unchanged_native_transport(self):
        client,net=self.network()
        with patch.object(client._embedding,'exchange',return_value=HTTPReply(200,b'{}')) as old:
            reply=client.exchange(BAILIAN_BEIJING_ENDPOINT,b'{}',{},2,1024)
        self.assertEqual(reply.status,200);old.assert_called_once()
        self.assertEqual(net.posts,0)

    def test_unapproved_endpoint_or_nonstream_payload_never_connects(self):
        client,net=self.network()
        for endpoint,body in [('https://example.org/v1/chat/completions',self.body),
                              (OPENAI_ENDPOINT,b'{"model":"other"}')]:
            with self.assertRaises(ValueError):client.exchange(endpoint,body,{},2,1024)
        self.assertEqual(net.posts,0);self.assertEqual(net.connects,0)

    def test_deadline_cancellation_prevents_late_post_and_keeps_slot_busy(self):
        client,net=self.network()
        entered=threading.Event();release=threading.Event();done=threading.Event()
        def stall(*args,**kwargs):
            entered.set();release.wait(2);done.set();return net
        with patch('httpcore._backends.sync.SyncBackend.connect_tcp',side_effect=stall):
            with self.assertRaises(ProviderFailure) as caught:self.exchange(client,timeout=.05)
            self.assertTrue(entered.is_set());self.assertEqual(caught.exception.phase,'deadline_unknown')
            with self.assertRaises(ProviderFailure):self.exchange(client)
            release.set();self.assertTrue(done.wait(1))
            until=time.monotonic()+1
            while not net.closed and time.monotonic()<until:time.sleep(.005)
        self.assertEqual(net.posts,0);self.assertTrue(net.closed)

    def test_close_prevents_further_network_use(self):
        client,net=self.network();self.exchange(client);client.close()
        with self.assertRaises(ProviderFailure):self.exchange(client)
        self.assertEqual(net.posts,1);self.assertTrue(net.closed)


if __name__=='__main__':unittest.main()
