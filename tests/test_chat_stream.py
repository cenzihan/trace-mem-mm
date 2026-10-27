import json
import http.client
import io
import unittest
from unittest.mock import patch

from agent_memory.chat_stream import assemble_chat_stream,CompleteChatStream
from agent_memory.mem0_transport import Mem0ServiceTransport
from agent_memory.provider_http import HTTPReply,NativeHTTPS,ProviderFailure,OPENAI_ENDPOINT,safe_failure_kind
import test_service_transport as fixtures


def events(text='{"memory":[{"text":"A synthetic teal notebook."}]}'):
    base={'id':'synthetic','model':'gpt-4o-mini-2024-07-18','usage':None}
    rows=[dict(base,choices=[{'index':0,'delta':{'role':'assistant','content':text[:12]},'finish_reason':None}]),
          dict(base,choices=[{'index':0,'delta':{'content':text[12:]},'finish_reason':None}]),
          dict(base,choices=[{'index':0,'delta':{},'finish_reason':'stop'}]),
          dict(base,choices=[],usage={'prompt_tokens':20,'completion_tokens':10,'total_tokens':30})]
    return rows


def wire(rows,done=True):
    return ('\n\n'.join('data: '+json.dumps(row) for row in rows)+
        ('\n\ndata: [DONE]\n\n' if done else '\n\n')).encode()


class StreamTests(unittest.TestCase):
    def test_unicode_and_usage_reconstruct_exactly(self):
        text='{"memory":[{"text":"独立合成记忆。"}]}'
        result=assemble_chat_stream(wire(events(text)).replace(b'\n',b'\r\n'))
        self.assertEqual(result['choices'][0]['message']['content'],text)
        self.assertEqual(result['usage']['total_tokens'],30)
        self.assertEqual(result['model'],'gpt-4o-mini-2024-07-18')

    def test_partial_stream_cannot_be_success(self):
        for raw in [wire(events(),False),wire(events()[:-1]),wire(events()[:2]),b'data: {bad}\n\n']:
            with self.subTest(raw_length=len(raw)):
                with self.assertRaises(ProviderFailure):assemble_chat_stream(raw)

    def test_identity_finish_and_tools_are_strict(self):
        mutations=[
            lambda r:r[1].update(model='unapproved-model'),
            lambda r:r[1].update(id='different'),
            lambda r:r[2]['choices'][0].update(finish_reason='length'),
            lambda r:r[1]['choices'][0]['delta'].update(refusal='synthetic refusal'),
            lambda r:r[1]['choices'][0]['delta'].update(tool_calls=[{}]),
            lambda r:r[1]['choices'][0].update(index=1),
        ]
        for mutate in mutations:
            r=events();mutate(r)
            with self.assertRaises(ProviderFailure):assemble_chat_stream(wire(r))
        with self.assertRaises(ProviderFailure):assemble_chat_stream(wire(events())+wire(events()))

    def test_received_invalid_stream_retains_receipt_and_stops(self):
        fixture=fixtures.ServiceTransportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        class FakeStream:
            is_fake_http=True
            def __init__(self):self.calls=0
            def exchange(self,endpoint,body,headers,timeout,limit):
                self.calls+=1
                payload=json.loads(body)
                assert payload['stream'] and payload['stream_options']=={'include_usage':True}
                assert payload['model']=='gpt-4o-mini' and payload['store'] is False
                return HTTPReply(200,wire(events(),False),{'content-type':'text/event-stream'})
        http=FakeStream()
        transport=Mem0ServiceTransport(fixture.p/'budget.sqlite',fixture.authority,fixture.prices,
            fixture.p/'stream-receipts',{'gpt-4o-mini':lambda:'SYNTHETIC'},http=http,offline_test=True)
        payload={'model':'gpt-4o-mini','messages':[{'role':'system','content':'Return JSON memory.'},{'role':'user','content':'Synthetic teal notebook.'}],
            'store':False,'n':1,'max_completion_tokens':512,'response_format':{'type':'json_object'}}
        with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,payload)
        with transport.journal.connect() as db:
            row=db.execute('SELECT state,status,body FROM operations').fetchone()
        self.assertEqual(row['state'],'failed');self.assertEqual(row['status'],200);self.assertTrue(row['body'])
        with self.assertRaises(ValueError):transport(OPENAI_ENDPOINT,payload)
        self.assertEqual(http.calls,1)

    def test_success_uses_original_ledger_and_minimizes_wire(self):
        fixture=fixtures.ServiceTransportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        class FakeStream:
            is_fake_http=True
            def exchange(self,*args):
                return HTTPReply(200,wire(events()),{'content-type':'text/event-stream; charset=utf-8'})
        transport=Mem0ServiceTransport(fixture.p/'budget.sqlite',fixture.authority,fixture.prices,
            fixture.p/'success-receipts',{'gpt-4o-mini':lambda:'SYNTHETIC'},http=FakeStream(),offline_test=True)
        payload={'model':'gpt-4o-mini','messages':[{'role':'system','content':'Return JSON memory.'},{'role':'user','content':'Synthetic teal notebook.'}],
            'store':False,'n':1,'max_completion_tokens':512,'response_format':{'type':'json_object'}}
        result=transport(OPENAI_ENDPOINT,payload)
        self.assertIn('memory',json.loads(result['choices'][0]['message']['content']))
        with transport.journal.connect() as db:row=db.execute('SELECT * FROM operations').fetchone()
        self.assertEqual(row['state'],'complete');self.assertIsNone(row['body'])
        self.assertEqual(transport.ledger.totals()['attempts'],1)
        manifest=json.loads(row['manifest'])
        self.assertTrue(manifest['streaming']);self.assertGreater(manifest['request_bytes'],0)
        self.assertNotIn('Synthetic teal notebook',row['manifest'])

    def test_post_header_disconnect_is_not_repeated_or_leaked(self):
        class Sock:
            def settimeout(self,value):pass
        class Connection:
            posts=0
            def __init__(self,*args,**kwargs):self.sock=Sock()
            def connect(self):pass
            def request(self,*args,**kwargs):Connection.posts+=1
            def getresponse(self):raise http.client.RemoteDisconnected('SECRET_SOURCE_OR_TOKEN')
            def close(self):pass
        with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
            with self.assertRaises(ProviderFailure) as caught:
                NativeHTTPS(handshake_attempts=3).exchange(OPENAI_ENDPOINT,b'{}',{},2,1024)
        self.assertEqual(Connection.posts,1)
        self.assertEqual(caught.exception.phase,'response_headers')
        self.assertEqual(caught.exception.transport_kind,'RemoteDisconnected')
        self.assertNotIn('SECRET',str(caught.exception))


    def test_error_classification_never_keeps_message(self):
        error=http.client.RemoteDisconnected('SECRET_SOURCE_OR_TOKEN')
        self.assertEqual(safe_failure_kind(error),'RemoteDisconnected')
        self.assertNotIn('SECRET',safe_failure_kind(error))


class StreamCompletionTests(unittest.TestCase):
    def test_only_evidence_factory_enables_new_transport_policy(self):
        from agent_memory.mem0_app import create_evidence_app
        from agent_memory.mem0_transport import Mem0EvidenceTransport,Mem0ServiceTransport
        from agent_memory.mem0_evidence import Mem0EvidenceSystem
        from agent_memory.http2_transport import HTTP2AcademicHTTPS
        with patch('agent_memory.mem0_app.create_app') as factory:
            env={'TRACEMEM_MEM0_EVIDENCE_ENABLED':'approved-native-mem0-evidence'}
            create_evidence_app(env)
        factory.assert_called_once_with(env,_system_type=Mem0EvidenceSystem,
            _transport_type=Mem0EvidenceTransport,_http_type=HTTP2AcademicHTTPS)
        self.assertEqual(Mem0ServiceTransport.receipt_scope,'native-mem0-v3.0.3')
        self.assertEqual(Mem0EvidenceTransport.receipt_scope,'native-mem0-v3.1.2')

    def test_unknown_diagnostics_remain_scalar_and_old_call_cannot_replay(self):
        fixture=fixtures.ServiceTransportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        class FailingHTTP:
            is_fake_http=True
            def __init__(self):self.calls=0
            def exchange(self,*args):
                self.calls+=1
                exc=ProviderFailure('network_error',phase='response_body')
                exc.transport_kind='IncompleteRead'
                exc.response_diagnostics={'received_bytes':81,'response_status':200,
                    'sse_done_frame':False,'sse_complete':False,
                    'unapproved':'SECRET_SOURCE','body':'SECRET_SOURCE'}
                raise exc
        http=FailingHTTP()
        transport=Mem0ServiceTransport(fixture.p/'budget.sqlite',fixture.authority,fixture.prices,
            fixture.p/'diagnostic-receipts',{'gpt-4o-mini':lambda:'SYNTHETIC'},http=http,offline_test=True)
        payload={'model':'gpt-4o-mini','messages':[{'role':'system','content':'Return JSON memory.'},
            {'role':'user','content':'Synthetic notebook.'}],'store':False,'n':1,
            'max_completion_tokens':512,'response_format':{'type':'json_object'}}
        with self.assertRaises(ProviderFailure):transport(OPENAI_ENDPOINT,payload)
        with self.assertRaises(ValueError):transport(OPENAI_ENDPOINT,payload)
        self.assertEqual(http.calls,1)
        with transport.journal.connect() as db:row=db.execute('SELECT * FROM operations').fetchone()
        self.assertEqual(row['state'],'unknown');self.assertIsNone(row['body']);self.assertIsNone(row['status'])
        diagnostic=(fixture.p/'diagnostic-receipts/transport-failure-phases.jsonl').read_text()
        self.assertNotIn('SECRET_SOURCE',diagnostic)
        self.assertEqual(json.loads(diagnostic)['received_bytes'],81)

    def exchange(self,raw,*,enabled=True,content_type='text/event-stream',status=200,stream=True,limit=8192):
        # Real Python HTTP/1.1 chunk decoder, with the closing zero chunk missing.
        # This models a transport envelope truncated AFTER the semantic response.
        wire_http=(f'HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\n'
                   'Transfer-Encoding: chunked\r\n\r\n').encode()
        wire_http+=f'{len(raw):x}\r\n'.encode()+raw+b'\r\n'
        class Sock:
            def makefile(self,*args):return io.BytesIO(wire_http)
            def settimeout(self,*args):pass
        class Connection:
            posts=0;closed=False
            def __init__(self,*args,**kwargs):self.sock=Sock()
            def connect(self):pass
            def request(self,*args,**kwargs):Connection.posts+=1
            def getresponse(self):
                response=http.client.HTTPResponse(self.sock);response.begin();return response
            def close(self):Connection.closed=True
        payload=json.dumps({'model':'gpt-4o-mini','stream':stream,
                            'stream_options':{'include_usage':True}}).encode()
        try:
            with patch('agent_memory.provider_http.http.client.HTTPSConnection',Connection):
                return NativeHTTPS(stop_at_chat_done=enabled).exchange(
                    OPENAI_ENDPOINT,payload,{},2,limit)
        finally:
            self.assertEqual(Connection.posts,1)
            self.assertTrue(Connection.closed)

    def test_complete_sse_does_not_wait_for_missing_http_terminator(self):
        raw=wire(events())
        result=self.exchange(raw)
        self.assertEqual(result.body,raw)
        self.assertEqual(assemble_chat_stream(result.body)['usage']['total_tokens'],30)

    def test_default_policy_keeps_legacy_read_to_eof(self):
        with self.assertRaises(ProviderFailure) as caught:self.exchange(wire(events()),enabled=False)
        self.assertEqual(caught.exception.transport_kind,'IncompleteRead')
        self.assertEqual(caught.exception.phase,'response_body')

    def test_incomplete_sse_remains_unknown_with_scalar_diagnostics(self):
        raw=wire(events(),False)
        with self.assertRaises(ProviderFailure) as caught:self.exchange(raw)
        self.assertEqual(caught.exception.transport_kind,'IncompleteRead')
        self.assertEqual(caught.exception.response_diagnostics,{
            'received_bytes':len(raw),'response_status':200,'sse_done_frame':False,'sse_complete':False})
        self.assertNotIn('synthetic',str(caught.exception).lower())

    def test_done_without_usage_or_finish_cannot_end_http_read(self):
        for rows in (events()[:-1],events()[:2]):
            with self.subTest(rows=len(rows)):
                with self.assertRaises(ProviderFailure) as caught:self.exchange(wire(rows))
                self.assertFalse(caught.exception.response_diagnostics['sse_complete'])
                self.assertTrue(caught.exception.response_diagnostics['sse_done_frame'])

    def test_bad_usage_cannot_be_a_complete_stream(self):
        for usage in ({},{'prompt_tokens':20,'completion_tokens':10,'total_tokens':31},
                      {'prompt_tokens':True,'completion_tokens':10,'total_tokens':11},
                      {'prompt_tokens':-1,'completion_tokens':10,'total_tokens':9}):
            rows=events();rows[-1]['usage']=usage
            self.assertFalse(CompleteChatStream().feed(wire(rows)))
            with self.assertRaises(ProviderFailure):assemble_chat_stream(wire(rows))

    def test_done_marker_inside_content_is_not_termination(self):
        raw=wire(events('data: [DONE]\n\n'),False)
        tracker=CompleteChatStream()
        self.assertFalse(tracker.feed(raw));self.assertFalse(tracker.done_frame)

    def test_every_byte_boundary_handles_split_utf8_and_crlf(self):
        raw=wire(events('{"memory":[{"text":"独立合成样例"}]}')).replace(b'\n',b'\r\n')
        for split in range(1,len(raw)):
            tracker=CompleteChatStream()
            self.assertFalse(tracker.feed(raw[:split]),split)
            self.assertTrue(tracker.feed(raw[split:]),split)

    def test_done_requires_full_event_boundary(self):
        raw=wire(events()).rstrip(b'\n')
        tracker=CompleteChatStream()
        self.assertFalse(tracker.feed(raw));self.assertFalse(tracker.feed(b'\n'))
        self.assertTrue(tracker.feed(b'\n'))

    def test_wrong_response_or_request_does_not_enable_early_end(self):
        for kwargs in ({'content_type':'application/json'},{'status':503},{'stream':False}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ProviderFailure):self.exchange(wire(events()),**kwargs)

    def test_size_cap_and_extra_event_are_not_silently_accepted(self):
        with self.assertRaises(ProviderFailure) as caught:self.exchange(wire(events()),limit=128)
        self.assertEqual(caught.exception.category,'response_limit')
        with self.assertRaises(ProviderFailure):self.exchange(wire(events())+wire(events()))

