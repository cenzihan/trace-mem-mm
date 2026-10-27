import json
import http.client
import unittest
from unittest.mock import patch

from agent_memory.chat_stream import assemble_chat_stream
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

