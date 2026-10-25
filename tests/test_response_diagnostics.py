"""Synthetic structure failures; no real response, source content, or credentials."""
import copy
import json
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from agent_memory.response_diagnostics import chat_contract_branch,describe_reply,DiagnosticHTTP
from agent_memory.provider_http import BudgetedJSONTransport,HTTPReply,OPENAI_ENDPOINT,ProviderFailure
from agent_memory.budget import BudgetLedger,VerifiedRate

def response():
    return dict(model='gpt-4o-mini-2024-07-18',usage=dict(prompt_tokens=40,completion_tokens=10,total_tokens=50),
                choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(summary='PRIVATE_UNIT_SOURCE',evidence_quotes=[]))))])

class ResponseDiagnosticTests(unittest.TestCase):
    def test_branch_partition_matches_frozen_transport(self):
        cases=[]
        for label,mutate in [
            ('choices_shape',lambda d:d.update(choices=[])),
            ('finish_length',lambda d:d['choices'][0].update(finish_reason='length')),
            ('finish_content_filter',lambda d:d['choices'][0].update(finish_reason='content_filter')),
            ('finish_other',lambda d:d['choices'][0].update(finish_reason='ARBITRARY_PRIVATE_TEXT')),
            ('message_shape',lambda d:d['choices'][0].update(message=None)),
            ('message_refusal',lambda d:d['choices'][0]['message'].update(refusal='PRIVATE_REASON')),
            ('message_tool_calls',lambda d:d['choices'][0]['message'].update(tool_calls=[dict(id='PRIVATE')])),
            ('content_not_json',lambda d:d['choices'][0]['message'].update(content='PRIVATE_NOT_JSON')),
            ('content_not_object',lambda d:d['choices'][0]['message'].update(content='[]')),
            ('summary_type_or_missing',lambda d:d['choices'][0]['message'].update(content='{"evidence_quotes":[]}')),
            ('quotes_type_or_missing',lambda d:d['choices'][0]['message'].update(content='{"summary":"PRIVATE"}')),
            ('quote_item_type',lambda d:d['choices'][0]['message'].update(content='{"summary":"PRIVATE","evidence_quotes":[0]}'))]:
            data=response();mutate(data);cases.append((label,data))
        with tempfile.TemporaryDirectory() as directory:
            ledger=BudgetLedger(Path(directory)/'budget.sqlite')
            client=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,
                rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat()),
                http=type('Fake',(),{'is_fake_http':True})(),mode='offline_fake')
            for label,data in cases:
                with self.subTest(label=label):
                    self.assertEqual(chat_contract_branch(data),label)
                    self.assertRaises(ProviderFailure,client._contract,data,{})
                    self.assertNotIn('PRIVATE',json.dumps(describe_reply('gpt-4o-mini',HTTPReply(200,json.dumps(data).encode()))))
            self.assertEqual(chat_contract_branch(response()),'accepted_frozen_chat_contract')
            client._contract(response(),{})
    def test_bad_raw_json_is_finite_and_redacted(self):
        for raw,branch in ((b'PRIVATE_NOT_JSON','response_not_json'),(b'\xff','response_not_json'),(b'[]','response_not_object')):
            r=describe_reply('gpt-4o-mini',HTTPReply(200,raw));self.assertEqual(r['branch'],branch)
            self.assertNotIn('PRIVATE',json.dumps(r))
    def test_wrapper_one_exchange_same_bytes_and_no_headers_in_sink(self):
        reply=HTTPReply(200,json.dumps(response()).encode());calls=[];records=[]
        class Fake:
            is_fake_http=True
            def exchange(self,*args):calls.append(args);return reply
        wrapper=DiagnosticHTTP(Fake(),'gpt-4o-mini',records.append)
        actual=wrapper.exchange(OPENAI_ENDPOINT,b'PRIVATE_SOURCE',{'Authorization':'PRIVATE_KEY'},1,1024)
        self.assertIs(actual,reply);self.assertEqual(len(calls),1);self.assertEqual(len(records),1)
        self.assertNotIn('PRIVATE',json.dumps(records))
    def test_diagnostic_sink_failure_never_resends(self):
        calls=[]
        class Fake:
            is_fake_http=True
            def exchange(self,*args):calls.append(1);return HTTPReply(200,b'{}')
        def failure(_):raise OSError('unit sink failure')
        wrapper=DiagnosticHTTP(Fake(),'gpt-4o-mini',failure)
        self.assertRaises(OSError,wrapper.exchange,OPENAI_ENDPOINT,b'{}',{},1,1024)
        self.assertEqual(len(calls),1)
    def test_optional_private_response_cache_excludes_errors_and_unexpected_fields(self):
        cache=[];data=response();data['unexpected_header_echo']='PRIVATE_KEY'
        class Fake:
            is_fake_http=True
            status=200
            def exchange(self,*args):return HTTPReply(self.status,json.dumps(data).encode())
        http=Fake();wrapper=DiagnosticHTTP(http,'gpt-4o-mini',lambda _:None,cache.append)
        wrapper.exchange(OPENAI_ENDPOINT,b'{}',{},1,1024)
        self.assertEqual(set(cache[0]),{'model','usage','choices'});self.assertNotIn('PRIVATE_KEY',json.dumps(cache))
        http.status=500;wrapper.exchange(OPENAI_ENDPOINT,b'{}',{},1,1024);self.assertEqual(len(cache),1)

if __name__=='__main__':unittest.main()
