"""Synthetic role-contract, original-source and bounded accounting regressions."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.evidence_roles import (role_payload, role_indices, checked_payload,
    RoleSelector, RoleSystem, RoleBudgetedTransport, RoleServiceTransport, create_role_app)
from agent_memory.joint_selection import selection_payload, candidate_count
from agent_memory.protocol import content_parts, text_of
from agent_memory.provider_http import ProviderFailure
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding, FakeCards


class RoleFixtureTransport:
    def __init__(self, roles):self.roles=roles;self.calls=[]
    def ready(self):pass
    def __call__(self, endpoint, payload):
        self.calls.append(copy.deepcopy(payload))
        if isinstance(self.roles,Exception):raise self.roles
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.roles)}}]}


class RoleTests(unittest.TestCase):
    def test_identity_and_fact_union_preserves_fact_order_and_deduplicates_roles(self):
        self.assertEqual(role_indices({'identity':[0,2],'facts':[2,1]},3),[2,1,0])
        self.assertEqual(role_indices({'identity':[1],'facts':[1]},2),[1])
        self.assertEqual(role_indices({'identity':[],'facts':[]},2),[])

    def test_contract_rejects_malformed_indices_extra_answers_and_combined_overflow(self):
        for bad in [{'identity':[True],'facts':[]},{'identity':[0,0],'facts':[]},
                    {'identity':[-1],'facts':[]},{'identity':[9],'facts':[]},
                    {'identity':[],'facts':[],'answer':'invented'},
                    {'identity':list(range(5)),'facts':list(range(5,9))},
                    {'selected':[0]}, {'identity':None,'facts':[]}]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):role_indices(bad,9)

    def test_views_pool_and_image_budget_unchanged_from_successful_baseline(self):
        rows=[source('Log note. '*300,True) for _ in range(32)]
        before=copy.deepcopy(rows);parts=[image()]+content_parts('query')
        old=selection_payload(parts,rows,['alternative']);new=role_payload(parts,rows,['alternative'])
        self.assertEqual(new['messages'][1],old['messages'][1])
        self.assertEqual(checked_payload(new),old);self.assertEqual(rows,before)
        t=object.__new__(RoleBudgetedTransport);t.model='gpt-4o-mini';t.output_token_cap=512
        self.assertEqual(t._bounds(new),(128000,512))
        self.assertEqual(candidate_count(checked_payload(new)),32)

    def test_exact_role_response_contract_and_dated_model(self):
        t=object.__new__(RoleBudgetedTransport);t.model='gpt-4o-mini';t.output_token_cap=512
        payload=role_payload(content_parts('q'),[source()])
        result=RoleFixtureTransport({'identity':[0],'facts':[0]})('',payload)
        t._contract(result,payload)
        self.assertEqual(t._response_model({'model':'gpt-4o-mini-2024-07-18'}),'gpt-4o-mini-2024-07-18')
        for change in ['length','tool_calls']:
            bad=copy.deepcopy(result);bad['choices'][0]['finish_reason']=change
            with self.assertRaises(ProviderFailure):t._contract(bad,payload)
        bad=copy.deepcopy(result);bad['choices'][0]['message']['content']='{"identity":[0],"identity":[],"facts":[]}'
        with self.assertRaises(ProviderFailure):t._contract(bad,payload)
        with self.assertRaises(ProviderFailure):t._response_model({'model':'other'})

    def test_single_call_and_no_semantic_claim_or_implicit_completion(self):
        rows=[source('The ochre sail is named Elsin.'),source('Elsin is in the east loft.')]
        t=RoleFixtureTransport({'identity':[0],'facts':[1]})
        self.assertEqual(RoleSelector(t).select(content_parts('Where is the ochre sail?'),rows),[1,0])
        self.assertEqual(len(t.calls),1)
        # A partial but structurally valid result stays visible to independent
        # quality checks instead of being padded, guessed or converted to503.
        self.assertEqual(RoleSelector(RoleFixtureTransport({'identity':[],'facts':[1]})).select(content_parts('q'),rows),[1])

    def test_empty_and_failures_keep_original_retry_accounting_scope(self):
        rows=[source()]
        self.assertEqual(RoleSelector(RoleFixtureTransport({'identity':[],'facts':[]})).select(content_parts('q'),rows),[])
        with self.assertRaises(ProviderFailure):RoleSelector(RoleFixtureTransport(ProviderFailure('network_error'))).select(content_parts('q'),rows)
        self.assertEqual(RoleServiceTransport.receipt_scope,'native-mem0-raw-v4')
        self.assertTrue(RoleServiceTransport.retry_failed_requests)
        self.assertEqual(len(RoleServiceTransport.retry_delays),5)
        with self.assertRaises(ValueError):create_role_app({})


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class RoleNativeTests(unittest.TestCase):
    def test_both_roles_return_complete_originals_and_isolate_other_user(self):
        with tempfile.TemporaryDirectory() as directory:
            class Dynamic(RoleFixtureTransport):
                def __call__(self,endpoint,payload):
                    rows=json.loads(payload['messages'][1]['content'][0]['text'])['candidates']
                    self.roles={'identity':[r['index'] for r in rows if 'named' in r['text_prefix']],
                                'facts':[r['index'] for r in rows if 'loft' in r['text_prefix']]}
                    return super().__call__(endpoint,payload)
            t=Dynamic({});s=RoleSystem(Path(directory)/'memory.sqlite',embedding=FakeEmbedding(),transport=t,card_adapter=FakeCards())
            texts=['The ochre sail is named Elsin.','Elsin is in the east loft.']
            try:
                for uid,req,text in [('a','one',texts[0]),('a','two',texts[1]),('b','private','FOREIGN Elsin at loft')]:
                    s.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=text)]))
                result=s.search(dict(user_id='a',query='Where is the ochre sail?',top_k=100))['data']
                self.assertEqual({text_of(r['content']) for r in result},set(texts))
                self.assertEqual(text_of(result[0]['content']),texts[1])
                self.assertNotIn('FOREIGN',json.dumps(t.calls));self.assertEqual(len(t.calls),1)
            finally:
                s.close();connection=getattr(s.native.db,'connection',None)
                if connection:connection.close()
