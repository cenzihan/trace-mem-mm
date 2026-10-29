"""Synthetic index-role, image-label and primary retention regressions."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.anchored_support import (AnchoredSelector, AnchoredSystem,
    AnchoredServiceTransport, AnchoredBudgetedTransport, anchored_payload,
    checked_anchored_payload, create_anchored_app)
from agent_memory.joint_selection import selection_payload, candidate_count
from agent_memory.protocol import content_parts, text_of
from agent_memory.provider_http import ProviderFailure
from test_completeness_selection import SequenceTransport
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding, FakeCards


class AnchoredTests(unittest.TestCase):
    def test_missing_alias_is_selectable_zero_without_reselecting_primary(self):
        sources=[source('Amber ferry has identifier Leto64.'),source('Leto64 is at west quay.')]
        t=SequenceTransport([[1],[0]])
        self.assertEqual(AnchoredSelector(t).select(content_parts('Where is amber ferry?'),sources),[1,0])
        header=json.loads(t.calls[1]['messages'][1]['content'][0]['text'])
        self.assertEqual([r['text_prefix'] for r in header['reference_anchors']],['Leto64 is at west quay.'])
        self.assertEqual([r['text_prefix'] for r in header['candidates']],['Amber ferry has identifier Leto64.'])
        self.assertNotIn('index',header['reference_anchors'][0])

    def test_noncontiguous_primary_order_and_support_mapping(self):
        t=SequenceTransport([[3,1],[2,0]])
        self.assertEqual(AnchoredSelector(t).select(content_parts('q'),[source(str(i)) for i in range(6)]),[3,1,4,0])

    def test_empty_support_retains_primary(self):
        self.assertEqual(AnchoredSelector(SequenceTransport([[1],[]])).select(content_parts('q'),[source(),source()]),[1])

    def test_no_second_call_for_empty_full_or_exhausted_pool(self):
        for primary,count in [([],2),([0],1),(list(range(8)),9)]:
            t=SequenceTransport([primary])
            self.assertEqual(AnchoredSelector(t).select(content_parts('q'),[source() for _ in range(count)]),primary)
            self.assertEqual(len(t.calls),1)

    def test_support_capacity_does_not_displace_primary(self):
        t=SequenceTransport([[9,8,7,6],[0,1,2,3,4,5]])
        self.assertEqual(AnchoredSelector(t).select(content_parts('q'),[source() for _ in range(10)]),[9,8,7,6,0,1,2,3])

    def test_local_support_bounds_reject_old_global_or_anchor_indices(self):
        for invalid in ([2],[True],[-1],[0,0]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                AnchoredSelector(SequenceTransport([[2],invalid])).select(content_parts('q'),[source() for _ in range(3)])

    def test_image_roles_preserve_preview_bytes_and_total_budget(self):
        sources=[source(str(i),True) for i in range(10)];parts=[image()];before=copy.deepcopy(sources)
        original=selection_payload(parts,sources,['choice'])
        payload,mapping=anchored_payload(parts,sources,['choice'],[4,0])
        content=payload['messages'][1]['content'];old=original['messages'][1]['content']
        self.assertEqual([x for x in content if x['type']=='image_url'],[x for x in old if x['type']=='image_url'])
        self.assertEqual(sum(x['type']=='image_url' for x in content),8)
        labels=[json.loads(x['text']) for x in content[1:] if x['type']=='text']
        self.assertEqual(labels[0],{'image_for':'query'})
        self.assertIn({'anchor_image_for':0},labels);self.assertIn({'anchor_image_for':1},labels)
        self.assertEqual(mapping,[1,2,3,5,6,7,8,9]);self.assertEqual(sources,before)
        header=json.loads(content[0]['text'])
        self.assertFalse(header['candidates'][-1]['image_preview_attached'])

    def test_checked_contract_preserves_payload_and_rejects_anchor_map_corruption(self):
        p,_=anchored_payload(content_parts('q'),[source() for _ in range(3)],[],[1])
        before=copy.deepcopy(p);self.assertEqual(candidate_count(checked_anchored_payload(p)),2);self.assertEqual(p,before)
        header=json.loads(p['messages'][1]['content'][0]['text']);header['reference_anchors'][0]['anchor_index']=True
        p['messages'][1]['content'][0]['text']=json.dumps(header)
        with self.assertRaises(ValueError):checked_anchored_payload(p)

    def test_unknown_response_is_not_primary_fallback(self):
        with self.assertRaises(ProviderFailure):
            AnchoredSelector(SequenceTransport([[0],ProviderFailure('network_error')])).select(content_parts('q'),[source(),source()])

    def test_model_contract_scope_and_explicit_factory(self):
        t=object.__new__(AnchoredBudgetedTransport);t.model='gpt-4o-mini'
        p,_=anchored_payload(content_parts('q'),[source(),source()],[],[1])
        r={'choices':[{'finish_reason':'stop','message':{'content':'{"selected":[0]}'}}]};t._contract(r,p)
        r['choices'][0]['message']['content']='{"selected":[1]}'
        with self.assertRaises(ProviderFailure):t._contract(r,p)
        with self.assertRaises(ProviderFailure):t._response_model({'model':'wrong'})
        self.assertEqual(AnchoredServiceTransport.receipt_scope,'native-mem0-raw-v4')
        with self.assertRaises(ValueError):create_anchored_app({})


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class AnchoredNativeTests(unittest.TestCase):
    def test_original_sources_across_sessions_and_user_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            t=SequenceTransport([[1],[0],[]])
            s=AnchoredSystem(Path(directory)/'memory.sqlite',embedding=FakeEmbedding(),transport=t,card_adapter=FakeCards())
            try:
                for uid,req,text in [('a','one','Amber ferry has identifier Leto64.'),('a','two','Leto64 is at west quay.'),('b','private','PRIVATE_OTHER_USER')]:
                    s.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=text)]))
                rows=s.search(dict(user_id='a',query='Where is amber ferry?',top_k=100))['data']
                self.assertEqual({text_of(r['content']) for r in rows},{'Amber ferry has identifier Leto64.','Leto64 is at west quay.'})
                self.assertNotIn('PRIVATE_OTHER_USER',json.dumps(t.calls))
                self.assertEqual(s.search(dict(user_id='a',query='Unrecorded launch?',top_k=100))['data'],[])
            finally:
                s.close()
                connection=getattr(s.native.db,'connection',None)
                if connection:connection.close()
