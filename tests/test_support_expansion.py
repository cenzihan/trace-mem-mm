"""Independent synthetic regressions for primary retention and support bounds."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.support_expansion import (SupportSelector, SupportSystem,
    SupportServiceTransport, SupportBudgetedTransport, support_payload,
    checked_support_payload, create_support_app, SUPPORT_SYSTEM)
from agent_memory.completeness_selection import REVIEW_SYSTEM
from agent_memory.joint_selection import candidate_count, selection_payload
from agent_memory.protocol import content_parts, text_of
from agent_memory.provider_http import ProviderFailure
from test_completeness_selection import SequenceTransport
from test_mem0_joint import source
from test_mem0_native import FakeEmbedding, FakeCards


class SupportTests(unittest.TestCase):
    def test_empty_review_cannot_delete_primary_direct_evidence(self):
        t=SequenceTransport([[1], []])
        self.assertEqual(SupportSelector(t).select(content_parts('Which hangar?'),
            [source('unrelated dock'), source('glider is in hangar')]), [1])
        self.assertEqual(len(t.calls),2)

    def test_review_cannot_reorder_primary_and_only_appends_unique_indices(self):
        t=SequenceTransport([[2,0], [0,1,2,3]])
        self.assertEqual(SupportSelector(t).select(content_parts('Compare states'),
            [source() for _ in range(4)]), [2,0,1,3])

    def test_empty_primary_stays_empty_without_second_call(self):
        t=SequenceTransport([[]])
        self.assertEqual(SupportSelector(t).select(content_parts('No such event'),[source()]), [])
        self.assertEqual(len(t.calls),1)

    def test_capacity_preserves_entire_primary_prefix(self):
        t=SequenceTransport([[3,2,1,0], [4,5,6,7,8,9]])
        self.assertEqual(SupportSelector(t).select(content_parts('query'),[source() for _ in range(10)]),[3,2,1,0,4,5,6,7])
        t=SequenceTransport([list(range(8))])
        self.assertEqual(SupportSelector(t).select(content_parts('query'),[source() for _ in range(9)]),list(range(8)))
        self.assertEqual(len(t.calls),1)

    def test_unknown_review_does_not_return_unreviewed_primary(self):
        t=SequenceTransport([[0],ProviderFailure('network_error')])
        with self.assertRaises(ProviderFailure):SupportSelector(t).select(content_parts('q'),[source()])

    def test_invalid_review_indices_are_rejected_before_union(self):
        for invalid in ([True],[-1],[2],[1,1],list(range(9))):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                SupportSelector(SequenceTransport([[0],invalid])).select(content_parts('q'),[source(),source()])

    def test_same_original_views_and_untrusted_choices(self):
        p=support_payload(content_parts('q'),[source('literal original')],['untrusted option'],[0])
        original=copy.deepcopy(p)
        checked=checked_support_payload(p)
        self.assertEqual(p,original)
        self.assertEqual(candidate_count(checked),1)
        first=selection_payload(content_parts('q'),[source('literal original')],['untrusted option'])
        header=json.loads(p['messages'][1]['content'][0]['text']);self.assertEqual(header.pop('draft_selected'),[0])
        self.assertEqual(header,json.loads(first['messages'][1]['content'][0]['text']))
        self.assertEqual(p['messages'][0]['content'],SUPPORT_SYSTEM)
        p['messages'][0]['content']=REVIEW_SYSTEM
        with self.assertRaises(ValueError):checked_support_payload(p)

    def test_dated_model_complete_output_and_scope_unchanged(self):
        t=object.__new__(SupportBudgetedTransport);t.model='gpt-4o-mini'
        p=support_payload(content_parts('q'),[source()],[],[0])
        r={'choices':[{'finish_reason':'stop','message':{'content':'{"selected":[]}'}}]}
        t._contract(r,p);r['choices'][0]['finish_reason']='length'
        with self.assertRaises(ProviderFailure):t._contract(r,p)
        with self.assertRaises(ProviderFailure):t._response_model({'model':'other'})
        self.assertEqual(SupportServiceTransport.receipt_scope,'native-mem0-raw-v4')
        with self.assertRaises(ValueError):create_support_app({})


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class SupportNativeTests(unittest.TestCase):
    def test_linked_originals_primary_order_and_user_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            t=SequenceTransport([[0],[1],[]])
            s=SupportSystem(Path(directory)/'memory.sqlite',embedding=FakeEmbedding(),transport=t,card_adapter=FakeCards())
            try:
                for uid,req,text in [('a','one','Copper cart has tag Juno81.'),
                                     ('a','two','Juno81 moved to the north depot.'),
                                     ('b','private','OTHER_USER_ONLY')]:
                    s.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=text)]))
                rows=s.search(dict(user_id='a',query='Where is copper cart?',top_k=100))['data']
                self.assertEqual({text_of(r['content']) for r in rows},{'Copper cart has tag Juno81.','Juno81 moved to the north depot.'})
                self.assertNotIn('OTHER_USER_ONLY',json.dumps(t.calls))
                self.assertEqual(s.search(dict(user_id='a',query='Unrecorded flight?',top_k=100))['data'],[])
                self.assertEqual(len(t.calls),3)
            finally:
                s.close()
                connection=getattr(s.native.db,'connection',None)
                if connection:connection.close()
