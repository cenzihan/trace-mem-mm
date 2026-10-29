"""Original-source packing regression, synthetic images only, no provider calls."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.mem0_raw import Mem0RawSystem
from agent_memory.mem0_evidence import pack_mixed
from agent_memory.mem0_joint import Mem0JointSystem
from agent_memory.recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport
from agent_memory.primary_first import PrimaryFirstSystem, create_primary_first_app
from agent_memory.protocol import content_parts, media_size, text_of
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding, FakeCards, FakeTransport


def row(mid, request, ordinal, session='session'):
    return (mid, '2026-01-01T00:00:00Z', session, request, ordinal, mid)


def order(system, rows, ids, anchors=12):
    instance = object.__new__(system)
    instance.context_anchor_limit = anchors
    return instance._context_order(rows, ids)


class PrimaryFirstTests(unittest.TestCase):
    def test_media_budget_at_top100_no_longer_lets_neighbor_displace_primary(self):
        rows = [row('first','a',0), row('neighbor','a',1), row('second','b',0)]
        old = order(Mem0RawSystem,rows,['first','second'])
        new = order(PrimaryFirstSystem,rows,['first','second'])
        part = image(); capacity = 2*media_size([part])
        originals = {r[0]:dict(id=r[0],content=[part]) for r in rows}
        def packed(ids):return [r['id'] for r in pack_mixed((originals[i] for i in ids),[],100,limit=capacity)]
        self.assertEqual(old,['first','neighbor','second'])
        self.assertEqual(new,['first','second','neighbor'])
        self.assertEqual(packed(old),['first','neighbor'])
        self.assertEqual(packed(new),['first','second'])
        # This does not make primary records exempt from the media budget.
        self.assertEqual([r['id'] for r in pack_mixed((originals[i] for i in new),[],100,limit=capacity//2)],['first'])

    def test_selected_neighbor_keeps_its_selector_rank(self):
        rows=[row('a','one',0),row('b','one',1),row('c','two',0)]
        self.assertEqual(order(PrimaryFirstSystem,rows,['a','c','b']),['a','c','b'])
        self.assertEqual(order(Mem0RawSystem,rows,['a','c','b']),['a','b','c'])

    def test_ample_capacity_preserves_members_and_whole_originals(self):
        rows=[row('a','one',0),row('n','one',1),row('b','two',0)]
        records={r[0]:dict(id=r[0],content=content_parts('  '+r[0]+' original  ')) for r in rows}
        results=[]
        for cls in [Mem0RawSystem,PrimaryFirstSystem]:
            packed=pack_mixed((records[i] for i in order(cls,rows,['a','b'])),[],100)
            results.append({r['id']:r['content'] for r in packed})
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[1],{k:v['content'] for k,v in records.items()})

    def test_neighbor_request_session_and_anchor_boundaries_unchanged(self):
        rows=[row('a','one',0),row('n','one',1),row('b','two',0),row('bn','two',1),row('foreign_session','one',1,'other')]
        self.assertEqual(order(PrimaryFirstSystem,rows,['a','b'],1),['a','b','n'])
        self.assertEqual(order(PrimaryFirstSystem,rows,['a','b'],0),['a','b'])
        self.assertEqual(order(PrimaryFirstSystem,rows,[]),[])

    def test_selector_exact_image_and_retry_implementation_are_inherited(self):
        self.assertIs(PrimaryFirstSystem._order_sources,Mem0JointSystem._order_sources)
        self.assertIs(PrimaryFirstSystem._select_sources,RecoverableOptionsSystem._select_sources)
        self.assertFalse(Mem0RawSystem.context_primary_first)
        self.assertTrue(PrimaryFirstSystem.context_primary_first)
        self.assertEqual(RetrySelectionTransport.receipt_scope,'native-mem0-raw-v4')
        self.assertEqual(len(RetrySelectionTransport.retry_delays),5)
        with self.assertRaises(ValueError):create_primary_first_app({})


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class PrimaryFirstNativeTests(unittest.TestCase):
    def test_search_preserves_primary_originals_then_neighbors_and_isolates_users(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding=FakeEmbedding();transport=FakeTransport()
            system=PrimaryFirstSystem(Path(directory)/'memory.sqlite',embedding=embedding,transport=transport,card_adapter=FakeCards())
            part=image()
            sources=[[dict(type='text',text=label),part] for label in ['primary alpha','optional neighbor','primary beta']]
            try:
                for uid,req,contents in [('u','first',sources[:2]),('u','second',sources[2:]),('other','private',[[dict(type='text',text='primary FOREIGN'),part]])]:
                    system.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=c) for c in contents]))
                class Selection:
                    def __init__(self):self.calls=0
                    def select(self,parts,candidates,options=None):
                        self.calls+=1
                        return [next(i for i,s in enumerate(candidates) if text_of(s['content'])==label) for label in ['primary alpha','primary beta']]
                selector=Selection();system.selector=selector
                with patch('agent_memory.mem0_raw.pack_mixed',side_effect=lambda originals,facts,top:pack_mixed(originals,facts,top,limit=2*media_size([part]))):
                    result=system.search(dict(user_id='u',query='primary',options=['alpha','beta'],top_k=100))['data']
                self.assertEqual([r['content'] for r in result],[sources[0],sources[2]])
                self.assertEqual(selector.calls,1)
                self.assertEqual(transport.calls,[])
                self.assertEqual(system.search(dict(user_id='absent',query='primary',top_k=100)),{'data':[]})
                self.assertEqual(selector.calls,1)
            finally:system.close()
