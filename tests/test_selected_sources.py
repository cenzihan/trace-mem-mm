"""Synthetic no-network ablation checks, including the expected recall tradeoff."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.selected_sources import SelectedSourcesSystem, create_selected_sources_app
from agent_memory.recoverable_options import RecoverableOptionsSystem
from agent_memory.joint_selection import selection_payload
from agent_memory.mem0_evidence import pack_mixed
from agent_memory.protocol import content_parts, media_size, text_of
from test_mem0_native import FakeEmbedding, FakeCards, FakeTransport
from test_mem0_evidence import image


class SelectedActivationTests(unittest.TestCase):
    def test_explicit_mode_preserves_baseline_selection_and_retry(self):
        with self.assertRaises(ValueError):create_selected_sources_app({})
        self.assertEqual(SelectedSourcesSystem.context_anchor_limit,0)
        self.assertEqual(RecoverableOptionsSystem.context_anchor_limit,12)
        self.assertIs(SelectedSourcesSystem._order_sources,RecoverableOptionsSystem._order_sources)
        self.assertIs(SelectedSourcesSystem._select_sources,RecoverableOptionsSystem._select_sources)
        self.assertIs(SelectedSourcesSystem.add,RecoverableOptionsSystem.add)
        caps=object.__new__(SelectedSourcesSystem).capabilities()
        self.assertNotIn('same-add-context',caps['retrieval'])


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class SelectedNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=SelectedSourcesSystem(Path(self.temp.name)/'memory.sqlite',embedding=FakeEmbedding(),transport=FakeTransport(),card_adapter=FakeCards())
        self.addCleanup(self.store.close)
        self.labels=[];self.payloads=[]
        outer=self
        class Selector:
            def select(self,parts,sources,options=None):
                outer.payloads.append(selection_payload(parts,sources,options))
                return [next(i for i,s in enumerate(sources) if text_of(s['content'])==label) for label in outer.labels]
        self.store.selector=Selector()

    def add(self,req,values,uid='u'):
        return self.store.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=v) for v in values]))

    def search(self,query='synthetic evidence',uid='u'):
        return self.store.search(dict(user_id=uid,query=query,options=['first','second'],top_k=100))['data']

    def test_only_primaries_preserve_originals_order_and_media_budget(self):
        pic=image()
        records=[content_parts(x)+[pic] for x in ['primary alpha','irrelevant neighbor','primary beta']]
        self.add('a',records[:2]);self.add('b',records[2:]);self.labels=['primary alpha','primary beta']
        with patch('agent_memory.mem0_raw.pack_mixed',side_effect=lambda originals,facts,top:pack_mixed(originals,facts,top,limit=2*media_size([pic]))):
            output=self.search()
        self.assertEqual([r['content'] for r in output],[records[0],records[2]])
        self.assertEqual(len(self.payloads),1)
        self.assertGreater(output[0]['score'],output[1]['score'])

    def test_selected_adjacent_source_is_kept_and_upstream_payload_unchanged(self):
        self.add('a',['selected alpha','selected neighbor']);self.add('b',['selected beta'])
        self.labels=['selected alpha','selected beta','selected neighbor']
        output=self.search()
        self.assertEqual([text_of(r['content']) for r in output],self.labels)
        self.store.context_anchor_limit=12
        self.search()
        self.assertEqual(self.payloads[0],self.payloads[1])
        self.assertEqual(len(self.payloads),2)

    def test_omitted_identity_recall_loss_is_an_explicit_tradeoff(self):
        self.add('a',['Alias Q7 means bronze pin','Q7 is in the maple case'])
        self.labels=['Q7 is in the maple case']
        new=self.search()
        self.store.context_anchor_limit=12
        old=self.search()
        self.assertEqual([text_of(r['content']) for r in new],self.labels)
        self.assertEqual(set(text_of(r['content']) for r in old),{'Alias Q7 means bronze pin','Q7 is in the maple case'})
        self.assertEqual(self.payloads[0],self.payloads[1])

    def test_empty_selection_exact_image_exception_and_isolation(self):
        original=content_parts('image primary')+[image()]
        self.add('a',[original,'unselected image neighbor'])
        self.add('b',['FOREIGN ONLY'],uid='other')
        self.labels=[]
        self.assertEqual(self.search(),[])
        output=self.search([image()])
        self.assertEqual([r['content'] for r in output],[original])
        self.assertNotIn('FOREIGN ONLY',json.dumps(self.payloads))
        calls=len(self.payloads);self.assertEqual(self.search(uid='absent'),[])
        self.assertEqual(len(self.payloads),calls)

    def test_complete_cross_session_pair_is_retained_when_actually_selected(self):
        self.add('identity',['The green token has identifier M8.'])
        self.add('location',['M8 is stored in the cedar drawer.'])
        self.labels=['The green token has identifier M8.','M8 is stored in the cedar drawer.']
        self.assertEqual([text_of(r['content']) for r in self.search()],self.labels)
        self.assertEqual(len(self.payloads),1)
