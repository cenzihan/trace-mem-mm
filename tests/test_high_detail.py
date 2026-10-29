"""Independent synthetic image-detail ablation and original-evidence guarantees."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from agent_memory.high_detail import high_detail_payload, HighDetailSelector, HighDetailSystem, create_high_detail_app
from agent_memory.joint_selection import selection_payload, candidate_count
from agent_memory.protocol import content_parts, text_of, ContractError
from agent_memory.provider_http import ProviderFailure
from agent_memory.recoverable_options import RecoverableOptionsSystem
from test_mem0_joint import source, SelectFake
from test_mem0_native import FakeEmbedding, FakeCards
from test_mem0_evidence import image


class HighDetailContractTests(unittest.TestCase):
    def test_only_detail_changes_in_bounded_multimodal_request(self):
        parts=content_parts('independent query')+[image()]
        sources=[source('source '+str(i),True) for i in range(32)]
        original=copy.deepcopy(sources)
        old=selection_payload(parts,sources,['alpha','beta'])
        new=high_detail_payload(parts,sources,['alpha','beta'])
        self.assertEqual(candidate_count(new),32)
        images=[p for p in new['messages'][1]['content'] if p['type']=='image_url']
        self.assertEqual(len(images),8)
        self.assertTrue(all(p['image_url']['detail']=='high' for p in images))
        self.assertEqual(sources,original)
        for part in images:part['image_url']['detail']='low'
        self.assertEqual(new,old) # indices, prompt, exact JPEG bytes and all other fields

    def test_no_image_request_is_exact_baseline_and_admission_still_early(self):
        parts=content_parts('evidence');sources=[source()]
        self.assertEqual(high_detail_payload(parts,sources),selection_payload(parts,sources))
        with self.assertRaises(ContractError):high_detail_payload(content_parts('x'*4001),sources)
        with self.assertRaises(ValueError):high_detail_payload(parts,[source()]*33)
        with self.assertRaises(ValueError):create_high_detail_app({})
        self.assertIs(HighDetailSystem._order_sources,RecoverableOptionsSystem._order_sources)
        self.assertEqual(HighDetailSystem.context_anchor_limit,RecoverableOptionsSystem.context_anchor_limit)

    def test_single_call_empty_valid_failure_and_invalid_indices_not_fallback(self):
        transport=SelectFake();selector=HighDetailSelector(transport)
        transport.pick=[]
        self.assertEqual(selector.select(content_parts('absent'),[source()]),[])
        self.assertEqual(len(transport.calls),1)
        transport.pick=[9]
        with self.assertRaises(ValueError):selector.select(content_parts('absent'),[source()])
        transport.failed=True
        with self.assertRaises(RuntimeError):selector.select(content_parts('absent'),[source()])
        self.assertEqual(len(transport.calls),3)


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class HighDetailNativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.t=SelectFake()
        self.store=HighDetailSystem(Path(self.tmp.name)/'memory.sqlite',embedding=FakeEmbedding(),transport=self.t,card_adapter=FakeCards())
        self.addCleanup(self.store.close)

    def add(self,values,uid='u',session='s'):
        self.store.add(dict(user_id=uid,request_id=session,session_id=session,
                       messages=[dict(role='user',content=v) for v in values]))

    def search(self,query,uid='u'):
        return self.store.search(dict(user_id=uid,query=query,top_k=100,options=[]))['data']

    def test_original_image_and_neighbor_preserved_other_user_excluded(self):
        original=content_parts('copper image evidence')+[image()]
        self.add([original,'neighbor support'])
        self.add(['FOREIGN_PRIVATE_SENTINEL'],uid='other')
        result=self.search('copper image evidence')
        self.assertEqual(result[0]['content'],original)
        self.assertEqual(text_of(result[1]['content']),'neighbor support')
        self.assertEqual(len(self.t.calls),1)
        self.assertNotIn('FOREIGN_PRIVATE_SENTINEL',json.dumps(self.t.calls))
        self.assertEqual(self.store.capabilities()['selection_image_detail'],'high')
        self.assertEqual(self.search('copper',uid='absent'),[])
        self.assertEqual(len(self.t.calls),1)

    def test_cross_session_selected_pair_and_empty_selection_are_unchanged(self):
        self.add(['copper alias K41'],session='a');self.add(['K41 location cedar'],session='b')
        self.t.pick=[0,1]
        result=self.search('copper K41 location')
        self.assertEqual(set(text_of(r['content']) for r in result),{'copper alias K41','K41 location cedar'})
        self.t.pick=[]
        self.assertEqual(self.search('copper K41 location'),[]) # nonempty pool, explicit empty model choice
        self.assertEqual(len(self.t.calls),2)
