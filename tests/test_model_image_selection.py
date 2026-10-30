"""Synthetic regression for image identity bypassing evidence selection."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from agent_memory.model_image_selection import ModelImageSelectionSystem, create_model_image_selection_app
from agent_memory.recoverable_options import RecoverableOptionsSystem
from agent_memory.protocol import content_parts
from test_mem0_evidence import image
from test_mem0_joint import SelectFake
from test_mem0_native import FakeEmbedding, FakeCards


class ImageSelectionPoolTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:');self.addCleanup(self.db.close)
        self.db.executescript('''CREATE TABLE memories(id TEXT,user_id TEXT,content TEXT,created_at TEXT,session_id TEXT);
            CREATE TABLE evidence(memory_id TEXT);''')
        for mid,uid,text in [('photo','u','original image source'),('fact','u','separate recorded fact'),('foreign','other','PRIVATE_SENTINEL')]:
            self.db.execute('INSERT INTO memories VALUES(?,?,?,?,?)',(mid,uid,json.dumps(content_parts(text)+([image()] if mid=='photo' else [])), '2024-02-01','s-'+mid))
        self.db.execute('INSERT INTO evidence VALUES(?)',('photo',))
        self.rows=[('photo','2024-02-01','s-photo','r-photo',0,'original image source'),('fact','2024-02-01','s-fact','r-fact',0,'separate recorded fact')]
        self.transport=SelectFake()
        from agent_memory.joint_selection import JointSelector
        self.selector=JointSelector(self.transport)

    def ordered(self,cls,pick,identities=None):
        store=object.__new__(cls);store.connect=lambda:self.db;store.selector=self.selector
        self.transport.pick=pick
        return store._order_sources({'user_id':'u','options':[]},content_parts('independent evidence query'),self.rows,['photo','fact'],{'photo'} if identities is None else identities)

    def test_empty_choice_is_no_longer_overridden_by_exact_image(self):
        self.assertEqual(self.ordered(RecoverableOptionsSystem,[]),['photo'])
        baseline_payload=self.transport.calls[-1]
        self.assertEqual(self.ordered(ModelImageSelectionSystem,[]),[])
        self.assertEqual(self.transport.calls[-1],baseline_payload)

    def test_unselected_identity_does_not_precede_a_selected_fact(self):
        self.assertEqual(self.ordered(ModelImageSelectionSystem,[1]),['fact'])
        self.assertEqual(self.ordered(RecoverableOptionsSystem,[1]),['photo','fact'])

    def test_selected_identity_retains_baseline_priority(self):
        self.assertEqual(self.ordered(ModelImageSelectionSystem,[1,0]),['photo','fact'])
        self.assertEqual(self.ordered(ModelImageSelectionSystem,[0]),['photo'])

    def test_no_identity_request_matches_baseline_and_excludes_other_user(self):
        self.assertEqual(self.ordered(ModelImageSelectionSystem,[1,0],set()),self.ordered(RecoverableOptionsSystem,[1,0],set()))
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(self.transport.calls))
        header=json.loads(self.transport.calls[-1]['messages'][1]['content'][0]['text'])
        self.assertEqual(len(header['candidates']),2)
        self.assertTrue(header['candidates'][0]['image_preview_attached'])

    def test_selector_error_never_falls_back_to_identity(self):
        self.transport.failed=True
        with self.assertRaises(RuntimeError):self.ordered(ModelImageSelectionSystem,[0])

    def test_factory_requires_opt_in_and_no_memory_capacity_expansion(self):
        with self.assertRaises(ValueError):create_model_image_selection_app({})
        self.assertEqual(ModelImageSelectionSystem.context_anchor_limit,RecoverableOptionsSystem.context_anchor_limit)


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class ImageSelectionNativeTests(unittest.TestCase):
    def test_exact_query_image_can_abstain_then_return_original_when_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            transport=SelectFake()
            store=ModelImageSelectionSystem(Path(tmp)/'memory.sqlite',embedding=FakeEmbedding(),transport=transport,card_adapter=FakeCards())
            try:
                original=content_parts('independent copper token image')+[image()]
                store.add({'user_id':'u','request_id':'new','session_id':'s','messages':[{'role':'user','content':original}]})
                query=content_parts('independent copper token image')+[image()]
                transport.pick=[]
                self.assertEqual(store.search({'user_id':'u','query':query,'top_k':100,'options':[]})['data'],[])
                transport.pick=[0]
                result=store.search({'user_id':'u','query':query,'top_k':100,'options':[]})['data']
                self.assertEqual(len(result),1);self.assertEqual(result[0]['content'],original)
                self.assertEqual(len(transport.calls),2)
                self.assertEqual(store.search({'user_id':'absent','query':query,'top_k':100,'options':[]})['data'],[])
                self.assertEqual(len(transport.calls),2)
            finally:store.close()
