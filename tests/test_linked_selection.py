"""Frozen synthetic source graphs and native integration, with no external calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from agent_memory.linked_selection import (linked_payload, linked_views, linked_indices,
    LinkedSystem, LinkedServiceTransport, LinkedBudgetedTransport, create_linked_app, VERSION)
from agent_memory.protocol import content_parts, text_of
from agent_memory.provider_http import ProviderFailure, HTTPReply, OPENAI_ENDPOINT
from agent_memory.joint_selection import strict_json
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding, FakeCards
from test_chat_stream import wire, events
import test_service_transport as receipt_fixtures


def node(index, witness, requires=(), kind='text'):
    return dict(index=index, kind=kind, witness=witness, requires=list(requires))


class LinkedContractTests(unittest.TestCase):
    def setUp(self):
        self.sources = [source('Slate is called Fern.'), source('Fern entered the south room.'),
                        source('The room was later locked.'), source('An unrelated boat.')]
        self.payload = linked_payload(content_parts('Where did Slate enter?'), self.sources)
        self.rows = linked_views(self.payload)
        self.value = {'roots':[1], 'evidence':[node(1,'Fern entered',[0]), node(0,'Slate is called Fern.')]}

    def test_terminal_and_required_alias_both_returned(self):
        self.assertEqual(linked_indices(self.value,self.rows),[1,0])

    def test_unordered_evidence_entries_do_not_change_closure(self):
        self.value['evidence'].reverse()
        self.assertEqual(linked_indices(self.value,self.rows),[1,0])

    def test_transitive_dependencies_all_returned_once(self):
        v={'roots':[2,1], 'evidence':[node(0,'Slate'),node(1,'Fern',[0]),node(2,'room',[1])]}
        self.assertEqual(linked_indices(v,self.rows),[2,1,0])

    def test_shared_prerequisite_deduplicated(self):
        v={'roots':[1,2], 'evidence':[node(0,'Slate'),node(1,'Fern',[0]),node(2,'room',[0])]}
        self.assertEqual(linked_indices(v,self.rows),[1,0,2])

    def test_missing_prerequisite_entry_is_not_silently_dropped(self):
        self.value['evidence'].pop()
        with self.assertRaises(ValueError):linked_indices(self.value,self.rows)

    def test_missing_root_rejected(self):
        self.value['roots']=[3]
        with self.assertRaises(ValueError):linked_indices(self.value,self.rows)

    def test_cycles_and_self_dependency_rejected(self):
        for deps in ([0],[1]):
            v=copy.deepcopy(self.value);v['evidence'][1]['requires']=deps
            with self.subTest(deps=deps),self.assertRaises(ValueError):linked_indices(v,self.rows)

    def test_unreachable_source_rejected_not_fallback_tail(self):
        self.value['evidence'].append(node(3,'boat'))
        with self.assertRaises(ValueError):linked_indices(self.value,self.rows)

    def test_empty_roots_with_sources_rejected(self):
        self.value['roots']=[]
        with self.assertRaises(ValueError):linked_indices(self.value,self.rows)

    def test_empty_selection_is_valid(self):
        self.assertEqual(linked_indices({'roots':[],'evidence':[]},self.rows),[])

    def test_bad_dependency_types_duplicates_and_range(self):
        for deps in ([True],[-1],[4],[0,0],['0'],None):
            v=copy.deepcopy(self.value);v['evidence'][0]['requires']=deps
            with self.subTest(deps=deps),self.assertRaises(ValueError):linked_indices(v,self.rows)

    def test_bad_root_types_duplicates_and_range(self):
        for roots in ([True],[-1],[4],[1,1],['1'],None):
            v=copy.deepcopy(self.value);v['roots']=roots
            with self.subTest(roots=roots),self.assertRaises(ValueError):linked_indices(v,self.rows)

    def test_duplicate_node_and_boolean_index_rejected(self):
        for entry in (node(1,'Fern'),node(True,'Fern')):
            v=copy.deepcopy(self.value);v['evidence'].append(entry)
            with self.assertRaises(ValueError):linked_indices(v,self.rows)

    def test_cross_source_quote_rejected(self):
        self.value['evidence'][0]['witness']='Slate'
        with self.assertRaises(ValueError):linked_indices(self.value,self.rows)

    def test_extra_answer_and_missing_requires_rejected(self):
        v=copy.deepcopy(self.value);v['answer']='south'
        with self.assertRaises(ValueError):linked_indices(v,self.rows)
        del self.value['evidence'][0]['requires']
        with self.assertRaises(ValueError):linked_indices(self.value,self.rows)

    def test_total_source_bound_includes_dependencies(self):
        rows=linked_views(linked_payload(content_parts('source'),[source('source')]*9))
        v={'roots':[8],'evidence':[node(i,'source',[i-1] if i else []) for i in range(9)]}
        with self.assertRaises(ValueError):linked_indices(v,rows)

    def test_missing_image_preview_rejected(self):
        v={'roots':[0],'evidence':[node(0,'visible shape',kind='image')]}
        with self.assertRaises(ValueError):linked_indices(v,self.rows)
        rows=linked_views(linked_payload(content_parts('shape'),[source(picture=True)]))
        self.assertEqual(linked_indices(v,rows),[0])

    def test_payload_preserves_options_and_inputs(self):
        parts=[image()]; sources=[source('untrusted text',True)]; options=['same','other','same']
        before=copy.deepcopy([parts,sources,options]);p=linked_payload(parts,sources,options)
        self.assertEqual([parts,sources,options],before)
        self.assertEqual(strict_json(p['messages'][1]['content'][0]['text'])['options'],options)
        self.assertEqual(p['max_completion_tokens'],1536)

    def test_prompt_and_index_map_fixed(self):
        p=copy.deepcopy(self.payload);p['messages'][0]['content']='altered'
        with self.assertRaises(ValueError):linked_views(p)
        p=copy.deepcopy(self.payload);h=strict_json(p['messages'][1]['content'][0]['text'])
        h['candidates'][0]['index']=True;p['messages'][1]['content'][0]['text']=json.dumps(h)
        with self.assertRaises(ValueError):linked_views(p)

    def test_explicit_activation_same_scope_and_model(self):
        with self.assertRaises(ValueError):create_linked_app({})
        self.assertEqual(LinkedServiceTransport.receipt_scope,'native-mem0-raw-v4')
        obj=object.__new__(LinkedBudgetedTransport);obj.model='gpt-4o-mini'
        with self.assertRaises(ProviderFailure):obj._response_model({'model':'gpt-4o-mini-latest'})

    def test_semantics_not_falsely_claimed_by_parser(self):
        # Structurally valid but unrelated roots remain the model's quality responsibility.
        self.assertEqual(linked_indices({'roots':[3],'evidence':[node(3,'boat')]},self.rows),[3])


class GraphFake:
    def __init__(self):self.calls=[];self.mode='chain';self.failed=False
    def ready(self):pass
    def __call__(self,endpoint,payload):
        self.calls.append(payload)
        if self.failed:raise RuntimeError('independent failure')
        rows=linked_views(payload);evidence=[];roots=[]
        if self.mode=='chain':
            alias=next(r for r in rows if 'called Fern' in r['text_prefix'])
            location=next(r for r in rows if 'south room' in r['text_prefix'])
            roots=[location['index']]
            evidence=[node(location['index'],'Fern entered',[alias['index']]),
                      node(alias['index'],'Slate is called Fern.')]
        return {'choices':[{'message':{'content':json.dumps(dict(roots=roots,evidence=evidence))}}]}


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned optional SDK')
class LinkedNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.e,self.t,self.c=FakeEmbedding(),GraphFake(),FakeCards()
        self.store=LinkedSystem(Path(self.temp.name)/'memory.sqlite',embedding=self.e,
                                transport=self.t,card_adapter=self.c)
        self.addCleanup(self.close)
    def close(self):
        self.store.close(); conn=getattr(self.store.native.db,'connection',None)
        if conn:conn.close()
    def add(self,text,session,uid='alice'):
        return self.store.add(dict(user_id=uid,session_id=session,request_id=session,
                                   messages=[dict(role='user',content=text)]))
    def search(self,query,uid='alice'):
        return self.store.search(dict(user_id=uid,query=query,options=['south','north'],top_k=100))['data']
    def seed(self):
        self.add('Slate is called Fern.','one');self.add('Fern entered the south room.','two')
    def test_dependency_from_different_add_returned_as_original(self):
        self.seed();self.add('A boat unrelated to these objects.','three')
        rows=self.search('Where did Slate enter?')
        self.assertEqual([text_of(r['content']) for r in rows],
                         ['Fern entered the south room.','Slate is called Fern.'])
        self.assertEqual(len(self.t.calls),1);self.assertEqual(self.c.calls,0)
        self.assertEqual(self.store.capabilities()['version'],VERSION)
    def test_empty_graph_no_fallback_and_image_identity_intact(self):
        original=[image()];self.add(original,'picture');self.t.mode='empty'
        self.assertEqual(self.search('absent named event'),[])
        self.assertEqual(self.search([image()])[0]['content'],original)
    def test_cross_user_isolation_and_query_not_archived(self):
        self.seed();self.add('PRIVATE_OTHER_SOURCE','other',uid='bob')
        self.search('Slate QUERY_NOT_ARCHIVED')
        self.assertNotIn('PRIVATE_OTHER_SOURCE',json.dumps(self.t.calls))
        with self.store.connect() as db:rows=db.execute('select content from memories').fetchall()
        self.assertNotIn('QUERY_NOT_ARCHIVED',str(rows))
    def test_provider_error_propagated_without_retry(self):
        self.seed();self.t.failed=True
        with self.assertRaises(RuntimeError):self.search('Slate')
        self.assertEqual(len(self.t.calls),1)


class LinkedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.f=receipt_fixtures.ServiceTransportTests();self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.payload=linked_payload(content_parts('Slate'),[source('Slate is called Fern.'),source('Fern entered.')])
    def transport(self,text,complete=True):
        class Http:
            is_fake_http=True
            def __init__(self):self.calls=0
            def exchange(self,*args):
                self.calls+=1
                return HTTPReply(200,wire(events(text),complete),{'content-type':'text/event-stream'})
        self.http=Http();f=self.f
        return LinkedServiceTransport(f.p/'budget.sqlite',f.authority,f.prices,
            f.p/'linked-receipts',{'gpt-4o-mini':lambda:'SYNTHETIC'},http=self.http,offline_test=True)
    def value(self):return json.dumps({'roots':[1],'evidence':[node(1,'Fern',[0]),node(0,'Slate')]})
    def test_success_minimized_original_receipt(self):
        t=self.transport(self.value());t(OPENAI_ENDPOINT,self.payload)
        with t.journal.connect() as db:r=db.execute('select * from operations').fetchone()
        self.assertEqual(r['state'],'complete');self.assertIsNone(r['body'])
        self.assertEqual(r['scope'],'native-mem0-raw-v4');self.assertEqual(self.http.calls,1)
    def test_missing_dependency_known_failure_not_retry(self):
        t=self.transport(json.dumps({'roots':[1],'evidence':[node(1,'Fern',[0])]}))
        with self.assertRaises(ProviderFailure):t(OPENAI_ENDPOINT,self.payload)
        with t.journal.connect() as db:r=db.execute('select * from operations').fetchone()
        self.assertEqual(r['state'],'failed');self.assertEqual(r['status'],200)
        with self.assertRaises(ValueError):t(OPENAI_ENDPOINT,self.payload)
        self.assertEqual(self.http.calls,1)
    def test_incomplete_stream_remains_unknown(self):
        t=self.transport(self.value(),False)
        with self.assertRaises(ProviderFailure):t(OPENAI_ENDPOINT,self.payload)
        with self.assertRaises(ValueError):t.ready()
        self.assertEqual(self.http.calls,1)

