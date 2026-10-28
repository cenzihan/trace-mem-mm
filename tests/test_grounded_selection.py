"""Independent synthetic controls; no official data and no network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from agent_memory.grounded_selection import (grounded_payload,candidate_views,
    evidence_indices,GroundedBudgetedTransport,GroundedServiceTransport,GroundedSystem,
    create_grounded_app,SYSTEM,VERSION)
from agent_memory.protocol import content_parts,text_of,ContractError
from agent_memory.provider_http import ProviderFailure
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding,FakeCards


def entry(index=0,kind='text',witness='anchor'):
    return dict(index=index,kind=kind,witness=witness)


class GroundedContractTests(unittest.TestCase):
    def setUp(self):
        self.payload=grounded_payload(content_parts('query'),[source('anchor source')],['a','b'])
        self.rows=candidate_views(self.payload)

    def test_quote_from_exact_selected_source(self):
        self.assertEqual(evidence_indices({'evidence':[entry()]},self.rows),[0])

    def test_other_source_quote_rejected(self):
        rows=candidate_views(grounded_payload(content_parts('query'),[source('first'),source('second')]))
        with self.assertRaises(ValueError):evidence_indices({'evidence':[entry(0,witness='second')]},rows)

    def test_options_are_not_source_quote(self):
        with self.assertRaises(ValueError):evidence_indices({'evidence':[entry(witness='b')]},self.rows)

    def test_missing_and_fabricated_quotes_rejected(self):
        for witness in ('',' ','fiction'):
            with self.subTest(witness=witness),self.assertRaises(ValueError):
                evidence_indices({'evidence':[entry(witness=witness)]},self.rows)

    def test_unicode_byte_bound(self):
        rows=candidate_views(grounded_payload(content_parts('query'),[source('界'*80)]))
        self.assertEqual(evidence_indices({'evidence':[entry(witness='界'*64)]},rows),[0])
        with self.assertRaises(ValueError):evidence_indices({'evidence':[entry(witness='界'*65)]},rows)

    def test_hidden_tail_quote_not_accepted(self):
        rows=candidate_views(grounded_payload(content_parts('query'),[source('x'*1600+'HIDDEN')]))
        with self.assertRaises(ValueError):evidence_indices({'evidence':[entry(witness='HIDDEN')]},rows)

    def test_image_requires_attached_preview(self):
        with self.assertRaises(ValueError):evidence_indices({'evidence':[entry(kind='image',witness='mountain')]},self.rows)
        rows=candidate_views(grounded_payload(content_parts('query'),[source(picture=True)]))
        self.assertEqual(evidence_indices({'evidence':[entry(kind='image',witness='visible shape')]},rows),[0])

    def test_ninth_image_not_accepted_as_seen(self):
        rows=candidate_views(grounded_payload(content_parts('query'),[source(picture=True) for _ in range(9)]))
        with self.assertRaises(ValueError):evidence_indices({'evidence':[entry(8,kind='image',witness='shape')]},rows)

    def test_index_type_range_duplicates_rejected(self):
        for values in ([entry(True)],[entry(-1)],[entry(1)],[entry(),entry()]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                evidence_indices({'evidence':values},self.rows)

    def test_extra_answer_or_choice_fields_rejected(self):
        for obj in ({'evidence':[],'answer':'a'},{'evidence':[dict(entry(),answer='a')]},
                    {'selected':[0]},{'evidence':'none'}):
            with self.subTest(obj=obj),self.assertRaises(ValueError):evidence_indices(obj,self.rows)

    def test_empty_selection_is_valid(self):
        self.assertEqual(evidence_indices({'evidence':[]},self.rows),[])

    def test_non_dict_and_unknown_kind_rejected(self):
        for item in (None,1,[],entry(kind='answer')):
            with self.subTest(item=item),self.assertRaises(ValueError):
                evidence_indices({'evidence':[item]},self.rows)

    def test_all_options_remain_equal_untrusted_context(self):
        p=grounded_payload(content_parts('query'),[source('anchor')],['B','A','B'])
        header=json.loads(p['messages'][1]['content'][0]['text'])
        self.assertEqual(header['options'],['B','A','B'])
        self.assertNotIn('B',p['messages'][0]['content'])
        self.assertEqual(p['model'],'gpt-4o-mini')
        self.assertEqual(p['max_completion_tokens'],1024)

    def test_payload_builder_does_not_mutate_inputs(self):
        parts=[image()];sources=[source(picture=True)];options=['one','two']
        before=copy.deepcopy([parts,sources,options])
        grounded_payload(parts,sources,options)
        self.assertEqual([parts,sources,options],before)

    def test_system_and_candidate_map_frozen(self):
        p=copy.deepcopy(self.payload);p['messages'][0]['content']='different'
        with self.assertRaises(ValueError):candidate_views(p)
        p=copy.deepcopy(self.payload)
        header=json.loads(p['messages'][1]['content'][0]['text']);header['candidates'][0]['index']=True
        p['messages'][1]['content'][0]['text']=json.dumps(header)
        with self.assertRaises(ValueError):candidate_views(p)

    def test_factory_requires_explicit_activation(self):
        with self.assertRaises(ValueError):create_grounded_app({})

    def test_original_scope_and_budget_transport_preserved(self):
        self.assertEqual(GroundedServiceTransport.receipt_scope,'native-mem0-raw-v4')
        self.assertEqual(GroundedServiceTransport.output_token_cap,1024)

    def test_response_stop_and_contract_required(self):
        obj=object.__new__(GroundedBudgetedTransport);obj.model='gpt-4o-mini'
        good={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'evidence':[entry()]})}}]}
        obj._contract(good,self.payload)
        for key,value in [('finish_reason','length'),('finish_reason',None)]:
            bad=copy.deepcopy(good);bad['choices'][0][key]=value
            with self.assertRaises(ProviderFailure):obj._contract(bad,self.payload)
        bad=copy.deepcopy(good);bad['choices'][0]['message']['content']='{"evidence":[],"evidence":[]}'
        with self.assertRaises(ProviderFailure):obj._contract(bad,self.payload)

    def test_dated_model_exact(self):
        obj=object.__new__(GroundedBudgetedTransport);obj.model='gpt-4o-mini'
        with self.assertRaises(ProviderFailure):obj._response_model({'model':'gpt-4o-mini-latest'})


class WitnessFake:
    def __init__(self):self.calls=[];self.mode='text';self.failed=False
    def ready(self):pass
    def __call__(self,endpoint,payload):
        self.calls.append(payload)
        if self.failed:raise RuntimeError('synthetic provider failure')
        rows=candidate_views(payload);evidence=[]
        if self.mode=='text':
            evidence=[entry(r['index'],witness=r['text_prefix'][:24]) for r in rows if r['text_prefix']]
        elif self.mode=='invalid':evidence=[entry(witness='not in source')]
        elif self.mode=='image':evidence=[entry(r['index'],'image','visible pattern') for r in rows if r['image_preview_attached']]
        return {'choices':[{'message':{'content':json.dumps({'evidence':evidence[:8]})}}]}


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0 dependency')
class GroundedNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.e,self.t,self.c=FakeEmbedding(),WitnessFake(),FakeCards()
        self.store=GroundedSystem(Path(self.temp.name)/'memory.sqlite',embedding=self.e,transport=self.t,card_adapter=self.c)
        self.addCleanup(self.close)
    def close(self):
        self.store.close()
        conn=getattr(self.store.native.db,'connection',None)
        if conn:conn.close()
    def add(self,text,uid='alice',session='s',request='r'):
        return self.store.add(dict(user_id=uid,session_id=session,request_id=request,
                                  messages=[dict(role='user',content=x) for x in text]))
    def search(self,query,uid='alice'):
        return self.store.search(dict(user_id=uid,query=query,options=['choice1','choice2'],top_k=100))['data']

    def test_complete_originals_retained_cross_session(self):
        one='Amber item entered the northern locker. '+'original '*500
        two='The same Amber item later moved to the courtyard.'
        self.add([one],session='first');self.add([two],session='second',request='r2')
        rows=self.search('Where did Amber move?')
        self.assertEqual({text_of(r['content']) for r in rows},{one,two})
        self.assertEqual(len(self.t.calls),1);self.assertEqual(self.c.calls,0)
        self.assertEqual(self.store.capabilities()['version'],VERSION)

    def test_empty_witness_selection_has_no_fallback(self):
        self.add(['an unrelated record']);self.t.mode='empty'
        self.assertEqual(self.search('missing information'),[])

    def test_invalid_quote_not_hidden_as_empty_or_original_tail(self):
        self.add(['an original record']);self.t.mode='invalid'
        with self.assertRaises(ValueError):self.search('original')

    def test_exception_propagates_once(self):
        self.add(['source']);self.t.failed=True
        with self.assertRaises(RuntimeError):self.search('source')
        self.assertEqual(len(self.t.calls),1)

    def test_cross_user_isolation_and_query_not_persisted(self):
        self.add(['allowed source']);self.add(['PRIVATE_OTHER'],uid='bob')
        self.search('NEW_QUERY_SENTINEL')
        self.assertNotIn('PRIVATE_OTHER',json.dumps(self.t.calls))
        with self.store.connect() as db:stored=db.execute('select content from memories').fetchall()
        self.assertNotIn('NEW_QUERY_SENTINEL',str(stored))

    def test_exact_image_identity_survives_empty_model_selection(self):
        original=[dict(type='text',text='original picture'),image()]
        self.add([original]);self.t.mode='empty'
        self.assertEqual(self.search([image()])[0]['content'],original)

    def test_generated_witness_not_returned_in_place_of_original_image(self):
        original=[image()];self.add([original]);self.t.mode='image'
        rows=self.search('visible pattern')
        self.assertEqual(rows[0]['content'],original)
        self.assertNotIn('witness',rows[0])

    def test_options_capacity_rejected_before_dispatch(self):
        self.add(['source']);before=len(self.e.calls)
        with self.assertRaises(ContractError):
            self.store.search(dict(user_id='alice',query='source',options=['x']*33,top_k=100))
        self.assertEqual(len(self.e.calls),before);self.assertEqual(self.t.calls,[])

