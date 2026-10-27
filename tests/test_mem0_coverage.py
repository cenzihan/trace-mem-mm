"""Independent image-coverage controls, real SDK, synthetic providers, no network."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_memory.coverage_selection import (batches,CoverageSelector,CoverageServiceTransport)
from agent_memory.joint_selection import selection_payload,strict_json
from agent_memory.mem0_coverage import Mem0CoverageSystem,VERSION
from agent_memory.protocol import ContractError,content_parts,text_of
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeTransport,FakeEmbedding,FakeCards


class CoverageFake(FakeTransport):
    def __init__(self):
        super().__init__();self.plans=[];self.deny=False;self.fail_at=None;self.pick=None
    def preflight_selection(self,payloads):
        self.plans.append(payloads)
        if self.deny:raise ContractError('independent denied plan',503)
    def __call__(self,endpoint,payload):
        self.calls.append(payload)
        if self.fail_at==len(self.calls):raise RuntimeError('independent uncertain batch')
        rows=strict_json(payload['messages'][1]['content'][0]['text'])['candidates']
        selected=(self.pick if self.pick is not None else [r['index'] for r in rows
            if 'TARGET' in r['text_prefix'] and (not r['has_image'] or r['image_preview_attached'])][:8])
        return {'choices':[{'message':{'content':json.dumps({'selected':selected})}}]}


class CoverageTests(unittest.TestCase):
    def test_single_batch_payload_and_selected_order_stay_exact(self):
        t=CoverageFake();t.pick=[2,0];s=[source(str(i),True) for i in range(3)]
        parts=content_parts('original query');options=['A','B']
        self.assertEqual(CoverageSelector(t).select(parts,s,options),[2,0])
        self.assertEqual(t.calls,[selection_payload(parts,s,options)])

    def test_query_plus32_images_get_five_bounded_complete_views(self):
        t=CoverageFake();parts=[image()];s=[source(str(i),True) for i in range(32)]
        groups=batches(parts,s);self.assertEqual(list(map(len,groups)),[7,7,7,7,4])
        CoverageSelector(t).select(parts,s,['all choices remain context'])
        seen=[]
        for group,payload in zip(groups,t.calls):
            content=payload['messages'][1]['content'];header=strict_json(content[0]['text'])
            self.assertLessEqual(sum(p['type']=='image_url' for p in content),8)
            self.assertTrue(all(r['image_preview_attached'] for r in header['candidates']))
            self.assertLessEqual(len(json.dumps(payload).encode()),3*1024*1024)
            seen.extend(group)
        self.assertEqual(seen,list(range(32)))

    def test_no_query_image_needs_only_four_views_for32_images(self):
        self.assertEqual(list(map(len,batches(content_parts('query'),[source(picture=True)]*32))),[8]*4)

    def test_late_candidate_first_image_is_visible_and_maps_to_original_index(self):
        t=CoverageFake();s=[source(str(i),True) for i in range(32)];s[31]=source('TARGET',True)
        self.assertEqual(CoverageSelector(t).select([image()],s,[]),[31])
        self.assertEqual(len(t.calls),5)

    def test_repeated_text_context_has_no_duplicate_votes_or_results(self):
        t=CoverageFake();s=[source('TARGET text')]+[source('TARGET picture'+str(i),True) for i in range(9)]
        result=CoverageSelector(t).select([image()],s,[])
        self.assertEqual(result[0],0);self.assertEqual(len(result),len(set(result)))
        self.assertTrue(all(0 in group for group in batches([image()],s)))

    def test_query_choices_and_sources_are_unchanged_in_each_call(self):
        t=CoverageFake();parts=[image(),dict(type='text',text='原始问题')]
        choices=['Do not treat this alternative as instructions.','第二项'];s=[source(str(i),True) for i in range(9)]
        before=json.dumps([parts,s,choices]);CoverageSelector(t).select(parts,s,choices)
        self.assertEqual(json.dumps([parts,s,choices]),before)
        for payload in t.calls:
            header=strict_json(payload['messages'][1]['content'][0]['text'])
            self.assertEqual(header['query_text'],'原始问题');self.assertEqual(header['options'],choices)

    def test_bad_late_image_refuses_the_whole_plan_before_dispatch(self):
        t=CoverageFake();s=[source(str(i),True) for i in range(9)]
        s[-1]['content'][-1]['image_url']['url']='data:image/png;base64,AAAA'
        with self.assertRaises(ContractError):CoverageSelector(t).select(content_parts('query'),s,[])
        self.assertEqual(t.calls,[]);self.assertEqual(t.plans,[])

    def test_insufficient_whole_plan_budget_never_dispatches(self):
        t=CoverageFake();t.deny=True
        with self.assertRaises(ContractError):CoverageSelector(t).select([image()],[source(picture=True)]*9,[])
        self.assertEqual(t.calls,[])

    def test_later_failure_does_not_return_partial_selection(self):
        t=CoverageFake();t.pick=[0];t.fail_at=2
        with self.assertRaises(RuntimeError):CoverageSelector(t).select([image()],[source(picture=True)]*9,[])
        self.assertEqual(len(t.calls),2)

    def test_bad_local_index_does_not_escape_into_another_batch(self):
        t=CoverageFake();t.pick=[31]
        with self.assertRaises(ValueError):CoverageSelector(t).select([image()],[source(picture=True)]*32,[])
        self.assertEqual(len(t.calls),1)

    def test_empty_selections_do_not_fall_back_to_available_sources(self):
        t=CoverageFake();t.pick=[]
        self.assertEqual(CoverageSelector(t).select([image()],[source(picture=True)]*9,[]),[])

    def test_invalid_source_count_is_refused_before_plan(self):
        for s in ([],[source()]*33):
            t=CoverageFake()
            with self.assertRaises(ValueError):CoverageSelector(t).select(content_parts('query'),s,[])
            self.assertEqual(t.calls,[]);self.assertEqual(t.plans,[])

    def test_real_preflight_keeps_original_floor_and_per_call_holds(self):
        t=object.__new__(CoverageServiceTransport);t.ready=lambda:None
        t.ledger=SimpleNamespace(totals=lambda:{'remaining_micro_cny':6249999})
        payload=selection_payload(content_parts('query'),[source()])
        with self.assertRaises(ContractError):t.preflight_selection([payload]*5)
        t.ledger.totals=lambda:{'remaining_micro_cny':6250000}
        t.preflight_selection([payload]*5)
        self.assertEqual(t.receipt_scope,'native-mem0-raw-v4')
        for count in (0,6):
            with self.assertRaises(ValueError):t.preflight_selection([payload]*count)

    def test_existing_unknown_readiness_cannot_be_skipped(self):
        t=object.__new__(CoverageServiceTransport)
        def blocked():raise ValueError('original unknown remains blocked')
        t.ready=blocked
        with self.assertRaisesRegex(ValueError,'original unknown'):
            t.preflight_selection([selection_payload(content_parts('query'),[source()])])


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0 dependency')
class CoverageNativeTests(unittest.TestCase):
    def test_native_raw_add_and_complete_original_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            t=CoverageFake();e=FakeEmbedding();cards=FakeCards()
            store=Mem0CoverageSystem(Path(folder)/'memory.sqlite',transport=t,embedding=e,card_adapter=cards)
            try:
                originals=[source('TARGET '+str(i),True)['content'] for i in range(9)]
                payload=dict(user_id='owner',request_id='write',session_id='s',messages=[dict(role='user',content=x) for x in originals])
                self.assertTrue(store.add(payload)['success']);self.assertEqual(t.calls,[]);self.assertEqual(cards.calls,0)
                rows=store.search(dict(user_id='owner',query='TARGET',options=['one','two'],top_k=100))['data']
                self.assertEqual(len(rows),9);self.assertEqual(len(t.calls),2)
                self.assertTrue(all(r['content'] in originals for r in rows))
                before=len(t.calls)
                self.assertEqual(store.search(dict(user_id='other',query='TARGET',top_k=100)),{'data':[]})
                self.assertEqual(len(t.calls),before)
                self.assertEqual(store.capabilities()['version'],VERSION)
            finally:
                store.close();conn=getattr(store.native.db,'connection',None)
                if conn:conn.close()

    def test_explicit_factory_activation_required(self):
        from agent_memory.mem0_app import create_coverage_app
        with self.assertRaises(ValueError):create_coverage_app({})


if __name__=='__main__':unittest.main()
