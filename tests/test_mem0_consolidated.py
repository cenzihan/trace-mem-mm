import json
import unittest
from types import SimpleNamespace
from agent_memory.consolidated_selection import (ConsolidatingSelector,
    ConsolidatingServiceTransport,finalists)
from agent_memory.joint_selection import selection_payload,strict_json
from agent_memory.protocol import content_parts,ContractError
from test_mem0_coverage import CoverageFake
from test_mem0_joint import source
from test_mem0_evidence import image

class Scripted(CoverageFake):
    def __init__(self,answers):
        super().__init__();self.answers=answers;self.consolidation_plans=[]
    def preflight_consolidation(self,payloads):
        self.consolidation_plans.append(payloads)
        if self.deny:raise ContractError('independent budget denial',503)
    def __call__(self,endpoint,payload):
        self.calls.append(payload)
        value=self.answers[len(self.calls)-1]
        if isinstance(value,Exception):raise value
        return {'choices':[{'message':{'content':json.dumps({'selected':value})}}]}

class ConsolidatedTests(unittest.TestCase):
    def test_final_comparison_can_reorder_locally_first_results(self):
        t=Scripted([[0],[0],[1]])
        self.assertEqual(ConsolidatingSelector(t).select(content_parts('query'),[source(picture=True)]*9),[8])
        self.assertEqual(len(t.calls),3)
    def test_all_first_images_then_bounded_global_view_with_query(self):
        t=Scripted([[0],[0],[0],[0],[0],[4]])
        self.assertEqual(ConsolidatingSelector(t).select([image()],[source(picture=True)]*32),[28])
        self.assertEqual(len(t.calls),6)
        self.assertTrue(all(sum(p['type']=='image_url' for p in v['messages'][1]['content'])<=8 for v in t.calls))
    def test_empty_batches_make_no_final_call(self):
        t=Scripted([[],[]])
        self.assertEqual(ConsolidatingSelector(t).select(content_parts('x'),[source(picture=True)]*9),[])
        self.assertEqual(len(t.calls),2)
    def test_failed_final_call_never_returns_partial_survivors(self):
        t=Scripted([[0],[0],RuntimeError('independent failure')])
        with self.assertRaises(RuntimeError):ConsolidatingSelector(t).select(content_parts('x'),[source(picture=True)]*9)
        self.assertEqual(len(t.calls),3)
    def test_single_batch_retains_previous_payload_and_selection(self):
        t=Scripted([[2,0]]);s=[source(picture=True)]*3;p=content_parts('x')
        self.assertEqual(ConsolidatingSelector(t).select(p,s,['a','b']),[2,0])
        self.assertEqual(t.calls,[selection_payload(p,s,['a','b'])]);self.assertEqual(t.consolidation_plans,[])
    def test_round_robin_limits_images_and_keeps_selected_text_without_votes(self):
        s=[source(picture=True)]*20+[source('text')]*12
        selected=[list(range(8))+[20,21],list(range(8,16))+[20,22],list(range(16,20))+[21,23]]
        result=finalists([image()],s,selected)
        self.assertEqual(len([i for i in result if i<20]),7)
        self.assertEqual([i for i in result if i>=20],[20,21,22,23])
        self.assertTrue({0,8,16}.issubset(result));self.assertEqual(len(result),len(set(result)))
    def test_six_call_conservative_funds_checked_before_dispatch(self):
        t=object.__new__(ConsolidatingServiceTransport);t.ready=lambda:None
        t.ledger=SimpleNamespace(totals=lambda:{'remaining_micro_cny':6499999})
        payload=selection_payload(content_parts('x'),[source()])
        with self.assertRaises(ContractError):t.preflight_consolidation([payload]*5)
        t.ledger.totals=lambda:{'remaining_micro_cny':6500000}
        t.preflight_consolidation([payload]*5)
    def test_bad_final_index_cannot_escape_mapping(self):
        t=Scripted([[0],[0],[2]])
        with self.assertRaises(ValueError):ConsolidatingSelector(t).select(content_parts('x'),[source(picture=True)]*9)
    def test_choices_and_originals_unchanged_across_all_calls(self):
        t=Scripted([[0],[0],[0]]);s=[source(picture=True)]*9;p=content_parts('Original?');choices=['choice1','choice2']
        before=json.dumps([p,s,choices]);ConsolidatingSelector(t).select(p,s,choices)
        self.assertEqual(json.dumps([p,s,choices]),before)
        self.assertTrue(all(strict_json(v['messages'][1]['content'][0]['text'])['options']==choices for v in t.calls))
    def test_explicit_activation_required(self):
        from agent_memory.mem0_app import create_consolidated_app
        with self.assertRaises(ValueError):create_consolidated_app({})

