"""Synthetic probes only: no keys, external provider, official data or replay."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.independent_control_probe import IndependentControlProbe
from agent_memory.consolidated_selection import ConsolidatingSelector
from agent_memory.coverage_selection import CoverageSelector
from agent_memory.joint_selection import JointSelector
from agent_memory.protocol import content_parts, ContractError
from test_mem0_consolidated import Scripted
from test_mem0_joint import source


class ProbeTests(unittest.TestCase):
    def setup_control(self, answers, target_index=8):
        sources = [source('independent-source-' + str(i), True) for i in range(9)]
        target = sources[target_index]['content'] if target_index is not None else content_parts('absent')
        probe = IndependentControlProbe(provenance='synthetic', target_content=target)
        transport = Scripted(answers)
        return sources, probe, transport, probe.wrap(ConsolidatingSelector(transport))

    def test_distinguishes_pool_batch_final_and_packed_ranks(self):
        s,p,t,w = self.setup_control([[0],[0],[1,0]])
        result = w.select(content_parts('query'), s)
        self.assertEqual(result, [8,0])
        p.observe_returned([s[0],s[8]])  # E.g. final packing differs from selection.
        v=p.snapshot()
        self.assertEqual(v['target_pool_rank'],9)
        self.assertEqual([c['target_input_rank'] for c in v['calls']],[None,1,2])
        self.assertEqual(v['target_selected_rank'],1)
        self.assertEqual(v['target_returned_rank'],2)

    def test_distinguishes_missing_pool_from_model_omission(self):
        s,p,t,w=self.setup_control([[0],[],[0]], None)
        w.select(content_parts('query'),s)
        self.assertEqual(p.snapshot()['target_pool_count'],0)
        self.assertIsNone(p.snapshot()['target_pool_rank'])

    def test_target_dropped_before_final_has_no_final_rank(self):
        s,p,t,w=self.setup_control([[0],[],[0]])
        w.select(content_parts('query'),s)
        v=p.snapshot()
        self.assertEqual(v['calls'][1]['target_input_rank'],1)
        self.assertIsNone(v['calls'][1]['target_selected_rank'])
        self.assertIsNone(v['calls'][2]['target_input_rank'])

    def test_empty_selection_is_complete_not_transport_failure(self):
        s,p,t,w=self.setup_control([[],[]])
        self.assertEqual(w.select(content_parts('query'),s),[])
        p.observe_returned([])
        self.assertEqual(p.snapshot()['outcome'],'selection_complete')
        self.assertEqual(p.snapshot()['returned_count'],0)

    def test_failure_keeps_original_exception_and_unknown_return_rank(self):
        failure=RuntimeError('synthetic secret exception text')
        s,p,t,w=self.setup_control([[0],[0],failure])
        with self.assertRaises(RuntimeError) as caught:w.select(content_parts('query'),s)
        self.assertIs(caught.exception,failure)
        self.assertEqual(len(t.calls),3)
        self.assertEqual(p.snapshot()['outcome'],'selection_failed')
        self.assertIsNone(p.snapshot()['returned_count'])
        self.assertNotIn('secret',json.dumps(p.snapshot()))

    def test_output_contains_no_content_ids_query_options_or_indices(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        w.select(content_parts('secret independent query'),s,['private option'])
        p.observe_returned([dict(s[8],id='private source identifier')])
        serialized=json.dumps(p.snapshot())
        for secret in ('independent-source','secret','private','data:image','selected_indices'):
            self.assertNotIn(secret,serialized)
        self.assertLess(len(serialized.encode()),8192)

    def test_probe_does_not_change_payloads_order_or_preflight(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        before=json.dumps(s)
        expected=Scripted([[0],[0],[1]])
        result=ConsolidatingSelector(expected).select(content_parts('query'),s,['a','b'])
        self.assertEqual(w.select(content_parts('query'),s,['a','b']),result)
        self.assertEqual(t.calls,expected.calls)
        self.assertEqual(t.plans,expected.plans)
        self.assertEqual(t.consolidation_plans,expected.consolidation_plans)
        self.assertEqual(json.dumps(s),before)

    def test_failed_diagnostic_mapping_does_not_mask_provider_or_reorder(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        with patch('scripts.independent_control_probe.selection_payload',side_effect=OSError('private')):
            self.assertEqual(w.select(content_parts('query'),s),[8])
        self.assertFalse(p.snapshot()['diagnostic_valid'])
        self.assertEqual(p.snapshot()['calls'],[])
        self.assertEqual(len(t.calls),3)

    def test_no_reuse_or_return_observation_before_completion(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        with self.assertRaises(ValueError):p.observe_returned([])
        w.select(content_parts('query'),s)
        with self.assertRaises(ValueError):w.select(content_parts('query'),s)
        self.assertEqual(len(t.calls),3)

    def test_output_is_exclusive_and_write_failure_does_not_change_result(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        self.assertEqual(w.select(content_parts('query'),s),[8])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scalars.json'
            self.assertTrue(p.write_once(path));before=path.read_bytes()
            self.assertFalse(p.write_once(path));self.assertEqual(path.read_bytes(),before)
            self.assertFalse(p.write_once(Path(d)/'missing'/'file'))

    def test_no_implicit_official_provenance(self):
        for provenance in ('official','smoke','unknown',''):
            with self.assertRaises(ValueError):
                IndependentControlProbe(provenance=provenance,target_content=content_parts('x'))

    def test_single_batch_coverage_mapping_and_no_extra_calls(self):
        s,p,t,_=self.setup_control([[2,0]],2)
        wrapped=p.wrap(CoverageSelector(t))
        self.assertEqual(wrapped.select(content_parts('query'),s[:3]),[2,0])
        self.assertEqual(len(t.calls),1)
        self.assertEqual(p.snapshot()['target_selected_rank'],1)

    def test_original_budget_denial_stops_before_any_model_call(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        t.deny=True
        with self.assertRaises(ContractError):w.select(content_parts('query'),s)
        self.assertEqual(t.calls,[])
        self.assertEqual(p.snapshot()['calls'],[])
        self.assertEqual(p.snapshot()['outcome'],'selection_failed')

    def test_invalid_final_response_remains_failure_not_partial_result(self):
        s,p,t,w=self.setup_control([[0],[0],[99]])
        with self.assertRaises(ValueError):w.select(content_parts('query'),s)
        self.assertEqual(len(t.calls),3)
        self.assertEqual(p.snapshot()['calls'][-1]['outcome'],'invalid_response')
        self.assertIsNone(p.snapshot()['selected_count'])

    def test_duplicate_targets_are_reported_as_multiple_not_silently_unique(self):
        s,p,t,_=self.setup_control([[1,0]],0)
        wrapped=p.wrap(JointSelector(t))
        self.assertEqual(wrapped.select(content_parts('query'),[s[0],s[0]]),[1,0])
        self.assertEqual(p.snapshot()['target_pool_count'],2)
        self.assertEqual(p.snapshot()['target_pool_rank'],1)

    def test_returned_missing_is_distinct_from_returned_not_observed(self):
        s,p,t,w=self.setup_control([[0],[0],[1]])
        w.select(content_parts('query'),s)
        self.assertIsNone(p.snapshot()['returned_count'])
        p.observe_returned([s[0]])
        self.assertEqual(p.snapshot()['returned_count'],1)
        self.assertIsNone(p.snapshot()['target_returned_rank'])
        with self.assertRaises(ValueError):p.observe_returned([s[8]])
