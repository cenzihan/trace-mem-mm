"""Synthetic contract fixtures only; no native corpus performance run."""
import unittest
from agent_memory.baseline_preparation import source_views,fuse_to_native_candidates,estimate_atm_plan,ComparisonBudget

class BaselinePreparationTests(unittest.TestCase):
    def record(self,identity='source',namespace='u'):
        return dict(namespace=namespace,native_id=identity,modality='image',timestamp='source date',short_text='unit short',detail_text='unit detail',
                    occurrence_ids=[identity+':image:0',identity+':image:1'],preprocessing_models=['Qwen3-VL-2B-Instruct'])
    def test_provenance_no_silent_strict_relabel_or_query_gold_fields(self):
        self.assertRaises(ValueError,source_views,self.record(),condition='strict_from_reviewed_source')
        r=self.record();r['answer']='unit forbidden';self.assertRaises(ValueError,source_views,r,condition='external_sgm_research')
        r=self.record();r['query']='unit forbidden';self.assertRaises(ValueError,source_views,r,condition='external_sgm_research')
    def test_all_occurrences_and_view_parent_identity_preserved(self):
        units=source_views(self.record(),condition='external_sgm_research');self.assertEqual(len(units),3)
        self.assertTrue(all(len(u['occurrence_ids'])==2 and u['native_id']=='source' for u in units))
        result=fuse_to_native_candidates([[u['unit_id'] for u in units]],units,'u')
        self.assertEqual(result['native_evidence_count'],1)
    def test_namespace_and_empty_or_oversized_inputs_fail_closed(self):
        units=source_views(self.record(),condition='external_sgm_research')
        self.assertRaises(ValueError,fuse_to_native_candidates,[[units[0]['unit_id']]],units,'other')
        r=self.record();r.update(short_text='',detail_text='');self.assertRaises(ValueError,source_views,r,condition='external_sgm_research')
        r=self.record();r['detail_text']='x'*8193;self.assertRaises(ValueError,source_views,r,condition='external_sgm_research')
    def test_single_global_candidate_budget_not_per_lane_inflation(self):
        units=[u for i in range(120) for u in source_views(self.record(str(i)),condition='external_sgm_research')]
        result=fuse_to_native_candidates([[u['unit_id'] for u in units],[u['unit_id'] for u in reversed(units)]],units,'u')
        self.assertEqual(len(result['candidates']),100);self.assertEqual(len(result['evidence']),20)
        self.assertRaises(ValueError,ComparisonBudget(candidates=200).validate)
    def test_explicit_conservative_plan_no_guard_or_provider_effects(self):
        p=estimate_atm_plan(native_artifacts=11034,questions=20)
        self.assertEqual(p['provider_request_upper_bound'],3407);self.assertEqual(p['cumulative_preflight_cny'],'188.75')
        self.assertTrue(p['planning_only']);self.assertFalse(p['existing_completed_guard_reusable'])
        self.assertRaises(ValueError,estimate_atm_plan,native_artifacts=11035,questions=20)

if __name__=='__main__':unittest.main()
