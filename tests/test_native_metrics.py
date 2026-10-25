"""Tiny ID-only unit tests, not a performance dataset."""
import unittest
from agent_memory.native_metrics import retrieval_metrics


class NativeMetricsTests(unittest.TestCase):
    def test_partial_recall_distinct_from_any_and_all(self):
        value=retrieval_metrics(['a','x'],['a','b'],2)
        self.assertEqual(value['fractional_recall'],.5);self.assertTrue(value['any_hit']);self.assertFalse(value['all_evidence_coverage'])
    def test_duplicate_cards_cannot_inflate_occurrence_recall(self):
        value=retrieval_metrics(['a','a','a'],['a','b'],3)
        self.assertEqual(value['hits'],1);self.assertEqual(value['fractional_recall'],.5)
    def test_top_k_cutoff(self):
        value=retrieval_metrics(['x','a','b'],['a','b'],2)
        self.assertEqual(value['fractional_recall'],.5)
    def test_no_gold_does_not_invent_perfect_recall(self):
        value=retrieval_metrics([],[],3)
        self.assertIsNone(value['fractional_recall']);self.assertIsNone(value['any_hit']);self.assertIsNone(value['all_evidence_coverage'])
    def test_invalid_inputs(self):
        for args in ((['a'],['a'],0),(['a'],['a'],True),([1],['a'],1),(['a'],[None],1)):
            self.assertRaises(ValueError,retrieval_metrics,*args)


if __name__=='__main__':unittest.main()
