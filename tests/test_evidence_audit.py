"""Synthetic identifier-only diagnostic contracts, not performance scenarios."""
import unittest
from agent_memory.evidence_audit import selection_trace

class EvidenceAuditTests(unittest.TestCase):
    def test_packing_and_citation_losses_are_distinct(self):
        r=selection_trace(['g1','g2'],['g1','g2','noise'],['g1','noise'],[])
        self.assertEqual(r['gold_lost_during_packing'],['g2']);self.assertEqual(r['packed_gold_not_cited'],['g1'])
        self.assertEqual(r['retrieval_fraction'],1);self.assertEqual(r['context_fraction'],.5)
    def test_empty_gold_not_false_zero_performance(self):
        self.assertIsNone(selection_trace([],['source'],['source'],[])['context_fraction'])
    def test_same_blob_different_occurrences_not_merged(self):
        r=selection_trace(['r'],['r'],['r'],['r'],gold_images=[('r',0),('r',1)],packed_images=[('r',0),('r',1)],sent_images=[('r',0)])
        self.assertEqual(r['gold_image_occurrences'],2);self.assertEqual(r['available_gold_images_not_sent'],1)
    def test_inconsistent_stage_or_remote_image_reject(self):
        self.assertRaises(ValueError,selection_trace,['g'],['g'],['other'],[])
        self.assertRaises(ValueError,selection_trace,['g'],['g'],['g'],['unknown'])
        self.assertRaises(ValueError,selection_trace,['g'],['g'],['g'],[],sent_images=[('unknown',0)])

if __name__=='__main__':unittest.main()
