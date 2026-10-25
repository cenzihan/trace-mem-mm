"""Synthetic interface fixtures only; no dataset performance answers."""
import unittest
from agent_memory.native_atm import split_native_qa,corpus_inventory,split_overlap

class NativeATMTests(unittest.TestCase):
    def row(self):return dict(id='unit-q',question='unit question',answer='unit gold',evidence_ids=['img1'],qtype='open_end')
    def test_questions_exclude_all_gold_and_pool(self):
        row=self.row();row['niah_evidence_ids']=['img1','noise'];questions,gold=split_native_qa([row])
        self.assertEqual(set(questions[0]),{'id','question','qtype'});self.assertIn('answer',gold[0]);self.assertIn('niah_evidence_ids',gold[0])
    def test_duplicate_ids_and_missing_pool_evidence_reject(self):
        self.assertRaises(ValueError,split_native_qa,[self.row(),self.row()])
        row=self.row();row['niah_evidence_ids']=['noise'];self.assertRaises(ValueError,split_native_qa,[row])
    def test_inventory_keeps_video_audio_and_sgm_explicit(self):
        r=corpus_inventory(['data/raw_memory/image/img1.jpg','data/raw_memory/video/vid1.mp4','data/raw_memory/audio/aud1.wav','data/raw_memory/email/emails.json','data/processed_memory/image_batch_results.json'])
        self.assertEqual(r['unsupported_modalities'],['video','audio']);self.assertEqual(r['counts']['external_processed'],1)
        self.assertTrue(r['external_sgm_never_used']);self.assertFalse(r['raw_media_complete'])
    def test_paths_and_evidence_collision_reject(self):
        for path in ('../escape','C:/absolute','data/../escape'):
            self.assertRaises(ValueError,corpus_inventory,[path])
        self.assertRaises(ValueError,corpus_inventory,['data/raw_memory/image/id.jpg','data/raw_memory/video/id.mp4'])
    def test_related_splits_are_not_independent(self):
        q,_=split_native_qa([self.row()]);r=split_overlap(q,q,{'25':q})
        self.assertEqual(r['hard_ids_in_regular'],1);self.assertEqual(r['niah']['25']['same_question_in_hard'],1)
    def test_changed_ids_cannot_hide_question_text_overlap(self):
        r=split_overlap([dict(id='a',question='A unit question')],[dict(id='b',question='A unit question')],{})
        self.assertEqual(r['hard_ids_in_regular'],0);self.assertEqual(r['hard_exact_question_text_in_regular'],1)

if __name__=='__main__':unittest.main()
