"""Synthetic field-shape fixtures only, no native dataset contents or provider calls."""
import copy
import unittest
from agent_memory.atm_sgm_preparation import prepare_sgm_sources

class ATMSGMPreparationTests(unittest.TestCase):
    def setUp(self):
        self.images=[dict(image_path='/author-root/image/unit-image.jpg',caption='unit caption',timestamp='source date',tags=[])]
        self.videos=[dict(video_path='/author-root/video/unit-video.mp4',caption='unit video caption',timestamp='source date')]
        self.emails=[dict(id='email-unit',timestamp='source date',short_summary='unit summary',detail='unit detail')]
    def run_prepare(self,**overrides):
        options=dict(namespace='native-unit-user',expected_images={'unit-image'},expected_videos={'unit-video'},expected_email_count=1,preprocessing_models=['Qwen3-VL-2B-Instruct'])
        options.update(overrides)
        return prepare_sgm_sources(self.images,self.videos,self.emails,**options)
    def test_complete_native_sources_no_invented_sessions_or_media_reads(self):
        r=self.run_prepare();self.assertEqual(r['source_counts'],dict(image=1,video=1,email=1))
        self.assertFalse(r['academic_strict_end_to_end']);self.assertFalse(r['media_loaded']);self.assertFalse(r['external_encoding_enabled'])
        self.assertTrue(all('session_id' not in x for x in r['records']))
    def test_partial_duplicate_or_remote_source_blocks(self):
        self.assertRaises(ValueError,self.run_prepare,expected_images={'unit-image','missing'})
        self.images.append(copy.deepcopy(self.images[0]));self.assertRaises(ValueError,self.run_prepare)
        self.images=self.images[:1];self.images[0]['image_path']='https://remote.example/unit-image.jpg';self.assertRaises(ValueError,self.run_prepare)
    def test_gold_fields_and_empty_visual_reject(self):
        self.images[0]['answer']='unit forbidden';self.assertRaises(ValueError,self.run_prepare)
        self.images[0].pop('answer');self.images[0]['caption']='';self.assertRaises(ValueError,self.run_prepare)
    def test_email_count_and_root_schema_explicit(self):
        self.assertRaises(ValueError,self.run_prepare,expected_email_count=2)
        self.images={'results':self.images};self.assertRaises(ValueError,self.run_prepare)

if __name__=='__main__':unittest.main()
