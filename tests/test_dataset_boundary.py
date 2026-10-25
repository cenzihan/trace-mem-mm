"""Synthetic bytes test the ingestion boundary only, no performance benchmark."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from agent_memory.dataset_boundary import DatasetBoundaryError,read_verified_evidence,validate_manifest


class DatasetBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'image.bin').write_bytes(b'fixture-only')
        self.manifest=dict(dataset='synthetic unit boundary',native_task='native task',source_url='https://example.org/native',
             split='unit-only',license='explicit unit permission',review_status='native_use_reviewed',
             intended_use='native_offline_evaluation',evaluation_material=False,source_overlap_review='completed',
             evidence_origin='raw_media_or_original_dialogue',preprocessing_models=[],
             files=[dict(path='image.bin',role='evidence',sha256=hashlib.sha256(b'fixture-only').hexdigest())])
    def tearDown(self):self.tmp.cleanup()
    def test_verified_evidence_only(self):self.assertEqual(read_verified_evidence(self.root,self.manifest,'image.bin'),b'fixture-only')
    def test_gold_and_questions_never_evidence(self):
        for role in ('questions','gold','native_container'):
            self.manifest['files'][0]['role']=role
            self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,self.manifest,'image.bin')
    def test_pending_reviews_block(self):
        for key,value in (('license','unverified'),('review_status','pending'),('source_overlap_review','pending')):
            item=copy.deepcopy(self.manifest);item[key]=value
            self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,item,'image.bin')
    def test_protected_material_blocked(self):
        self.manifest['evaluation_material']=True
        self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,self.manifest,'image.bin')
    def test_training_or_tuning_not_supported(self):
        self.manifest['intended_use']='training'
        self.assertRaises(DatasetBoundaryError,validate_manifest,self.manifest)
    def test_escaping_path_blocked(self):
        for path in ('../image.bin',str(self.root/'image.bin')):
            self.manifest['files'][0]['path']=path
            self.assertRaises(DatasetBoundaryError,validate_manifest,self.manifest)
    def test_hash_mismatch_blocked(self):
        self.manifest['files'][0]['sha256']='0'*64
        self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,self.manifest,'image.bin')
    def test_missing_or_duplicate_file_blocked(self):
        self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,self.manifest,'unknown.bin')
        self.manifest['files']*=2
        self.assertRaises(DatasetBoundaryError,validate_manifest,self.manifest)
    def test_qwen_preprocessed_sgm_not_academic_raw_input(self):
        self.manifest.update(evidence_origin='external_derived',preprocessing_models=['Qwen3-VL-2B'])
        self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,self.manifest,'image.bin')
    def test_unknown_or_conflicting_preprocessing_blocks(self):
        self.manifest['preprocessing_models']=['gpt-4o-mini']
        self.assertRaises(DatasetBoundaryError,validate_manifest,self.manifest)
        self.manifest.update(evidence_origin='external_derived',preprocessing_models=[])
        self.assertRaises(DatasetBoundaryError,read_verified_evidence,self.root,self.manifest,'image.bin')


if __name__=='__main__':unittest.main()
