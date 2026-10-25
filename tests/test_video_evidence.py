"""Unit-only fake media process; no real dataset video or image generation benchmark."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from agent_memory.video_evidence import extract_video_frames,verified_video,FramePolicy,bounded_process

class VideoEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.raw=b'\0\0\0\x18ftypisomUNIT_ONLY'
        (self.root/'unit.mp4').write_bytes(self.raw);self.digest=hashlib.sha256(self.raw).hexdigest();self.calls=[];self.audio=False
    def tearDown(self):self.tmp.cleanup()
    def runner(self,command,timeout,max_bytes):
        self.calls.append(command)
        if command[0]=='unit-ffprobe':return json.dumps(dict(streams=[dict(codec_type='video')]+([dict(codec_type='audio')] if self.audio else []),format=dict(duration='12'))).encode()
        return (Path(__file__).resolve().parents[1]/'fixtures/red.png').read_bytes()
    def extract(self):return extract_video_frames(self.root,'unit.mp4',self.digest,'native-unit-video',ffprobe='unit-ffprobe',ffmpeg='unit-ffmpeg',runner=self.runner,policy=FramePolicy(frames=3))
    def test_frames_keep_one_native_evidence_and_offsets(self):
        result=self.extract();self.assertEqual(len(result['frames']),3);self.assertEqual(result['native_evidence_count'],1)
        self.assertEqual([r['frame_offset_seconds'] for r in result['frames']],[2,6,10])
        self.assertEqual(len({r['frame_occurrence_id'] for r in result['frames']}),3)
        self.assertFalse(result['complete_temporal_coverage']);self.assertFalse(result['audio_dropped'])
    def test_audio_blocks_before_frame_decode(self):
        self.audio=True;self.assertRaises(ValueError,self.extract);self.assertEqual(len(self.calls),1)
    def test_path_hash_and_policy_reject_before_media_process(self):
        for relative in ('../unit.mp4','https://remote.example/a.mp4','C:/unit.mp4'):
            self.assertRaises(ValueError,verified_video,self.root,relative,self.digest)
        self.assertRaises(ValueError,verified_video,self.root,'unit.mp4','0'*64)
        self.assertRaises(ValueError,verified_video,self.root,'unit.mp4',self.digest,FramePolicy(frames=100))
        self.assertEqual(self.calls,[])
    def test_unverified_missing_tool_does_not_launch(self):
        self.assertRaises(ValueError,bounded_process,['unit-ffmpeg'],1,1024)

if __name__=='__main__':unittest.main()
