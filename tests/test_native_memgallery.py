"""Schema-only synthetic unit cases; no native performance questions or media."""
import json
import unittest
from agent_memory.native_memgallery import plan_rounds,materialize_round


class NativeMemGalleryTests(unittest.TestCase):
    def container(self):
        return {'human-annotated QAs':[dict(question='GOLD_QUESTION_SENTINEL',answer='GOLD_ANSWER_SENTINEL',clue=['GOLD_CLUE_SENTINEL'])],
            'character_profile':dict(name='PROFILE_NOT_A_DIALOGUE'),
            'multi_session_dialogues':[dict(session_id='S1',date='2025-01-01',dialogues=[dict(round='D1',
                user='Original user message',assistant='Original assistant message',image_id=['I1','I2'],
                input_image=['../image/unit/a.jpg','../image/unit/b.jpg'],image_caption=['caption A','caption B'])])]}
    def test_gold_and_profile_do_not_enter_plans(self):
        plans=plan_rounds(self.container(),'dataset-task-u')
        for forbidden in ('GOLD_QUESTION','GOLD_ANSWER','GOLD_CLUE','PROFILE_NOT_A_DIALOGUE'):
            self.assertNotIn(forbidden,json.dumps(plans))
    def test_all_images_and_source_order_preserved(self):
        plan=plan_rounds(self.container(),'u')[0];parts=plan['messages'][0]['content']
        self.assertEqual([p['type'] for p in parts],['text','native_image_ref','text','native_image_ref','text'])
        self.assertEqual([p['part_ordinal'] for p in plan['provenance']['images']],[1,3])
        self.assertEqual(plan['provenance']['source_date'],'2025-01-01');self.assertNotIn('timestamp',plan['messages'][0])
    def test_explicit_materialization_and_namespace(self):
        plan=plan_rounds(self.container(),'u')[0];reads=[]
        def reader(path):reads.append(path);return b'\xff\xd8\xffunit'
        payload=materialize_round(plan,reader)
        self.assertEqual(len(reads),2);self.assertEqual(payload['user_id'],'u');self.assertNotIn('provenance',payload)
        self.assertEqual(plan['messages'][0]['content'][1]['type'],'native_image_ref')
        self.assertNotEqual(plan['request_id'],plan_rounds(self.container(),'other-u')[0]['request_id'])
    def test_array_mismatch_never_silently_drops_images(self):
        data=self.container();data['multi_session_dialogues'][0]['dialogues'][0]['image_caption'].pop()
        self.assertRaises(ValueError,plan_rounds,data,'u')
    def test_path_escape_rejected(self):
        data=self.container();data['multi_session_dialogues'][0]['dialogues'][0]['input_image'][0]='../image/../../secret'
        self.assertRaises(ValueError,plan_rounds,data,'u')
    def test_duplicate_round_and_unconfigured_reader_rejected(self):
        data=self.container();data['multi_session_dialogues'][0]['dialogues']*=2
        self.assertRaises(ValueError,plan_rounds,data,'u')
        self.assertRaises(ValueError,materialize_round,plan_rounds(self.container(),'u')[0],None)


if __name__=='__main__':unittest.main()
