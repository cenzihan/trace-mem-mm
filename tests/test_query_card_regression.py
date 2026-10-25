"""Image/OCR query regression, synthetic unit bytes only, no performance cases."""
import tempfile
import unittest
from pathlib import Path
from agent_memory.system import AgentMemorySystem,RetrievalConfig
from agent_memory.development_run import DevelopmentRun


class QueryCardRegression(unittest.TestCase):
    def test_image_quote_not_in_query_can_guide_search_but_never_become_fact(self):
        class Cards:
            model='gpt-4o-mini'
            def extract(self,parts):return dict(summary='source evidence visible label',evidence_quotes=['OCR NOT IN QUERY'])
        with tempfile.TemporaryDirectory() as tmp:
            store=AgentMemorySystem(Path(tmp)/'m.sqlite',RetrievalConfig(visual_query_cards=True),card_adapter=Cards())
            # Add keeps its grounding safety boundary; no model needed for source construction.
            raw=AgentMemorySystem(store.path);raw.add(dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content='source evidence visible label')]))
            result,trace=store.retrieve(dict(user_id='u',top_k=1,query=[dict(type='image_url',image_url=dict(url='data:image/png;base64,iVBORw0KGgo='))]))
            self.assertTrue(result['data']);self.assertEqual(trace['query_unverified_visual_quotes'],1)
            self.assertFalse(trace['query_summary_is_verified_fact']);self.assertEqual(store.inspect_user('u')['memories'],1)
    def test_add_still_rejects_unverified_text_quote(self):
        class Cards:
            model='gpt-4o-mini'
            def extract(self,parts):return dict(summary='summary',evidence_quotes=['not in source'])
        with tempfile.TemporaryDirectory() as tmp:
            store=AgentMemorySystem(Path(tmp)/'m.sqlite',card_adapter=Cards())
            self.assertRaises(ValueError,store.add,dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content='source')]))
            self.assertEqual(store.inspect_user('u')['memories'],0)
    def test_explicit_native_scope_keeps_default_old_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertRaises(ValueError,DevelopmentRun,Path(tmp)/'bad.sqlite',create=True,max_requests=100,max_reservation_cny='20')
            run=DevelopmentRun(Path(tmp)/'native.sqlite',create=True,max_requests=100,max_reservation_cny='20',approval_scope='native_100_20')
            self.assertEqual(run.summary()['max_requests'],100)


if __name__=='__main__':unittest.main()
