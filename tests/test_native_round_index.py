"""Synthetic interface/regression bytes only; no performance dataset or live calls."""
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime,timezone
from agent_memory.native_round_index import NativeRoundIndex
from agent_memory.native_memgallery import plan_rounds
from agent_memory.protocol import ContractError,image_bytes
from agent_memory.visual_analysis import GPT4oMiniVisualAnalysis
from agent_memory.provider_http import BudgetedJSONTransport,OPENAI_ENDPOINT
from agent_memory.budget import BudgetLedger,VerifiedRate


class NativeRoundTests(unittest.TestCase):
    def plans(self):
        return plan_rounds({'multi_session_dialogues':[dict(session_id='s',date='2025-01-01',dialogues=[
            dict(round='r1',user='PanelX source',assistant='original reply',image_id=['i1'],
                 input_image=['../image/unit/a.png'],image_caption=['source caption']),
            dict(round='r2',user='PanelX second source',assistant='second reply')])]},'u')
    def setup_index(self,root):
        store=NativeRoundIndex(root/'index.sqlite',64);store.register_originals(self.plans(),lambda path:b'\x89PNG\r\n\x1a\nunit')
        rows=store.units('u');digest=json.loads(rows[0]['image_hashes'])[0]
        visual=dict(global_caption='visible panel',visible_text=['PanelX'],regions=['panel at center'],relations=[],named_entities=['PanelX'],uncertainties=[])
        docs=store.documents('u',{digest:visual});vectors=[[1.]*64 for _ in docs]
        return store,docs,vectors
    def test_atomic_index_original_image_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store,docs,vectors=self.setup_index(root)
            self.assertRaises(ContractError,store.search,'u','PanelX',vectors[0])
            store.publish('u',docs,vectors,vectors,dict(embedding_model='text-embedding-v4'))
            result,trace=NativeRoundIndex(root/'index.sqlite',64).search('u','PanelX',vectors[0],top_k=2)
            self.assertEqual(trace['candidate_count'],2);self.assertEqual(len(result['data']),2)
            images=[p for row in result['data'] for p in row['content'] if p['type']=='image_url']
            self.assertEqual(image_bytes(images[0]['image_url']['url'])[0],b'\x89PNG\r\n\x1a\nunit')
    def test_namespace_and_purged_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,docs,vectors=self.setup_index(Path(tmp));store.publish('u',docs,vectors,vectors,{})
            self.assertEqual(store.search('other','PanelX',vectors[0])[0],{'data':[]})
            store.source.purge_user('u')
            self.assertRaises(ContractError,store.search,'u','PanelX',vectors[0])
    def test_c_expansion_bounded_and_no_query_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,docs,vectors=self.setup_index(Path(tmp));store.publish('u',docs,vectors,vectors,{})
            _,trace=store.search('u','PanelX',vectors[0],variant='C')
            self.assertLessEqual(len(trace['neighbors']),20);self.assertFalse(trace['query_persisted']);self.assertFalse(trace['natural_time_inference'])
    def test_visual_schema_and_order(self):
        item=dict(index=0,global_caption='visible',visible_text=[],regions=[],relations=[],named_entities=[],uncertainties=[])
        response=dict(choices=[dict(message=dict(content=json.dumps(dict(images=[item]))))])
        self.assertEqual(GPT4oMiniVisualAnalysis.validated_items(response,1)[0]['index'],0)
        item['index']=True;response['choices'][0]['message']['content']=json.dumps(dict(images=[item]))
        self.assertRaises(ValueError,GPT4oMiniVisualAnalysis.validated_items,response,1)
    def test_higher_output_budget_requires_explicit_bounded_cap(self):
        class Fake:
            is_fake_http=True
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'budget.sqlite');rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat())
            args=dict(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,rate=rate,http=Fake(),mode='offline_fake')
            body=dict(model='gpt-4o-mini',max_completion_tokens=1536,store=False,n=1,
                      messages=[dict(role='system',content='source'),dict(role='user',content='original')])
            self.assertRaises(ValueError,BudgetedJSONTransport(**args)._bounds,body)
            self.assertEqual(BudgetedJSONTransport(**args,output_token_cap=1536)._bounds(body),(128000,1536))
            self.assertEqual(ledger.totals()['attempts'],0)


if __name__=='__main__':unittest.main()
