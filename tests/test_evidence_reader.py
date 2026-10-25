"""Synthetic interface-only tests: source budgets, gold isolation and blind scoring."""
import json
import unittest
from agent_memory.evidence_reader import pack_evidence,pack_matched_pair,GPT4oMiniEvidenceReader,ReaderBudget,READER_PROMPT
from agent_memory.evidence_scoring import supplemental_em_f1,GPT4oMiniProxyJudge

URI='data:image/png;base64,iVBORw0KGgo='


class EvidenceReaderTests(unittest.TestCase):
    def records(self):return [dict(id='m1',content=[dict(type='text',text='Original source'),dict(type='image_url',image_url=dict(url=URI))])]
    def metadata(self):return {'m1':dict(source_session='s',source_round='r',source_date='2025-01-01')}
    def transform(self,uri):return uri,dict(original_mapping_verified=True)
    def test_source_only_metadata_and_no_gold_question_object(self):
        meta=self.metadata();meta['m1']['answer']='GOLD_SENTINEL'
        self.assertRaises(ValueError,pack_evidence,'question',self.records(),meta,len,self.transform)
        self.assertRaises(ValueError,pack_evidence,dict(question='q',answer='gold'),self.records(),self.metadata(),len,self.transform)
    def test_budgets_and_matched_image_count(self):
        pair=pack_matched_pair('question',{'A':self.records(),'C':self.records()},self.metadata(),len,self.transform)
        self.assertEqual(pair['A']['image_count'],pair['C']['image_count']);self.assertEqual(pair['A']['image_count'],1)
        self.assertLessEqual(pair['A']['text_token_count'],8192);self.assertFalse(pair['A']['context_gold_access'])
        pair=pack_matched_pair('question',{'A':self.records(),'C':[]},self.metadata(),len,self.transform)
        self.assertEqual(pair['A']['image_count'],0);self.assertEqual(pair['C']['image_count'],0)
    def test_whole_unit_prefix_not_text_crop(self):
        base=len(READER_PROMPT)+len('ORIGINAL QUESTION:\nquestion')+32
        pack=pack_evidence('question',self.records(),self.metadata(),len,self.transform,
                           ReaderBudget(text_tokens=base+1,utf8_bytes=24000))
        self.assertEqual(pack['source_memory_ids'],[]);self.assertEqual(pack['dropped_records'],1)
    def test_reader_aliases_are_validated_not_invented(self):
        pack=pack_evidence('question',self.records(),self.metadata(),len,self.transform)
        value=dict(answer='response',evidence_ids=['E9'],confidence='supported')
        response=dict(model='gpt-4o-mini-2024-07-18',choices=[dict(message=dict(content=json.dumps(value)))])
        result=GPT4oMiniEvidenceReader.decode(response,pack)
        self.assertFalse(result['citations_valid']);self.assertFalse(result['cited_memory_ids'])
    def test_proxy_supplemental_metrics_and_blinding(self):
        self.assertEqual(supplemental_em_f1('The red dog.','red dog')['normalized_em'],1)
        self.assertAlmostEqual(supplemental_em_f1('red','red dog')['token_f1'],2/3)
        class Transport:
            def __call__(self,*args):raise AssertionError('no dispatch')
        self.assertRaises(ValueError,GPT4oMiniProxyJudge(Transport()).request,[dict(item_id=0,question='q',prediction='p',reference='r',variant='A')])


if __name__=='__main__':unittest.main()
