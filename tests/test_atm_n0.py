"""Synthetic contract fixtures only; no native performance data or model calls."""
import unittest
from agent_memory.atm_n0 import native_sgm_corpus,NativeBM25
from agent_memory.retrieval import bm25,ranked
from scripts.run_atm_n0_local import chunks_utf8

class ATMN0Tests(unittest.TestCase):
    def corpus(self,images=None):
        return native_sgm_corpus(images if images is not None else [dict(image_path='/author/img.jpg',caption='unit caption')],
            [dict(video_path='/author/vid.mp4',caption='unit video')],[dict(id='email-unit',detail='unit email')],
            expected_images={'img'},expected_videos={'vid'},expected_emails=1)
    def test_indexed_formula_matches_existing_baseline(self):
        docs={'one':'unit alpha alpha beta','two':'unit beta gamma','three':'测试内容 alpha'}
        index=NativeBM25(docs)
        for query in ('alpha beta','测试','gamma','unseen'):
            expected=bm25(query,docs);got=index.search(query)
            order=[key for key in ranked(expected) if expected[key]>0]
            self.assertEqual([key for key,_ in got],order)
            for key,score in got:self.assertAlmostEqual(score,expected[key],places=12)
    def test_tie_order_stable_and_no_match_empty(self):
        a=NativeBM25({'b':'same','a':'same'});b=NativeBM25({'a':'same','b':'same'})
        self.assertEqual(a.search('same'),b.search('same'));self.assertEqual(a.search('absent'),[])
    def test_complete_native_source_no_gold_and_no_clipping(self):
        rows,meta=self.corpus([dict(image_path='/author/img.jpg',caption='x'*20000)])
        self.assertEqual(len(rows),3);self.assertIn('x'*20000,rows[0]['text']);self.assertTrue(meta['local_text_not_truncated'])
        bad=[dict(image_path='/author/img.jpg',caption='unit',answer='forbidden')]
        self.assertRaises(ValueError,self.corpus,bad)
        self.assertRaises(ValueError,self.corpus,[])
    def test_metadata_only_source_counted_not_silently_dropped(self):
        rows,meta=self.corpus([dict(image_path='/author/img.jpg',timestamp='unit date')])
        self.assertEqual(len(rows),3);self.assertEqual(meta['metadata_only_sources']['image'],1)
    def test_planning_utf8_chunks_preserve_source_without_clipping(self):
        text='测试🙂'*3000;parts=list(chunks_utf8(text))
        self.assertEqual(b''.join(parts).decode('utf-8'),text)
        self.assertTrue(all(len(p)<=8192 and p.decode('utf-8') for p in parts))

if __name__=='__main__':unittest.main()
