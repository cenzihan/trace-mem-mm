import json
import unittest
from agent_memory.online_cards import VerifiedQuoteCards

class Tests(unittest.TestCase):
    def test_unverified_quote_omitted_not_rewritten_and_summary_preserved(self):
        raw={'summary':'model description','evidence_quotes':['exact source','paraphrased claim','']};calls=[]
        def transport(endpoint,payload):
            calls.append(payload)
            return {'model':'gpt-4o-mini-2024-07-18','choices':[{'message':{'content':json.dumps(raw)}}]}
        result=VerifiedQuoteCards(transport).extract('An exact source sentence.')
        self.assertEqual(result,dict(summary='model description',evidence_quotes=['exact source'],omitted_unverified_quote_count=2))
        self.assertEqual(len(calls),1);self.assertEqual(raw['evidence_quotes'],['exact source','paraphrased claim',''])
    def test_all_invalid_quotes_does_not_invent_replacement(self):
        def transport(*args):return {'model':'gpt-4o-mini','choices':[{'message':{'content':json.dumps(dict(summary='unchanged',evidence_quotes=['not present']))}}]}
        self.assertEqual(VerifiedQuoteCards(transport).extract('original')['evidence_quotes'],[])
    def test_bad_type_is_not_silently_accepted(self):
        def transport(*args):return {'model':'gpt-4o-mini','choices':[{'message':{'content':json.dumps(dict(summary='unchanged',evidence_quotes=[5]))}}]}
        with self.assertRaises(ValueError):VerifiedQuoteCards(transport).extract('original')
