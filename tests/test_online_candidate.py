"""Synthetic unit contracts only; no dataset scores, model calls or network."""
import json
from pathlib import Path
import tempfile
import unittest

from agent_memory.online_candidate import OnlineDenseRerankSystem
from agent_memory.protocol import ContractError, text_of


class Embedding:
    model = 'text-embedding-v4'
    dimensions = 256

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1., 0.] + [0.] * 254 if 'blue' in text or text == 'needle'
                else [0., 1.] + [0.] * 254 for text in texts]


class Cards:
    model = 'gpt-4o-mini'

    def __init__(self):
        self.calls = 0

    def extract(self, parts):
        self.calls += 1
        return {'summary': 'derived description', 'evidence_quotes': [text_of(parts)]}


class Scorer:
    model = 'Qwen/Qwen3-VL-Reranker-2B'

    def __init__(self):
        self.calls = []
        self.override = None

    def score(self, parts, candidates, base):
        self.calls.append((parts, candidates, base))
        return base if self.override is None else self.override


class OnlineCandidateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embedding, self.cards, self.scorer = Embedding(), Cards(), Scorer()
        self.path = Path(self.directory.name) / 'memory.sqlite'
        self.system = self.reopen()

    def reopen(self):
        return OnlineDenseRerankSystem(self.path, embedding=self.embedding,
                    card_adapter=self.cards, scoring_adapter=self.scorer)

    def add(self, texts, uid='owner', request_id='request'):
        return self.system.add(dict(user_id=uid, session_id='session', request_id=request_id,
            messages=[dict(role='user', content=text) for text in texts]))

    def query(self, uid='owner', top=100):
        return dict(user_id=uid, query='needle', top_k=top)

    def test_dense_candidates_are_not_replaced_by_keyword_fusion(self):
        self.add(['blue source without keyword', 'needle needle needle red source'])
        result = self.system.search(self.query())
        self.assertIn('blue source', text_of(result['data'][0]['content']))
        self.assertNotIn('retrieval_text', result['data'][0])
        self.assertNotIn('derived description', text_of(result['data'][0]['content']))

    def test_native_source_parts_survive_ranking_and_recreation(self):
        self.add(['blue source', 'red source'])
        before = self.system.search(self.query())
        self.system = self.reopen()
        self.assertEqual(self.system.search(self.query()), before)
        self.assertTrue(all('derived description' in row['retrieval_text'] for row in self.scorer.calls[-1][1]))

    def test_cross_user_and_empty_search_make_no_model_calls(self):
        self.add(['blue source'])
        before = (len(self.embedding.calls), self.cards.calls, len(self.scorer.calls))
        self.assertEqual(self.system.search(self.query('another-user')), {'data': []})
        self.assertEqual((len(self.embedding.calls), self.cards.calls, len(self.scorer.calls)), before)

    def test_idempotent_add_uses_no_more_models_and_conflict_rejected(self):
        self.add(['blue source'])
        before = (len(self.embedding.calls), self.cards.calls)
        self.add(['blue source'])
        self.assertEqual((len(self.embedding.calls), self.cards.calls), before)
        with self.assertRaises(ContractError):
            self.add(['changed source'])

    def test_missing_or_foreign_vectors_fail_without_model_or_lexical_fallback(self):
        self.add(['blue source'])
        with self.system.connect() as db:
            db.execute("UPDATE cards SET vector_model='another-model'")
        before = len(self.embedding.calls)
        with self.assertRaises(ContractError) as caught:
            self.system.search(self.query())
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(len(self.embedding.calls), before)

    def test_only_top100_candidates_are_sent_to_the_scorer(self):
        for start in range(0, 105, 15):
            self.add(['blue source ' + str(i) for i in range(start, min(start+15, 105))],
                     request_id='request-' + str(start))
        result = self.system.search(self.query(top=200))
        self.assertEqual(len(self.scorer.calls[-1][1]), 100)
        self.assertEqual(len(result['data']), 100)

    def test_bad_reranker_scores_fail_explicitly(self):
        self.add(['blue source'])
        for values in ([], [float('nan')], [True], ['high']):
            self.scorer.override = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.system.search(self.query())

    def test_search_does_not_persist_question_or_options(self):
        self.add(['blue source'])
        query = dict(self.query(), query='private query marker', options=['private option marker'])
        self.system.search(query)
        with self.system.connect() as db:
            dump = '\n'.join(db.iterdump())
        self.assertNotIn('private query marker', dump)
        self.assertNotIn('private option marker', dump)

    def test_malformed_vector_is_rejected_before_paid_query_embedding(self):
        self.add(['blue source'])
        with self.system.connect() as db:
            db.execute("UPDATE cards SET vector='[1,0]'")
        before = len(self.embedding.calls)
        with self.assertRaises(ContractError) as caught:
            self.system.search(self.query())
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(len(self.embedding.calls), before)

    def test_image_only_query_uses_scoped_identity_without_invented_caption(self):
        image = dict(type='image_url', image_url=dict(url='data:image/png;base64,'
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII='))
        content = [dict(type='text', text='blue before'), image, dict(type='text', text='after')]
        self.add([content])
        before = len(self.embedding.calls)
        result = self.system.search(dict(user_id='owner', query=[image], top_k=100))
        self.assertEqual(result['data'][0]['content'], content)
        self.assertEqual(len(self.embedding.calls), before)
        self.assertEqual(self.scorer.calls, [])
        self.assertEqual(self.system.search(dict(user_id='another', query=[image], top_k=100)), {'data': []})


if __name__ == '__main__':
    unittest.main()
