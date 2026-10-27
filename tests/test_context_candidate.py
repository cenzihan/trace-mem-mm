"""Synthetic regression cases; not a benchmark or real-model quality result."""
import json
from pathlib import Path
import tempfile
import unittest

from agent_memory.context_candidate import EvidenceContextCandidate, pack_originals
from agent_memory.protocol import ContractError, text_of
from test_online_candidate import Cards, Embedding, Scorer


IMAGE = dict(type='image_url', image_url=dict(url='data:image/png;base64,'
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII='))


class HostedScorer(Scorer):
    model = 'qwen3-vl-rerank'


class ContextCandidateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'memory.sqlite'
        self.embedding, self.cards, self.scorer = Embedding(), Cards(), HostedScorer()
        self.system = self.reopen()

    def reopen(self):
        return EvidenceContextCandidate(self.path, embedding=self.embedding,
            card_adapter=self.cards, scoring_adapter=self.scorer)

    def add(self, texts, uid='owner', session='session', request='request'):
        return self.system.add(dict(user_id=uid, session_id=session, request_id=request,
            messages=[dict(role='user' if i % 2 == 0 else 'assistant', content=text)
                      for i, text in enumerate(texts)]))

    def search(self, query='needle', uid='owner', top=100):
        return self.system.search(dict(user_id=uid, query=query, top_k=top))

    def test_mixed_query_keeps_image_match_outside_dense_top100(self):
        for batch in range(6):
            self.add(['blue distractor ' + str(i) for i in range(20)], request='d'+str(batch))
        target = [dict(type='text', text='red original'), IMAGE]
        self.add([target], request='image')
        result = self.search([dict(type='text', text='needle'), IMAGE])
        self.assertEqual(len(self.scorer.calls[-1][1]), 100)
        self.assertIn(target, [c['content'] for c in self.scorer.calls[-1][1]])
        self.assertEqual(result['data'][0]['content'], target)

    def test_lexical_anchor_survives_dense_candidate_cutoff(self):
        for batch in range(6):
            self.add(['blue distractor ' + str(i) for i in range(20)], request='d'+str(batch))
        self.add(['needle rare direct evidence'], request='target')
        self.search()
        self.assertIn('needle rare direct evidence',
                      [text_of(c['content']) for c in self.scorer.calls[-1][1]])

    def test_original_neighbours_are_returned_next_to_anchor(self):
        self.add(['red earlier original', 'blue anchor', 'red later original'])
        result = self.search(top=3)
        self.assertEqual([text_of(c['content']) for c in result['data']],
                         ['blue anchor', 'red earlier original', 'red later original'])
        self.assertEqual([c['score'] for c in result['data']], [1.0, 0.5, 1/3])
        self.assertTrue(all('derived description' not in text_of(c['content']) for c in result['data']))

    def test_context_does_not_cross_request_session_or_user(self):
        self.add(['blue anchor'], request='one')
        self.add(['red different request'], request='two')
        self.add(['red different session'], session='elsewhere', request='three')
        self.add(['blue private neighbour', 'blue private anchor'], uid='other')
        result, diag = self.system.retrieve(dict(user_id='owner', query='needle', top_k=100))
        self.assertEqual(diag['context_candidates'], 0)
        self.assertFalse(any('private' in text_of(c['content']) for c in result['data']))

    def test_context_can_recover_neighbour_outside_rerank_pool(self):
        self.add(['blue anchor', 'red neighbour'], request='pair')
        for batch in range(6):
            self.add(['blue distractor ' + str(i) for i in range(20)], request='d'+str(batch))
        def score(parts, candidates, base):
            return [1 if text_of(c['content']) == 'blue anchor' else 0 for c in candidates]
        self.scorer.score = score
        result = self.search(top=2)
        self.assertEqual([text_of(c['content']) for c in result['data']], ['blue anchor', 'red neighbour'])

    def test_image_only_query_has_no_embedding_or_text_rerank(self):
        content = [dict(type='text', text='red original'), IMAGE]
        self.add([content])
        calls = len(self.embedding.calls)
        self.assertEqual(self.search([IMAGE])['data'][0]['content'], content)
        self.assertEqual(len(self.embedding.calls), calls)
        self.assertEqual(len(self.scorer.calls), 0)

    def test_absent_user_returns_empty_without_calls(self):
        self.add(['blue source'])
        before = (len(self.embedding.calls), self.cards.calls, len(self.scorer.calls))
        self.assertEqual(self.search(uid='missing'), {'data': []})
        self.assertEqual(before, (len(self.embedding.calls), self.cards.calls, len(self.scorer.calls)))

    def test_model_contract_stays_fixed(self):
        self.scorer.model = 'unapproved-alias'
        with self.assertRaises(ValueError):
            self.reopen()

    def test_invalid_source_vector_blocks_before_model(self):
        self.add(['blue source'])
        with self.system.connect() as db:
            db.execute("UPDATE cards SET vector='[1,2]'")
        before = len(self.embedding.calls)
        with self.assertRaises(ContractError):
            self.search()
        self.assertEqual(len(self.embedding.calls), before)

    def test_invalid_rerank_score_rejected(self):
        self.add(['blue source'])
        for values in ([], [float('nan')], [True]):
            self.scorer.override = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.search()

    def test_restart_idempotence_and_query_privacy(self):
        self.add(['blue source'])
        before = (len(self.embedding.calls), self.cards.calls)
        self.add(['blue source'])
        self.assertEqual(before, (len(self.embedding.calls), self.cards.calls))
        result = self.search('private-question-marker')
        self.system = self.reopen()
        self.assertEqual(self.search('private-question-marker'), result)
        with self.system.connect() as db:
            self.assertNotIn('private-question-marker', '\n'.join(db.iterdump()))

    def test_response_cap_and_originals_are_preserved(self):
        self.add(['blue source', 'red source', 'red context'])
        result = self.search(top=1)
        self.assertEqual(len(result['data']), 1)
        self.assertEqual(set(result['data'][0]), {'id', 'content', 'created_at', 'score'})

    def test_media_budget_skips_whole_record_and_keeps_later_text(self):
        content = [dict(type='text', text='image caption'), IMAGE]
        originals = {key: dict(id=key, content=value, created_at='2026-01-01T00:00:00Z')
                     for key, value in [('a', content), ('b', content),
                                        ('c', [dict(type='text', text='later evidence')])]}
        before = json.dumps(originals, sort_keys=True)
        selected, used, skipped = pack_originals(['a', 'b', 'c'], 2, originals.__getitem__, limit=100)
        self.assertEqual([c['id'] for c in selected], ['a', 'c'])
        self.assertLessEqual(used, 100)
        self.assertEqual(skipped, 1)
        self.assertEqual(json.dumps(originals, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
