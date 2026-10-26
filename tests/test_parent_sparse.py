"""Synthetic source fixtures only; no native benchmark questions or gold."""
import copy
import unittest

from agent_memory.atm_n0 import NativeBM25, native_sgm_corpus
from agent_memory.parent_sparse import (
    DETAIL_STRIDE, DETAIL_WINDOW, ParentSparseIndex, build_sgm_views,
)
from agent_memory.retrieval import tokenize


class ParentSparseTests(unittest.TestCase):
    def setUp(self):
        self.images = [dict(image_path='/source/image-one.jpg', short_caption='short alpha',
                            caption='detailed beta', ocr_text='visible gamma', tags=['delta'],
                            timestamp='2026-01-02', location_name='unit place')]
        self.videos = [dict(video_path='/source/video-one.mp4', caption='video epsilon')]
        self.emails = [dict(id='email-one', short_summary='summary zeta', detail='detail eta')]

    def build(self, **options):
        settings = dict(expected_images={'image-one'}, expected_videos={'video-one'},
                        expected_emails=1, namespace='unit-user')
        settings.update(options)
        return build_sgm_views(self.images, self.videos, self.emails, **settings)

    def make_index(self, texts, extra=(), namespace='unit-user'):
        records = [dict(namespace=namespace, native_id=key, text=value) for key, value in texts.items()]
        views = [dict(namespace=namespace, native_id=key, lane='flat', text=value) for key, value in texts.items()]
        views.extend(dict(namespace=namespace, native_id=key, lane=lane, text=text)
                     for key, lane, text in extra)
        return ParentSparseIndex(records, views, namespace), records, views

    def test_flat_text_and_lane_scores_match_frozen_n0(self):
        records, views, metadata = self.build()
        expected, _ = native_sgm_corpus(self.images, self.videos, self.emails,
            expected_images={'image-one'}, expected_videos={'video-one'}, expected_emails=1)
        self.assertEqual({r['id']: r['text'] for r in expected}, {r['id']: r['text'] for r in records})
        baseline = NativeBM25({r['id']: r['text'] for r in expected})
        candidate = ParentSparseIndex(records, views, 'unit-user')
        for query in ('alpha beta', 'video', '2026', 'not_present'):
            got = candidate.search(query)['lanes']['flat']
            wanted = baseline.search(query)
            self.assertEqual(got['ranked_ids'], [key for key, _ in wanted])
            for key, score in wanted:
                self.assertAlmostEqual(got['scores'][key], score, places=12)
        self.assertEqual(metadata['source_count'], 3)
        self.assertEqual(metadata['provider_calls'], 0)

    def test_summary_fallback_and_detail_fields_are_source_only(self):
        records, views, _ = self.build()
        mapping = {(v['native_id'], v['lane']): v['text'] for v in views}
        self.assertEqual(mapping['image-one', 'summary'], 'short alpha')
        self.assertEqual(mapping['video-one', 'summary'], 'video epsilon')
        self.assertEqual(mapping['email-one', 'summary'], 'summary zeta')
        self.assertEqual(mapping['image-one', 'detail'],
                         'detailed beta\nvisible gamma\ndelta\n2026-01-02\nunit place')
        self.assertEqual(mapping['email-one', 'detail'], 'detail eta')
        self.assertNotIn('short alpha', mapping['image-one', 'detail'])

    def test_metadata_only_sources_remain_in_flat(self):
        self.images = [dict(image_path='/source/image-one.jpg')]
        self.videos = [dict(video_path='/source/video-one.mp4')]
        self.emails = [dict(id='email-one')]
        records, views, _ = self.build()
        self.assertEqual(len(records), 3)
        self.assertEqual([v['lane'] for v in views], ['flat', 'flat', 'flat'])
        index = ParentSparseIndex(records, views, 'unit-user')
        self.assertEqual(index.source_count, 3)
        self.assertEqual(index.unit_counts, dict(flat=3, summary=0, detail=0))

    def test_duplicate_views_do_not_change_statistics_or_votes(self):
        index, records, views = self.make_index(
            {'a': 'unit needle', 'b': 'unit needle'}, [('a', 'detail', 'needle ' * 900),
                                                      ('b', 'summary', 'needle')])
        doubled = ParentSparseIndex(records, views + copy.deepcopy(views) * 7, 'unit-user')
        self.assertEqual(index.unit_counts, doubled.unit_counts)
        self.assertEqual(index.view_counts, doubled.view_counts)
        self.assertEqual(index.search('needle'), doubled.search('needle'))

    def test_all_chunks_scored_before_parent_top100(self):
        texts = {'a-heavy': 'neutral'}
        unique_chunks = ' '.join('needle marker%05d' % i for i in range(DETAIL_STRIDE * 80))
        extra = [('a-heavy', 'detail', unique_chunks)]
        for position in range(105):
            key = 'b-%03d' % position
            texts[key] = 'neutral'
            extra.append((key, 'detail', 'needle filler filler filler'))
        index, _, _ = self.make_index(texts, extra)
        lane = index.search('needle')['lanes']['detail']
        self.assertEqual(len(lane['ranked_ids']), 100)
        self.assertEqual(len(set(lane['ranked_ids'])), 100)
        self.assertIn('a-heavy', lane['ranked_ids'])
        self.assertGreaterEqual(sum(key.startswith('b-') for key in lane['ranked_ids']), 99)

    def test_duplicate_detail_windows_do_not_change_any_parent_score(self):
        texts = {'a': 'neutral', 'b': 'neutral', 'c': 'neutral'}
        other_views = [('b', 'detail', 'needle alpha'), ('c', 'detail', 'needle beta beta')]
        single, _, _ = self.make_index(texts, [('a', 'detail', 'needle ' * 512)] + other_views)
        # 896 terms produce two identical 512-term windows at offsets 0 and 384.
        repeated, _, _ = self.make_index(texts, [('a', 'detail', 'needle ' * 896)] + other_views)
        self.assertEqual(single.unit_counts, repeated.unit_counts)
        for query in ('needle', 'needle alpha', 'beta'):
            self.assertEqual(single.search(query), repeated.search(query))

    def test_long_source_tail_and_exact_token_windows_preserved(self):
        self.emails[0]['detail'] = 'ordinary ' * 1700 + 'tailmarker 测试词项'
        records, views, _ = self.build()
        detail = next(v['text'] for v in views if v['native_id'] == 'email-one' and v['lane'] == 'detail')
        self.assertEqual(detail, self.emails[0]['detail'])
        self.assertGreater(len(detail.encode('utf-8')), 8192)
        index = ParentSparseIndex(records, views, 'unit-user')
        self.assertIn('email-one', index.search('tailmarker')['lanes']['detail']['ranked_ids'])
        self.assertIn('email-one', index.search('试词')['lanes']['detail']['ranked_ids'])
        self.assertEqual(DETAIL_WINDOW, 512)
        self.assertEqual(DETAIL_STRIDE, 384)
        # The appended CJK bigrams from tokenize must not be tokenized a second time.
        expected_terms = tokenize(detail)
        detail_index = index._lanes['detail']
        observed = {term for term, postings in detail_index.postings.items()
                    if any(detail_index.parents[pos] == 'email-one' for pos, _ in postings)}
        self.assertEqual(observed, set(expected_terms))

    def test_equal_rrf_math_and_one_vote_per_lane(self):
        index, _, _ = self.make_index(
            {'a': 'needle', 'b': 'needle'}, [('b', 'summary', 'needle'),
                                           ('b', 'detail', 'needle ' * 1300)])
        result = index.search('needle')
        self.assertEqual(result['ranked_ids'], ['b', 'a'])
        self.assertAlmostEqual(result['scores']['a'], 1 / 61, places=15)
        self.assertAlmostEqual(result['scores']['b'], 1 / 62 + 2 / 61, places=15)
        for parent in result['ranked_ids']:
            self.assertLessEqual(result['scores'][parent], 3 / 61)

    def test_namespace_isolation_and_unknown_parent_rejected(self):
        index, records, views = self.make_index({'a': 'needle'})
        other = copy.deepcopy(views)
        other[0]['namespace'] = 'another-user'
        self.assertRaises(ValueError, ParentSparseIndex, records, other, 'unit-user')
        other = copy.deepcopy(records)
        other[0]['namespace'] = 'another-user'
        self.assertRaises(ValueError, ParentSparseIndex, other, views, 'unit-user')
        self.assertRaises(ValueError, ParentSparseIndex, records, views, '')
        foreign = dict(namespace='unit-user', native_id='absent', lane='summary', text='needle')
        self.assertRaises(ValueError, ParentSparseIndex, records, views + [foreign], 'unit-user')
        isolated, _, _ = self.make_index({'a': 'different'}, namespace='another-user')
        self.assertEqual(isolated.search('needle')['ranked_ids'], [])
        self.assertEqual(index.search('needle')['ranked_ids'], ['a'])
        self.images[0]['namespace'] = 'another-user'
        self.assertRaises(ValueError, self.build)

    def test_duplicate_records_missing_flat_and_changed_flat_rejected(self):
        _, records, views = self.make_index({'a': 'needle'})
        self.assertRaises(ValueError, ParentSparseIndex, records + copy.deepcopy(records), views, 'unit-user')
        self.assertRaises(ValueError, ParentSparseIndex, records, [], 'unit-user')
        changed = copy.deepcopy(views)
        changed[0]['text'] = 'other'
        self.assertRaises(ValueError, ParentSparseIndex, records, changed, 'unit-user')
        self.images.append(copy.deepcopy(self.images[0]))
        self.assertRaises(ValueError, self.build)

    def test_query_gold_and_nested_evaluation_fields_rejected(self):
        _, records, views = self.make_index({'a': 'needle'})
        for field in ('query', 'question', 'answer', 'gold', 'evidence_ids'):
            bad = copy.deepcopy(records)
            bad[0][field] = 'forbidden'
            self.assertRaises(ValueError, ParentSparseIndex, bad, views, 'unit-user')
        bad = copy.deepcopy(records)
        bad[0]['arbitrary_metadata'] = 'not a declared source field'
        self.assertRaises(ValueError, ParentSparseIndex, bad, views, 'unit-user')
        self.images[0]['metadata'] = {'Gold': ['forbidden']}
        self.assertRaises(ValueError, self.build)
        self.images[0].pop('metadata')
        self.emails[0]['query'] = 'forbidden'
        self.assertRaises(ValueError, self.build)

    def test_deterministic_order_bounded_results_and_empty_query(self):
        texts = {'unit-%03d' % i: 'needle' for i in range(120)}
        index, records, views = self.make_index(texts)
        reversed_index = ParentSparseIndex(list(reversed(records)), list(reversed(views)), 'unit-user')
        self.assertEqual(index.search('needle'), reversed_index.search('needle'))
        self.assertEqual(index.search('needle')['ranked_ids'], sorted(texts)[:100])
        self.assertEqual(index.search('needle', 3)['ranked_ids'], sorted(texts)[:3])
        self.assertEqual(index.search('')['ranked_ids'], [])
        self.assertEqual(index.search('unknown')['ranked_ids'], [])
        for value in (0, 101, True, 1.5):
            self.assertRaises(ValueError, index.search, 'needle', value)
        self.assertRaises(ValueError, index.search, None)


if __name__ == '__main__':
    unittest.main()
