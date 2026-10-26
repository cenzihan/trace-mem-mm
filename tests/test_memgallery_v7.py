"""Synthetic V7 interface tests: no benchmark, gold file, provider or image I/O."""
import copy
import json
import math
import unittest
from unittest.mock import patch

from agent_memory.memgallery_v7 import (
    base_rankings, native_id, project_queries, rerank_variants, select_pilot,
    source_documents, source_mapping, union_pairs,
)
from agent_memory.native_metrics import retrieval_metrics
from agent_memory.shared_dense import SharedDenseIndex


def source():
    item = dict(round='round-repeated', user='  Original user\nUnicode 原文.  ',
                assistant='Original assistant\nunchanged.', image_id=['image-1', 'image-2'],
                input_image=['../image/synthetic/a.jpg', '../image/synthetic/b.png'],
                image_caption=['Full author caption\nline two.', ''])
    return dict(multi_session_dialogues=[
        dict(session_id='session-a', date='2025-01-01', dialogues=[item]),
        dict(session_id='session-b', date='2025-01-02', dialogues=[copy.deepcopy(item)]),
    ])


class SyntheticPublishedCache:
    def read_published(self):
        ids = [f'synthetic-source-{i:03}' for i in range(130)]
        return dict(scope=dict(namespace='synthetic-v7', dimensions=64),
                    chunks=[dict(key=identity, text='cobalt source ' + str(i),
                                 vector=[1.0, i / 130] + [0.0] * 62)
                            for i, identity in enumerate(ids)],
                    source_chunks={identity: [identity] for identity in ids})


class MemGalleryV7Tests(unittest.TestCase):
    def test_complete_source_text_caption_and_composite_image_mapping(self):
        original = source()
        before = copy.deepcopy(original)
        documents = source_documents(original)
        self.assertEqual(original, before)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]['id'], native_id('session-a', 'round-repeated'))
        self.assertNotEqual(documents[0]['id'], documents[1]['id'])
        self.assertEqual(json.loads(documents[0]['id']), ['session-a', 'round-repeated'])
        item = original['multi_session_dialogues'][0]['dialogues'][0]
        for field in ('user', 'assistant'):
            self.assertIn(item[field], documents[0]['text'])
            self.assertEqual(documents[0][field], item[field])
        self.assertEqual(documents[0]['image_captions'], item['image_caption'])
        self.assertEqual(documents[0]['image_ids'], item['image_id'])
        self.assertEqual(documents[0]['image_refs'], item['input_image'])
        self.assertEqual(len(documents[0]['images']), 2)
        self.assertEqual(documents[0]['images'][1]['caption'], '')
        self.assertIn(item['image_caption'][0], documents[0]['text'])
        self.assertFalse(documents[0]['raw_media_loaded'])
        mapping = source_mapping(documents)
        self.assertEqual(mapping['ambiguous_round_labels'], ['round-repeated'])
        self.assertEqual(len(mapping['by_round_label']['round-repeated']), 2)
        self.assertEqual(len(mapping['by_image_id']['"image-1"']), 2)

    def test_source_rejects_gold_profile_mismatch_duplicates_and_path_escape(self):
        for key in ('human-annotated QAs', 'character_profile', 'gold'):
            value = source()
            value[key] = 'FORBIDDEN_SYNTHETIC_VALUE'
            self.assertRaises(ValueError, source_documents, value)
        value = source()
        value['multi_session_dialogues'][0]['dialogues'][0]['image_caption'].pop()
        self.assertRaises(ValueError, source_documents, value)
        value = source()
        rows = value['multi_session_dialogues'][0]['dialogues']
        rows.append(copy.deepcopy(rows[0]))
        self.assertRaises(ValueError, source_documents, value)
        value = source()
        value['multi_session_dialogues'][0]['dialogues'][0]['input_image'][0] = '../image/../../secret'
        self.assertRaises(ValueError, source_documents, value)
        value = source()
        value['multi_session_dialogues'][0]['dialogues'][0]['answer'] = 'FORBIDDEN'
        self.assertRaises(ValueError, source_documents, value)

    def test_public_query_keeps_original_image_fields_and_point_without_atm_conversion(self):
        rows = [dict(index=0, question=' Original synthetic question?\n', point='VS',
                     question_image=['../image/synthetic/query.png'], image_caption=['caption unchanged']),
                dict(index=1, question='Other question?', point='CD', query_image=None)]
        result = project_queries(rows)
        self.assertEqual(result, rows)
        self.assertIsNot(result[0], rows[0])
        self.assertNotIn('qtype', result[0])
        for key in ('answer', 'clue', 'session_id', 'gold'):
            bad = copy.deepcopy(rows)
            bad[0][key] = 'FORBIDDEN_SYNTHETIC_VALUE'
            self.assertRaises(ValueError, project_queries, bad)
        self.assertRaises(ValueError, project_queries, rows + [rows[0]])

    def test_metadata_selection_representative_per_point_stable_under_order_and_extends(self):
        metadata = [dict(index=i, point=('VS', 'CD', 'AR', 'KR')[i % 4]) for i in range(24)]
        small = select_pilot(metadata, count=8)
        large = select_pilot(list(reversed(metadata)), count=16)
        self.assertEqual(select_pilot(metadata), large)
        self.assertEqual(small['selected'], large['selected'][:8])
        self.assertEqual({row['point'] for row in large['selected'][:4]}, {'VS', 'CD', 'AR', 'KR'})
        self.assertEqual(len({row['index'] for row in large['selected']}), 16)
        self.assertFalse(large['question_or_gold_used'])
        self.assertEqual(large['planned_reader_answers'], 64)

    def test_metadata_selection_rejects_question_gold_and_unattainable_coverage(self):
        metadata = [dict(index=i, point='p' + str(i)) for i in range(4)]
        self.assertRaises(ValueError, select_pilot, metadata, count=3)
        self.assertRaises(ValueError, select_pilot, metadata, count=8)
        for key in ('question', 'answer', 'clue', 'image_caption'):
            bad = copy.deepcopy(metadata)
            bad[0][key] = 'FORBIDDEN_SYNTHETIC_VALUE'
            self.assertRaises(ValueError, select_pilot, bad, count=4)
        self.assertRaises(ValueError, select_pilot, metadata, count=4, seed=1)

    def test_single_dense_pass_matches_shared_index_hybrid_and_uses_native_top100(self):
        index = SharedDenseIndex(SyntheticPublishedCache())
        query_vector = [1.0, 0.0] + [0.0] * 62
        expected_dense = index.dense(query_vector, namespace='synthetic-v7')
        expected_hybrid = index.hybrid('cobalt 129', query_vector, namespace='synthetic-v7')
        with patch.object(index, 'dense', wraps=index.dense) as call:
            result = base_rankings(index, 'cobalt 129', query_vector, namespace='synthetic-v7')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result['dense'], expected_dense)
        self.assertEqual(result['hybrid'], expected_hybrid)
        self.assertEqual(len(result['dense']['ranked_ids']), 100)
        self.assertEqual(len(result['hybrid']['ranked_ids']), 100)
        self.assertRaises(ValueError, base_rankings, index, 'cobalt', query_vector, namespace='other-scene')

    def test_union_scores_once_stable_ties_and_each_top100_pool_retained(self):
        bases = dict(dense=dict(ranked_ids=['a', 'b', 'c'], scores=[3., 2., 1.]),
                     hybrid=dict(ranked_ids=['c', 'a', 'd'], scores=[3., 2., 1.]))
        original = copy.deepcopy(bases)
        pairs = union_pairs('synthetic-scene:q0', bases)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len({row['pair_id'] for row in pairs}), 4)
        self.assertEqual([row['source_id'] for row in pairs], ['a', 'b', 'c', 'd'])
        result = rerank_variants(bases, dict(a=1.0, b=2.0, c=1.0, d=-3.0))
        self.assertEqual(result['dense_reranked']['ranked_ids'], ['b', 'a', 'c'])
        self.assertEqual(result['hybrid_reranked']['ranked_ids'], ['c', 'a', 'd'])
        self.assertEqual(bases, original)
        for base in ('dense', 'hybrid'):
            self.assertEqual(set(result[base]['ranked_ids']), set(result[base + '_reranked']['ranked_ids']))
            self.assertEqual(retrieval_metrics(result[base]['ranked_ids'], ['a', 'd'], 100),
                             retrieval_metrics(result[base + '_reranked']['ranked_ids'], ['a', 'd'], 100))
        self.assertRaises(ValueError, rerank_variants, bases, dict(a=1., b=2., c=3.))
        self.assertRaises(ValueError, rerank_variants, bases, dict(a=1., b=2., c=3., d=math.nan))
        bad = copy.deepcopy(bases)
        bad['dense']['ranked_ids'][1] = 'a'
        self.assertRaises(ValueError, union_pairs, 'synthetic-scene:q0', bad)


if __name__ == '__main__':
    unittest.main()
