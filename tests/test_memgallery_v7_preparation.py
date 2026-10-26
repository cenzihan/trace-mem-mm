"""Synthetic data only; no native benchmark reads or external calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_memgallery_v7 as v7


def container():
    return {'character_profile': {'name': 'PROFILE_SENTINEL'},
            'multi_session_dialogues': [{'session_id': 'S1', 'date': '2020-01-01', 'dialogues': [
                {'round': 'R1', 'user': 'Original user source.', 'assistant': 'Original assistant source.',
                 'input_image': ['../image/synthetic/a.png'], 'image_caption': ['Original caption.'], 'image_id': ['I1']}]}],
            'human-annotated QAs': [dict(point='VS' if i % 2 else 'FAKE', question='QUESTION_SENTINEL',
                question_image='../image/synthetic/query.png' if i % 2 else None,
                image_caption=['QUERY_CAPTION_SENTINEL'] if i % 2 else [],
                answer='GOLD_SENTINEL', clue=['R1'], session_id=['S1']) for i in range(20)]}


class MemGalleryV7PreparationTests(unittest.TestCase):
    def test_complete_separation_and_question_fields_preserved(self):
        data = container(); before = copy.deepcopy(data)
        source, questions, gold, schema = v7.split_container(data)
        self.assertEqual(data, before)
        self.assertEqual(set(source), {'multi_session_dialogues'})
        self.assertNotIn('SENTINEL', json.dumps(source))
        self.assertNotIn('GOLD_SENTINEL', json.dumps(questions))
        self.assertNotIn('PROFILE_SENTINEL', json.dumps(questions))
        self.assertEqual(questions[1]['question_image'], data['human-annotated QAs'][1]['question_image'])
        self.assertEqual(questions[1]['image_caption'], data['human-annotated QAs'][1]['image_caption'])
        self.assertEqual(set(questions[0]), {'index', 'question', 'point', 'question_image', 'image_caption'})
        self.assertIsNone(questions[0]['question_image'])
        self.assertIsNone(questions[0]['image_caption'])
        self.assertEqual(set(gold[0]), {'index', 'answer', 'clue', 'session_id'})
        self.assertEqual(schema['source_rounds'], 1)
        self.assertEqual(schema['questions_with_image'], 10)
        self.assertEqual(schema['questions_with_image_caption'], 10)

    def test_unknown_question_fields_stop_without_value_echo(self):
        data = container(); data['human-annotated QAs'][0]['unclassified'] = 'SECRET_VALUE'
        with self.assertRaises(v7.PreparationError) as raised:
            v7.split_container(data)
        self.assertEqual(raised.exception.reason, 'unclassified_question_fields')
        self.assertEqual(raised.exception.schema, {'unclassified': {'str': 1}})
        self.assertNotIn('SECRET_VALUE', str(raised.exception))

    def test_embedded_source_annotation_is_rejected(self):
        data = container(); data['multi_session_dialogues'][0]['answer'] = 'GOLD_SENTINEL'
        with self.assertRaises(v7.PreparationError):
            v7.split_container(data)

    def test_strict_json_and_reference_types(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.assertRaises(v7.PreparationError):
                v7.strict_load(raw)
        data = container(); data['human-annotated QAs'][0]['clue'] = 'R1'
        with self.assertRaises(v7.PreparationError):
            v7.split_container(data)

    def test_query_image_requires_caption_and_no_orphan_caption(self):
        data = container(); data['human-annotated QAs'][1]['image_caption'] = []
        with self.assertRaises(v7.PreparationError):
            v7.split_container(data)
        data = container(); data['human-annotated QAs'][0]['image_caption'] = 'orphan caption'
        with self.assertRaises(v7.PreparationError):
            v7.split_container(data)

    def test_selection_metadata_only_deterministic_and_stratified(self):
        rows = [{'index': i, 'point': str(i % 9)} for i in range(60)]
        chosen = v7.select_reader(rows)
        self.assertEqual(chosen, v7.select_reader(list(reversed(rows))))
        self.assertEqual(len(set(chosen['selected_indices'])), 16)
        self.assertEqual(len(chosen['selected_point_counts']), 9)
        self.assertEqual(chosen['planned_reader_answers'], 64)
        with self.assertRaises(v7.PreparationError):
            v7.select_reader([dict(x, question='forbidden selection input') for x in rows])

    def test_selection_is_durable_before_fetch_and_failure_never_reselects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fetch(_):
                frozen = root / v7.WORK / 'FROZEN_SELECTION_RULE.json'
                self.assertTrue(frozen.is_file())
                raise RuntimeError('must not appear in report')
            with patch.object(v7, 'selection_rule', return_value={'state': 'synthetic frozen rule'}):
                with self.assertRaises(RuntimeError):
                    v7.prepare(root, fetcher=fetch)
            failure = (root / v7.WORK / 'PREPARATION_FAILED.json').read_text()
            self.assertNotIn('must not appear', failure)
            with self.assertRaises(v7.PreparationError):
                v7.prepare(root, fetcher=lambda _: self.fail('fetch must not run twice'))

    def test_outputs_are_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'out.json'
            v7.save_new(p, {'a': 1})
            with self.assertRaises(FileExistsError):
                v7.save_new(p, {'a': 2})
            self.assertEqual(json.loads(p.read_text()), {'a': 1})


if __name__ == '__main__':
    unittest.main()
