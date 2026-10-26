"""Temporary synthetic data only; never opens native Hard questions or answers."""
import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from scripts.score_atm_v4_hard_isolated import (
    DETAIL_FILE, RUN_FILE, SCORE_FILE, score_project, sha256, verify_before_gold,
)


class ATMV4ScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.gold_file = 'data/atm-schema/scorer-only/atm-bench-hard.json'
        self.query_file = 'data/public-questions.json'
        self.ranking_file = 'runtime/atm-v4-hard/retrieval-results.jsonl'
        self.frozen_file = 'runtime/atm-v4-hard/FROZEN_CONFIG.json'
        self.index_file = 'runtime/atm-n0/source-index.sqlite'
        self.write('data/source.json', b'{"synthetic":true}')
        self.write('scripts/frozen.py', b'# synthetic source\n')
        self.write('docs/preserved-plan.md', b'Synthetic frozen plan.\n')
        index = self.root / self.index_file
        index.parent.mkdir(parents=True)
        with closing(sqlite3.connect(index)) as db:
            db.execute('CREATE TABLE sources(id TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO sources VALUES(?)', [('a',), ('b',), ('c',)])
            db.commit()
        self.questions = [dict(id='q1', question='SYNTHETIC_QUESTION_SENTINEL', qtype='open_end'),
                          dict(id='q2', question='another synthetic question', qtype='list_recall')]
        self.gold = [dict(id='q1', question='SYNTHETIC_QUESTION_SENTINEL', qtype='open_end',
                          answer='SYNTHETIC_ANSWER_SENTINEL', evidence_ids=['a', 'b']),
                     dict(id='q2', question='another synthetic question', qtype='list_recall',
                          answer='SYNTHETIC_ANSWER_SENTINEL', evidence_ids=['c'])]
        self.rows = [dict(id='q1', qtype='open_end', methods={
            'flat': dict(ranked_ids=['a', 'c', 'b'], scores=[3., 2., 1.], elapsed_ms=1),
            'v4': dict(ranked_ids=['b', 'a', 'c'], scores=[3., 2., 1.], elapsed_ms=1)}),
            dict(id='q2', qtype='list_recall', methods={
            'flat': dict(ranked_ids=['a', 'b', 'c'], scores=[3., 2., 1.], elapsed_ms=1),
            'v4': dict(ranked_ids=['c', 'a', 'b'], scores=[3., 2., 1.], elapsed_ms=1)})]
        self.refresh()

    def write(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def dump(self, relative, value):
        self.write(relative, json.dumps(value).encode())

    def digest(self, relative):
        return sha256((self.root / relative).read_bytes())

    def refresh(self):
        self.dump(self.query_file, self.questions)
        self.dump(self.gold_file, self.gold)
        self.gold_sha256 = self.digest(self.gold_file)
        self.write(self.ranking_file, ('\n'.join(json.dumps(row) for row in self.rows) + '\n').encode())
        self.frozen = dict(gold_file=self.gold_file, gold_sha256=self.gold_sha256,
            query_file=self.query_file, query_sha256=self.digest(self.query_file),
            source_files=[dict(file='data/source.json', sha256=self.digest('data/source.json'))],
            source_index_file=self.index_file,
            source_index_sha256=self.digest(self.index_file),
            source_ids_sha256=sha256(json.dumps(['a', 'b', 'c'], separators=(',', ':')).encode()),
            code_hashes={'scripts/frozen.py': self.digest('scripts/frozen.py')},
            preserved_hashes={'docs/preserved-plan.md': self.digest('docs/preserved-plan.md')})
        self.dump(self.frozen_file, self.frozen)
        self.run = dict(state='retrieval_complete_gold_not_loaded', questions=2, source_count=3,
            source_ids_sha256=sha256(json.dumps(['a', 'b', 'c'], separators=(',', ':')).encode()),
            ranking_file=self.ranking_file, ranking_sha256=self.digest(self.ranking_file),
            frozen_file=self.frozen_file, frozen_sha256=self.digest(self.frozen_file),
            source_index_sha256=self.digest(self.index_file),
            question_sha256=self.digest(self.query_file),
            code_hashes={'scripts/frozen.py': self.digest('scripts/frozen.py')})
        self.dump(RUN_FILE, self.run)

    def options(self):
        return dict(expected_questions=2, expected_sources=3, expected_gold_sha256=self.gold_sha256)

    def assert_rejected_before_gold(self):
        real_read = Path.read_bytes
        gold_path = (self.root / self.gold_file).resolve()

        def guarded_read(path):
            if path.resolve() == gold_path:
                raise AssertionError('Gold opened before invalid retrieval was rejected')
            return real_read(path)

        with patch.object(Path, 'read_bytes', guarded_read):
            with self.assertRaises(ValueError):
                verify_before_gold(self.root, **self.options())

    def test_known_paired_metrics_and_private_outputs(self):
        report = score_project(self.root, bootstrap_samples=100, **self.options())
        self.assertEqual(report['methods']['flat']['1']['fractional_recall'], .25)
        self.assertEqual(report['methods']['v4']['1']['fractional_recall'], .75)
        paired = report['paired']['1']['fractional_recall']
        self.assertEqual((paired['wins'], paired['ties'], paired['losses']), (1, 1, 0))
        self.assertEqual(paired['mean_delta'], .5)
        self.assertEqual(paired['paired_query_bootstrap_95_ci'], [0., 1.])
        self.assertEqual(report['methods']['v4']['5']['all_evidence_coverage'], 1.)
        for relative in (SCORE_FILE, DETAIL_FILE):
            output = (self.root / relative).read_text()
            self.assertNotIn('SYNTHETIC_QUESTION_SENTINEL', output)
            self.assertNotIn('SYNTHETIC_ANSWER_SENTINEL', output)

    def test_rankings_tampered_after_freeze_rejected_before_gold(self):
        self.rows[0]['methods']['flat']['ranked_ids'][0] = 'c'
        self.write(self.ranking_file, ('\n'.join(json.dumps(row) for row in self.rows) + '\n').encode())
        self.assert_rejected_before_gold()

    def test_signed_duplicate_questions_rejected_before_gold(self):
        self.rows[1] = copy.deepcopy(self.rows[0])
        self.refresh()
        self.assert_rejected_before_gold()

    def test_signed_duplicate_or_unknown_sources_rejected_before_gold(self):
        original = copy.deepcopy(self.rows)
        for ids in (['a', 'a', 'b'], ['a', 'unknown-source', 'b']):
            with self.subTest(ids=ids):
                self.rows = copy.deepcopy(original)
                self.rows[0]['methods']['flat']['ranked_ids'] = ids
                self.refresh()
                self.assert_rejected_before_gold()

    def test_every_frozen_input_is_checked_before_gold(self):
        for relative in ('data/source.json', 'scripts/frozen.py', self.query_file, self.index_file,
                         'docs/preserved-plan.md'):
            with self.subTest(relative=relative):
                original = (self.root / relative).read_bytes()
                self.write(relative, original + b' ')
                self.assert_rejected_before_gold()
                self.write(relative, original)

    def test_run_manifest_cannot_replace_frozen_identity_or_code_hashes(self):
        original = copy.deepcopy(self.run)
        for field, value in (('code_hashes', {}), ('source_index_sha256', '0' * 64),
                             ('source_ids_sha256', '0' * 64), ('question_sha256', '0' * 64)):
            with self.subTest(field=field):
                self.run = copy.deepcopy(original)
                self.run[field] = value
                self.dump(RUN_FILE, self.run)
                self.assert_rejected_before_gold()

    def test_same_id_with_changed_question_is_not_scored_as_native_question(self):
        self.questions[0]['question'] = 'different synthetic question with the same ID'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Public question text differs'):
            score_project(self.root, bootstrap_samples=10, **self.options())
        self.assertFalse((self.root / SCORE_FILE).exists())
        self.assertFalse((self.root / DETAIL_FILE).exists())

    def test_question_type_mismatch_rejected_before_gold(self):
        self.rows[0]['qtype'] = 'number'
        self.refresh()
        self.assert_rejected_before_gold()

    def test_missing_or_unknown_gold_is_rejected_without_outputs(self):
        original = copy.deepcopy(self.gold)
        cases = (original[:1], [dict(original[0], evidence_ids=[]), original[1]],
                 [dict(original[0], evidence_ids=['unknown-source']), original[1]])
        for gold in cases:
            with self.subTest(gold_count=len(gold)):
                self.gold = copy.deepcopy(gold)
                self.refresh()
                with self.assertRaises(ValueError):
                    score_project(self.root, bootstrap_samples=10, **self.options())
                self.assertFalse((self.root / SCORE_FILE).exists())
                self.assertFalse((self.root / DETAIL_FILE).exists())

    def test_missing_gold_file_preserves_inputs_and_produces_no_scores(self):
        (self.root / self.gold_file).unlink()
        with self.assertRaises(FileNotFoundError):
            score_project(self.root, bootstrap_samples=10, **self.options())
        self.assertFalse((self.root / SCORE_FILE).exists())

    def test_finished_report_and_details_are_never_overwritten(self):
        score_project(self.root, bootstrap_samples=10, **self.options())
        before = {relative: (self.root / relative).read_bytes() for relative in (SCORE_FILE, DETAIL_FILE)}
        with self.assertRaises(FileExistsError):
            score_project(self.root, bootstrap_samples=10, **self.options())
        self.assertEqual(before, {relative: (self.root / relative).read_bytes() for relative in before})


if __name__ == '__main__':
    unittest.main()
