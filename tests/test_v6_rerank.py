"""Synthetic frozen inputs only; no real questions, gold, model, SSH or API."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from agent_memory import v6_rerank as v6
from agent_memory.native_metrics import retrieval_metrics
from scripts.prepare_atm_v5_queries import digest, query_id


class V6RerankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in v6.CODE:
            self.write(name, b'# synthetic code identity\n')
        self.spec = dict(model='synthetic-scoring-only', revision='fixed-test-revision',
                         max_length=4096, score_direction='higher_is_better', workers=4)
        self.source_ids = [f's{index:02d}' for index in range(26)]
        index = self.root / v6.SOURCE_INDEX
        index.parent.mkdir(parents=True)
        with closing(sqlite3.connect(index)) as db, db:
            db.execute('CREATE TABLE sources(id TEXT PRIMARY KEY,text TEXT)')
            db.executemany('INSERT INTO sources VALUES(?,?)', ((identity, 'SOURCE_TEXT_'+identity) for identity in self.source_ids))
        specs, rankings, groups = [], [], {}
        inputs = {v6.SOURCE_INDEX: self.hash(v6.SOURCE_INDEX)}
        for dataset in ('regular', 'hard'):
            question = dict(id='q_'+dataset, qtype='open_end', question='PRIVATE_ORIGINAL_QUESTION_'+dataset)
            qpath = f'data/{dataset}.questions.json'
            gpath = f'data/scorer-only/{dataset}.json'
            self.dump(qpath, [question])
            self.dump(gpath, [dict(question, evidence_ids=['s20'], answer='PRIVATE_GOLD_ANSWER')])
            specs.append(dict(name=dataset, count=1, query_file=qpath, query_sha256=self.hash(qpath),
                              gold_file=gpath, gold_sha256=self.hash(gpath)))
            inputs[qpath] = self.hash(qpath)
            methods = {base: dict(ranked_ids=self.source_ids[:25] if base == 'dense' else self.source_ids[1:],
                                 scores=[float(25-i) for i in range(25)]) for base in v6.BASES}
            rankings.append(dict(dataset=dataset, id=question['id'], qtype=question['qtype'], methods=methods))
            groups[dataset] = dict(questions=1, methods={base: {str(k): {
                metric: retrieval_metrics(row['ranked_ids'], ['s20'], k)[metric] for metric in v6.METRICS}
                for k in v6.KS} for base, row in methods.items()})
        rankfile = 'runtime/v5/retrieval-results.jsonl'
        freezefile = 'runtime/v5/FROZEN_CONFIG.json'
        self.lines(rankfile, rankings)
        self.dump(freezefile, dict(input_hashes=inputs, code_hashes={}, datasets=specs,
            source_count=26, query_count=2, source_index_sha256=self.hash(v6.SOURCE_INDEX),
            condition='synthetic-only', exposure='synthetic unit test, not benchmark performance'))
        self.dump(v6.V5_RUN, dict(state='retrieval_complete_gold_not_loaded', source_count=26, questions=2,
            frozen_file=freezefile, frozen_sha256=self.hash(freezefile), ranking_file=rankfile,
            ranking_sha256=self.hash(rankfile)))
        self.dump(v6.V5_SCORES, dict(state='complete', datasets=groups))

    def write(self, name, raw):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    def dump(self, name, value):
        self.write(name, json.dumps(value).encode())

    def lines(self, name, rows):
        self.write(name, ('\n'.join(json.dumps(row) for row in rows)+'\n').encode())

    def hash(self, name):
        return digest((self.root / name).read_bytes())

    def read(self, name):
        return json.loads((self.root / name).read_bytes())

    def no_gold(self):
        read = Path.read_bytes
        def guarded(path):
            if 'scorer-only' in path.parts:
                raise AssertionError('Gold opened before independent validated scoring')
            return read(path)
        return patch.object(Path, 'read_bytes', guarded)

    def prepare(self):
        return v6.prepare_project(self.root, reranker_spec=self.spec, expected_sources=26, expected_questions=2)

    def model_output(self, transform=lambda rows: rows):
        rows = [dict(pair_id=row['pair_id'], score=1.0 if row['source_id'] == 's20' else 0.)
                for row in v6.json_lines((self.root / v6.PAIRS).read_bytes())]
        self.lines(v6.MODEL_SCORES, transform(rows))
        frozen = self.read(v6.FROZEN)
        self.dump(v6.MODEL_RUN, dict(state='complete', frozen_config_sha256=self.hash(v6.FROZEN),
            pairs_sha256=self.hash(v6.PAIRS), reranker_spec=self.spec, completed_pairs=frozen['pair_count'],
            scores_file=v6.MODEL_SCORES, scores_sha256=self.hash(v6.MODEL_SCORES)))

    def test_stable_ties_preserve_each_retrievers_order(self):
        self.assertEqual(v6.stable_rerank(['c', 'a', 'b'], {'a': .3, 'b': .3, 'c': .2}), ['a', 'b', 'c'])
        self.assertEqual(v6.stable_rerank(['b', 'a', 'c'], {'a': .3, 'b': .3, 'c': .2}), ['b', 'a', 'c'])

    def test_union_once_fixed_candidates_and_separate_gold_scoring(self):
        with self.no_gold():
            prep = self.prepare()
            self.assertEqual(prep['pair_count'], 52)  # 26 union pairs, not 50 per question.
            self.model_output()
            v6.rank_project(self.root)
        rankings = v6.json_lines((self.root / v6.RANKINGS).read_bytes())
        for row in rankings:
            for base in v6.BASES:
                self.assertEqual(set(row['methods'][base]['ranked_ids']), set(row['methods'][base+'_reranked']['ranked_ids']))
                self.assertEqual(row['methods'][base+'_reranked']['ranked_ids'][0], 's20')
        result = v6.score_project(self.root, bootstrap_samples=50)
        regular = result['datasets']['regular']
        self.assertEqual(regular['paired']['dense_reranked_minus_dense']['20']['fractional_recall']['mean_delta'], 1.)
        self.assertEqual(regular['paired']['hybrid_reranked_minus_hybrid']['20']['fractional_recall']['mean_delta'], 0.)
        self.assertEqual(regular['paired']['dense_reranked_minus_dense']['100']['fractional_recall']['ci95'], [0., 0.])
        for sentinel in ('PRIVATE_ORIGINAL_QUESTION', 'SOURCE_TEXT', 'PRIVATE_GOLD_ANSWER'):
            self.assertNotIn(sentinel, (self.root / v6.SCORE_REPORT).read_text())
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaises(FileExistsError):
            v6.score_project(self.root)

    def test_duplicate_pair_score_rejected_before_gold(self):
        self.prepare()
        self.model_output(lambda rows: rows + [rows[0]])
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'duplicate'):
            v6.rank_project(self.root)
        self.assertFalse((self.root / v6.RANKINGS).exists())

    def test_missing_pair_rejected_before_gold(self):
        self.prepare()
        self.model_output(lambda rows: rows[:-1])
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'exactly once'):
            v6.rank_project(self.root)

    def test_unknown_or_nonfinite_model_score_rejected(self):
        self.prepare()
        for bad in ('unknown', 'nonfinite'):
            with self.subTest(bad=bad):
                def change(rows):
                    rows[0] = dict(pair_id='f'*64, score=0.) if bad == 'unknown' else dict(rows[0], score=float('nan'))
                    return rows
                self.model_output(change)
                with self.no_gold(), self.assertRaises(ValueError):
                    v6.rank_project(self.root)

    def test_changed_source_or_query_projection_is_rejected_before_gold(self):
        self.prepare()
        self.model_output()
        self.write(v6.QUERIES, (self.root / v6.QUERIES).read_bytes()+b' ')
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'hash mismatch'):
            v6.rank_project(self.root)

    def test_manually_reordered_ties_rejected_even_with_updated_ranking_hash(self):
        self.prepare()
        self.model_output()
        v6.rank_project(self.root)
        rows = v6.json_lines((self.root / v6.RANKINGS).read_bytes())
        ids = rows[0]['methods']['dense_reranked']['ranked_ids']
        ids[1], ids[2] = ids[2], ids[1]
        self.lines(v6.RANKINGS, rows)
        run = self.read(v6.RUN_REPORT)
        run['ranking_sha256'] = self.hash(v6.RANKINGS)
        self.dump(v6.RUN_REPORT, run)
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'stable ties'):
            v6.score_project(self.root, bootstrap_samples=50)
        self.assertFalse((self.root / v6.SCORE_REPORT).exists())

    def test_wrong_model_revision_is_rejected(self):
        self.prepare()
        self.model_output()
        run = self.read(v6.MODEL_RUN)
        run['reranker_spec']['revision'] = 'unreviewed-revision'
        self.dump(v6.MODEL_RUN, run)
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'frozen reranker'):
            v6.rank_project(self.root)


if __name__ == '__main__':
    unittest.main()
