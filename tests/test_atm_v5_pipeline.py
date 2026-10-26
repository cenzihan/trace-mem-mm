"""Synthetic private caches only; no native questions, gold, providers, or credentials."""
import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from agent_memory.atm_n0 import NativeBM25
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache
from scripts.prepare_atm_v5_queries import (
    CODE, DATASETS, QUERY_CACHE, QUERY_FROZEN, REVISION, SOURCE_CACHE, SOURCE_FROZEN,
    digest, prepare_queries,
)
from scripts.score_atm_v5_dense import (
    DETAILS, RANKINGS, RUN_REPORT, SCORE_REPORT, SOURCE_INDEX, retrieve, score,
)


class ATMV5PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in CODE:
            self.write(relative, b'# Synthetic code identity fixture.\n')
        self.documents = {'a': 'alpha', 'b': 'beta', 'c': 'gamma'}
        index = self.root / SOURCE_INDEX
        index.parent.mkdir(parents=True)
        with closing(sqlite3.connect(index)) as db, db:
            db.execute('CREATE TABLE sources(id TEXT PRIMARY KEY,text TEXT)')
            db.executemany('INSERT INTO sources VALUES(?,?)', self.documents.items())
        self.source_scope = CacheScope('synthetic-source', REVISION, 'n0-flat-utf8-8192-v1', dimensions=64)
        cache = SourceEmbeddingCache(self.root / SOURCE_CACHE, self.source_scope, offline_test=True)
        prepared = cache.prepare_sources([dict(id=key, text=text) for key, text in self.documents.items()])
        self.fill(cache, lambda text: {'alpha': 0, 'beta': 1, 'gamma': 2}[text])
        cache.publish()
        self.dump(SOURCE_FROZEN, dict(scope=asdict(self.source_scope), source_count=3,
            source_index_sha256=self.hash(SOURCE_INDEX), manifest_sha256=prepared['manifest_sha256'],
            code_hashes={'agent_memory/embedding_cache.py': self.hash('agent_memory/embedding_cache.py')}))
        self.specs = copy.deepcopy(list(DATASETS))
        for spec in self.specs:
            spec['count'] = 1
            question = 'alpha' if spec['name'] == 'regular' else 'beta'
            row = dict(id='unit-' + spec['name'], question=question, qtype='open_end')
            self.dump(spec['query_file'], [row])
            self.dump(spec['gold_file'], [dict(row, answer='PRIVATE_ANSWER_SENTINEL', evidence_ids=['b'])])
            spec['gold_sha256'] = self.hash(spec['gold_file'])
            ranked = NativeBM25(self.documents).search(question)
            flat = dict(ranked_ids=[key for key, _ in ranked], scores=[value for _, value in ranked])
            baseline = dict(id=row['id'], qtype=row['qtype'])
            if spec['baseline_method'] is None:
                baseline.update(flat)
            else:
                baseline['methods'] = {'flat': flat}
            self.write(spec['baseline_file'], (json.dumps(baseline) + '\n').encode())
            self.dump(spec['anchor_file'], {
                spec['query_hash_field']: self.hash(spec['query_file']),
                spec['baseline_hash_field']: self.hash(spec['baseline_file'])})

    def write(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def dump(self, relative, value):
        self.write(relative, json.dumps(value).encode())

    def hash(self, relative):
        return digest((self.root / relative).read_bytes())

    def fill(self, cache, axis):
        for batch in cache.missing_batches():
            vectors = []
            for row in batch:
                vector = [0.] * 64
                vector[axis(row['text'])] = 1.
                vectors.append(vector)
            cache.import_batch([row['key'] for row in batch], vectors,
                dict(returned_model='text-embedding-v4', receipt_sha256='a' * 64,
                     budget_attempt_id='synthetic-offline-attempt'))

    def prepare(self, complete=True):
        report = prepare_queries(self.root, datasets=self.specs, offline_test=True)
        frozen = json.loads((self.root / QUERY_FROZEN).read_text())
        cache = SourceEmbeddingCache(self.root / QUERY_CACHE, CacheScope(**frozen['scope']), offline_test=True)
        if complete:
            self.fill(cache, lambda _text: 1)
            cache.publish()
        return report

    def no_gold(self):
        original = Path.read_bytes
        def guarded(path):
            if 'scorer-only' in path.parts:
                raise AssertionError('Preparation/retrieval must never open gold')
            return original(path)
        return patch.object(Path, 'read_bytes', guarded)

    def test_preparation_and_retrieval_never_open_gold_then_separate_scores(self):
        with self.no_gold():
            preparation = self.prepare()
            run = retrieve(self.root, offline_test=True, expected_sources=3)
        self.assertEqual(preparation['query_count'], 2)
        self.assertEqual(run['questions'], 2)
        scored = score(self.root, expected_sources=3)
        regular = scored['datasets']['regular']
        self.assertEqual(regular['methods']['flat']['1']['fractional_recall'], 0.)
        self.assertEqual(regular['methods']['dense']['1']['fractional_recall'], 1.)
        self.assertEqual(regular['paired_vs_flat']['dense']['1']['fractional_recall']['wins'], 1)
        self.assertTrue(scored['no_new_heldout_claim'])
        self.assertNotIn('PRIVATE_ANSWER_SENTINEL', (self.root / SCORE_REPORT).read_text())
        self.assertNotIn('PRIVATE_ANSWER_SENTINEL', (self.root / DETAILS).read_text())
        with self.assertRaises(FileExistsError):
            score(self.root, expected_sources=3)

    def test_incomplete_query_cache_cannot_claim_retrieval(self):
        self.prepare(complete=False)
        with self.assertRaises(ValueError):
            retrieve(self.root, offline_test=True, expected_sources=3)
        self.assertFalse((self.root / RUN_REPORT).exists())
        self.assertFalse((self.root / RANKINGS).exists())

    def test_question_only_projection_rejects_answer_even_with_matching_anchor(self):
        spec = self.specs[0]
        rows = json.loads((self.root / spec['query_file']).read_text())
        rows[0]['answer'] = 'forbidden gold'
        self.dump(spec['query_file'], rows)
        anchor = json.loads((self.root / spec['anchor_file']).read_text())
        anchor[spec['query_hash_field']] = self.hash(spec['query_file'])
        self.dump(spec['anchor_file'], anchor)
        with self.no_gold(), self.assertRaises(ValueError):
            prepare_queries(self.root, datasets=self.specs, offline_test=True)
        self.assertFalse((self.root / QUERY_CACHE).exists())

    def test_changed_ranking_hash_is_rejected_before_gold(self):
        self.prepare()
        retrieve(self.root, offline_test=True, expected_sources=3)
        self.write(RANKINGS, (self.root / RANKINGS).read_bytes() + b' ')
        with self.no_gold(), self.assertRaises(ValueError):
            score(self.root, expected_sources=3)
        self.assertFalse((self.root / SCORE_REPORT).exists())

    def test_unknown_native_source_is_rejected_even_if_run_hash_is_updated(self):
        self.prepare()
        retrieve(self.root, offline_test=True, expected_sources=3)
        rows = [json.loads(line) for line in (self.root / RANKINGS).read_text().splitlines()]
        rows[0]['methods']['dense']['ranked_ids'][0] = 'unknown-source'
        self.write(RANKINGS, ('\n'.join(json.dumps(row) for row in rows) + '\n').encode())
        run = json.loads((self.root / RUN_REPORT).read_text())
        run['ranking_sha256'] = self.hash(RANKINGS)
        self.dump(RUN_REPORT, run)
        with self.no_gold(), self.assertRaises(ValueError):
            score(self.root, expected_sources=3)

    def test_cache_change_after_retrieval_rejected_before_gold(self):
        self.prepare()
        retrieve(self.root, offline_test=True, expected_sources=3)
        self.write(QUERY_CACHE, (self.root / QUERY_CACHE).read_bytes() + b'changed')
        with self.no_gold(), self.assertRaises(ValueError):
            score(self.root, expected_sources=3)


if __name__ == '__main__':
    unittest.main()
