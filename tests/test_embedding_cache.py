"""Temporary synthetic source embeddings only; no provider or native questions."""
import json
import math
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache, CHUNK_BYTES


class SourceEmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='embedding-cache-test-')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'cache.sqlite'
        self.scope = CacheScope('test-user', 'frozen-revision', 'flat-v5', dimensions=64)
        self.cache = SourceEmbeddingCache(self.path, self.scope, offline_test=True)
        self.proof = dict(returned_model='text-embedding-v4', receipt_sha256='a' * 64,
                          budget_attempt_id='synthetic-attempt-1')

    def vector(self, value=1.0):
        return [value] + [0.0] * (self.scope.dimensions - 1)

    def fill(self, cache=None):
        cache = cache or self.cache
        for batch in cache.missing_batches():
            cache.import_batch([row['key'] for row in batch],
                               [self.vector() for _ in batch], self.proof)

    def test_successful_vectors_survive_restart_and_reimport(self):
        prepared = self.cache.prepare_sources([dict(id='a', text='first source'), dict(id='b', text='second source')])
        self.fill()
        original = self.cache.publish()
        restarted = SourceEmbeddingCache(self.path, self.scope, offline_test=True)
        self.assertEqual(restarted.missing_batches(), [])
        self.assertEqual(restarted.read_published(), original)
        self.assertEqual(restarted.prepare_sources([dict(id='b', text='second source'), dict(id='a', text='first source')]), prepared)
        chunk = original['chunks'][0]
        self.assertEqual(restarted.import_batch([chunk['key']], [chunk['vector']], self.proof), dict(imported=0, reused=1))
        self.assertRaises(ValueError, restarted.import_batch, [chunk['key']], [self.vector(2)], self.proof)
        self.assertEqual(restarted.read_published(), original)

    def test_exact_utf8_reconstruction_and_chunk_deduplication(self):
        text = '前缀🙂\r\n' + '甲🙂abc ' * 1800 + '\n尾部 不裁剪'
        prepared = self.cache.prepare_sources([dict(id='a', text=text), dict(id='b', text=text)])
        self.assertGreater(prepared['chunk_count'], 1)
        sources = prepared['manifest']['sources']
        self.assertEqual(sources[0]['chunk_keys'], sources[1]['chunk_keys'])
        batches = self.cache.missing_batches()
        chunks = {row['key']: row['text'] for batch in batches for row in batch}
        self.assertEqual(''.join(chunks[key] for key in sources[0]['chunk_keys']), text)
        self.assertTrue(all(0 < len(value.encode('utf-8')) <= CHUNK_BYTES for value in chunks.values()))
        self.fill()
        snapshot = self.cache.publish()
        self.assertEqual(snapshot['source_chunks']['a'], snapshot['source_chunks']['b'])

    def test_every_scope_component_is_isolated(self):
        self.cache.prepare_sources([dict(id='a', text='shared original')])
        self.fill()
        self.cache.publish()
        original_key = self.cache.read_published()['chunks'][0]['key']
        for other in (replace(self.scope, namespace='other-user'),
                      replace(self.scope, dataset_revision='other-revision'),
                      replace(self.scope, representation='other-view'),
                      replace(self.scope, dimensions=128)):
            isolated = SourceEmbeddingCache(self.path, other, offline_test=True)
            isolated.prepare_sources([dict(id='a', text='shared original')])
            batch = isolated.missing_batches()[0]
            self.assertNotEqual(batch[0]['key'], original_key)
            self.assertRaises(ValueError, isolated.import_batch, [original_key], [[1.0] * other.dimensions], self.proof)
            self.assertRaises(ValueError, isolated.read_published)

    def test_duplicate_source_id_and_extra_fields_rejected(self):
        self.assertRaises(ValueError, self.cache.prepare_sources,
                          [dict(id='a', text='first'), dict(id='a', text='first')])
        for field in ('question', 'answer', 'gold', 'namespace', 'prompt'):
            self.assertRaises(ValueError, self.cache.prepare_sources, [dict(id='a', text='source', **{field: 'extra'})])
        self.assertRaises(ValueError, self.cache.prepare_sources, [dict(id='a', text='')])

    def test_different_manifest_cannot_replace_frozen_sources(self):
        self.cache.prepare_sources([dict(id='a', text='first')])
        for records in ([dict(id='a', text='changed')], [dict(id='a', text='first'), dict(id='b', text='new')]):
            self.assertRaises(ValueError, self.cache.prepare_sources, records)
        self.assertEqual(len(self.cache.missing_batches()[0]), 1)

    def test_whitespace_sources_or_chunks_fail_without_partial_manifest(self):
        for number, text in enumerate((' \t\r\n', 'valid' + ' ' * 16384)):
            path = Path(self.temporary.name) / ('whitespace-%d.sqlite' % number)
            cache = SourceEmbeddingCache(path, self.scope, offline_test=True)
            self.assertRaises(ValueError, cache.prepare_sources,
                              [dict(id='valid-first', text='original source'), dict(id='bad', text=text)])
            self.assertRaises(ValueError, cache.missing_batches)
            with closing(sqlite3.connect(path)) as db:
                self.assertIsNone(db.execute('SELECT manifest FROM source_cache_scopes').fetchone()[0])
                self.assertEqual(db.execute('SELECT count(*) FROM source_cache_chunks').fetchone()[0], 0)
            # A failed preparation did not claim this scope or silently drop bad sources.
            restarted = SourceEmbeddingCache(path, self.scope, offline_test=True)
            result = restarted.prepare_sources([dict(id='valid-first', text='original source')])
            self.assertEqual(result['source_count'], 1)

    def test_cached_whitespace_chunk_is_rejected_even_with_matching_text_hash(self):
        self.cache.prepare_sources([dict(id='a', text='valid')])
        from agent_memory.embedding_cache import _sha
        whitespace = ' ' * 10
        digest = _sha(whitespace)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('UPDATE source_cache_chunks SET key=?,text=?,text_hash=?',
                       (self.cache._key(digest), whitespace, digest))
        self.assertRaises(ValueError, self.cache.missing_batches)

    def test_incomplete_cache_cannot_publish(self):
        self.cache.prepare_sources([dict(id='a', text='one'), dict(id='b', text='two')])
        batch = self.cache.missing_batches()[0]
        self.cache.import_batch([batch[0]['key']], [self.vector()], self.proof)
        self.assertRaises(ValueError, self.cache.publish)
        self.assertRaises(ValueError, self.cache.read_published)
        self.assertEqual(len(self.cache.missing_batches()[0]), 1)

    def test_batch_validation_and_write_failures_are_atomic(self):
        self.cache.prepare_sources([dict(id='a', text='one'), dict(id='b', text='two')])
        batch = self.cache.missing_batches()[0]
        keys = [row['key'] for row in batch]
        self.assertRaises(ValueError, self.cache.import_batch, keys, [self.vector(), [0.0] * 64], self.proof)
        self.assertEqual(len(self.cache.missing_batches()[0]), 2)
        # Force a failure after the first UPDATE; the whole transaction rolls back.
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TRIGGER reject_second BEFORE UPDATE OF vector ON source_cache_chunks "
                       "WHEN NEW.key='" + keys[1] + "' BEGIN SELECT RAISE(ABORT,'synthetic fault'); END")
        self.assertRaises(sqlite3.IntegrityError, self.cache.import_batch, keys,
                          [self.vector(), self.vector()], self.proof)
        self.assertEqual(len(self.cache.missing_batches()[0]), 2)

    def test_vector_dimension_numbers_norm_and_overflow(self):
        self.cache.prepare_sources([dict(id='a', text='one')])
        key = self.cache.missing_batches()[0][0]['key']
        invalid = ([1.0] * 63, [0.0] * 64, [math.nan] * 64, [math.inf] * 64,
                   [True] * 64, ['1'] * 64, [1e308] * 64, [10 ** 1000] * 64)
        for vector in invalid:
            self.assertRaises(ValueError, self.cache.import_batch, [key], [vector], self.proof)
        self.assertEqual(len(self.cache.missing_batches()[0]), 1)
        self.cache.import_batch([key], [[1e-200] * 64], self.proof)
        self.assertEqual(self.cache.publish()['chunks'][0]['vector'], [1e-200] * 64)

    def test_provenance_is_narrow_and_validated(self):
        self.cache.prepare_sources([dict(id='a', text='one')])
        key = self.cache.missing_batches()[0][0]['key']
        invalid = ({}, dict(self.proof, returned_model='other'), dict(self.proof, receipt_sha256='bad'),
                   dict(self.proof, budget_attempt_id=''), dict(self.proof, prompt='not allowed'),
                   dict(self.proof, headers={'key': 'not allowed'}))
        for proof in invalid:
            self.assertRaises(ValueError, self.cache.import_batch, [key], [self.vector()], proof)

    def test_batches_are_at_most_ten_and_keys_unique(self):
        self.cache.prepare_sources([dict(id=str(i), text='source ' + str(i)) for i in range(23)])
        self.assertEqual([len(batch) for batch in self.cache.missing_batches()], [10, 10, 3])
        for size in (0, 11, True, 1.5):
            self.assertRaises(ValueError, self.cache.missing_batches, size)
        key = self.cache.missing_batches()[0][0]['key']
        self.assertRaises(ValueError, self.cache.import_batch, [key, key], [self.vector()] * 2, self.proof)
        self.assertRaises(ValueError, self.cache.import_batch, [key], [], self.proof)

    def test_tampered_source_vector_or_provenance_fails_integrity(self):
        for column, value in (('text', 'changed'), ('vector', json.dumps([2.0] * 64)),
                              ('provenance', json.dumps(dict(self.proof, budget_attempt_id='changed')))):
            path = Path(self.temporary.name) / (column + '.sqlite')
            cache = SourceEmbeddingCache(path, self.scope, offline_test=True)
            cache.prepare_sources([dict(id='a', text='one')])
            self.fill(cache)
            cache.publish()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('UPDATE source_cache_chunks SET ' + column + '=?', (value,))
            self.assertRaises(ValueError, cache.read_published)

    def test_manifest_and_publication_hash_tampering_is_rejected(self):
        for column in ('manifest', 'published'):
            path = Path(self.temporary.name) / (column + '.sqlite')
            cache = SourceEmbeddingCache(path, self.scope, offline_test=True)
            cache.prepare_sources([dict(id='a', text='one')])
            self.fill(cache)
            cache.publish()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('UPDATE source_cache_scopes SET ' + column + "='{}'")
            self.assertRaises(ValueError, cache.read_published)

    def test_scope_and_path_validation(self):
        for changes in (dict(namespace=''), dict(dataset_revision=' '), dict(representation=''),
                        dict(model='other'), dict(dimensions=True), dict(dimensions=65)):
            self.assertRaises(ValueError, replace, self.scope, **changes)
        self.assertRaises(ValueError, SourceEmbeddingCache, self.path, self.scope)
        self.assertRaises(ValueError, SourceEmbeddingCache,
                          Path(__file__).resolve().parents[1] / 'outside-runtime.sqlite', self.scope, True)
        self.assertRaises(ValueError, SourceEmbeddingCache,
                          Path(self.temporary.name) / 'not-sqlite.txt', self.scope, True)
        unrelated = Path(self.temporary.name) / 'unrelated.sqlite'
        with closing(sqlite3.connect(unrelated)) as db, db:
            db.execute('CREATE TABLE private_records(value)')
        self.assertRaises(ValueError, SourceEmbeddingCache, unrelated, self.scope, True)


if __name__ == '__main__':
    unittest.main()
