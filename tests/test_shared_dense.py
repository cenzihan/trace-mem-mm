"""Synthetic vectors verify ranking/contracts, never a benchmark performance claim."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache
from agent_memory.shared_dense import SharedDenseIndex


class SharedDenseTests(unittest.TestCase):
    def vector(self, axis):
        value = [0.0]*64
        value[axis] = 1.0
        return value

    def prepared(self, temporary, records, choose_axis):
        cache = SourceEmbeddingCache(Path(temporary)/'cache.sqlite',
            CacheScope(namespace='u', dataset_revision='revision1', representation='flat-utf8-v1', dimensions=64),
            offline_test=True)
        cache.prepare_sources(records)
        for batch in cache.missing_batches():
            cache.import_batch([row['key'] for row in batch],
                [self.vector(choose_axis(row['text'])) for row in batch],
                dict(returned_model='text-embedding-v4', receipt_sha256='a'*64, budget_attempt_id='synthetic-only'))
        cache.publish()
        return cache

    def test_dense_and_hybrid_have_expected_independent_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = self.prepared(temporary, [dict(id='a',text='alpha'), dict(id='b',text='beta')],
                                  lambda text: 0 if text=='alpha' else 1)
            index = SharedDenseIndex(cache)
            dense = index.dense(self.vector(1), namespace='u')
            self.assertEqual(dense['ranked_ids'], ['b','a'])
            self.assertEqual(dense['scores'], [1.0,0.0])
            hybrid = index.hybrid('alpha', self.vector(1), namespace='u')
            self.assertEqual(hybrid['ranked_ids'], ['a','b'])
            self.assertAlmostEqual(hybrid['scores'][0],1/61+1/62)
            self.assertEqual(hybrid['lanes'],dict(flat=['a'],dense=['b','a']))

    def test_long_source_keeps_all_chunks_but_one_native_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            long_text='needle '*1600
            cache=self.prepared(temporary,[dict(id='a',text=long_text),dict(id='b',text='other')],lambda text:0)
            index=SharedDenseIndex(cache)
            self.assertEqual(index.documents['a'],long_text)
            self.assertGreater(len(index.source_chunks['a']),1)
            self.assertEqual(index.dense(self.vector(0),namespace='u')['ranked_ids'],['a','b'])

    def test_no_query_persistence_and_shared_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache=self.prepared(temporary,[dict(id='a',text='source')],lambda text:0)
            path=Path(temporary)/'cache.sqlite'
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            first=SharedDenseIndex(cache);second=SharedDenseIndex(cache)
            first.hybrid('transient never store this',self.vector(0),namespace='u')
            second.dense(self.vector(0),namespace='u')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            self.assertEqual(cache.missing_batches(),[])

    def test_query_admission_and_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache=self.prepared(temporary,[dict(id='a',text='source')],lambda text:0)
            index=SharedDenseIndex(cache)
            for vector in ([0.0]*64,[float('nan')]*64,[1e308]*64,[10**400]*64,[1.0]):
                with self.assertRaises(ValueError):index.dense(vector,namespace='u')
            with self.assertRaises(ValueError):index.dense(self.vector(0),namespace='other')
            with self.assertRaises(ValueError):index.hybrid('',self.vector(0),namespace='u')
            with self.assertRaises(ValueError):index.dense(self.vector(0),namespace='u',top_k=101)


if __name__=='__main__':
    unittest.main()
