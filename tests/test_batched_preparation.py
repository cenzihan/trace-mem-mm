import importlib.util
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from agent_memory.batched_preparation import prepare_embeddings
from agent_memory.mem0_bridge import BudgetEmbedding
from agent_memory.parallel_options import RequestEmbedding
from agent_memory.disk_options import DiskOptionsSystem
from test_mem0_native import FakeEmbedding, FakeTransport, FakeCards


def adapter(provider):
    original = object.__new__(BudgetEmbedding)
    original.config = None
    original.embedding = provider
    original.failed = False
    return RequestEmbedding(original)


class BatchedPreparationTests(unittest.TestCase):
    def test_order_duplicates_unicode_long_chunks_and_vectors_are_identical(self):
        texts = ['Synthetic short '+str(i) for i in range(21)] + [
            'Synthetic repeated', 'Synthetic repeated', ' ',
            '\u4e2d\U0001f642'*1700, 'long synthetic '*9000, 'tail']
        old, new = FakeEmbedding(), FakeEmbedding()
        before = list(prepare_embeddings(adapter(old), iter(texts), 1))
        after = list(prepare_embeddings(adapter(new), iter(texts), 10))
        self.assertEqual(before, after)
        self.assertLess(len(new.calls), len(old.calls))
        self.assertTrue(all(len(call) <= 10 for call in new.calls))
        self.assertTrue(all(len(text.encode()) <= 8000 for call in new.calls for text in call))

    def test_small_messages_reduce_calls_without_altering_pooling(self):
        texts = ['Synthetic small '+str(i) for i in range(30)]
        old, new = FakeEmbedding(), FakeEmbedding()
        self.assertEqual(list(prepare_embeddings(adapter(old),texts,1)),
                         list(prepare_embeddings(adapter(new),texts,10)))
        self.assertEqual((len(old.calls),len(new.calls)),(30,3))

    def test_failed_later_batch_does_not_change_another_request_flag(self):
        provider = FakeEmbedding(); shared = adapter(provider)
        original = provider.embed
        barrier = threading.Barrier(2)
        def embed(texts):
            if 'bad' in texts[0]:raise RuntimeError('synthetic provider failure')
            return original(texts)
        def call(prefix):
            shared.failed=False;barrier.wait(3)
            try:return list(prepare_embeddings(shared,[prefix+str(i) for i in range(20)],10))
            finally:self.assertEqual(shared.failed,prefix=='bad')
        with patch.object(provider,'embed',side_effect=embed):
            with ThreadPoolExecutor(max_workers=2) as pool:
                good=pool.submit(call,'good');bad=pool.submit(call,'bad')
                self.assertEqual(len(good.result(5)),20)
                with self.assertRaises(RuntimeError):bad.result(5)


@unittest.skipUnless(importlib.util.find_spec('mem0'), 'pinned Mem0 required')
class BatchedNativeTests(unittest.TestCase):
    def setUp(self):
        class Store(DiskOptionsSystem):embedding_preparation_batch_size=10
        self.temp=tempfile.TemporaryDirectory();self.provider=FakeEmbedding()
        self.store=Store(Path(self.temp.name)/'memory.sqlite',embedding=self.provider,
                         transport=FakeTransport(),card_adapter=FakeCards())
    def tearDown(self):
        self.store.close();self.store.native.db.close();self.temp.cleanup()
    def payload(self, user='u', request='r'):
        return dict(user_id=user,request_id=request,session_id='s',
            messages=[dict(role='user',content=f'Synthetic {user} notebook {i}') for i in range(23)])
    def test_native_memory_vectors_duplicate_and_user_isolation(self):
        first=self.store.add(self.payload())
        self.assertEqual(len(self.provider.calls),3)
        self.assertEqual(self.store.add(self.payload()),first)
        self.assertEqual(len(self.provider.calls),3)
        self.store.add(self.payload('other'))
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0],46)
            self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],2)
        self.assertTrue(all(not any('other' in t for t in call) for call in self.provider.calls[:3]))
        self.assertTrue(all(all('other' in t for t in call) for call in self.provider.calls[3:]))
    def test_failure_after_first_batch_leaves_no_partial_memory(self):
        original=self.provider.embed;calls=[]
        def embed(texts):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('synthetic failed batch')
            return original(texts)
        with patch.object(self.provider,'embed',side_effect=embed):
            with self.assertRaises(RuntimeError):self.store.add(self.payload())
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0)
        self.store.add(self.payload('good'))
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0],23)
