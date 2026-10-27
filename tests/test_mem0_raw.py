"""Real pinned Mem0 SDK, synthetic vectors and images, no provider traffic."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.mem0_raw import Mem0RawSystem, Mem0RawTransport, VERSION
from agent_memory.protocol import ContractError, content_parts, text_of
from test_mem0_native import FakeEmbedding, FakeTransport, FakeCards
from test_mem0_evidence import image


class RawModeBoundaryTests(unittest.TestCase):
    def test_llm_dispatch_blocked_before_credentials_or_journal(self):
        transport = object.__new__(Mem0RawTransport)
        with self.assertRaises(ValueError):
            transport('https://api.openai.com/v1/chat/completions', {'model': 'gpt-4o-mini'})

    def test_explicit_activation(self):
        from agent_memory.mem0_app import create_raw_app
        with self.assertRaises(ValueError):create_raw_app({})
        self.assertEqual(Mem0RawTransport.receipt_scope, 'native-mem0-raw-v4')


@unittest.skipUnless(importlib.util.find_spec('mem0'), 'optional pinned Mem0 dependency')
class RawNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'raw.sqlite'
        self.e, self.t, self.c = FakeEmbedding(), FakeTransport(), FakeCards()
        self.store = self.open()

    def open(self):
        return Mem0RawSystem(self.path, embedding=self.e, transport=self.t, card_adapter=self.c)

    def close(self):
        self.store.close()
        conn = getattr(self.store.native.db, 'connection', None)
        if conn:conn.close()

    def tearDown(self):
        self.assertEqual(self.t.calls, [])
        self.assertEqual(self.c.calls, 0)
        self.close()
        self.temp.cleanup()

    def payload(self, contents, uid='alice', req='r', session='s'):
        return dict(user_id=uid, request_id=req, session_id=session,
                    messages=[dict(role='user', content=c) for c in contents])

    def search(self, query, uid='alice', top=100):
        return self.store.search(dict(user_id=uid, query=query, top_k=top))['data']

    def test_raw_text_persists_and_native_result_maps_to_exact_original(self):
        original = '  Mira stores the cobalt atlas on shelf four.  '
        self.store.add(self.payload([original]))
        rows = self.search('cobalt atlas')
        self.assertEqual(rows[0]['content'], content_parts(original))
        self.assertEqual(len(self.e.calls), 2)  # one source + one query
        self.assertTrue(rows[0]['id'].startswith('mem_'))
        self.assertEqual(self.store.capabilities()['version'], VERSION)
        self.assertFalse(self.store.capabilities()['llm_calls_enabled'])

    def test_mixed_parts_order_images_and_source_timestamp_are_preserved(self):
        parts = [dict(type='text', text='before'), image(), dict(type='text', text='after')]
        payload = self.payload([parts])
        payload['messages'][0]['timestamp'] = 1700000000123
        self.store.add(payload)
        rows = self.search([dict(type='text', text='before'), image()])
        self.assertEqual(rows[0]['content'], parts)
        self.assertEqual(rows[0]['created_at'], '2023-11-14T22:13:20.123000Z')

    def test_image_only_source_and_query_need_no_model_and_do_not_invent_text(self):
        self.store.add(self.payload([[image()]]))
        self.assertEqual(self.e.calls, [])
        rows = self.search([image()])
        self.assertEqual(rows[0]['content'], [image()])
        self.assertEqual(self.e.calls, [])
        self.assertEqual(self.search('unknown semantic description'), [])
        self.assertEqual(self.e.calls, [])

    def test_native_dense_candidate_maps_even_when_lexical_channel_empty(self):
        self.store.add(self.payload(['Mira keeps a cobalt atlas']))
        with patch('agent_memory.mem0_raw.bm25', return_value={}):
            self.assertEqual(text_of(self.search('cobalt atlas')[0]['content']), 'Mira keeps a cobalt atlas')

    def test_user_scope_cross_session_and_absent_user_no_paid_query(self):
        self.store.add(self.payload(['anchor first'], session='one'))
        self.store.add(self.payload(['anchor second'], req='two', session='two'))
        self.store.add(self.payload(['anchor private'], uid='other'))
        text = ' '.join(text_of(r['content']) for r in self.search('anchor'))
        self.assertIn('first', text);self.assertIn('second', text);self.assertNotIn('private', text)
        before = len(self.e.calls)
        self.assertEqual(self.search('anchor', uid='absent'), [])
        self.assertEqual(len(self.e.calls), before)

    def test_restart_idempotence_and_query_never_archived(self):
        payload = self.payload(['unmodified source'])
        self.store.add(payload)
        self.close();self.store = self.open()
        before = len(self.e.calls)
        self.store.add(payload)
        self.assertEqual(len(self.e.calls), before)
        with self.assertRaises(ContractError):self.store.add(self.payload(['changed']))
        self.search('unmodified query sentinel')
        with self.store.connect() as db:
            self.assertEqual(db.execute('select indexed_text from mem0_source_index').fetchall(), [('unmodified source',)])

    def test_partial_add_unknown_blocks_archive_and_replay(self):
        payload = self.payload(['first source', 'second source'])
        original = self.e.embed
        count = 0
        def fail_second(texts):
            nonlocal count
            count += 1
            if count == 2:raise RuntimeError('synthetic second provider failure')
            return original(texts)
        with patch.object(self.e, 'embed', side_effect=fail_second):
            with self.assertRaises(Exception):self.store.add(payload)
        with self.store.connect() as db:
            self.assertEqual(db.execute('select count(*) from memories').fetchone()[0], 0)
            self.assertEqual(db.execute('select state from add_operations').fetchone()[0], 'uncertain')
        before = len(self.e.calls)
        with self.assertRaises(ContractError):self.store.add(payload)
        with self.assertRaises((ContractError, ValueError)):self.search('first')
        self.assertEqual(len(self.e.calls), before)

    def test_native_source_mismatch_rejected(self):
        self.store.add(self.payload(['correct source']))
        result = self.store.native.search('correct', filters={'user_id': __import__('agent_memory.mem0_bridge', fromlist=['native_scope']).native_scope('alice')})
        result['results'][0]['metadata']['tracemem_source_id'] = 'not-an-original'
        with patch.object(self.store.native, 'search', return_value=result):
            with self.assertRaises(ContractError):self.search('correct')

    def test_context_and_top_k_are_bounded(self):
        self.store.add(self.payload(['earlier support', 'rare lexical anchor', 'later support']))
        rows = self.search('rare lexical', top=3)
        self.assertEqual([text_of(r['content']) for r in rows], ['rare lexical anchor', 'earlier support', 'later support'])
        self.assertEqual(len(self.search('rare lexical', top=1)), 1)

    def test_archive_index_failure_is_atomic(self):
        with patch.object(self.store, '_persist_descriptions', side_effect=RuntimeError('synthetic')):
            with self.assertRaises(RuntimeError):self.store.add(self.payload(['source']))
        with self.store.connect() as db:
            self.assertEqual(db.execute('select count(*) from memories').fetchone()[0], 0)
            self.assertEqual(db.execute('select count(*) from mem0_source_index').fetchone()[0], 0)

    def test_unicode_source_is_not_truncated(self):
        original = '原始书架记录 cobalt '*900
        self.store.add(self.payload([original]))
        self.assertGreater(len(self.e.calls[0]), 1)
        rows = self.search('cobalt')
        self.assertEqual(rows[0]['content'], content_parts(original))

    def test_incomplete_index_rejected_before_embedding(self):
        self.store.add(self.payload(['source']))
        with self.store.connect() as db:db.execute('delete from mem0_source_index')
        before = len(self.e.calls)
        with self.assertRaises(ContractError):self.search('source')
        self.assertEqual(len(self.e.calls), before)
