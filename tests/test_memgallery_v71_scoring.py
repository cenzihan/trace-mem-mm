"""Additive wrapper contracts only; synthetic metadata and no benchmark reads."""
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import score_memgallery_v71 as wrapper


class MemGalleryV71ScorerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        names = [*wrapper.PATHS, 'reader_payload', 'decode_reader', 'CODE', 'verify_reader_complete', 'save_new']
        for name in names:
            self.stack.enter_context(patch.object(wrapper.original, name, getattr(wrapper.original, name)))
        self.native = SimpleNamespace(reader_payload=Mock(name='new_reader_payload'), decode_reader=Mock(name='new_decode'))
        self.stack.enter_context(patch.object(wrapper, 'native_reader', return_value=self.native))

    def test_install_only_rebinds_chat_and_decoder_idempotently(self):
        preserved = {name: getattr(wrapper.original, name) for name in
            ('GOLD', 'QUERIES', 'SELECTION', 'RANKINGS', 'RETRIEVAL_SCORES', 'RETRIEVAL_DETAILS',
             'paired', 'aggregate_reader', 'judge_batches', 'build_proxy_judge', 'decode_proxy_judge')}
        wrapper.install(); first_code = wrapper.original.CODE; wrapper.install()
        for name, value in wrapper.PATHS.items():
            self.assertEqual(getattr(wrapper.original, name), value)
        for name, value in preserved.items():
            self.assertIs(getattr(wrapper.original, name), value)
        self.assertIs(wrapper.original.reader_payload, self.native.reader_payload)
        self.assertIs(wrapper.original.decode_reader, self.native.decode_reader)
        self.assertEqual(wrapper.original.CODE, first_code)
        self.assertTrue(set(wrapper.ORIGINAL_CODE + wrapper.EXTRA_CODE) <= set(first_code))

    def test_new_interface_verified_before_original_completion_and_frozen_into_inputs(self):
        complete = {'input_hashes': {'synthetic': 'old'}}; order = []
        with (patch.object(wrapper, 'verify_interface', side_effect=lambda _: order.append('interface')),
             patch.object(wrapper, 'BASE_VERIFY_READER', side_effect=lambda *a, **k: (order.append('reader') or complete)),
             patch.object(wrapper.original, 'sha', return_value='interface-hash')):
            result = wrapper.verify_reader_complete(Path('synthetic'), final=True)
        self.assertEqual(order, ['interface', 'reader'])
        self.assertEqual(result['input_hashes'][wrapper.INTERFACE], 'interface-hash')

    def test_incomplete_new_reader_never_reaches_gold(self):
        wrapper.install()
        with tempfile.TemporaryDirectory() as tmp, patch.object(wrapper, 'verify_interface'), \
                patch.object(wrapper, 'BASE_VERIFY_READER', side_effect=ValueError('new reader incomplete')), \
                patch.object(wrapper.original, 'read_gold_after_completion') as gold:
            with self.assertRaises(ValueError):
                wrapper.prepare_judge(Path(tmp), count_tokens=lambda _: 1)
            gold.assert_not_called()

    def test_new_reports_annotated_before_exclusive_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / wrapper.PATHS['JUDGE_FROZEN']
            value = {'input_hashes': {wrapper.INTERFACE: 'test-only-hash'}}
            digest = wrapper.annotated_save(path, value)
            stored = json.loads(path.read_text(encoding='utf-8'))
            self.assertFalse(stored['independent_replication'])
            self.assertEqual(stored['new_reader_inferences'], 64)
            self.assertEqual(stored['original_v7_reader_receipts_replayed'], 0)
            self.assertEqual(stored['interface_frozen_file'], wrapper.INTERFACE)
            self.assertEqual(wrapper.original.digest(path.read_bytes()), digest)
            with self.assertRaises(FileExistsError):
                wrapper.annotated_save(path, value)

    def test_private_rows_do_not_get_report_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / wrapper.PATHS['READER_DETAILS']
            value = [{'index': 0, 'score': .25}]
            wrapper.annotated_save(path, value)
            self.assertEqual(json.loads(path.read_text()), value)

    def test_cli_cannot_rescore_shared_retrieval(self):
        with patch.object(wrapper, 'install') as install, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            wrapper.main(['retrieval'])
        install.assert_not_called()

    def test_interface_scope_rejects_repair_or_missing_old_receipts(self):
        valid = dict(state='new_alias_interface_frozen_no_calls', same_queries_rankings_complete20sources=True,
            new_provider_calls_max=72, new_preflight_cny='18', original_reader_calls_replayed=0,
            answer_text_repair=False, gold_used_for_interface_design=False,
            hashes={name: 'synthetic-hash' for name in (
                'runtime/memgallery-v7/chat-temperature0/FAILURE_COMPLETED.json',
                'runtime/memgallery-v7/chat-temperature0/FROZEN_READER.json',
                'runtime/memgallery-v7/chat-temperature0/FROZEN_LIVE.json',
                'runtime/memgallery-v7/chat-temperature0/reader-packs.jsonl',
                'runtime/memgallery-v7/chat-temperature0/reader-answers.jsonl',
                'runtime/memgallery-v7/chat-temperature0/run.sqlite',
                'runtime/memgallery-v7/chat-temperature0/receipts/private-embedding.sqlite',
                wrapper.PATHS['READER_FROZEN'], wrapper.PATHS['PACKS'])})
        with patch.object(wrapper.original, 'load', return_value=valid), \
                patch.object(wrapper.original, 'check_hashes') as hashes:
            self.assertIs(wrapper.verify_interface(Path('synthetic')), valid)
            hashes.assert_called_once()
            valid['answer_text_repair'] = True
            with self.assertRaises(ValueError): wrapper.verify_interface(Path('synthetic'))
            valid['answer_text_repair'] = False
            valid['hashes'].pop('runtime/memgallery-v7/chat-temperature0/receipts/private-embedding.sqlite')
            with self.assertRaises(ValueError): wrapper.verify_interface(Path('synthetic'))

    def test_real_new_decoder_binding_preserves_original_answer_and_native_citation(self):
        from agent_memory import memgallery_v71_reader as native
        identity = '["synthetic-session","synthetic-round"]'
        query = dict(index=0, point='FR', question='Synthetic question?', question_image=None, image_caption=None)
        source = {identity: dict(text='Synthetic source.', source_session='synthetic-session',
            source_round='synthetic-round', source_date='2020-01-01', image_ids=[])}
        pack = native.pack_memgallery_evidence(query, [identity], source, lambda s: max(1, len(s)//4))
        value = dict(summary='', evidence_quotes=[], answer='Original exact answer.', cited_ids=['M0001'], confidence='supported')
        response = dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message=dict(content=json.dumps(value)))])
        with patch.object(wrapper, 'native_reader', return_value=native):
            wrapper.install()
            wrapper.original.reader_payload(pack)
            decoded = wrapper.original.decode_reader(response, pack)
        self.assertEqual(decoded['answer'], 'Original exact answer.')
        self.assertEqual(decoded['cited_ids'], [identity])
        self.assertEqual(decoded['cited_aliases'], ['M0001'])
        self.assertEqual(decoded['question_index'], 0)


if __name__ == '__main__':
    unittest.main()
