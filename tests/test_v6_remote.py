"""Pure synthetic remote-worker contracts; no SSH, models, GPUs, or native data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_atm_v6_remote as worker


def render(messages):
    system = messages[0]['content'][0]['text']
    user = ''.join(row['text'] for row in messages[1]['content'])
    return '<system>' + system + '</system><user>' + user + '</user><assistant>'


class RemoteWorkerTests(unittest.TestCase):
    def test_no_truncation_preserves_exact_template_and_full_originals(self):
        query, document = ' Q?中文 ', 'Original\n document 😀'
        result = worker.fit_document(query, document, render, lambda s: list(map(ord, s)))
        self.assertFalse(result['truncated'])
        text = ''.join(map(chr, result['input_ids']))
        self.assertEqual(text, render(worker.messages(query, document)))
        self.assertIn(query, text)
        self.assertTrue(text.endswith('</user><assistant>'))

    def test_document_only_prefix_truncation_keeps_query_and_assistant_suffix(self):
        query, document = 'UNCHANGED_QUESTION 😀', '本文🙂' * 1000
        overhead = len(render(worker.messages(query, '')))
        result = worker.fit_document(query, document, render, lambda s: list(map(ord, s)), overhead + 17)
        self.assertTrue(result['truncated'])
        self.assertLessEqual(len(result['input_ids']), overhead + 17)
        selected = document[:result['document_characters']]
        self.assertEqual(''.join(map(chr, result['input_ids'])), render(worker.messages(query, selected)))
        self.assertEqual(selected, document[:17])

    def test_oversized_query_is_rejected_never_truncated(self):
        with self.assertRaises(worker.WorkerBlocked):
            worker.fit_document('query' * 100, 'document', render, lambda s: list(map(ord, s)), 50)

    def test_batches_enforce_padded_budget_and_stable_order(self):
        lengths = [4096, 4096, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 100]
        items = [dict(pair_id=str(i), input_ids=[1] * n) for i, n in enumerate(lengths)]
        actual = list(worker.batches(items))
        self.assertEqual([row['pair_id'] for batch in actual for row in batch], [row['pair_id'] for row in items])
        for batch in actual:
            self.assertLessEqual(len(batch), 8)
            self.assertLessEqual(len(batch) * max(len(row['input_ids']) for row in batch), 8192)

    def test_four_shards_and_pilot_resume_cover_exact_ordinals(self):
        pairs = [dict(pair_id=str(i)) for i in range(1044)]
        shards = [worker.shard(pairs, i) for i in range(4)]
        self.assertEqual(sorted(int(row['pair_id']) for rows in shards for row in rows), list(range(1044)))
        self.assertEqual(shards[0][255]['pair_id'], '1020')
        self.assertEqual(shards[0][256]['pair_id'], '1024')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scores-0.jsonl'
            worker.append_scores(path, shards[0][:256], [0.] * 256)
            self.assertEqual(worker.committed_prefix(path, shards[0]), 256)

    def test_committed_scores_require_order_finite_and_exact_schema(self):
        expected = [dict(pair_id='a'), dict(pair_id='b')]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scores.jsonl'
            for records in ([dict(pair_id='b', score=1.)], [dict(pair_id='a', score=float('inf'))],
                            [dict(pair_id='a', score=1., query='forbidden')],
                            [dict(pair_id='a', score=1.), dict(pair_id='a', score=2.)]):
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
                with self.assertRaises((worker.WorkerBlocked, ValueError)):
                    worker.committed_prefix(path, expected)
            with self.assertRaises(worker.WorkerBlocked):
                worker.append_scores(path, expected, [1., float('nan')])

    def test_partial_tail_fails_closed_until_explicit_repair_with_backup(self):
        expected = [dict(pair_id='a'), dict(pair_id='b')]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scores.jsonl'
            worker.append_scores(path, expected[:1], [0.5])
            valid = path.read_bytes()
            tail = b'{"pair_id":"b","score":'
            path.write_bytes(valid + tail)
            with self.assertRaises(worker.PartialScoreTail):
                worker.committed_prefix(path, expected)
            self.assertEqual(path.read_bytes(), valid + tail)
            self.assertEqual(worker.committed_prefix(path, expected, repair_partial_tail=True), 1)
            self.assertEqual(path.read_bytes(), valid)
            self.assertEqual(next(Path(temporary).glob('*.bin')).read_bytes(), tail)
            worker.append_scores(path, expected[1:], [0.9])
            self.assertEqual(worker.committed_prefix(path, expected), 2)
            corrupt = b'{"pair_id":"wrong","score":1}\n' + tail
            path.write_bytes(corrupt)
            with self.assertRaises(worker.WorkerBlocked):
                worker.committed_prefix(path, expected, repair_partial_tail=True)
            self.assertEqual(path.read_bytes(), corrupt)

    def fixture(self, root):
        query_id = json.dumps(['regular', 'synthetic'], separators=(',', ':'))
        rows = {'queries.jsonl': [dict(query_id=query_id, dataset='regular', id='synthetic', qtype='open_end', question='Which source?')],
                'sources.jsonl': [dict(source_id='source', text='Synthetic source only')],
                'candidates.jsonl': [dict(synthetic=True)],
                'pairs.jsonl': [dict(pair_id=worker.pair_identity(query_id, 'source'), query_id=query_id, source_id='source')]}
        for name, values in rows.items():
            (root / name).write_text(''.join(worker.canonical(value) + '\n' for value in values), encoding='utf-8')
        frozen = dict(reranker_spec=copy.deepcopy(worker.SPEC), workers=4, query_count=1, source_count=1, pair_count=1,
            remote_runner_sha256=worker.file_hash(Path(worker.__file__)),
            artifact_hashes={'runtime/atm-v6-rerank/' + name: worker.file_hash(root / name) for name in rows})
        (root / 'FROZEN_CONFIG.json').write_text(worker.canonical(frozen), encoding='utf-8')
        return frozen

    def test_frozen_reference_inputs_and_runner_fingerprint_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self.fixture(root)
            self.assertEqual(len(worker.load_inputs(root)['pairs']), 1)
            frozen['remote_runner_sha256'] = '0' * 64
            (root / 'FROZEN_CONFIG.json').write_text(worker.canonical(frozen))
            with self.assertRaises(worker.WorkerBlocked):
                worker.load_inputs(root)

    def test_changed_data_or_added_gold_is_rejected_even_if_hash_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self.fixture(root)
            query = worker.strict_json((root / 'queries.jsonl').read_bytes())
            query['answer'] = 'forbidden'
            (root / 'queries.jsonl').write_text(worker.canonical(query) + '\n')
            with self.assertRaises(worker.WorkerBlocked):
                worker.load_inputs(root)
            frozen['artifact_hashes']['runtime/atm-v6-rerank/queries.jsonl'] = worker.file_hash(root / 'queries.jsonl')
            (root / 'FROZEN_CONFIG.json').write_text(worker.canonical(frozen))
            with self.assertRaises(worker.WorkerBlocked):
                worker.load_inputs(root)

    def test_explicit_execute_required_before_worker_or_gpu_entry(self):
        with patch.object(worker, 'run_worker') as execution:
            with self.assertRaises(SystemExit):
                worker.main(['--work', 'unused', '--gpu-uuid', 'unused', '--worker-index', '0'])
            execution.assert_not_called()


if __name__ == '__main__':
    unittest.main()
