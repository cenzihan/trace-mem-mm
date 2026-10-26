"""Synthetic V7 remote admission/control contracts; never SSH, GPU or provider.

Every data/config/code identity fixture is temporary. The original worker is
entered only through its already-complete-shard branch, with locks and GPU/model
entry points mocked. No native experiment artifact is opened by these tests.
"""
import copy
import io
import tempfile
import unittest
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import control_memgallery_v7_remote as control
from scripts import run_memgallery_v7_remote as wrapper

worker = wrapper.worker
PREFIX = 'runtime/memgallery-v7/rerank/'
GPU_UUIDS = ['GPU-' + str(index).zfill(8) + '-0000-0000-0000-000000000000' for index in range(4)]


class MemGalleryV7RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / PREFIX
        self.work.mkdir(parents=True)
        self.runner = self.root / 'scripts/run_atm_v6_remote.py'
        self.wrapper = self.root / 'scripts/run_memgallery_v7_remote.py'
        self.controller = self.root / 'scripts/control_memgallery_v7_remote.py'
        for code in (self.runner, self.wrapper, self.controller):
            code.parent.mkdir(parents=True, exist_ok=True)
            code.write_text('# Synthetic execution identity: ' + code.name + '\n', encoding='utf-8')
        sources = [worker.canonical(['synthetic-session', 'round-' + str(i)]) for i in range(103)]
        self.rows = {
            'queries.jsonl': [dict(query_id='memgallery-v7:' + str(i), index=i, point=point,
                                   question='Synthetic original query ' + str(i))
                              for i, point in enumerate(('VS', 'CD'))],
            'sources.jsonl': [dict(source_id=identity, text='Synthetic source ' + str(i))
                              for i, identity in enumerate(sources)],
            'candidates.jsonl': [], 'pairs.jsonl': [],
        }
        for query in self.rows['queries.jsonl']:
            methods = dict(dense=dict(ranked_ids=sources[:100], scores=list(map(float, range(100, 0, -1)))),
                           hybrid=dict(ranked_ids=sources[2:102], scores=list(map(float, range(100, 0, -1)))))
            self.rows['candidates.jsonl'].append(dict(query_id=query['query_id'], index=query['index'],
                                                      point=query['point'], methods=methods))
            for source in sorted(set(methods['dense']['ranked_ids']) | set(methods['hybrid']['ranked_ids'])):
                self.rows['pairs.jsonl'].append(dict(pair_id=worker.pair_identity(query['query_id'], source),
                    query_id=query['query_id'], source_id=source))
        self.freeze = dict(experiment='memgallery-v7', reranker_spec=copy.deepcopy(wrapper.SPEC), workers=4,
                           query_count=2, source_count=103, pair_count=204,
                           remote_runner_sha256=worker.file_hash(self.runner),
                           remote_wrapper_sha256=worker.file_hash(self.wrapper),
                           artifact_hashes={}, code_hashes={name: worker.file_hash(self.root / name)
                                                          for name in control.CODE}, input_hashes={})
        self.rebind()

    def dump(self, path, value):
        path.write_text(worker.canonical(value), encoding='utf-8')

    def rebind(self):
        for name, rows in self.rows.items():
            (self.work / name).write_text(''.join(worker.canonical(row) + '\n' for row in rows), encoding='utf-8')
        self.freeze['artifact_hashes'] = {PREFIX + name: worker.file_hash(self.work / name) for name in self.rows}
        self.dump(self.work / 'FROZEN_CONFIG.json', self.freeze)

    def admitted(self):
        return wrapper.load_inputs(self.work, self.runner, self.wrapper)

    def controlled(self):
        stack = ExitStack()
        for name, value in (('ROOT', self.root), ('WORK', self.work), ('RUNNER', self.runner), ('WRAPPER', self.wrapper)):
            stack.enter_context(patch.object(control, name, value))
        # Any accidental subprocess/SSH action outside an explicit test spy fails.
        stack.enter_context(patch.object(control.subprocess, 'run', side_effect=AssertionError('No subprocess allowed')))
        return stack

    def staged(self):
        files = [self.work / name for name in control.INPUTS] + [self.runner, self.wrapper]
        self.dump(self.work / 'REMOTE_STAGE.json', dict(remote_directory=control.REMOTE,
            hashes={path.name: worker.file_hash(path) for path in files}))

    def inventory(self):
        return dict(gpus=[dict(index=i, uuid=GPU_UUIDS[i], used=0, total=16000, compute_process=False) for i in range(4)])

    def test_native_index_point_inputs_and_distinct_union_admitted_without_atm_aliases(self):
        data = self.admitted()
        self.assertEqual(set(data['queries']), {'memgallery-v7:0', 'memgallery-v7:1'})
        self.assertEqual(len(data['sources']), 103)
        self.assertEqual(len(data['pairs']), 204)  # 102 union pairs, not 200 per question.
        self.assertEqual(len({row['pair_id'] for row in data['pairs']}), 204)
        for row in self.rows['queries.jsonl']:
            self.assertEqual(set(row), {'query_id', 'index', 'point', 'question'})
            self.assertNotIn('qtype', row)

    def test_both_runner_hashes_and_data_hash_are_independently_checked(self):
        for field in ('remote_runner_sha256', 'remote_wrapper_sha256'):
            with self.subTest(field=field):
                prior = self.freeze[field]
                self.freeze[field] = '0' * 64
                self.rebind()
                with self.assertRaises(wrapper.WorkerBlocked):
                    self.admitted()
                self.freeze[field] = prior
        self.rebind()
        path = self.work / 'sources.jsonl'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(wrapper.WorkerBlocked):
            self.admitted()

    def test_colocated_execution_hashes_remain_in_end_of_worker_recheck_map(self):
        runner, wrapped = self.work / self.runner.name, self.work / self.wrapper.name
        runner.write_bytes(self.runner.read_bytes())
        wrapped.write_bytes(self.wrapper.read_bytes())
        hashes = wrapper.load_inputs(self.work, runner, wrapped)['hashes']
        self.assertEqual(hashes[runner.name], self.freeze['remote_runner_sha256'])
        self.assertEqual(hashes[wrapped.name], self.freeze['remote_wrapper_sha256'])

    def test_gold_and_atm_alias_fields_rejected_even_after_rebinding_hashes(self):
        cases = [('queries.jsonl', field) for field in ('answer', 'clue', 'gold', 'dataset', 'qtype', 'id')]
        cases += [(name, 'gold') for name in ('sources.jsonl', 'candidates.jsonl', 'pairs.jsonl')]
        for name, field in cases:
            with self.subTest(file=name, field=field):
                self.rows[name][0][field] = 'Synthetic forbidden field'
                self.rebind()
                with self.assertRaises(wrapper.WorkerBlocked):
                    self.admitted()
                del self.rows[name][0][field]
        self.rebind()

    def test_duplicate_query_source_candidate_or_pair_rejected_with_valid_file_hash(self):
        for name in self.rows:
            with self.subTest(file=name):
                self.rows[name].append(copy.deepcopy(self.rows[name][0]))
                self.rebind()
                with self.assertRaises(wrapper.WorkerBlocked):
                    self.admitted()
                self.rows[name].pop()
        self.rebind()

    def test_pairs_must_cover_exact_candidate_union_not_just_match_a_count(self):
        original = copy.deepcopy(self.rows['pairs.jsonl'])
        del self.rows['pairs.jsonl'][-1]
        self.freeze['pair_count'] -= 1
        self.rebind()
        with self.assertRaises(wrapper.WorkerBlocked):
            self.admitted()
        self.rows['pairs.jsonl'] = original
        source = self.rows['sources.jsonl'][-1]['source_id']  # Known source, absent from both pools.
        qid = 'memgallery-v7:0'
        self.rows['pairs.jsonl'].append(dict(pair_id=worker.pair_identity(qid, source), query_id=qid, source_id=source))
        self.freeze['pair_count'] = len(self.rows['pairs.jsonl'])
        self.rebind()
        with self.assertRaises(wrapper.WorkerBlocked):
            self.admitted()

    def test_candidate_native_top100_duplicate_unknown_and_nonfinite_rejected(self):
        original = copy.deepcopy(self.rows['candidates.jsonl'][0]['methods']['dense'])
        for change in ('short', 'duplicate', 'unknown', 'nonfinite'):
            with self.subTest(change=change):
                ranking = copy.deepcopy(original)
                if change == 'short':
                    ranking['ranked_ids'].pop(); ranking['scores'].pop()
                elif change == 'duplicate':
                    ranking['ranked_ids'][0] = ranking['ranked_ids'][1]
                elif change == 'unknown':
                    ranking['ranked_ids'][0] = 'unknown-native-source'
                else:
                    ranking['scores'][0] = 'not-a-finite-number'
                self.rows['candidates.jsonl'][0]['methods']['dense'] = ranking
                self.rebind()
                with self.assertRaises(wrapper.WorkerBlocked):
                    self.admitted()

    def test_candidate_index_requires_native_integer_not_bool_or_float_alias(self):
        for alias in (True, 1.0):
            with self.subTest(index=alias):
                self.rows['candidates.jsonl'][1]['index'] = alias
                self.rebind()
                with self.assertRaises(wrapper.WorkerBlocked):
                    self.admitted()

    def test_wrapper_execute_gate_precedes_original_worker_entry_and_restores_hooks(self):
        loader, writer = worker.load_inputs, worker.atomic_json
        with patch.object(worker, 'run_worker') as execution, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                wrapper.main(['--work', 'unused', '--gpu-uuid', 'unused', '--worker-index', '0'])
            self.assertEqual(stopped.exception.code, 2)
            execution.assert_not_called()
        self.assertIs(worker.load_inputs, loader)
        self.assertIs(worker.atomic_json, writer)

    def test_wrapper_calls_original_worker_complete_prefix_without_gpu_or_model(self):
        self.freeze['remote_runner_sha256'] = worker.file_hash(Path(worker.__file__))
        self.freeze['remote_wrapper_sha256'] = worker.file_hash(Path(wrapper.__file__))
        self.rebind()
        loader, writer = worker.load_inputs, worker.atomic_json
        actual = worker.run_worker
        with patch.object(worker, 'REMOTE_ROOT', self.root), \
                patch.object(worker, 'exclusive_lock', return_value=nullcontext()), \
                patch.object(worker, 'committed_prefix', side_effect=lambda _path, pairs, **_kw: len(pairs)), \
                patch.object(worker, 'check_gpu_idle', side_effect=AssertionError('No GPU entry')) as gpu, \
                patch.object(worker, 'verify_model', side_effect=AssertionError('No model entry')) as model, \
                patch.object(worker, 'run_worker', wraps=actual) as execution, redirect_stdout(io.StringIO()):
            code = wrapper.main(['--work', str(self.work), '--gpu-uuid', GPU_UUIDS[0],
                                 '--worker-index', '0', '--execute'])
        self.assertEqual(code, 0)
        execution.assert_called_once()
        gpu.assert_not_called(); model.assert_not_called()
        self.assertIs(worker.load_inputs, loader)
        self.assertIs(worker.atomic_json, writer)
        report = worker.strict_json((self.work / 'report-0.json').read_bytes())
        self.assertEqual(report['state'], 'complete')
        self.assertEqual(report['experiment'], 'memgallery-v7')
        self.assertEqual(report['remote_wrapper_sha256'], self.freeze['remote_wrapper_sha256'])
        self.assertEqual(report['remote_runner_sha256'], self.freeze['remote_runner_sha256'])

    def test_controller_verify_checks_code_hashes_without_contacting_remote(self):
        with self.controlled(), patch.object(control, 'remote', side_effect=AssertionError('No SSH')) as remote:
            result = control.verify()
            self.assertEqual(result['state'], 'verified_not_staged')
            self.assertEqual(result['gpu_jobs_started'], 0)
            self.controller.write_text('# changed reviewed controller', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'frozen local input changed'):
                control.verify()
            remote.assert_not_called()

    def test_controller_mutation_cli_requires_execute_before_any_action(self):
        with patch.object(control, 'stage') as stage, patch.object(control, 'launch') as launch, \
                patch.object(control, 'collect') as collect, patch.object(control, 'remote') as remote, \
                redirect_stderr(io.StringIO()):
            for action in ('stage', 'pilot', 'start', 'resume', 'collect'):
                with self.subTest(action=action), self.assertRaises(SystemExit) as stopped:
                    control.main([action])
                self.assertEqual(stopped.exception.code, 2)
            for action in (stage, launch, collect, remote):
                action.assert_not_called()

    def test_unknown_dispatch_keeps_intent_and_cannot_replay(self):
        self.staged()
        intent = self.work / 'PILOT_LAUNCH.intent.json'
        def uncertain_dispatch(_code):
            self.assertTrue(intent.is_file())
            self.assertEqual(worker.strict_json(intent.read_bytes())['state'], 'dispatch_intent_no_automatic_replay')
            raise TimeoutError('synthetic unknown SSH result')
        with self.controlled(), patch.object(control, 'inventory', side_effect=self.inventory), \
                patch.object(control, 'remote', side_effect=uncertain_dispatch) as remote:
            with self.assertRaises(TimeoutError):
                control.launch(pilot=True)
            remote.assert_called_once()
            self.assertFalse((self.work / 'PILOT_LAUNCH.json').exists())
            self.assertTrue(intent.exists())
            remote.reset_mock()
            with self.assertRaises(FileExistsError):
                control.launch(pilot=True)
            remote.assert_not_called()

    def test_controller_stage_mismatch_blocks_before_inventory_or_dispatch(self):
        self.staged()
        path = self.work / 'REMOTE_STAGE.json'
        staged = worker.strict_json(path.read_bytes())
        staged['hashes'][self.wrapper.name] = '0' * 64
        self.dump(path, staged)
        with self.controlled(), patch.object(control, 'inventory') as inventory, patch.object(control, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'stage does not match'):
                control.launch(pilot=True)
            inventory.assert_not_called(); remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
