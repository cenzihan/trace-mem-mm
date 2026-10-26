"""Native Mem-Gallery admission wrapper around the unchanged V6 GPU worker.

Only input schema admission and report provenance differ. Model verification,
tokenization, scoring, durable prefixes, GPU locks and deadlines use the original
worker functions. Importing either file never loads GPU libraries or a model.
"""
import importlib.util
import math
from pathlib import Path


ORIGINAL = Path(__file__).with_name('run_atm_v6_remote.py')
_spec = importlib.util.spec_from_file_location('_memgallery_v7_original_worker', ORIGINAL)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
WorkerBlocked = worker.WorkerBlocked
SPEC = worker.SPEC


def load_inputs(work, runner_path=None, wrapper_path=None):
    work = Path(work)
    runner_path = Path(runner_path or ORIGINAL)
    wrapper_path = Path(wrapper_path or __file__)
    frozen_path = work / 'FROZEN_CONFIG.json'
    if frozen_path.is_symlink() or not frozen_path.is_file() or frozen_path.stat().st_size > 2 * 1024 * 1024:
        raise WorkerBlocked('bounded ordinary V7 freeze required')
    frozen = worker.strict_json(frozen_path.read_bytes())
    if (frozen.get('experiment') != 'memgallery-v7' or frozen.get('reranker_spec') != SPEC
            or frozen.get('workers') != 4
            or frozen.get('remote_runner_sha256') != worker.file_hash(runner_path)
            or frozen.get('remote_wrapper_sha256') != worker.file_hash(wrapper_path)):
        raise WorkerBlocked('fixed V7 experiment, spec and both execution code hashes required')
    hashes = frozen.get('artifact_hashes')
    if not isinstance(hashes, dict):
        raise WorkerBlocked('all frozen input hashes required')
    names = {}
    for relative, expected in hashes.items():
        if (not isinstance(relative, str) or '\\' in relative or relative.startswith('/')
                or '..' in relative.split('/')):
            raise WorkerBlocked('relative frozen artifact path required')
        name = relative.rsplit('/', 1)[-1]
        path = work / name
        if (name not in worker.FILES or name in names or not worker.valid_digest(expected)
                or path.is_symlink() or not path.is_file() or worker.file_hash(path) != expected):
            raise WorkerBlocked('exact distinct frozen V7 input files required')
        names[name] = expected
    if set(names) != worker.FILES:
        raise WorkerBlocked('all four V7 inputs required')
    queries, metadata, sources = {}, {}, {}
    for row in worker.json_lines(work / 'queries.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'query_id', 'index', 'point', 'question'}
                or type(row['index']) is not int or row['index'] < 0
                or row['query_id'] != 'memgallery-v7:' + str(row['index'])
                or not worker.nonempty(row['point']) or not worker.nonempty(row['question'])
                or row['query_id'] in queries):
            raise WorkerBlocked('exact native question-only V7 records required')
        queries[row['query_id']] = row['question']
        metadata[row['query_id']] = (row['index'], row['point'])
    for row in worker.json_lines(work / 'sources.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'source_id', 'text'}
                or not worker.nonempty(row['source_id']) or not worker.nonempty(row['text'])
                or row['source_id'] in sources):
            raise WorkerBlocked('exact unique source-only records required')
        sources[row['source_id']] = row['text']
    expected_pairs, candidates_seen = set(), set()
    for row in worker.json_lines(work / 'candidates.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'query_id', 'index', 'point', 'methods'}
                or type(row['index']) is not int
                or row['query_id'] not in metadata or row['query_id'] in candidates_seen
                or (row['index'], row['point']) != metadata[row['query_id']]
                or not isinstance(row['methods'], dict) or set(row['methods']) != {'dense', 'hybrid'}):
            raise WorkerBlocked('complete original V7 candidates required')
        candidates_seen.add(row['query_id'])
        for ranking in row['methods'].values():
            if not isinstance(ranking, dict) or set(ranking) != {'ranked_ids', 'scores'}:
                raise WorkerBlocked('exact native ranked ID/score arrays required')
            ids, scores = ranking['ranked_ids'], ranking['scores']
            if (not isinstance(ids, list) or not isinstance(scores, list)
                    or len(ids) != min(100, len(sources)) or len(scores) != len(ids)
                    or any(not isinstance(identity, str) for identity in ids)
                    or len(set(ids)) != len(ids) or not set(ids) <= set(sources)
                    or any(type(score) not in (float, int) or not math.isfinite(score) for score in scores)):
                raise WorkerBlocked('complete unique native Top100 required')
            expected_pairs.update((row['query_id'], identity) for identity in ids)
    if candidates_seen != set(queries):
        raise WorkerBlocked('candidates must cover every native question')
    pairs, seen, actual_pairs = [], set(), set()
    for row in worker.json_lines(work / 'pairs.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'pair_id', 'query_id', 'source_id'}
                or not all(isinstance(value, str) for value in row.values())
                or row['query_id'] not in queries or row['source_id'] not in sources
                or row['pair_id'] != worker.pair_identity(row['query_id'], row['source_id'])
                or row['pair_id'] in seen):
            raise WorkerBlocked('unique frozen query/source pair required')
        seen.add(row['pair_id'])
        actual_pairs.add((row['query_id'], row['source_id']))
        pairs.append(row)
    if actual_pairs != expected_pairs:
        raise WorkerBlocked('pairs must exactly cover both frozen candidate pools once')
    for field, count in (('query_count', len(queries)), ('source_count', len(sources)), ('pair_count', len(pairs))):
        if type(frozen.get(field)) is not int or frozen[field] != count or not count:
            raise WorkerBlocked('frozen V7 input count differs')
    # The original worker checks every returned hash again before completion.
    if runner_path.parent.resolve() == work.resolve() and wrapper_path.parent.resolve() == work.resolve():
        names[runner_path.name] = frozen['remote_runner_sha256']
        names[wrapper_path.name] = frozen['remote_wrapper_sha256']
    return dict(frozen=frozen, frozen_sha256=worker.file_hash(frozen_path), hashes=names,
                queries=queries, sources=sources, pairs=pairs)


def main(argv=None):
    previous_loader, previous_writer = worker.load_inputs, worker.atomic_json
    def report_with_wrapper_provenance(path, value):
        if isinstance(value, dict) and 'worker_index' in value and 'frozen_sha256' in value:
            value = dict(value, experiment='memgallery-v7',
                         remote_wrapper_sha256=worker.file_hash(Path(__file__)))
        return previous_writer(path, value)
    worker.load_inputs, worker.atomic_json = load_inputs, report_with_wrapper_provenance
    try:
        return worker.main(argv)
    finally:
        worker.load_inputs, worker.atomic_json = previous_loader, previous_writer


if __name__ == '__main__':
    raise SystemExit(main())
