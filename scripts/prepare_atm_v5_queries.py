"""Prepare a private, question-only V5 cache; never read gold or send requests.

The complete regular/Hard projections are verified against their frozen runs.
This is an evaluation cache, separate from all source memory namespaces.
"""
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.embedding_cache import CacheScope, SourceEmbeddingCache

REVISION = '78e826dc07e97466b2f54443831ef9a83ab8b27c'
SOURCE_FROZEN = 'runtime/atm-v5-source-cache/FROZEN_SOURCE.json'
SOURCE_CACHE = 'runtime/atm-v5-source-cache/source-cache.sqlite'
QUERY_WORK = 'runtime/atm-v5-query-cache'
QUERY_CACHE = QUERY_WORK + '/query-cache.sqlite'
QUERY_FROZEN = QUERY_WORK + '/FROZEN_QUERIES.json'
QUERY_REPORT = 'reports/atm-v5-query-preparation.json'
CODE = ('agent_memory/embedding_cache.py', 'agent_memory/shared_dense.py',
        'agent_memory/atm_n0.py', 'agent_memory/retrieval.py', 'agent_memory/native_metrics.py',
        'scripts/prepare_atm_v5_queries.py', 'scripts/score_atm_v5_dense.py')
DATASETS = (
    dict(name='regular', count=1013,
         query_file='data/atm-schema/atm-bench.json.questions.json',
         anchor_file='reports/atm-n0-run.json', query_hash_field='question_public_sha256',
         baseline_file='runtime/atm-n0/retrieval-results.jsonl', baseline_hash_field='retrieval_result_sha256',
         baseline_method=None, gold_file='data/atm-schema/scorer-only/atm-bench.json',
         gold_sha256='ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762'),
    dict(name='hard', count=31,
         query_file='data/atm-schema/atm-bench-hard.json.questions.json',
         anchor_file='reports/atm-v4-hard-run.json', query_hash_field='question_sha256',
         baseline_file='runtime/atm-v4-hard/retrieval-results.jsonl', baseline_hash_field='ranking_sha256',
         baseline_method='flat', gold_file='data/atm-schema/scorer-only/atm-bench-hard.json',
         gold_sha256='acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a'),
)


def path_at(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('project-relative path required')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('artifact escapes project root')
    return path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(root, relative):
    return digest(path_at(root, relative).read_bytes())


def load_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    def bad_constant(_value):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def verified_bytes(root, relative, expected):
    raw = path_at(root, relative).read_bytes()
    if digest(raw) != expected:
        raise ValueError('frozen artifact hash mismatch: ' + relative)
    return raw


def save_new(root, relative, value):
    path = path_at(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def query_id(dataset, identity):
    # JSON identity avoids delimiter ambiguity while preserving original IDs separately.
    return json.dumps([dataset, identity], ensure_ascii=False, separators=(',', ':'))


def read_questions(root, datasets):
    """Load only exact id/question/qtype projections; reject every extra field."""
    output = []
    seen = set()
    for spec in datasets:
        if 'scorer-only' in path_at(root, spec['query_file']).parts:
            raise ValueError('query preparation cannot read a scorer-only file')
        rows = load_json(verified_bytes(root, spec['query_file'], spec['query_sha256']))
        if not isinstance(rows, list) or len(rows) != spec['count']:
            raise ValueError('complete native question count required')
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'id', 'question', 'qtype'}
                    or not isinstance(row['id'], str) or not row['id']
                    or not isinstance(row['question'], str) or not row['question'].strip()
                    or len(row['question'].encode('utf-8')) > 8192
                    or row['qtype'] not in ('number', 'open_end', 'list_recall')):
                raise ValueError('unmodified question-only record within one embedding input required')
            key = query_id(spec['name'], row['id'])
            if key in seen:
                raise ValueError('duplicate native question identity')
            seen.add(key)
            output.append(dict(dataset=spec['name'], cache_id=key, **row))
    return output


def prepare_queries(root=ROOT, *, datasets=DATASETS, offline_test=False):
    root = Path(root).resolve()
    if path_at(root, QUERY_WORK).exists() or path_at(root, QUERY_REPORT).exists():
        raise FileExistsError('query cache preparation already exists; preserve it')
    source_raw = path_at(root, SOURCE_FROZEN).read_bytes()
    source = load_json(source_raw)
    source_scope = CacheScope(**source['scope'])
    if source_scope.dataset_revision != REVISION:
        raise ValueError('pinned source dataset revision required')
    inputs = {SOURCE_FROZEN: digest(source_raw)}
    specs = []
    for original in datasets:
        spec = dict(original)
        if spec['name'] not in ('regular', 'hard'):
            raise ValueError('only original regular and Hard datasets are supported')
        anchor_raw = path_at(root, spec['anchor_file']).read_bytes()
        anchor = load_json(anchor_raw)
        spec['query_sha256'] = anchor[spec['query_hash_field']]
        spec['baseline_sha256'] = anchor[spec['baseline_hash_field']]
        inputs[spec['anchor_file']] = digest(anchor_raw)
        inputs[spec['query_file']] = spec['query_sha256']
        inputs[spec['baseline_file']] = spec['baseline_sha256']
        verified_bytes(root, spec['baseline_file'], spec['baseline_sha256'])
        specs.append(spec)
    if {spec['name'] for spec in specs} != {'regular', 'hard'} or len(specs) != 2:
        raise ValueError('exactly one regular and one Hard dataset required')
    # Freeze method hashes before parsing public question contents.
    code_hashes = {name: sha(root, name) for name in CODE}
    questions = read_questions(root, specs)
    scope = CacheScope(namespace='atm-78e826dc-public-query-v5', dataset_revision=REVISION,
                       representation='native-question-utf8-v1', model=source_scope.model,
                       dimensions=source_scope.dimensions)
    if scope.namespace == source_scope.namespace:
        raise ValueError('query and source namespaces must be separate')
    cache = SourceEmbeddingCache(path_at(root, QUERY_CACHE), scope, offline_test=offline_test)
    prepared = cache.prepare_sources([dict(id=row['cache_id'], text=row['question']) for row in questions])
    if any(len(row['chunk_keys']) != 1 for row in prepared['manifest']['sources']):
        raise ValueError('each original query must map to exactly one embedding chunk')
    for name, expected in {**inputs, **code_hashes}.items():
        verified_bytes(root, name, expected)
    frozen = dict(version='V5-query-cache', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_revision=REVISION, scope=asdict(scope), source_scope=asdict(source_scope),
        source_frozen_file=SOURCE_FROZEN, source_frozen_sha256=inputs[SOURCE_FROZEN],
        source_cache_file=SOURCE_CACHE, query_cache_file=QUERY_CACHE,
        manifest_sha256=prepared['manifest_sha256'], query_count=len(questions),
        unique_chunk_count=prepared['chunk_count'], datasets=specs,
        query_map=[{key: row[key] for key in ('dataset', 'id', 'qtype', 'cache_id')} for row in questions],
        input_hashes=inputs, code_hashes=code_hashes, gold_loaded=False,
        source_memory_modified=False, provider_requests=0,
        exposure='Regular and Hard are previously evaluated public development datasets; not new heldout data',
        methods=dict(dense='all-source max-chunk cosine', hybrid='flat/dense native Top100 lanes, equal RRF60'))
    save_new(root, QUERY_FROZEN, frozen)
    report = dict(state='query_cache_prepared_no_vectors', query_count=len(questions),
        counts=dict(Counter(row['dataset'] for row in questions)),
        unique_chunks=prepared['chunk_count'], missing_batches=len(cache.missing_batches()),
        query_cache_file=QUERY_CACHE, frozen_file=QUERY_FROZEN, frozen_sha256=sha(root, QUERY_FROZEN),
        provider_requests=0, gold_loaded=False, source_memory_modified=False,
        missing_is_not_dispatch_authorization=True, original_questions_local_private_only=True)
    save_new(root, QUERY_REPORT, report)
    return report


if __name__ == '__main__':
    print(json.dumps(prepare_queries(), indent=2))
