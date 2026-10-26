"""Score frozen Hard31 rankings only after verifying source/query/code integrity.

No retriever, provider, credential, or GPU imports. Native question/answer text
is never included in outputs. The CLI is fixed to the pinned Hard release.
"""
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.native_metrics import retrieval_metrics

GOLD_SHA256 = 'acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a'
RUN_FILE = 'reports/atm-v4-hard-run.json'
SCORE_FILE = 'reports/atm-v4-hard-scores.json'
DETAIL_FILE = 'runtime/atm-v4-hard/isolated-per-question-scores.json'
METHODS = ('flat', 'v4')
KS = (1, 5, 10, 20, 50, 100)
METRICS = ('fractional_recall', 'any_hit', 'all_evidence_coverage')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    def invalid_constant(_value):
        raise ValueError('Non-finite JSON value')
    return json.loads(raw, object_pairs_hook=_unique_object,
                      parse_constant=invalid_constant)


def _path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Project-relative artifact path required')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact path escapes project')
    return path


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Canonical SHA256 required')
    return value


def _verified(root, relative, digest, checked):
    path = _path(root, relative)
    raw = path.read_bytes()
    if sha256(raw) != _digest(digest):
        raise ValueError('Artifact SHA256 mismatch: ' + relative)
    checked[relative] = digest
    return raw


def _identity(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Nonempty native ID required')
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_before_gold(root, run_file=RUN_FILE, *, expected_questions=31,
                       expected_sources=11034, expected_gold_sha256=GOLD_SHA256):
    """Validate every non-gold input first; never open the frozen gold file here.

    Count/hash overrides support temporary synthetic contract tests. The CLI
    uses the fixed production defaults and exposes no overrides.
    """
    root = Path(root).resolve()
    raw = _path(root, run_file).read_bytes()
    checked = {run_file: sha256(raw)}
    run = _json(raw)
    if (run.get('state') != 'retrieval_complete_gold_not_loaded'
            or run.get('questions') != expected_questions
            or run.get('source_count') != expected_sources):
        raise ValueError('Complete run with expected corpus/question counts required')
    frozen = _json(_verified(root, run['frozen_file'], run['frozen_sha256'], checked))
    for field in ('code_hashes', 'source_index_sha256', 'source_ids_sha256'):
        if field not in frozen or frozen[field] != run.get(field):
            raise ValueError('Run and frozen manifest disagree: ' + field)
    if not run.get('question_sha256') or run['question_sha256'] != frozen.get('query_sha256'):
        raise ValueError('Run and frozen public question hashes disagree')
    if frozen.get('gold_sha256') != _digest(expected_gold_sha256):
        raise ValueError('Unrecognized frozen gold release')
    gold_path = _path(root, frozen['gold_file'])
    code_hashes = run.get('code_hashes')
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise ValueError('Frozen code hashes required')
    for relative, digest in code_hashes.items():
        if _path(root, relative).suffix != '.py':
            raise ValueError('Code manifest may only contain Python sources')
        _verified(root, relative, digest, checked)
    preserved = frozen.get('preserved_hashes', {})
    if not isinstance(preserved, dict):
        raise ValueError('Preserved artifact hash mapping required')
    for relative, digest in preserved.items():
        path = _path(root, relative)
        if path == gold_path or 'scorer-only' in path.parts:
            raise ValueError('Gold cannot be opened by the preservation audit')
        _verified(root, relative, digest, checked)
    sources = frozen.get('source_files')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Frozen source manifest required')
    seen_files = set()
    for item in sources:
        relative = item['file']
        path = _path(root, relative)
        if path == gold_path or 'scorer-only' in path.parts or path in seen_files:
            raise ValueError('Distinct source-only files required')
        seen_files.add(path)
        _verified(root, relative, item['sha256'], checked)
    query_file = frozen['query_file']
    if _path(root, query_file) == gold_path or 'scorer-only' in _path(root, query_file).parts:
        raise ValueError('Public question file must be isolated from gold')
    questions = _json(_verified(root, query_file, frozen['query_sha256'], checked))
    if not isinstance(questions, list) or len(questions) != expected_questions:
        raise ValueError('Public question count mismatch')
    public = {}
    public_text = {}
    for question in questions:
        if not isinstance(question, dict) or set(question) != {'id', 'question', 'qtype'}:
            raise ValueError('Public questions must not contain gold fields')
        identity = _identity(question['id'])
        if (identity in public or question['qtype'] not in ('number', 'open_end', 'list_recall')
                or not isinstance(question['question'], str) or not question['question'].strip()):
            raise ValueError('Invalid or duplicate public question')
        public[identity] = question['qtype']
        public_text[identity] = question['question']
    index_file = frozen['source_index_file']
    index_path = _path(root, index_file)
    if index_path == gold_path or 'scorer-only' in index_path.parts:
        raise ValueError('Source index must be isolated from gold')
    _verified(root, index_file, run['source_index_sha256'], checked)
    with closing(sqlite3.connect(index_path.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        source_ids = [_identity(row[0]) for row in db.execute('SELECT id FROM sources')]
    if len(source_ids) != expected_sources or len(set(source_ids)) != expected_sources:
        raise ValueError('Source ID count or uniqueness mismatch')
    identity_hash = sha256(json.dumps(sorted(source_ids), separators=(',', ':')).encode('utf-8'))
    if identity_hash != run['source_ids_sha256']:
        raise ValueError('Source ID manifest mismatch')
    known_ids = set(source_ids)
    ranking_file = run['ranking_file']
    if _path(root, ranking_file) == gold_path or 'scorer-only' in _path(root, ranking_file).parts:
        raise ValueError('Ranking file must be isolated from gold')
    ranking_raw = _verified(root, ranking_file, run['ranking_sha256'], checked)
    rankings = [_json(line) for line in ranking_raw.splitlines() if line.strip()]
    if len(rankings) != expected_questions:
        raise ValueError('Ranking question count mismatch')
    seen_questions = set()
    for row in rankings:
        identity = _identity(row['id'])
        if identity in seen_questions or identity not in public or row['qtype'] != public[identity]:
            raise ValueError('Duplicate/unknown question or qtype mismatch in rankings')
        seen_questions.add(identity)
        if not isinstance(row.get('methods'), dict) or set(row['methods']) != set(METHODS):
            raise ValueError('Both frozen methods required')
        for method in METHODS:
            result = row['methods'][method]
            ids, scores = result['ranked_ids'], result['scores']
            if not isinstance(ids, list) or not isinstance(scores, list) or len(ids) > 100 or len(ids) != len(scores):
                raise ValueError('Invalid ranked source/scores sequence')
            for source_id in ids:
                _identity(source_id)
            if len(ids) != len(set(ids)) or not set(ids) <= known_ids:
                raise ValueError('Duplicate or unknown native source ID in rankings')
            if (any(not _number(score) for score in scores)
                    or not _number(result['elapsed_ms']) or result['elapsed_ms'] < 0):
                raise ValueError('Finite scores and nonnegative latency required')
    if seen_questions != set(public):
        raise ValueError('Ranking/public question sets differ')
    return dict(root=root, run=run, frozen=frozen, checked=checked,
                public=public, public_text=public_text, source_ids=known_ids, rankings=rankings)


def _aggregate(rows, method):
    return {str(k): {metric: sum(float(row['methods'][method][str(k)][metric]) for row in rows) / len(rows)
                    for metric in METRICS} for k in KS}


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _paired(rows, samples, seed):
    if type(samples) is not int or samples < 1:
        raise ValueError('Positive bootstrap sample count required')
    generator = random.Random(seed)
    indices = [[generator.randrange(len(rows)) for _ in rows] for _ in range(samples)]
    result = {}
    for k in KS:
        result[str(k)] = {}
        for metric in METRICS:
            delta = [float(row['methods']['v4'][str(k)][metric]) - float(row['methods']['flat'][str(k)][metric]) for row in rows]
            means = sorted(sum(delta[i] for i in sample) / len(rows) for sample in indices)
            result[str(k)][metric] = dict(
                mean_delta=sum(delta) / len(rows),
                wins=sum(value > 1e-12 for value in delta),
                ties=sum(abs(value) <= 1e-12 for value in delta),
                losses=sum(value < -1e-12 for value in delta),
                paired_query_bootstrap_95_ci=[_percentile(means, .025), _percentile(means, .975)])
    return result


def score_verified(verified, *, bootstrap_samples=5000, bootstrap_seed=20261025):
    """Open pinned gold only after verify_before_gold succeeds; return aggregates/details."""
    root, frozen = verified['root'], verified['frozen']
    gold_raw = _verified(root, frozen['gold_file'], frozen['gold_sha256'], verified['checked'])
    gold_rows = _json(gold_raw)
    if not isinstance(gold_rows, list) or len(gold_rows) != len(verified['public']):
        raise ValueError('Gold question count mismatch')
    gold = {}
    for row in gold_rows:
        identity = _identity(row['id'])
        labels = row.get('evidence_ids')
        if (identity in gold or identity not in verified['public']
                or row.get('qtype') != verified['public'][identity]):
            raise ValueError('Gold question identity/qtype mismatch')
        if row.get('question') != verified['public_text'][identity]:
            raise ValueError('Public question text differs from pinned native gold question')
        if not isinstance(labels, list) or not labels:
            raise ValueError('Nonempty native gold labels required for Hard evaluation')
        for label in labels:
            _identity(label)
        if len(labels) != len(set(labels)) or not set(labels) <= verified['source_ids']:
            raise ValueError('Duplicate or unknown gold source ID')
        gold[identity] = labels
    if set(gold) != set(verified['public']):
        raise ValueError('Gold/public question sets differ')
    rows = []
    for ranking in verified['rankings']:
        identity = ranking['id']
        rows.append(dict(id=identity, qtype=ranking['qtype'], gold_count=len(gold[identity]),
            methods={method: {str(k): retrieval_metrics(ranking['methods'][method]['ranked_ids'], gold[identity], k)
                              for k in KS} for method in METHODS}))
    paired = _paired(rows, bootstrap_samples, bootstrap_seed)
    # Detect concurrent changes too; reports are written only after this check.
    for relative, digest in verified['checked'].items():
        _verified(root, relative, digest, {})
    groups = {qtype: [row for row in rows if row['qtype'] == qtype] for qtype in sorted({row['qtype'] for row in rows})}
    report = dict(
        state='complete', scored_at_utc=datetime.now(timezone.utc).isoformat(),
        condition='external_Qwen_SGM_three_view_sparse_RRF_not_strict_from_raw',
        comparison='v4 minus frozen flat BM25; same complete source corpus',
        questions=len(rows), source_count=len(verified['source_ids']),
        qtypes=dict(Counter(row['qtype'] for row in rows)),
        methods={method: _aggregate(rows, method) for method in METHODS}, paired=paired,
        by_qtype={qtype: dict(questions=len(group), methods={method: _aggregate(group, method) for method in METHODS})
                  for qtype, group in groups.items()},
        metric_formula=dict(fractional_recall='macro mean |unique(topK) intersect G| / |G|',
            any_hit='macro mean indicator(nonempty intersection)',
            all_evidence_coverage='macro mean indicator(G subset unique(topK))',
            unit='native artifact ID; no card/view/frame inflation'),
        bootstrap=dict(unit='paired query', samples=bootstrap_samples, seed=bootstrap_seed,
            method='percentile', confidence=.95,
            limitations='31 queries from one shared corpus; descriptive small-sample uncertainty, not cross-corpus generalization'),
        exposure='Hard QA previously machine-inspected for schema and overlap; this is a same-corpus new-question performance comparison, not never-read blind data',
        verified_input_sha256=dict(verified['checked']),
        source_ids_sha256=verified['run']['source_ids_sha256'],
        gold_only_opened_after_rankings_verified=True, inputs_unchanged_after_scoring=True,
        unresolved_gold_ids_count=0, duplicate_gold_labels=0,
        provider_requests=0, reader_answer_accuracy=None, raw_media_or_audio_processed=False,
        official_smoke=0, official_full=0)
    return report, rows


def score_project(root=ROOT, *, run_file=RUN_FILE, expected_questions=31,
                  expected_sources=11034, expected_gold_sha256=GOLD_SHA256,
                  bootstrap_samples=5000, bootstrap_seed=20261025):
    """Validate, score, and write exclusively; refuses to replace either output."""
    root = Path(root).resolve()
    report_path, detail_path = _path(root, SCORE_FILE), _path(root, DETAIL_FILE)
    if report_path.exists() or detail_path.exists():
        raise FileExistsError('V4 scoring already recorded; preserve frozen outputs')
    verified = verify_before_gold(root, run_file, expected_questions=expected_questions,
        expected_sources=expected_sources, expected_gold_sha256=expected_gold_sha256)
    report, rows = score_verified(verified, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open('x', encoding='utf-8') as target:
        json.dump(rows, target, ensure_ascii=False, indent=2, allow_nan=False)
    with report_path.open('x', encoding='utf-8') as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    report = score_project()
    print(json.dumps({key: report[key] for key in ('state', 'questions', 'source_count', 'methods', 'paired', 'provider_requests')}, indent=2))


if __name__ == '__main__':
    main()
