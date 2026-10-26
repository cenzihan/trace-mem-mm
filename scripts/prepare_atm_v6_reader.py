"""Two offline phases: freeze stable IDs, then pack frozen V6 reader inputs.

No gold, credentials, provider construction, or network. Never run a tokenizer
download fallback. Existing artifacts are never replaced. The selected public
development questions are not held-out data. Proxy judging is a later process.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.atm_reader import (
    ATM_READER_PROMPT, ATMReaderBudget, VARIANTS, pack_atm_evidence, query_id, select_pilot,
)
from scripts.prepare_atm_v5_queries import digest, load_json, path_at, save_new, sha, verified_bytes

V6_FROZEN = 'runtime/atm-v6-rerank/FROZEN_CONFIG.json'
V6_PREP = 'reports/atm-v6-rerank-preparation.json'
V6_RUN = 'reports/atm-v6-rerank-run.json'
QUERIES = 'runtime/atm-v6-rerank/queries.jsonl'
SOURCES = 'runtime/atm-v6-rerank/sources.jsonl'
RANKINGS = 'runtime/atm-v6-rerank/retrieval-results.jsonl'
WORK = 'runtime/atm-v6-reader'
SELECTION = WORK + '/FROZEN_SELECTION.json'
PACKS = WORK + '/reader-packs.jsonl'
FROZEN = WORK + '/FROZEN_READER.json'
CODE = ('agent_memory/atm_reader.py', 'scripts/prepare_atm_v6_reader.py')


def _lines(raw):
    return [load_json(line) for line in raw.splitlines() if line.strip()]


def _check(root, hashes):
    for name, expected in hashes.items():
        if 'scorer-only' in path_at(root, name).parts:
            raise ValueError('gold file is forbidden in reader preparation')
        verified_bytes(root, name, expected)


def _metadata(rows):
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'query_id', 'dataset', 'id', 'qtype', 'question'}:
            raise ValueError('exact V6 question-only projection required')
        # Do not inspect the question value to select questions.
        result.append({key: row[key] for key in ('dataset', 'id', 'qtype', 'query_id')})
    return result


def prepare_selection(root=ROOT):
    root = Path(root).resolve()
    if path_at(root, SELECTION).exists() or path_at(root, FROZEN).exists() or path_at(root, PACKS).exists():
        raise FileExistsError('reader preparation already exists; preserve it')
    prep_raw = path_at(root, V6_PREP).read_bytes()
    prep = load_json(prep_raw)
    if prep.get('state') != 'prepared_gold_not_loaded' or prep.get('frozen_file') != V6_FROZEN:
        raise ValueError('completed V6 source preparation required')
    frozen_raw = verified_bytes(root, V6_FROZEN, prep['frozen_sha256'])
    frozen = load_json(frozen_raw)
    if frozen.get('artifact_hashes') != prep.get('artifact_hashes') or frozen.get('methods') != list(VARIANTS):
        raise ValueError('V6 preparation binding mismatch')
    queries_raw = verified_bytes(root, QUERIES, frozen['artifact_hashes'][QUERIES])
    metadata = _metadata(_lines(queries_raw))
    if len(metadata) != frozen['query_count']:
        raise ValueError('full V6 query map required before selection')
    selection = select_pilot(metadata)
    inputs = {V6_PREP: digest(prep_raw), V6_FROZEN: digest(frozen_raw), QUERIES: digest(queries_raw)}
    codes = {name: sha(root, name) for name in CODE}
    _check(root, {**inputs, **codes})
    result = dict(state='selection_frozen_no_reader_calls', selection=selection,
        input_hashes=inputs, code_hashes=codes, reader_budget=asdict(ATMReaderBudget()),
        reader_prompt_sha256=digest(ATM_READER_PROMPT.encode('utf-8')),
        source_file=SOURCES, source_sha256=frozen['artifact_hashes'][SOURCES],
        source_count=frozen['source_count'], query_count=frozen['query_count'],
        selection_uses_question_text=False, gold_loaded=False, provider_requests=0,
        proxy_judge_requests_included=False)
    save_new(root, SELECTION, result)
    return dict(state=result['state'], selection_file=SELECTION, selection_sha256=sha(root, SELECTION),
        selected_ids=selection['selected'], planned_reader_answers=selection['planned_reader_answers'],
        provider_requests=0, gold_loaded=False, proxy_judge_requests_included=False)


def _local_tokenizer(root):
    report_file = 'reports/reader-tokenizer.json'
    raw = path_at(root, report_file).read_bytes()
    report = load_json(raw)
    if report.get('encoding') != 'o200k_base' or report.get('remote_fallback_disabled') is not True:
        raise ValueError('reviewed local tokenizer required')
    cache_name = 'runtime/tokenizer-cache/' + hashlib.sha1(report['source'].encode()).hexdigest()
    verified_bytes(root, cache_name, report['cache_sha256'])
    sys.path.insert(0, str(path_at(root, 'runtime/reader-deps')))
    os.environ['TIKTOKEN_CACHE_DIR'] = str(path_at(root, 'runtime/tokenizer-cache'))
    import tiktoken
    import tiktoken.load
    def no_network(_):
        raise RuntimeError('tokenizer network fallback forbidden')
    tiktoken.load.read_file = no_network
    if tiktoken.__version__ != report['version']:
        raise ValueError('tokenizer version changed')
    encoding = tiktoken.get_encoding('o200k_base')
    return lambda value: len(encoding.encode(value, disallowed_special=())), {
        report_file: digest(raw), cache_name: report['cache_sha256']}


def prepare_packs(root=ROOT, *, count_tokens=None, offline_test=False):
    root = Path(root).resolve()
    if path_at(root, PACKS).exists() or path_at(root, FROZEN).exists():
        raise FileExistsError('reader packs already exist; preserve them')
    selection_raw = path_at(root, SELECTION).read_bytes()
    frozen = load_json(selection_raw)
    _check(root, {**frozen['input_hashes'], **frozen['code_hashes']})
    if frozen['reader_prompt_sha256'] != digest(ATM_READER_PROMPT.encode('utf-8')):
        raise ValueError('reader prompt changed after selection')
    rows = _lines(verified_bytes(root, QUERIES, frozen['input_hashes'][QUERIES]))
    if select_pilot(_metadata(rows)) != frozen['selection']:
        raise ValueError('selected IDs changed after freezing')
    questions = {row['query_id']: row for row in rows}
    run_raw = path_at(root, V6_RUN).read_bytes()
    run = load_json(run_raw)
    if (run.get('state') != 'retrieval_complete_gold_not_loaded' or run.get('ranking_file') != RANKINGS
            or run.get('frozen_file') != V6_FROZEN or run.get('frozen_sha256') != frozen['input_hashes'][V6_FROZEN]
            or run.get('preparation_sha256') != frozen['input_hashes'][V6_PREP]):
        raise ValueError('complete V6 ranking binding required')
    raw_rankings = verified_bytes(root, RANKINGS, run['ranking_sha256'])
    rankings = {}
    for row in _lines(raw_rankings):
        if not isinstance(row, dict) or set(row) != {'dataset', 'id', 'qtype', 'methods'}:
            raise ValueError('exact four-variant V6 ranking required')
        key = query_id(row['dataset'], row['id'])
        if (key in rankings or key not in questions or row['qtype'] != questions[key]['qtype']
                or set(row['methods']) != set(VARIANTS)):
            raise ValueError('V6 ranking identity mismatch')
        rankings[key] = row
    if set(rankings) != set(questions) or len(questions) != frozen['query_count']:
        raise ValueError('complete original query coverage required')
    source_raw = verified_bytes(root, SOURCES, frozen['source_sha256'])
    sources = {}
    for row in _lines(source_raw):
        if (not isinstance(row, dict) or set(row) != {'source_id', 'text'} or not isinstance(row['source_id'], str)
                or not row['source_id'] or row['source_id'] in sources
                or not isinstance(row['text'], str) or not row['text'].strip()):
            raise ValueError('complete unique source-only text records required')
        sources[row['source_id']] = row['text']
    if len(sources) != frozen['source_count']:
        raise ValueError('complete V6 source corpus required')
    tokenizer_hashes = {}
    if count_tokens is None:
        count_tokens, tokenizer_hashes = _local_tokenizer(root)
    elif (not offline_test or root == ROOT or not root.is_relative_to(Path(tempfile.gettempdir()).resolve())):
        raise ValueError('injected tokenizer is restricted to temporary synthetic tests')
    packs = []
    budget = ATMReaderBudget(**frozen['reader_budget'])
    for item in frozen['selection']['selected']:
        row = questions[item['query_id']]
        question = {key: row[key] for key in ('id', 'question', 'qtype')}
        for variant in VARIANTS:
            ranking = rankings[item['query_id']]['methods'][variant]
            ids, scores = ranking.get('ranked_ids'), ranking.get('scores')
            if (set(ranking) != {'ranked_ids', 'scores'} or not isinstance(ids, list)
                    or not isinstance(scores, list) or len(ids) != len(scores)
                    or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
                raise ValueError('invalid frozen candidate ordering')
            pack = pack_atm_evidence(question, ids, sources, count_tokens, budget)
            packs.append(dict(query_id=item['query_id'], dataset=item['dataset'], id=item['id'],
                qtype=item['qtype'], variant=variant, pack=pack))
    inputs = {**frozen['input_hashes'], SELECTION: digest(selection_raw), SOURCES: digest(source_raw),
              V6_RUN: digest(run_raw), RANKINGS: digest(raw_rankings), **tokenizer_hashes}
    _check(root, {**inputs, **frozen['code_hashes']})
    path_at(root, PACKS).parent.mkdir(parents=True, exist_ok=True)
    with path_at(root, PACKS).open('x', encoding='utf-8') as handle:
        for row in packs:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    result = dict(state='reader_inputs_frozen_no_calls', input_hashes=inputs,
        code_hashes=frozen['code_hashes'], packs_file=PACKS, packs_sha256=sha(root, PACKS),
        selection_sha256=digest(selection_raw), reader_budget=asdict(budget),
        reader_prompt_sha256=frozen['reader_prompt_sha256'], variants=list(VARIANTS),
        questions=len(frozen['selection']['selected']), planned_reader_answers=len(packs),
        exposure=frozen['selection']['exposure'], source_condition='author external Qwen SGM text; no raw media',
        gold_loaded=False, provider_requests=0, proxy_judge_requests_included=False,
        tokenizer='injected_offline_test' if offline_test else 'local_o200k_base',
        packing_summary={variant: dict(
            packs=sum(row['variant'] == variant for row in packs),
            empty_contexts=sum(not row['pack']['source_memory_ids'] for row in packs if row['variant'] == variant),
            dropped_records=sum(row['pack']['dropped_records'] for row in packs if row['variant'] == variant),
            max_text_tokens=max(row['pack']['text_token_count'] for row in packs if row['variant'] == variant),
            max_utf8_bytes=max(row['pack']['utf8_text_bytes'] for row in packs if row['variant'] == variant))
            for variant in VARIANTS})
    save_new(root, FROZEN, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('select', 'pack'))
    args = parser.parse_args()
    print(json.dumps(prepare_selection() if args.phase == 'select' else prepare_packs(),
                     ensure_ascii=False, indent=2))
