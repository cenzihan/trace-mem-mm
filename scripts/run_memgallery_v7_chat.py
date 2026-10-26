"""Frozen native Mem-Gallery reader plus separately prepared blind proxy judge.

The reader preparation/execution cannot load gold. Exactly one durable receipt
per request; invalid substantive answers are retained, never regenerated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.encode_memgallery_v7 import (BASE, now, sha, load, save, lines, write_lines,
                                         history, singleton, verify as verify_encoding)
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.development_run import DevelopmentRun, ScopedTransport
from agent_memory.durable_chat import DurableChat
from agent_memory.memgallery_v7 import VARIANTS
from agent_memory.memgallery_v7_reader import (pack_memgallery_evidence, reader_payload,
                                             decode_reader, proxy_judge_payload)
from agent_memory.provider_http import BudgetedJSONTransport, NativeHTTPS, OPENAI_ENDPOINT, SHARED_LEDGER
from agent_memory.windows_credentials import read_key

WORK = BASE / 'chat'
PROGRESS = ROOT / 'reports/memgallery-v7-chat-progress.json'
PRICE = 'https://developers.openai.com/api/docs/models/gpt-4o-mini'
CHECKED = '2026-10-26T04:21:00+00:00'
CODE = ('scripts/run_memgallery_v7_chat.py', 'scripts/encode_memgallery_v7.py',
        'agent_memory/memgallery_v7_reader.py', 'agent_memory/durable_chat.py',
        'agent_memory/embedding_executor.py', 'agent_memory/provider_http.py',
        'agent_memory/development_run.py', 'agent_memory/budget.py', 'agent_memory/visual_recovery.py')


def tokenizer():
    report = load(ROOT / 'reports/reader-tokenizer.json')
    deps = ROOT / 'runtime/reader-deps'; cache = ROOT / 'runtime/tokenizer-cache'
    sys.path.insert(0, str(deps)); os.environ['TIKTOKEN_CACHE_DIR'] = str(cache)
    import tiktoken, tiktoken.load
    def no_network(_): raise RuntimeError('tokenizer remote fallback forbidden')
    tiktoken.load.read_file = no_network
    encoder = tiktoken.get_encoding('o200k_base')
    if tiktoken.__version__ != report['version']: raise ValueError('frozen tokenizer version mismatch')
    return lambda text: len(encoder.encode(text, disallowed_special=()))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha(ROOT / name) != expected: raise ValueError('frozen input/code hash changed')


def prepare():
    if (WORK / 'FROZEN_READER.json').exists(): raise FileExistsError('reader already frozen')
    verify_encoding()
    retrieval = BASE / 'rerank/retrieval-results.jsonl'
    # The collector publishes a complete checked immutable ranking manifest.
    complete = load(BASE / 'rerank/COMPLETED.json')
    if complete.get('state') != 'complete' or complete.get('retrieval_results_sha256') != sha(retrieval):
        raise ValueError('complete frozen GPU/ranking results required')
    from scripts.control_memgallery_v7_remote import frozen as verify_rerank
    verify_rerank()
    rerank_report = load(ROOT / 'reports/memgallery-v7-rerank-run.json')
    if (rerank_report['frozen_sha256'] != sha(BASE / 'rerank/FROZEN_CONFIG.json')
            or rerank_report['ranking_sha256'] != sha(retrieval)):
        raise ValueError('ranking completion must bind exact frozen experiment')
    rankings = lines(retrieval)
    docs = lines(BASE / 'encoding/source-documents.jsonl')
    queries = {q['query_id']: q['query'] for q in lines(BASE / 'encoding/query-documents.jsonl')}
    if len(rankings) != len(queries) or {r['query_id'] for r in rankings} != set(queries):
        raise ValueError('full unique retrieval question set required')
    by_query = {r['query_id']: r for r in rankings}
    chosen = load(BASE / 'READER_SELECTION.json')['selected_indices']
    if len(chosen) != 16 or len(set(chosen)) != 16: raise ValueError('fixed16 pilot required')
    sources = {d['id']: {key: d[key] for key in ('text','source_session','source_round','source_date','image_ids')} for d in docs}
    counter = tokenizer(); packs = []
    for index in chosen:
        identity = 'memgallery-v7:' + str(index); query = queries[identity]
        row = by_query[identity]
        if set(row['methods']) != set(VARIANTS) or row['index'] != index or row['point'] != query['point']:
            raise ValueError('native query/ranking identity mismatch')
        for variant in VARIANTS:
            pack = pack_memgallery_evidence(query, row['methods'][variant]['ranked_ids'], sources, counter)
            reader_payload(pack)
            packs.append(dict(query_id=identity, index=index, point=query['point'], variant=variant, pack=pack))
    WORK.mkdir(parents=True, exist_ok=True)
    write_lines(WORK / 'reader-packs.jsonl', packs)
    files = [BASE / 'PREPARATION.json', BASE / 'FROZEN_SELECTION_RULE.json', BASE / 'READER_SELECTION.json',
             BASE / 'encoding/FROZEN_ENCODING.json', BASE / 'encoding/source-documents.jsonl',
             BASE / 'encoding/query-documents.jsonl', BASE / 'rerank/FROZEN_CONFIG.json',
             BASE / 'rerank/COMPLETED.json', ROOT / 'reports/memgallery-v7-rerank-run.json',
             ROOT / 'reports/memgallery-v7-encoding-audit.json',
             retrieval, WORK / 'reader-packs.jsonl', ROOT / 'reports/reader-tokenizer.json']
    files.extend(p for p in (ROOT / 'runtime/tokenizer-cache').iterdir() if p.is_file())
    frozen = dict(state='reader_inputs_frozen_no_calls', created_at_utc=now(), planned_reader_answers=64,
        max_judge_batches=8, max_requests=72, max_preflight_cny='18', fixed_hold_cny='.25',
        input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        code_hashes={name: sha(ROOT / name) for name in CODE},
        packs_file=(WORK / 'reader-packs.jsonl').relative_to(ROOT).as_posix(),
        packs_sha256=sha(WORK / 'reader-packs.jsonl'), source_condition='original dialogue + author source/query image captions; zero raw image inference',
        score='gpt-4o-mini five-grade blind proxy, not official score',
        query_retrieval_representation='original question followed by QUERY_IMAGE_CAPTIONS header and provided captions',
        query_reader_representation='same original question/captions with numbered QUERY IMAGE n CAPTION headers',
        gold_loaded=False, automatic_retries=False, reader_output_tokens=768, judge_output_tokens=1536,
        max_input_tokens=8192, max_utf8_bytes=24000, image_count=0, tokenizer='tiktoken0.12.0/o200k_base',
        price_source=PRICE, price_checked_at=CHECKED, usd_per_million_input='.15', usd_per_million_output='.60',
        conversion='10 CNY/USD conservative budget assumption, not market FX')
    save(WORK / 'FROZEN_READER.json', frozen, exclusive=True)
    return dict(state=frozen['state'], reader_requests=64, max_judge_requests=8,
                min_packed_sources=min(len(p['pack']['source_memory_ids']) for p in packs),
                max_packed_sources=max(len(p['pack']['source_memory_ids']) for p in packs), gold_loaded=False)


def verify_reader():
    frozen = load(WORK / 'FROZEN_READER.json')
    verify_hashes({**frozen['input_hashes'], **frozen['code_hashes']})
    if frozen['planned_reader_answers'] != 64: raise ValueError('fixed reader size required')
    return frozen, lines(ROOT / frozen['packs_file'])


def saved_row(request, result, phase):
    saved = {key: result[key] for key in ('attempt_id','receipt_sha256','run_ordinal')}
    saved['request_id'] = request['request_id']
    if phase == 'reader':
        row = request['metadata']
        saved.update({key: row[key] for key in ('query_id','index','point','variant')})
        try: answer = decode_reader(result['response'], row['pack'])
        except (ValueError, KeyError, TypeError):
            answer = dict(answer='', cited_ids=[], confidence='invalid', schema_status='invalid',
                          citations_valid=False, answer_ids_valid=False, model=result['response']['model'])
        saved['answer'] = answer
    else: saved['response'] = result['response']
    return saved


def requests_for(phase):
    _, packs = verify_reader()
    if phase == 'reader':
        return [dict(request_id=hashlib.sha256((p['query_id'] + '\0' + p['variant']).encode()).hexdigest(),
                     payload=reader_payload(p['pack']), metadata=p) for p in packs]
    frozen = load(WORK / 'FROZEN_JUDGE.json')
    verify_hashes({**frozen['input_hashes'], **frozen['code_hashes']})
    path = ROOT / frozen['requests_file']
    if sha(path) != frozen['requests_sha256']: raise ValueError('judge requests changed')
    requests = lines(path)
    if not 1 <= len(requests) <= 8: raise ValueError('judge request cap exceeded')
    return [dict(request_id=r['request_id'], payload=proxy_judge_payload(r['pack'])) for r in requests]


def execute(phase, max_requests=None):
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise ValueError('positive invocation request limit required')
    with singleton(WORK):
        frozen, _ = verify_reader(); ledger = BudgetLedger(SHARED_LEDGER)
        live = WORK / 'FROZEN_LIVE.json'
        if not live.exists():
            if (WORK / 'run.sqlite').exists(): raise ValueError('run without live freeze')
            baseline = ledger.totals()
            save(live, dict(reader_sha256=sha(WORK / 'FROZEN_READER.json'), baseline=baseline,
                historical_sha256=history(baseline['attempts']), created_at_utc=now(),
                authorization='User autonomous V7 experiment continuation;72calls/CNY18 within original500.'), exclusive=True)
        started = load(live)
        if started['reader_sha256'] != sha(WORK / 'FROZEN_READER.json') or history(started['baseline']['attempts']) != started['historical_sha256']:
            raise ValueError('historical ledger or frozen reader changed')
        runpath = WORK / 'run.sqlite'
        run = DevelopmentRun(runpath) if runpath.exists() else DevelopmentRun(runpath, create=True,
              max_requests=72, max_reservation_cny='18', approval_scope='native_100_20')
        rate = VerifiedRate('gpt-4o-mini', '1.5', '6', PRICE, CHECKED, conversion_basis='budget_assumption_not_market_fx')
        def factory(http, observer):
            base = BudgetedJSONTransport(model='gpt-4o-mini', endpoint=OPENAI_ENDPOINT,
                ledger=ledger, rate=rate, credential_reader=read_key, http=http, mode='live', enabled=True,
                prices_confirmed=True, fixed_reserve_cny='.25', usd_rates=('.15','.60'), output_token_cap=1536,
                timeout_seconds=120, max_response_bytes=1024*1024, attempt_observer=observer)
            return ScopedTransport(base, run, '.25')
        executor = DurableChat(WORK / 'receipts', 'memgallery-v7:' + sha(live), NativeHTTPS(), factory, run)
        requests = requests_for(phase)
        output = WORK / ('reader-answers.jsonl' if phase == 'reader' else 'judge-results.jsonl')
        existing = lines(output) if output.exists() else []
        if [r['request_id'] for r in existing] != [r['request_id'] for r in requests[:len(existing)]]:
            raise ValueError('saved outputs must be frozen request prefix')
        for request, row in zip(requests, existing):
            if saved_row(request, executor.recover(request['request_id'], request['payload']), phase) != row:
                raise ValueError('saved output differs from durable receipt')
        if run.summary()['state'] != 'active' and len(existing) != len(requests):
            raise ValueError('closed guard cannot dispatch')
        dispatched = 0; start = time.monotonic()
        def progress(state, error=None):
            value = dict(state=state, phase=phase, completed=len(existing), expected=len(requests),
                updated_at_utc=now(), new_this_invocation=dispatched, run=run.summary(), budget=ledger.totals())
            if phase == 'reader': value['invalid_answers'] = sum(r['answer']['schema_status'] != 'valid' for r in existing)
            if error: value['error_type'] = error
            save(PROGRESS, value); print(json.dumps(value), flush=True)
        try:
            for request in requests[len(existing):]:
                if time.monotonic() - start > 7200: raise ValueError('invocation deadline reached')
                result = executor.execute(request['request_id'], request['payload'])
                row = saved_row(request, result, phase)
                with output.open('a', encoding='utf-8') as out:
                    out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                    out.flush(); os.fsync(out.fileno())
                existing.append(row); dispatched += not result['recovered']; progress('running')
                if max_requests and dispatched >= max_requests and len(existing) < len(requests):
                    progress('bounded_invocation_finished_guard_active'); return
            if history(started['baseline']['attempts']) != started['historical_sha256']: raise ValueError('historical accounting changed')
            if phase == 'judge': run.close()
            progress(phase + '_complete')
            destination = WORK / (phase.upper() + '_COMPLETED.json')
            if not destination.exists():
                save(destination, dict(state='complete', phase=phase, count=len(existing),
                     output_sha256=sha(output), frozen_live_sha256=sha(live), run=run.summary()), exclusive=True)
        except BaseException as exc:
            category = getattr(exc, 'category', type(exc).__name__)
            progress('stopped_no_automatic_retry', category)
            raise SystemExit('Chat stopped: ' + category + '; inspect saved receipt, never replay unknown.') from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare','reader','judge'))
    parser.add_argument('--live', action='store_true'); parser.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    if args.phase == 'prepare': print(json.dumps(prepare()))
    elif not args.live: raise SystemExit('Explicit --live required')
    else: execute(args.phase, args.max_requests)
