"""Eight new JSON-prompt-compatible V7.1 judge batches; no automatic retry.

The old 65-call guard and rejected HTTP response remain closed and preserved.
Only the separately frozen eight request payloads may use this independent
8-request/CNY2 guard. Preflight never reads credentials or dispatches HTTP.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.encode_memgallery_v7 import now, sha, load, save, lines, singleton
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.development_run import DevelopmentRun, ScopedTransport
from agent_memory.durable_chat import DurableChat
from agent_memory.memgallery_v7_transport import MemGalleryV7TemperatureZeroTransport
from agent_memory.provider_http import NativeHTTPS, OPENAI_ENDPOINT, SHARED_LEDGER
from agent_memory.windows_credentials import read_key

WORK = ROOT / 'runtime/memgallery-v7/judge-json-v71'
PROGRESS = ROOT / 'reports/memgallery-v71-judge-json-progress.json'
OLD = 'runtime/memgallery-v7/chat-citation-v71'
AUDIT = 'reports/memgallery-v71-judge-json-rejection.json'
BASELINE = 1612
PRICE = 'https://developers.openai.com/api/docs/models/gpt-4o-mini'
CHECKED = '2026-10-26T04:21:00+00:00'
CODE = ('scripts/run_memgallery_v71_judge_json.py', 'scripts/encode_memgallery_v7.py',
        'agent_memory/memgallery_v7_transport.py', 'agent_memory/durable_chat.py',
        'agent_memory/embedding_executor.py', 'agent_memory/provider_http.py',
        'agent_memory/development_run.py', 'agent_memory/budget.py',
        'agent_memory/visual_recovery.py', 'agent_memory/windows_credentials.py')


def local_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('project-relative frozen path required')
    root = Path(root).resolve(); path = (root / name).resolve()
    if not path.is_relative_to(root): raise ValueError('frozen path escapes project')
    return path


def verify_hashes(root, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('nonempty frozen hashes required')
    for name, expected in hashes.items():
        if (not isinstance(expected, str) or len(expected) != 64
                or sha(local_path(root, name)) != expected):
            raise ValueError('frozen input/code hash changed')


def validate_payload(payload):
    """Full local admission before a guard claim, journal intent or credential."""
    if not isinstance(payload, dict): raise ValueError('frozen judge payload required')
    json.dumps(payload, allow_nan=False)
    if (payload.get('response_format') != {'type': 'json_object'}
            or type(payload.get('max_completion_tokens')) is not int
            or payload['max_completion_tokens'] != 1536):
        raise ValueError('fixed JSON judge format and output cap required')
    # This invokes the exact live transport admission without constructing a
    # ledger, credential reader, HTTP object or live transport.
    admission = object.__new__(MemGalleryV7TemperatureZeroTransport)
    admission.model = 'gpt-4o-mini'; admission.output_token_cap = 1536
    admission._bounds(payload)
    messages = payload['messages']
    if any(set(message) != {'role', 'content'} for message in messages):
        raise ValueError('fixed text message schema required')
    texts = [messages[0]['content']]
    content = messages[1]['content']
    if isinstance(content, str): texts.append(content)
    elif isinstance(content, list) and content:
        for part in content:
            if (not isinstance(part, dict) or set(part) != {'type', 'text'}
                    or part['type'] != 'text' or not isinstance(part['text'], str)):
                raise ValueError('judge permits text only, no media')
            texts.append(part['text'])
    else: raise ValueError('text judge content required')
    if (not messages[0]['content'].endswith('\nReturn JSON.')
            or not any('json' in text.lower() for text in texts)):
        raise ValueError('frozen JSON instruction required before provider dispatch')
    return payload


def historical_sha(ledger_path, limit):
    with closing(sqlite3.connect(Path(ledger_path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        rows = db.execute('SELECT * FROM attempts WHERE rowid<=? ORDER BY rowid', (limit,)).fetchall()
        if len(rows) != limit: raise ValueError('historical ledger prefix incomplete')
        observations = db.execute('SELECT o.* FROM provider_observations o JOIN attempts a '
            'ON a.id=o.attempt_id WHERE a.rowid<=? ORDER BY a.rowid', (limit,)).fetchall()
    return hashlib.sha256(json.dumps([rows, observations], separators=(',', ':')).encode()).hexdigest()


def verify(root=ROOT, work=None):
    root = Path(root); work = Path(work) if work is not None else root / WORK.relative_to(ROOT)
    frozen = load(work / 'FROZEN_JUDGE.json')
    inputs = frozen.get('input_hashes'); codes = frozen.get('code_hashes')
    verify_hashes(root, inputs); verify_hashes(root, codes)
    preserved = {OLD + '/run.sqlite', OLD + '/receipts/private-embedding.sqlite',
                 OLD + '/FROZEN_JUDGE.json', OLD + '/READER_COMPLETED.json', AUDIT}
    if not preserved <= set(inputs): raise ValueError('old closed attempt and reader audit must be frozen')
    with closing(sqlite3.connect(local_path(root, OLD + '/run.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        state = db.execute('SELECT state FROM run').fetchone()
        counts = dict(db.execute('SELECT state,COUNT(*) FROM calls GROUP BY state'))
        if state != ('complete',) or counts != {'complete': 64, 'rejected_response': 1}:
            raise ValueError('old 65-call guard must be closed with its rejected response preserved')
    path = local_path(root, frozen.get('requests_file'))
    if (path != (work / 'judge-requests.jsonl').resolve()
            or sha(path) != frozen.get('requests_sha256')):
        raise ValueError('fixed judge request file changed')
    requests = lines(path)
    if (len(requests) != 8 or any(not isinstance(r, dict) or set(r) != {'request_id', 'payload'} for r in requests)
            or any(not isinstance(r['request_id'], str) or not r['request_id'] for r in requests)
            or len({r['request_id'] for r in requests}) != 8):
        raise ValueError('exactly eight unique frozen requests required')
    for request in requests: validate_payload(request['payload'])
    return frozen, requests


def make_executor(work, live_sha, ledger, run, *, http=None, offline_test=False,
                  credential_reader=read_key, after_received=None):
    rate = VerifiedRate('gpt-4o-mini', '1.5', '6', PRICE, CHECKED,
                        conversion_basis='budget_assumption_not_market_fx')
    def factory(wrapped_http, observer):
        base = MemGalleryV7TemperatureZeroTransport(model='gpt-4o-mini', endpoint=OPENAI_ENDPOINT,
            ledger=ledger, rate=rate, credential_reader=credential_reader, http=wrapped_http,
            mode='offline_fake' if offline_test else 'live', enabled=not offline_test,
            prices_confirmed=not offline_test, fixed_reserve_cny='.25', usd_rates=('.15', '.60'),
            output_token_cap=1536, timeout_seconds=120, max_response_bytes=1024*1024,
            attempt_observer=observer)
        return ScopedTransport(base, run, '.25')
    return DurableChat(Path(work) / 'receipts', 'memgallery-v71-judge-json:' + live_sha,
        http if http is not None else NativeHTTPS(), factory, run,
        offline_test=offline_test, after_received=after_received)


def saved_row(request, result):
    return dict(request_id=request['request_id'], response=result['response'],
        attempt_id=result['attempt_id'], receipt_sha256=result['receipt_sha256'],
        run_ordinal=result['run_ordinal'])


def verify_run(run):
    summary = run.summary()
    if (summary['max_requests'] != 8 or summary['max_reservation_micro_cny'] != 2000000
            or summary['cumulative_preflight_micro_cny'] != 250000 * summary['request_claims']
            or set(summary['calls_by_model']) - {'gpt-4o-mini'}
            or summary['state'] not in ('active', 'complete')):
        raise ValueError('independent eight-request CNY2 guard changed')
    return summary


def execute(max_requests=None, *, root=ROOT, work=None, progress_file=None):
    if max_requests is not None and (type(max_requests) is not int or not 1 <= max_requests <= 8):
        raise ValueError('invocation limit must be 1..8')
    root = Path(root); work = Path(work) if work is not None else root / WORK.relative_to(ROOT)
    progress_file = Path(progress_file) if progress_file is not None else root / PROGRESS.relative_to(ROOT)
    # Validate every input before creating live state or claiming a request.
    _, requests = verify(root, work)
    with singleton(work):
        _, requests = verify(root, work)
        ledger = BudgetLedger(SHARED_LEDGER); live = work / 'FROZEN_LIVE.json'
        if not live.exists():
            if (work / 'run.sqlite').exists() or (work / 'judge-results.jsonl').exists():
                raise ValueError('execution state without live freeze')
            baseline = ledger.totals()
            if baseline['attempts'] != BASELINE: raise ValueError('expected historical baseline1612 required')
            save(live, dict(state='eight_judge_batches_frozen_no_calls', created_at_utc=now(),
                judge_sha256=sha(work / 'FROZEN_JUDGE.json'), baseline=baseline,
                historical_sha256=historical_sha(ledger.path, BASELINE),
                code_hashes={name: sha(local_path(root, name)) for name in CODE},
                max_requests=8, max_preflight_cny='2', fixed_hold_cny='.25',
                price_source=PRICE, price_checked_at=CHECKED,
                usd_per_million_input='.15', usd_per_million_output='.60',
                conversion='10 CNY/USD conservative budget assumption, not market FX',
                automatic_retries=False, old_guard_reopened=False,
                authorization='Autonomous experiment continuation: eight corrected JSON-format judge batches, separate CNY2 guard.'), exclusive=True)
        started = load(live)
        if (started.get('judge_sha256') != sha(work / 'FROZEN_JUDGE.json')
                or started.get('baseline', {}).get('attempts') != BASELINE
                or started.get('max_requests') != 8 or started.get('max_preflight_cny') != '2'
                or started.get('fixed_hold_cny') != '.25' or started.get('automatic_retries') is not False
                or started.get('old_guard_reopened') is not False
                or set(started.get('code_hashes', {})) != set(CODE)
                or historical_sha(ledger.path, BASELINE) != started.get('historical_sha256')):
            raise ValueError('live freeze or historical accounting changed')
        verify_hashes(root, started['code_hashes'])
        runpath = work / 'run.sqlite'
        run = DevelopmentRun(runpath) if runpath.exists() else DevelopmentRun(runpath, create=True,
            max_requests=8, max_reservation_cny='2', approval_scope='native_100_20')
        summary = verify_run(run)
        if ledger.totals()['attempts'] != BASELINE + summary['request_claims']:
            raise ValueError('ledger/guard dispatch count differs; manual audit required')
        executor = make_executor(work, sha(live), ledger, run)
        output = work / 'judge-results.jsonl'
        existing = lines(output) if output.exists() else []
        if [r.get('request_id') for r in existing] != [r['request_id'] for r in requests[:len(existing)]]:
            raise ValueError('saved outputs must be an exact frozen request prefix')
        for request, row in zip(requests, existing):
            if saved_row(request, executor.recover(request['request_id'], request['payload'])) != row:
                raise ValueError('saved judge output differs from durable receipt')
        if summary['state'] != 'active' and len(existing) != 8:
            raise ValueError('closed guard cannot dispatch')
        dispatched = 0; start = time.monotonic()
        def progress(state, error=None):
            value = dict(state=state, phase='judge', completed=len(existing), expected=8,
                updated_at_utc=now(), new_this_invocation=dispatched, run=run.summary(), budget=ledger.totals())
            if error: value['error_type'] = error
            save(progress_file, value); print(json.dumps(value), flush=True)
        try:
            for request in requests[len(existing):]:
                if time.monotonic() - start > 7200: raise ValueError('invocation deadline reached')
                validate_payload(request['payload'])
                result = executor.execute(request['request_id'], request['payload'])
                row = saved_row(request, result)
                with output.open('a', encoding='utf-8') as out:
                    out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                    out.flush(); os.fsync(out.fileno())
                existing.append(row); dispatched += not result['recovered']; progress('running')
                if max_requests and dispatched >= max_requests and len(existing) < 8:
                    progress('bounded_invocation_finished_guard_active'); return
            if historical_sha(ledger.path, BASELINE) != started['historical_sha256']:
                raise ValueError('historical accounting changed')
            if verify_run(run)['request_claims'] != 8 or ledger.totals()['attempts'] != BASELINE + 8:
                raise ValueError('complete judge requires eight exact accounted calls')
            run.close(); progress('judge_complete')
            completed = dict(state='complete', phase='judge', count=8, output_sha256=sha(output),
                frozen_live_sha256=sha(live), judge_frozen_sha256=sha(work / 'FROZEN_JUDGE.json'), run=run.summary())
            destination = work / 'JUDGE_COMPLETED.json'
            if destination.exists():
                if load(destination) != completed: raise ValueError('existing completion changed')
            else: save(destination, completed, exclusive=True)
        except BaseException as exc:
            category = getattr(exc, 'category', type(exc).__name__)
            progress('stopped_no_automatic_retry', category)
            raise SystemExit('Judge stopped: ' + category + '; inspect saved receipt, never replay unknown.') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('verify', 'judge'))
    parser.add_argument('--live', action='store_true'); parser.add_argument('--max-requests', type=int)
    args = parser.parse_args(argv)
    try:
        if args.phase == 'verify':
            _, requests = verify()
            print(json.dumps(dict(state='verified_no_calls', requests=len(requests), max_preflight_cny='2', json_instruction=True)))
        elif not args.live: raise ValueError('explicit --live required')
        else: execute(args.max_requests)
    except Exception as exc:
        raise SystemExit('V7.1 JSON judge stopped: ' + type(exc).__name__) from None


if __name__ == '__main__': main()
