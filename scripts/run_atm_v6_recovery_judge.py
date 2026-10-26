"""Five additive judge calls within the original V6 aggregate 80-call/CNY20 cap.

The original 74-call guard stays closed. No reader call is repeated. Existing
receipts and all earlier budget rows are immutable; uncertain calls never replay.
"""
import argparse
from contextlib import closing, contextmanager
from pathlib import Path
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.development_run import DevelopmentRun, ScopedTransport
from agent_memory.durable_chat import DurableChat
from agent_memory.provider_http import BudgetedJSONTransport, NativeHTTPS, OPENAI_ENDPOINT, SHARED_LEDGER
from agent_memory.windows_credentials import read_key
from scripts.run_atm_v6_reader import sha, load, lines, save, append, now, historical_digest

WORK = ROOT / 'runtime/atm-v6-reader-recovery'
OLD = ROOT / 'runtime/atm-v6-reader'
FROZEN = WORK / 'FROZEN_RECOVERY_LIVE.json'
JUDGE = WORK / 'FROZEN_RECOVERY_JUDGE.json'
OUTPUT = WORK / 'judge-results.jsonl'
REPORT = ROOT / 'reports/atm-v6-reader-recovery-judge-run.json'
PROGRESS = ROOT / 'reports/atm-v6-reader-recovery-live-progress.json'
PRICE_SOURCE = 'https://developers.openai.com/api/docs/models/gpt-4o-mini'
PRICE_CHECKED = '2026-10-26T03:42:55+00:00'
CODE = ('scripts/run_atm_v6_recovery_judge.py', 'scripts/recover_atm_v6_reader.py',
        'agent_memory/atm_reader_recovery.py', 'agent_memory/durable_chat.py',
        'agent_memory/embedding_executor.py', 'agent_memory/provider_http.py',
        'agent_memory/visual_recovery.py', 'agent_memory/development_run.py',
        'agent_memory/budget.py', 'scripts/run_atm_v6_reader.py')


def old_summary():
    with closing(sqlite3.connect((OLD/'run.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        limit, cap, state = db.execute('SELECT max_requests,cap,state FROM run').fetchone()
        count, held = db.execute('SELECT count(*),coalesce(sum(held),0) FROM calls').fetchone()
        unfinished = db.execute("SELECT count(*) FROM calls WHERE state!='complete'").fetchone()[0]
    return dict(max_requests=limit, cap=cap, state=state, count=count, held=held, unfinished=unfinished)


def aggregate_cap(old, new_count=5, new_hold=1250000):
    if (old != dict(max_requests=80, cap=20000000, state='complete', count=74,
                    held=18500000, unfinished=0) or new_count != 5 or new_hold != 1250000
            or old['count']+new_count > 80 or old['held']+new_hold > 20000000):
        raise ValueError('original closed run plus recovery exceeds frozen aggregate scope')


def verify_prepared():
    prepared = load(JUDGE)
    for name, digest in {**prepared['input_hashes'], **prepared['code_hashes']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('recovery judge frozen input changed')
    if prepared['judge_requests'] != 5 or prepared['model'] != 'gpt-4o-mini':
        raise ValueError('exact five specified-model recovery judge requests required')
    requests_path = ROOT/prepared['requests_file']
    if requests_path != WORK/'judge-requests.jsonl' or sha(requests_path) != prepared['requests_sha256']:
        raise ValueError('recovery request file identity mismatch')
    requests = lines(requests_path)
    if len(requests) != 5 or len({row['request_id'] for row in requests}) != 5:
        raise ValueError('five unique recovery requests required')
    for request in requests:
        payload = request['payload']
        if (not isinstance(request.get('request_id'), str) or not request['request_id']
                or not isinstance(payload, dict)
                or set(payload) != {'model', 'messages', 'max_completion_tokens', 'store', 'n', 'response_format'}
                or payload['model'] != 'gpt-4o-mini'
                or type(payload['max_completion_tokens']) is not int or payload['max_completion_tokens'] != 1536
                or payload['store'] is not False or type(payload['n']) is not int or payload['n'] != 1
                or payload['response_format'] != {'type': 'json_object'}):
            raise ValueError('bounded specified judge required')
        messages = payload['messages']
        if (not isinstance(messages, list) or len(messages) != 2
                or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
                       or not isinstance(m['content'], str) or not m['content'].strip() for m in messages)
                or [m['role'] for m in messages] != ['system', 'user']):
            raise ValueError('exact system plus user text-only judge messages required')
        if sum(len(m['content'].encode('utf-8')) for m in payload['messages']) > 32768:
            raise ValueError('judge batch exceeds existing transport text bound')
    aggregate_cap(old_summary())
    return requests


def freeze(ledger):
    requests = verify_prepared()
    expected = dict(judge_frozen_sha256=sha(JUDGE), code_hashes={name:sha(ROOT/name) for name in CODE},
        preserved_old_files={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in
            (OLD/'run.sqlite', OLD/'receipts/private-embedding.sqlite', OLD/'FROZEN_LIVE.json',
             OLD/'reader-answers.jsonl', OLD/'judge-results.jsonl', ROOT/'reports/atm-v6-reader-scores.json')},
        model='gpt-4o-mini', endpoint=OPENAI_ENDPOINT, max_requests=5,
        max_cumulative_preflight_cny='1.25', fixed_request_hold_cny='.25',
        aggregate_v6_calls_max=80, aggregate_v6_hold_cny_max='20',
        aggregate_v6_calls_planned=79, aggregate_v6_hold_cny_planned='19.75',
        shared_budget_cny='500', old_calls_preserved=74, automatic_retries=False,
        usd_input_per_million='.15', usd_output_per_million='.60',
        conversion='10 CNY/USD conservative budget assumption, not market FX',
        price_source=PRICE_SOURCE, price_checked_at=PRICE_CHECKED)
    if FROZEN.exists():
        frozen = load(FROZEN)
        if any(frozen.get(k) != v for k,v in expected.items()):
            raise ValueError('additive recovery live freeze changed')
    else:
        if (WORK/'run.sqlite').exists():
            raise ValueError('recovery guard exists without immutable freeze')
        baseline = ledger.totals()
        if baseline['attempts'] != 1451:
            raise ValueError('inspect concurrent or unexpected calls before recovery')
        frozen = dict(expected, created_at_utc=now(), baseline_budget=baseline,
            historical_attempts=baseline['attempts'],
            historical_ledger_sha256=historical_digest(baseline['attempts']),
            purpose='Regrade existing completed reader answers after auxiliary-only parser recovery; no new reader inference.',
            authorization='Existing V6 pilot request and 80-call/CNY20 limit; five additional calls remain within that scope.',
            transfer='Same selected public questions/references and unchanged original model answers for isolated blinded proxy grading.')
        save(FROZEN, frozen, exclusive=True)
    if historical_digest(frozen['historical_attempts']) != frozen['historical_ledger_sha256']:
        raise ValueError('historical budget rows changed')
    return frozen, requests


def saved(request, result):
    from scripts.recover_atm_v6_reader import decode_judge
    decode_judge(result['response'], set(range(8)))
    return dict(request_id=request['request_id'], **{k:result[k] for k in
        ('attempt_id','receipt_sha256','run_ordinal','response')})


def dispatch(requests, executor, run, output, progress):
    summary = run.summary()
    if (summary['state'] not in ('active', 'complete') or summary['max_requests'] != 5
            or summary['max_reservation_micro_cny'] != 1250000
            or not 0 <= summary['request_claims'] <= 5
            or summary['cumulative_preflight_micro_cny'] != summary['request_claims'] * 250000):
        raise ValueError('recovery guard differs from fixed five-request scope')
    existing = lines(output)
    if [r['request_id'] for r in existing] != [r['request_id'] for r in requests[:len(existing)]]:
        raise ValueError('saved judge output is not the frozen prefix')
    for request, row in zip(requests, existing):
        if saved(request, executor.recover(request['request_id'],request['payload'])) != row:
            raise ValueError('saved result differs from original durable receipt')
    if run.summary()['state'] == 'complete' and len(existing) != len(requests):
        raise ValueError('completed recovery guard cannot dispatch missing requests')
    for request in requests[len(existing):]:
        result = executor.execute(request['request_id'],request['payload'])
        row = saved(request,result)
        append(output,row)
        existing.append(row)
        progress(len(existing))
    if run.summary()['state'] == 'active':
        run.close()
    return existing


@contextmanager
def singleton():
    import msvcrt
    with (WORK/'live-worker.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try: yield
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if not args.live: raise SystemExit('Explicit --live required; no calls made.')
    with singleton():
        ledger=BudgetLedger(SHARED_LEDGER)
        frozen,requests=freeze(ledger)
        run_path=WORK/'run.sqlite'
        run=DevelopmentRun(run_path) if run_path.exists() else DevelopmentRun(run_path,
            create=True,max_requests=5,max_reservation_cny='1.25')
        rate=VerifiedRate('gpt-4o-mini','1.5','6',PRICE_SOURCE,PRICE_CHECKED,
                          conversion_basis='budget_assumption_not_market_fx')
        def factory(http,observer):
            base=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,
                rate=rate,credential_reader=read_key,http=http,mode='live',enabled=True,
                prices_confirmed=True,fixed_reserve_cny='.25',usd_rates=('.15','.60'),
                output_token_cap=1536,timeout_seconds=120,max_response_bytes=1024*1024,attempt_observer=observer)
            return ScopedTransport(base,run,'.25')
        namespace='atm-v6-recovery:'+sha(FROZEN)
        executor=DurableChat(WORK/'receipts',namespace,NativeHTTPS(),factory,run)
        def progress(completed,state='running',error=None):
            value=dict(state=state,updated_at_utc=now(),completed=completed,expected=5,
                       run=run.summary(),budget=ledger.totals(),old_v6_calls_preserved=74,
                       aggregate_v6_calls=74+run.summary()['request_claims'])
            if error: value['error_type']=error
            save(PROGRESS,value); print(json.dumps(value),flush=True)
        try:
            progress(len(lines(OUTPUT)))
            results=dispatch(requests,executor,run,OUTPUT,progress)
            # Includes all immutable old-run files and accounting in final validation.
            freeze(ledger)
            report=dict(state='judge_complete',completed_requests=len(results),
                results_file=str(OUTPUT.relative_to(ROOT)).replace('\\','/'),results_sha256=sha(OUTPUT),
                judge_frozen_sha256=sha(JUDGE),live_frozen_sha256=sha(FROZEN),receipt_namespace=namespace,
                run_guard=run.summary(),finished_at_utc=now(),official_score=False,
                original_reader_calls_repeated=0,aggregate_v6_calls=79)
            if REPORT.exists():
                prior=load(REPORT)
                if any(prior.get(k)!=v for k,v in report.items() if k!='finished_at_utc'):
                    raise ValueError('existing completion report differs')
            else: save(REPORT,report,exclusive=True)
            progress(len(results),'judge_complete')
        except BaseException as exc:
            progress(len(lines(OUTPUT)),'stopped_no_automatic_retry',getattr(exc,'category',type(exc).__name__))
            raise SystemExit('Recovery judge stopped; inspect original receipts, never replay uncertain calls.') from None


if __name__=='__main__': main()
