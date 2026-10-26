"""Explicit bounded real reader/judge requests; immutable inputs and no retries.

The reader phase has no gold-file access. Only a separate scorer may prepare
gold-informed judge requests, after all four conditions' answers are complete.
"""
import argparse
from contextlib import closing,contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_memory.atm_reader import ATMTextReader,reader_payload
from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.development_run import DevelopmentRun,ScopedTransport
from agent_memory.durable_chat import DurableChat
from agent_memory.provider_http import BudgetedJSONTransport,NativeHTTPS,OPENAI_ENDPOINT,SHARED_LEDGER
from agent_memory.windows_credentials import read_key

WORK=ROOT/'runtime/atm-v6-reader'
READER_FROZEN=WORK/'FROZEN_READER.json'
LIVE_FROZEN=WORK/'FROZEN_LIVE.json'
PROGRESS=ROOT/'reports/atm-v6-reader-live-progress.json'
CODE=('scripts/run_atm_v6_reader.py','agent_memory/atm_reader.py','agent_memory/durable_chat.py',
      'agent_memory/embedding_executor.py','agent_memory/provider_http.py','agent_memory/visual_recovery.py',
      'agent_memory/development_run.py','agent_memory/budget.py')
PRICE_SOURCE='https://developers.openai.com/api/docs/models/gpt-4o-mini'
PRICE_CHECKED='2026-10-25T15:14:11+00:00'


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value,exclusive=False):
    path=Path(path); temporary=path if exclusive else path.with_suffix('.tmp')
    with temporary.open('x' if exclusive else 'w',encoding='utf-8') as out:
        json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    if not exclusive: os.replace(temporary,path)
def lines(path):
    return [json.loads(row) for row in Path(path).read_text(encoding='utf-8').splitlines() if row.strip()] if Path(path).exists() else []
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as out:
        out.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())


def saved_row(request,result,phase):
    saved={key:result[key] for key in ('attempt_id','receipt_sha256','run_ordinal')}
    saved['request_id']=request['request_id']
    if phase=='reader':
        row=request['metadata'];saved.update({key:row[key] for key in ('query_id','dataset','id','qtype','variant')})
        try: answer=ATMTextReader.decode(result['response'],row['pack'])
        except (ValueError,KeyError,TypeError):
            answer=dict(answer='',answer_ids=[],cited_ids=[],schema_status='invalid',
                citations_valid=False,answer_ids_valid=False,model=result['response']['model'])
        saved['answer']=answer
    else: saved['response']=result['response']
    return saved


@contextmanager
def singleton():
    import msvcrt
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'live-worker.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try: yield
        finally: lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def historical_digest(limit):
    with closing(sqlite3.connect(SHARED_LEDGER.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        rows=db.execute('SELECT * FROM attempts WHERE rowid<=? ORDER BY rowid',(limit,)).fetchall()
        observations=db.execute('SELECT o.* FROM provider_observations o JOIN attempts a ON a.id=o.attempt_id WHERE a.rowid<=? ORDER BY a.rowid',(limit,)).fetchall()
    return hashlib.sha256(json.dumps([rows,observations],separators=(',',':')).encode()).hexdigest()


def verify_reader():
    frozen=load(READER_FROZEN)
    if frozen['state']!='reader_inputs_frozen_no_calls' or frozen['planned_reader_answers']!=64:
        raise ValueError('exact frozen 64-answer pilot required')
    for name,digest in {**frozen['input_hashes'],**frozen['code_hashes']}.items():
        if sha(ROOT/name)!=digest: raise ValueError('frozen reader source or code changed')
    packs=ROOT/frozen['packs_file']
    if sha(packs)!=frozen['packs_sha256']: raise ValueError('frozen reader packs changed')
    return frozen,lines(packs)


def live_freeze(ledger):
    reader,_=verify_reader()
    expected=dict(reader_frozen_sha256=sha(READER_FROZEN),code_hashes={name:sha(ROOT/name) for name in CODE},
        max_requests=80,max_cumulative_preflight_cny='20',shared_budget_cny='500',
        model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,fixed_request_hold_cny='.25',
        usd_input_per_million='.15',usd_output_per_million='.60',
        conversion='10 CNY/USD conservative budget assumption, not market FX',
        price_source=PRICE_SOURCE,price_checked_at=PRICE_CHECKED)
    if LIVE_FROZEN.exists():
        frozen=load(LIVE_FROZEN)
        if any(frozen.get(key)!=value for key,value in expected.items()):
            raise ValueError('V6 live execution freeze changed')
    else:
        if (WORK/'run.sqlite').exists(): raise ValueError('run without immutable execution freeze')
        baseline=ledger.totals()
        frozen=dict(expected,created_at_utc=now(),baseline_budget=baseline,historical_attempts=baseline['attempts'],
            historical_ledger_sha256=historical_digest(baseline['attempts']),
            authorization='User requested advancing the proposed V6 real rerank/reader experiment within existing CNY500 budget.',
            transfer='Frozen public ATM question/source text to established OpenAI endpoint; isolated reference answers only for later proxy judging.',
            condition=reader['source_condition'],exposure=reader['exposure'],automatic_retries=False)
        save(LIVE_FROZEN,frozen,exclusive=True)
    if historical_digest(frozen['historical_attempts'])!=frozen['historical_ledger_sha256']:
        raise ValueError('historical accounting changed')
    return frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('reader','judge'))
    parser.add_argument('--live',action='store_true');parser.add_argument('--max-requests',type=int)
    args=parser.parse_args()
    if not args.live: raise SystemExit('Explicit --live required; no calls made.')
    if args.max_requests is not None and args.max_requests<1: raise SystemExit('Positive invocation limit required.')
    with singleton():
        ledger=BudgetLedger(SHARED_LEDGER);frozen=live_freeze(ledger)
        run_path=WORK/'run.sqlite'
        run=DevelopmentRun(run_path) if run_path.exists() else DevelopmentRun(run_path,create=True,
            max_requests=80,max_reservation_cny='20',approval_scope='native_100_20')
        guard_state=run.summary()['state']
        if guard_state not in ('active','complete'): raise ValueError('invalid V6 guard state')
        rate=VerifiedRate('gpt-4o-mini','1.5','6',PRICE_SOURCE,PRICE_CHECKED,conversion_basis='budget_assumption_not_market_fx')
        def factory(http,observer):
            base=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,rate=rate,
                credential_reader=read_key,http=http,mode='live',enabled=True,prices_confirmed=True,
                fixed_reserve_cny='.25',usd_rates=('.15','.60'),output_token_cap=1536,timeout_seconds=120,
                max_response_bytes=1024*1024,attempt_observer=observer)
            return ScopedTransport(base,run,'.25')
        executor=DurableChat(WORK/'receipts','atm-v6:'+sha(LIVE_FROZEN),NativeHTTPS(),factory,run)
        _,packs=verify_reader()
        output=WORK/('reader-answers.jsonl' if args.phase=='reader' else 'judge-results.jsonl')
        if args.phase=='reader':
            requests=[dict(request_id=hashlib.sha256((row['query_id']+'\0'+row['variant']).encode()).hexdigest(),
                           payload=reader_payload(row['pack']),metadata=row) for row in packs]
        else:
            judge=load(WORK/'FROZEN_JUDGE.json')
            for name,digest in judge['input_hashes'].items():
                if sha(ROOT/name)!=digest: raise ValueError('isolated judge input changed')
            request_path=ROOT/judge['requests_file']
            if sha(request_path)!=judge['requests_sha256']: raise ValueError('judge requests changed')
            requests=lines(request_path)
            if len(requests)>16: raise ValueError('proxy judge request cap exceeded')
        existing=lines(output)
        if [row['request_id'] for row in existing]!=[row['request_id'] for row in requests[:len(existing)]]:
            raise ValueError('saved outputs are not the frozen request prefix')
        for request,row in zip(requests,existing):
            recovered=executor.recover(request['request_id'],request['payload'])
            if saved_row(request,recovered,args.phase)!=row:
                raise ValueError('saved answer differs from its completed durable receipt')
        if guard_state=='complete' and len(existing)!=len(requests):
            raise ValueError('completed V6 guard cannot dispatch missing requests')
        start=time.monotonic();new=0
        def progress(state,error=None):
            totals=ledger.totals()
            result=dict(state=state,phase=args.phase,updated_at_utc=now(),pid=os.getpid(),
                completed=len(existing),expected=len(requests),new_this_invocation=new,run=run.summary(),budget=totals,
                new_provider_requests=totals['attempts']-frozen['historical_attempts'],
                new_accounted_micro_cny=totals['accounted_micro_cny']-frozen['baseline_budget']['accounted_micro_cny'])
            if error: result['error_type']=error
            save(PROGRESS,result);print(json.dumps(result),flush=True)
        try:
            progress('running')
            for request in requests[len(existing):]:
                if time.monotonic()-start>7200: raise ValueError('invocation deadline reached')
                result=executor.execute(request['request_id'],request['payload'])
                saved=saved_row(request,result,args.phase)
                append(output,saved);existing.append(saved);new+=not result['recovered'];progress('running')
                if args.max_requests is not None and new>=args.max_requests and len(existing)<len(requests):
                    progress('invocation_limit_guard_remains_active');return
            if args.phase=='reader':
                report=dict(state='reader_complete_waiting_isolated_scoring',completed_answers=len(existing),questions=16,
                    reader_frozen_file='runtime/atm-v6-reader/FROZEN_READER.json',reader_frozen_sha256=sha(READER_FROZEN),
                    answers_file='runtime/atm-v6-reader/reader-answers.jsonl',answers_sha256=sha(output),finished_at_utc=now(),
                    gold_loaded=False,official_score=False,live_frozen_sha256=sha(LIVE_FROZEN))
                path=ROOT/'reports/atm-v6-reader-run.json'
            else:
                if run.summary()['state']=='active': run.close()
                report=dict(state='judge_complete',completed_requests=len(existing),results_file='runtime/atm-v6-reader/judge-results.jsonl',
                    results_sha256=sha(output),judge_frozen_sha256=sha(WORK/'FROZEN_JUDGE.json'),finished_at_utc=now(),official_score=False)
                path=ROOT/'reports/atm-v6-reader-judge-run.json'
            if not path.exists(): save(path,report,exclusive=True)
            progress(report['state'])
        except BaseException as exc:
            progress('stopped_no_automatic_retry',getattr(exc,'category',type(exc).__name__))
            raise SystemExit('V6 paid run stopped; inspect saved receipt and accounting before any continuation.') from None


if __name__=='__main__': main()
