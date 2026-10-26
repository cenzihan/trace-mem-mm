"""One bounded continuation for the already authorized V6 experiment.

Waits for the four existing GPU shards, merges complete scores, runs isolated
retrieval scoring, then the frozen reader and proxy judge. No new embeddings,
parameter search, unknown-call replay, recurring schedule, push or deployment.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.control_atm_v6_remote import status,launch,collect

WORK=ROOT/'runtime/atm-v6-rerank'
CONFIG=WORK/'PIPELINE_CONFIG.json'
PROGRESS=ROOT/'reports/atm-v6-followup-progress.json'
CODE=('scripts/finish_atm_v6.py','scripts/control_atm_v6_remote.py',
      'scripts/run_atm_v6_remote.py','agent_memory/v6_rerank.py','scripts/score_atm_v6_rerank.py',
      'scripts/prepare_atm_v6_reader.py','agent_memory/atm_reader.py','scripts/run_atm_v6_reader.py',
      'agent_memory/durable_chat.py','scripts/score_atm_v6_reader.py','agent_memory/atm_answer_metrics.py',
      'agent_memory/embedding_executor.py','agent_memory/provider_http.py','agent_memory/development_run.py',
      'agent_memory/visual_recovery.py','agent_memory/budget.py')
STEPS=(('retrieval_rank','scripts/score_atm_v6_rerank.py','rank'),
       ('retrieval_score','scripts/score_atm_v6_rerank.py','score'),
       ('reader_pack','scripts/prepare_atm_v6_reader.py','pack'),
       ('reader','scripts/run_atm_v6_reader.py','reader','--live'),
       ('judge_prepare','scripts/score_atm_v6_reader.py','prepare'),
       ('judge','scripts/run_atm_v6_reader.py','judge','--live'),
       ('final_scores','scripts/score_atm_v6_reader.py','finalize'))


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    tmp=Path(path).with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as out:
        json.dump(value,out,indent=2,ensure_ascii=False,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)
def expected_config():
    return dict(code_hashes={name:sha(ROOT/name) for name in CODE},
        rerank_frozen_sha256=sha(WORK/'FROZEN_CONFIG.json'),
        reader_selection_sha256=sha(ROOT/'runtime/atm-v6-reader/FROZEN_SELECTION.json'),
        max_wait_seconds=18000,max_deadline_continuations=1,steps=[list(step) for step in STEPS],
        max_new_provider_requests=80,max_cumulative_preflight_cny='20',shared_budget_cny='500',
        automatic_paid_retries=False,new_embeddings=False,official_submission=False)
def verify():
    frozen=json.loads(CONFIG.read_text(encoding='utf-8'))
    if frozen!=expected_config(): raise ValueError('frozen followup inputs or execution code changed')
    if (WORK/'STOP_FOLLOWUP').exists(): raise ValueError('explicit followup stop marker')


@contextmanager
def singleton():
    import msvcrt
    with (WORK/'followup.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try: yield
        finally: lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def run():
    with singleton():
        verify()
        if PROGRESS.exists(): raise ValueError('continuation already claimed; inspect instead of duplicate dispatch')
        start=time.monotonic()
        record=dict(state='running',phase='waiting_for_existing_gpu_jobs',pid=os.getpid(),started_at_utc=now(),
                    pipeline_config_sha256=sha(CONFIG),deadline_continuations=0,completed_steps=[])
        def progress(state=None,phase=None,**extra):
            if state: record['state']=state
            if phase: record['phase']=phase
            record.update(updated_at_utc=now(),elapsed_seconds=round(time.monotonic()-start,3),**extra)
            save(PROGRESS,record)
        try:
            progress()
            failures=0
            while True:
                verify()
                if time.monotonic()-start>18000: raise TimeoutError('bounded followup wait elapsed')
                try:
                    snapshot=status();failures=0
                except (OSError,RuntimeError,subprocess.SubprocessError,ValueError):
                    failures+=1
                    if failures>=3: raise
                    progress(status_read_failures=failures);time.sleep(30);continue
                rows={row['worker']:row['data'] for row in snapshot['workers'] if row['kind']=='report'}
                progress(gpu_completed_pairs=sum(row.get('completed_pairs',0) for row in rows.values()),
                         gpu_total_pairs=145629,gpu_states={str(i):row['state'] for i,row in rows.items()})
                if len(rows)==4 and all(row['state']=='complete' for row in rows.values()): break
                stopped=[row for row in rows.values() if row['state']=='stopped']
                if any(row.get('error_type')!='WorkerDeadline' for row in stopped):
                    raise RuntimeError('GPU worker stopped for a non-deadline reason')
                if len(rows)==4 and stopped and all(row['state'] in ('complete','stopped') for row in rows.values()):
                    if record['deadline_continuations']>=1: raise TimeoutError('bounded GPU continuation exhausted')
                    # Local deterministic inference only: continue the committed
                    # shard prefix once. Paid/unknown calls are never retried.
                    time.sleep(30);verify();launch(False,True)
                    record['deadline_continuations']+=1;progress()
                time.sleep(30)
            verify();progress(phase='collecting_complete_scores')
            collect()
            for phase,*command in STEPS:
                verify();progress(phase=phase)
                log=WORK/('followup-'+phase+'.log')
                with log.open('xb') as out:
                    result=subprocess.run([sys.executable,'-B',*command],cwd=ROOT,stdout=out,stderr=out,
                                          timeout=7500 if phase in ('reader','judge') else 1800)
                if result.returncode: raise RuntimeError('phase failed: '+phase)
                record['completed_steps'].append(phase);progress()
            progress(state='complete',phase='all_frozen_v6_steps_complete',finished_at_utc=now())
        except BaseException as exc:
            progress(state='stopped_no_automatic_retry',error_type=type(exc).__name__)
            raise SystemExit('V6 followup stopped; inspect bounded phase report.') from None


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-only',action='store_true');group.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:
        with CONFIG.open('x',encoding='utf-8') as out:
            json.dump(expected_config(),out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        print(json.dumps(dict(state='followup_frozen',config_sha256=sha(CONFIG))))
    else: run()
