"""Explicit private-host staging and bounded V6 GPU jobs; no provider API calls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runtime/atm-v6-rerank'
REMOTE = '/software/cenzihan/agent-memory-multimodal/atm-v6-rerank-20261025'
PYTHON = '/user/cenzihan/projects/zindi-underwater-noise/env/bin/python3.10'
HOST = 'GPU1_zhilian'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UpdateHostKeys=no', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=10']
INPUTS = ('FROZEN_CONFIG.json', 'queries.jsonl', 'sources.jsonl', 'candidates.jsonl', 'pairs.jsonl')
RUNNER = ROOT / 'scripts/run_atm_v6_remote.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with Path(path).open('x', encoding='utf-8') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
    else:
        tmp = Path(path).with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
        os.replace(tmp, path)


def remote(code, timeout=45):
    script = shlex.quote(PYTHON) + " -I -S - <<'V6PY'\n" + code + '\nV6PY\n'
    reply = subprocess.run(['ssh', *OPTIONS, HOST, 'bash -s'], input=script.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if reply.returncode:
        raise RuntimeError('remote_control_exit_' + str(reply.returncode))
    return json.loads(reply.stdout)


def frozen():
    value = json.loads((WORK / 'FROZEN_CONFIG.json').read_text(encoding='utf-8'))
    if sha(RUNNER) != value['remote_runner_sha256'] or value['reranker_spec']['workers'] != 4:
        raise ValueError('frozen remote runner/spec changed')
    for name, digest in value['artifact_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('frozen input changed')
    return value


def stage():
    value = frozen()
    paths = [WORK / name for name in INPUTS] + [RUNNER]
    expected = {path.name: sha(path) for path in paths}
    remote(f'''import pathlib,hashlib,json,os
p=pathlib.Path({REMOTE!r}); p.mkdir(mode=0o700,parents=True,exist_ok=True)
assert not p.is_symlink()
expected={expected!r}
for name,digest in expected.items():
 f=p/name
 if f.exists(): assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
os.chmod(p,0o700)
print(json.dumps({{'state':'ready_for_same_content_upload'}}))''')
    result = subprocess.run(['scp', *OPTIONS, *map(str, paths), HOST + ':' + REMOTE + '/'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if result.returncode:
        raise RuntimeError('private_upload_failed')
    verified = remote(f'''import pathlib,hashlib,json,os
p=pathlib.Path({REMOTE!r}); expected={expected!r}
for name,digest in expected.items():
 f=p/name; assert hashlib.sha256(f.read_bytes()).hexdigest()==digest; os.chmod(f,0o600)
print(json.dumps({{'state':'staged_verified','files':len(expected),'hashes':expected}}))''')
    verified.update(remote_directory=REMOTE, host=HOST, pair_count=value['pair_count'],
                    transfer='public source text and questions only; no gold, raw media, keys or ledger')
    save(WORK / 'REMOTE_STAGE.json', verified)
    return {key: verified[key] for key in ('state', 'files', 'host', 'pair_count')}


def launch(pilot, resume=False):
    local_frozen = frozen()
    launch_record = WORK / ('PILOT_LAUNCH.json' if pilot else 'RESUME_LAUNCH.json' if resume else 'FULL_LAUNCH.json')
    if launch_record.exists():
        raise ValueError('dispatch already recorded; inspect worker status instead of duplicate launch')
    resume_workers = None
    if resume:
        if not (WORK / 'FULL_LAUNCH.json').exists():
            raise ValueError('initial full dispatch required')
        snapshot = status()
        prior = {row['worker']: row['data'] for row in snapshot['workers'] if row['kind']=='report'}
        if any(row.get('worker_index') != i or row.get('frozen_sha256') != sha(WORK/'FROZEN_CONFIG.json')
                or row.get('remote_runner_sha256') != sha(RUNNER) or row.get('spec') != local_frozen['reranker_spec']
                for i,row in prior.items()):
            raise ValueError('deadline report does not match the frozen worker')
        if set(prior) != set(range(4)) or any(row['state'] != 'complete' and
                not (row['state']=='stopped' and row.get('error_type')=='WorkerDeadline') for row in prior.values()):
            raise ValueError('only completed local deadline stops may continue')
        resume_workers = [i for i,row in prior.items() if row['state']!='complete']
        if not resume_workers:
            raise ValueError('no unfinished shard')
        save(WORK / 'DEADLINE_REPORTS.json', prior, exclusive=True)
    elif not pilot:
        pilot_report = remote(f'''import pathlib,json
p=pathlib.Path({REMOTE!r})/'report-0.json'
print(json.dumps(json.loads(p.read_text()) if p.exists() else {{}}))''')
        if pilot_report.get('state') not in ('pair_limit_reached', 'complete'):
            raise ValueError('pilot must finish successfully before continuation')
        save(WORK / 'PILOT_REPORT.json', pilot_report, exclusive=True)
    stage_record = json.loads((WORK / 'REMOTE_STAGE.json').read_text())
    if stage_record['hashes']['FROZEN_CONFIG.json'] != sha(WORK / 'FROZEN_CONFIG.json'):
        raise ValueError('staging does not match freeze')
    binding = WORK / 'GPU_BINDINGS.json'
    if not binding.exists():
        inventory = remote('''import subprocess,json
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
print(json.dumps([dict(index=int(r.split(',')[0]),uuid=r.split(',')[1].strip(),used=int(r.split(',')[2])) for r in rows.splitlines()]))''')
        available = {row['index']: row for row in inventory if row['used'] <= 128}
        order = (1, 0, 2, 3)
        if any(index not in available for index in order):
            raise ValueError('four specified idle GPUs unavailable; no substitution')
        save(binding, [available[index]['uuid'] for index in order], exclusive=True)
    uuids = json.loads(binding.read_text())
    workers = [0] if pilot else resume_workers if resume else list(range(4))
    code = f'''import pathlib,subprocess,json,os,hashlib
p=pathlib.Path({REMOTE!r}); uuids={uuids!r}; workers={workers!r}
assert hashlib.sha256((p/'run_atm_v6_remote.py').read_bytes()).hexdigest()=={sha(RUNNER)!r}
assert hashlib.sha256((p/'FROZEN_CONFIG.json').read_bytes()).hexdigest()=={sha(WORK/'FROZEN_CONFIG.json')!r}
result=[]
for i in workers:
 progress=p/('progress-'+str(i)+'.json')
 if progress.exists():
  old=json.loads(progress.read_text())
  if old.get('state')=='complete':
   result.append(dict(worker=i,state='already_complete')); continue
 cmd=['timeout','7260',{PYTHON!r},'-I','-S',str(p/'run_atm_v6_remote.py'),'--work',str(p),'--gpu-uuid',uuids[i],'--worker-index',str(i),'--workers','4','--max-seconds','7200','--execute']
 if {pilot!r}: cmd += ['--max-pairs','256']
 log=p/('worker-'+str(i)+'.log')
 with log.open('ab') as out:
  os.chmod(log,0o600)
  child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True,cwd=p)
 result.append(dict(worker=i,pid=child.pid,gpu_uuid=uuids[i],mode='pilot' if {pilot!r} else 'continuation'))
print(json.dumps(result))'''
    result = remote(code)
    save(launch_record, result, exclusive=True)
    return result


def status():
    result = remote(f'''import pathlib,json,subprocess
p=pathlib.Path({REMOTE!r}); rows=[]
for i in range(4):
 for kind in ('progress','report'):
  f=p/(kind+'-'+str(i)+'.json')
  if f.exists(): rows.append(dict(worker=i,kind=kind,data=json.loads(f.read_text())))
print(json.dumps(dict(workers=rows)))''')
    save(ROOT / 'reports/atm-v6-live-progress.json', result)
    return result


def collect():
    value = frozen()
    incoming = WORK / 'remote-results'
    incoming.mkdir(exist_ok=True)
    for i in range(4):
        for kind, suffix in (('scores', 'jsonl'), ('report', 'json')):
            name = f'{kind}-{i}.{suffix}'
            result = subprocess.run(['scp', *OPTIONS, HOST + ':' + REMOTE + '/' + name, str(incoming / name)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            if result.returncode:
                raise ValueError('remote result incomplete')
    pairs = [json.loads(line) for line in (WORK / 'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    found = {}
    reports = []
    for i in range(4):
        report = json.loads((incoming / f'report-{i}.json').read_text())
        if (report.get('state') != 'complete' or report.get('worker_index') != i
                or report.get('frozen_sha256') != sha(WORK / 'FROZEN_CONFIG.json')
                or report.get('remote_runner_sha256') != sha(RUNNER)
                or report.get('spec') != value['reranker_spec']
                or report.get('scores_sha256') != sha(incoming / f'scores-{i}.jsonl')):
            raise ValueError('worker not complete')
        reports.append(report)
        rows = [json.loads(line) for line in (incoming / f'scores-{i}.jsonl').read_text().splitlines()]
        if [row['pair_id'] for row in rows] != [row['pair_id'] for row in pairs[i::4]]:
            raise ValueError('worker output not exact frozen shard')
        for row in rows:
            if set(row) != {'pair_id', 'score'} or row['pair_id'] in found or type(row['score']) not in (int,float) or not math.isfinite(row['score']):
                raise ValueError('invalid or duplicate relevance score')
            found[row['pair_id']] = row['score']
    destination = WORK / 'reranker-scores.jsonl'
    with destination.open('x', encoding='utf-8') as out:
        for row in pairs:
            out.write(json.dumps(dict(pair_id=row['pair_id'],score=found[row['pair_id']]), separators=(',',':'),allow_nan=False)+'\n')
        out.flush(); os.fsync(out.fileno())
    run = dict(state='complete', frozen_config_sha256=sha(WORK/'FROZEN_CONFIG.json'),
               pairs_sha256=sha(WORK/'pairs.jsonl'), reranker_spec=value['reranker_spec'],
               completed_pairs=len(found), scores_file='runtime/atm-v6-rerank/reranker-scores.jsonl',
               scores_sha256=sha(destination), remote_runner_sha256=sha(RUNNER), workers=reports,
               provider_requests=0, new_embedding_calls=0, raw_media=False)
    save(WORK/'RERANK_RUN.json', run, exclusive=True)
    return dict(state=run['state'],completed_pairs=len(found),provider_requests=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('stage','pilot','start','resume','status','collect'))
    args=parser.parse_args()
    action={'stage':stage,'pilot':lambda:launch(True),'start':lambda:launch(False),'resume':lambda:launch(False,True),'status':status,'collect':collect}[args.action]
    try:
        print(json.dumps(action(),ensure_ascii=False,allow_nan=False))
    except Exception as exc:
        # Remote stderr may include model paths or source payload; never echo it.
        raise SystemExit('V6 control stopped: '+type(exc).__name__) from None


if __name__=='__main__':
    main()
