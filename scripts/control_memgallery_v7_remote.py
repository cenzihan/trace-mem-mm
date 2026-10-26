"""Explicit, separate V7 staging/GPU control; importing never contacts a host.

verify is local and inventory/status are read-only. Mutating actions additionally
require --execute. All inference uses the unchanged V6 worker through the pinned
native-schema wrapper; no API provider, raw media, gold or credentials are sent.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.control_atm_v6_remote import HOST, OPTIONS, PYTHON, remote, save, sha
from scripts import run_memgallery_v7_remote as wrapper

WORK = ROOT / 'runtime/memgallery-v7/rerank'
REMOTE = '/software/cenzihan/agent-memory-multimodal/memgallery-v7-20261026'
RUNNER = ROOT / 'scripts/run_atm_v6_remote.py'
WRAPPER = ROOT / 'scripts/run_memgallery_v7_remote.py'
INPUTS = ('FROZEN_CONFIG.json', 'queries.jsonl', 'sources.jsonl', 'candidates.jsonl', 'pairs.jsonl')
CODE = ('scripts/run_atm_v6_remote.py', 'scripts/run_memgallery_v7_remote.py',
        'scripts/control_memgallery_v7_remote.py')


def local_path(relative):
    if not isinstance(relative, str) or '\\' in relative or '..' in relative.split('/'):
        raise ValueError('safe project-relative input path required')
    path = (ROOT / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(ROOT.resolve()) or 'scorer-only' in path.parts:
        raise ValueError('only project-local non-gold inputs may enter GPU control')
    return path


def frozen():
    data = wrapper.load_inputs(WORK, RUNNER, WRAPPER)
    value = data['frozen']
    codes = value.get('code_hashes')
    if not isinstance(codes, dict) or not set(CODE) <= set(codes):
        raise ValueError('both workers and V7 controller must be code-hash frozen')
    if (codes[CODE[0]] != value['remote_runner_sha256']
            or codes[CODE[1]] != value['remote_wrapper_sha256']):
        raise ValueError('execution identities differ across freeze fields')
    for group in ('artifact_hashes', 'code_hashes', 'input_hashes'):
        hashes = value.get(group, {})
        if not isinstance(hashes, dict):
            raise ValueError('frozen input hash mapping required')
        for name, expected in hashes.items():
            if not wrapper.worker.valid_digest(expected) or sha(local_path(name)) != expected:
                raise ValueError('frozen local input changed')
    return data


def verify():
    data = frozen()
    return dict(state='verified_not_staged', experiment='memgallery-v7',
                queries=len(data['queries']), sources=len(data['sources']), pairs=len(data['pairs']),
                remote_runner_sha256=sha(RUNNER), remote_wrapper_sha256=sha(WRAPPER),
                provider_requests=0, gpu_jobs_started=0)


def inventory():
    return remote('''import subprocess,json
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
occupied={line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
rows=[]
for line in raw.splitlines():
 fields=[v.strip() for v in line.split(',')]
 rows.append(dict(index=int(fields[0]),uuid=fields[1],used=int(fields[2]),total=int(fields[3]),compute_process=fields[1] in occupied))
print(json.dumps(dict(host='GPU1_zhilian',gpus=rows)))''')


def stage():
    data = frozen()
    paths = [WORK / name for name in INPUTS] + [RUNNER, WRAPPER]
    expected = {path.name: sha(path) for path in paths}
    record = WORK / 'REMOTE_STAGE.json'
    if record.exists():
        prior = wrapper.worker.strict_json(record.read_bytes())
        if prior.get('hashes') != expected or prior.get('remote_directory') != REMOTE:
            raise ValueError('existing staging identity differs; preserve it')
    remote(f'''import pathlib,hashlib,json,os
p=pathlib.Path({REMOTE!r}); p.mkdir(mode=0o700,parents=True,exist_ok=True)
assert not p.is_symlink(); os.chmod(p,0o700)
for name,digest in {expected!r}.items():
 f=p/name
 assert not f.is_symlink()
 if f.exists(): assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
print(json.dumps(dict(state='ready_for_same_content_upload')))''')
    if not record.exists():
        result = subprocess.run(['scp', *OPTIONS, *map(str, paths), HOST + ':' + REMOTE + '/'],
                                capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError('private_V7_upload_failed')
    verified = remote(f'''import pathlib,hashlib,json,os
p=pathlib.Path({REMOTE!r}); expected={expected!r}
for name,digest in expected.items():
 f=p/name; assert not f.is_symlink(); assert hashlib.sha256(f.read_bytes()).hexdigest()==digest; os.chmod(f,0o600)
print(json.dumps(dict(state='staged_verified',files=len(expected),hashes=expected)))''')
    verified.update(remote_directory=REMOTE, host=HOST, pair_count=len(data['pairs']),
                    experiment='memgallery-v7', transfer='source/caption/query text only; no gold, raw media, keys or ledger')
    if not record.exists():
        save(record, verified, exclusive=True)
    return dict(state='staged_verified', files=len(expected), pairs=len(data['pairs']), host=HOST)


def status():
    return remote(f'''import pathlib,json
p=pathlib.Path({REMOTE!r}); rows=[]
for i in range(4):
 for kind in ('progress','report'):
  f=p/(kind+'-'+str(i)+'.json')
  if f.exists(): rows.append(dict(worker=i,kind=kind,data=json.loads(f.read_text())))
print(json.dumps(dict(experiment='memgallery-v7',workers=rows)))''')


def validate_report(report, worker_index, frozen_value):
    if (report.get('worker_index') != worker_index or report.get('workers') != 4
            or report.get('experiment') != 'memgallery-v7'
            or report.get('frozen_sha256') != sha(WORK / 'FROZEN_CONFIG.json')
            or report.get('remote_runner_sha256') != sha(RUNNER)
            or report.get('remote_wrapper_sha256') != sha(WRAPPER)
            or report.get('spec') != frozen_value['reranker_spec']):
        raise ValueError('worker report differs from exact V7 freeze and both code hashes')


def launch(pilot=False, resume=False):
    data = frozen()
    if pilot and resume:
        raise ValueError('pilot and resume are separate phases')
    record = WORK / ('PILOT_LAUNCH.json' if pilot else 'RESUME_LAUNCH.json' if resume else 'FULL_LAUNCH.json')
    if record.exists():
        raise ValueError('dispatch already recorded; inspect before any further launch')
    staged = wrapper.worker.strict_json((WORK / 'REMOTE_STAGE.json').read_bytes())
    if (staged.get('remote_directory') != REMOTE
            or staged.get('hashes', {}).get('FROZEN_CONFIG.json') != sha(WORK / 'FROZEN_CONFIG.json')
            or staged['hashes'].get(RUNNER.name) != sha(RUNNER)
            or staged['hashes'].get(WRAPPER.name) != sha(WRAPPER)):
        raise ValueError('stage does not match current V7 freeze')
    workers = [0] if pilot else list(range(4))
    if not pilot:
        prior = {row['worker']: row['data'] for row in status()['workers'] if row['kind'] == 'report'}
        for index, report in prior.items():
            validate_report(report, index, data['frozen'])
        if resume:
            if (not (WORK / 'FULL_LAUNCH.json').exists() or set(prior) != set(range(4))
                    or any(row['state'] != 'complete' and not
                           (row['state'] == 'stopped' and row.get('error_type') == 'WorkerDeadline')
                           for row in prior.values())):
                raise ValueError('only known completed or local deadline-stopped shards may resume')
            workers = [index for index, row in prior.items() if row['state'] != 'complete']
            if not workers:
                raise ValueError('all frozen shards already complete')
        elif (not (WORK / 'PILOT_LAUNCH.json').exists() or set(prior) != {0}
              or prior[0]['state'] not in ('complete', 'pair_limit_reached')):
            raise ValueError('successful exact pilot required before full continuation')
        save(WORK / ('DEADLINE_REPORTS.json' if resume else 'PILOT_REPORT.json'), prior, exclusive=True)
    current = inventory()['gpus']
    available = {row['index']: row for row in current
                 if row['used'] <= 128 and row['total'] >= 12000 and not row['compute_process']}
    binding = WORK / 'GPU_BINDINGS.json'
    if not binding.exists():
        order = (1, 0, 2, 3)
        if any(index not in available for index in order):
            raise ValueError('four specified idle GPUs unavailable; no substitution')
        save(binding, [available[index]['uuid'] for index in order], exclusive=True)
    uuids = wrapper.worker.strict_json(binding.read_bytes())
    if not isinstance(uuids, list) or len(uuids) != 4 or len(set(uuids)) != 4:
        raise ValueError('four distinct frozen GPU UUIDs required')
    idle_uuids = {row['uuid'] for row in available.values()}
    if any(uuids[index] not in idle_uuids for index in workers):
        raise ValueError('assigned GPU is no longer idle')
    # Persist intent before SSH: an unknown dispatch must never automatically replay.
    intent = record.with_suffix('.intent.json')
    save(intent, dict(workers=workers, gpu_uuids=uuids, frozen_sha256=sha(WORK / 'FROZEN_CONFIG.json'),
                      pilot=pilot, resume=resume, state='dispatch_intent_no_automatic_replay'), exclusive=True)
    expected = staged['hashes']
    result = remote(f'''import pathlib,subprocess,json,os,hashlib
p=pathlib.Path({REMOTE!r}); uuids={uuids!r}; workers={workers!r}
for name,digest in {expected!r}.items():
 f=p/name; assert not f.is_symlink(); assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
result=[]
for i in workers:
 cmd=['timeout','7260',{PYTHON!r},'-I','-S',str(p/'run_memgallery_v7_remote.py'),'--work',str(p),'--gpu-uuid',uuids[i],'--worker-index',str(i),'--workers','4','--max-seconds','7200','--execute']
 if {pilot!r}: cmd+=['--max-pairs','256']
 log=p/('worker-'+str(i)+'.log')
 with log.open('ab') as out:
  os.chmod(log,0o600)
  child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True,cwd=p)
 result.append(dict(worker=i,pid=child.pid,gpu_uuid=uuids[i],mode='pilot' if {pilot!r} else 'continuation'))
print(json.dumps(result))''')
    save(record, result, exclusive=True)
    return result


def collect():
    data = frozen()
    if (WORK / 'RERANK_RUN.json').exists() or (WORK / 'reranker-scores.jsonl').exists():
        raise FileExistsError('preserve completed V7 artifacts')
    incoming = WORK / 'remote-results'
    incoming.mkdir(exist_ok=True)
    for i in range(4):
        for kind, suffix in (('scores', 'jsonl'), ('report', 'json')):
            name = f'{kind}-{i}.{suffix}'
            destination = incoming / name
            if destination.exists():
                raise FileExistsError('preserve already collected shard; inspect partial collection')
            reply = subprocess.run(['scp', *OPTIONS, HOST + ':' + REMOTE + '/' + name, str(destination)],
                                   capture_output=True, timeout=90)
            if reply.returncode:
                raise RuntimeError('V7_result_download_incomplete')
    found, reports = {}, []
    for i in range(4):
        report = wrapper.worker.strict_json((incoming / f'report-{i}.json').read_bytes())
        validate_report(report, i, data['frozen'])
        path = incoming / f'scores-{i}.jsonl'
        rows = list(wrapper.worker.json_lines(path))
        expected = data['pairs'][i::4]
        if (report.get('state') != 'complete' or report.get('scores_sha256') != sha(path)
                or report.get('completed_pairs') != len(expected)
                or [row.get('pair_id') for row in rows] != [row['pair_id'] for row in expected]):
            raise ValueError('complete exact frozen V7 shard required')
        for row in rows:
            if (set(row) != {'pair_id', 'score'} or row['pair_id'] in found
                    or type(row['score']) not in (int, float) or not math.isfinite(row['score'])):
                raise ValueError('unique finite relevance scores required')
            found[row['pair_id']] = row['score']
        reports.append(report)
    destination = WORK / 'reranker-scores.jsonl'
    with destination.open('x', encoding='utf-8') as out:
        for pair in data['pairs']:
            out.write(wrapper.worker.canonical(dict(pair_id=pair['pair_id'], score=found[pair['pair_id']])) + '\n')
        out.flush(); os.fsync(out.fileno())
    run = dict(state='complete', experiment='memgallery-v7', frozen_config_sha256=sha(WORK / 'FROZEN_CONFIG.json'),
               pairs_sha256=sha(WORK / 'pairs.jsonl'), reranker_spec=data['frozen']['reranker_spec'],
               completed_pairs=len(found), scores_file='runtime/memgallery-v7/rerank/reranker-scores.jsonl',
               scores_sha256=sha(destination), remote_runner_sha256=sha(RUNNER), remote_wrapper_sha256=sha(WRAPPER),
               workers=reports, provider_requests=0, new_embedding_calls=0, raw_media=False)
    save(WORK / 'RERANK_RUN.json', run, exclusive=True)
    return dict(state='complete', completed_pairs=len(found), provider_requests=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'inventory', 'stage', 'pilot', 'start', 'resume', 'status', 'collect'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.action not in ('verify', 'inventory', 'status') and not args.execute:
        parser.error('mutating V7 control requires explicit --execute')
    actions = dict(verify=verify, inventory=inventory, stage=stage, pilot=lambda: launch(pilot=True),
                   start=launch, resume=lambda: launch(resume=True), status=status, collect=collect)
    try:
        print(json.dumps(actions[args.action](), ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        raise SystemExit('V7 control stopped: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
