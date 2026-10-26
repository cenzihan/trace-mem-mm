"""Own-process supervision for the fixed-model submission service only."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/user/cenzihan/projects/zindi-underwater-noise/env/bin/python3.10'
EXPIRES_AT = 1793896200
if time.time() >= EXPIRES_AT:
    raise SystemExit(0)
os.umask(0o077)
lock = (ROOT/'supervisor.lock').open('a')
try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(0)
(ROOT/'supervisor.pid').write_text(str(os.getpid()))
children = {}


def stop_signal(*args):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop_signal)
signal.signal(signal.SIGINT, stop_signal)
try:
    while time.time() < EXPIRES_AT:
        for name in ('qwen_service', 'tunnel_runner'):
            child = children.get(name)
            if child is not None and child.poll() is None:
                continue
            log = ROOT/(name+'.log')
            if log.exists() and log.stat().st_size > 1048576:
                log.replace(ROOT/(name+'.previous.log'))
            with log.open('ab') as stream:
                children[name] = subprocess.Popen([PYTHON, '-I', '-S', str(ROOT/(name+'.py'))],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                    start_new_session=True)
        (ROOT/'processes.json').write_text(json.dumps({name:child.pid for name,child in children.items()}))
        time.sleep(30)
finally:
    for child in children.values():
        if child.poll() is None and os.getpgid(child.pid) == child.pid:
            os.killpg(child.pid, signal.SIGTERM)
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
