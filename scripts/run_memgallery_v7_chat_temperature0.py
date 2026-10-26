"""Explicit additive V7 compatibility entry; original failed guard stays intact.

Both reader/judge payloads retain the previously frozen temperature=0. The
preflight failure sent no HTTP request. A separate execution directory prevents
any guard reset or replay of an unresolved original journal entry.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import run_memgallery_v7_chat as original
from agent_memory.memgallery_v7_transport import MemGalleryV7TemperatureZeroTransport

OLD = original.BASE / 'chat'
WORK = original.BASE / 'chat-temperature0'
EXTRA_CODE = ('scripts/run_memgallery_v7_chat_temperature0.py', 'agent_memory/memgallery_v7_transport.py')


def install():
    original.WORK = WORK
    original.CODE = (*original.CODE, *EXTRA_CODE)
    original.BudgetedJSONTransport = MemGalleryV7TemperatureZeroTransport


def verify_compatibility():
    fixed = original.load(WORK / 'FROZEN_COMPATIBILITY.json')
    original.verify_hashes(fixed['hashes'])
    if original.sha(OLD / 'reader-packs.jsonl') != original.sha(WORK / 'reader-packs.jsonl'):
        raise ValueError('compatibility changed a frozen reader payload')
    if fixed['provider_requests_before_recovery'] != 1483:
        raise ValueError('preflight failure provider baseline differs')
    return fixed


def prepare():
    audit_file = ROOT / 'reports/memgallery-v7-chat-preflight-failure-audit.json'
    audit = original.load(audit_file)
    if audit.get('state') != 'confirmed_no_provider_dispatch':
        raise ValueError('confirmed local-only original rejection required')
    result = original.prepare()
    if original.sha(OLD / 'reader-packs.jsonl') != original.sha(WORK / 'reader-packs.jsonl'):
        raise ValueError('native question/evidence/prompt/payload changed; refuse recovery')
    paths = [audit_file, OLD / 'FROZEN_READER.json', OLD / 'FROZEN_LIVE.json',
             OLD / 'reader-packs.jsonl', OLD / 'run.sqlite', OLD / 'receipts/private-embedding.sqlite',
             WORK / 'FROZEN_READER.json', WORK / 'reader-packs.jsonl']
    original.save(WORK / 'FROZEN_COMPATIBILITY.json', dict(
        state='identical_reader_payloads_new_bounded_execution',
        created_at_utc=original.now(), provider_requests_before_recovery=1483,
        original_failure='temperature0 rejected by legacy local admission before credential/budget/HTTP',
        reader_payloads_byte_identical=True, original_claims_preserved=1,
        provider_requests_max=99, provider_hold_max_cny='19.35',
        nominal_guard_claims_max_including_local_failure=100,
        nominal_guard_preflight_max_including_local_failure_cny='19.60',
        hashes={p.relative_to(ROOT).as_posix(): original.sha(p) for p in paths}), exclusive=True)
    verify_compatibility()
    return dict(**result, compatibility='temperature0 admitted; payloads unchanged', original_failed_guard_preserved=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare','reader','judge'))
    parser.add_argument('--live', action='store_true'); parser.add_argument('--max-requests', type=int)
    args = parser.parse_args(); install()
    if args.phase == 'prepare': print(json.dumps(prepare()))
    elif not args.live: raise SystemExit('Explicit --live required')
    else:
        verify_compatibility()
        original.execute(args.phase, args.max_requests)
