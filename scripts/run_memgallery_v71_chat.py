"""Same-scene citation-interface regression; a new experiment, never a replay.

Sources, queries, ranked pools, selected questions and budgets stay fixed. A
short opaque source alias replaces serialized composite citation IDs, and the
prompt adds only citation-format instructions. All V7 failures remain intact.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import run_memgallery_v7_chat as original
from agent_memory.memgallery_v7_transport import MemGalleryV7TemperatureZeroTransport
from agent_memory import memgallery_v71_reader as native
from agent_memory.development_run import DevelopmentRun

OLD = original.BASE / 'chat-temperature0'
WORK = original.BASE / 'chat-citation-v71'
EXTRA = ('scripts/run_memgallery_v71_chat.py','agent_memory/memgallery_v71_reader.py',
         'agent_memory/memgallery_v7_transport.py')


def install():
    original.WORK = WORK
    original.PROGRESS = ROOT / 'reports/memgallery-v71-chat-progress.json'
    original.CODE = (*original.CODE, *EXTRA)
    original.BudgetedJSONTransport = MemGalleryV7TemperatureZeroTransport
    original.pack_memgallery_evidence = native.pack_memgallery_evidence
    original.reader_payload = native.reader_payload
    original.decode_reader = native.decode_reader


def verify():
    value = original.load(WORK / 'FROZEN_INTERFACE.json')
    original.verify_hashes(value['hashes'])
    if not value['same_queries_rankings_complete20sources'] or value['new_provider_calls_max'] != 72:
        raise ValueError('fixed interface-regression scope changed')
    return value


def prepare():
    failure = original.load(OLD / 'FAILURE_COMPLETED.json')
    if (failure['state'] != 'reader_failed_no_judge' or failure['count'] != 64
            or failure['invalid_answers'] != 58 or failure['run']['state'] != 'complete'):
        raise ValueError('original failed reader experiment must be preserved and closed')
    audit_path = ROOT / 'reports/memgallery-v7-reader-failure-accounting-audit.json'
    audit = original.load(audit_path)
    if audit.get('state') != 'passed': raise ValueError('original failure accounting audit required')
    result = original.prepare()
    previous = original.lines(OLD / 'reader-packs.jsonl')
    current = original.lines(WORK / 'reader-packs.jsonl')
    if len(previous) != 64 or len(current) != 64: raise ValueError('same64 complete comparison required')
    for old, new in zip(previous, current):
        for field in ('query_id','index','point','variant'):
            if old[field] != new[field]: raise ValueError('question/condition identity changed')
        if (old['pack']['source_memory_ids'] != new['pack']['source_memory_ids']
                or len(new['pack']['source_memory_ids']) != 20
                or old['pack']['question_sha256'] != new['pack']['question_sha256']):
            raise ValueError('same ordered complete sources and original question required')
    paths = [audit_path, OLD / 'FAILURE_COMPLETED.json', OLD / 'FROZEN_READER.json', OLD / 'FROZEN_LIVE.json',
             OLD / 'reader-packs.jsonl', OLD / 'reader-answers.jsonl', OLD / 'run.sqlite',
             OLD / 'receipts/private-embedding.sqlite', WORK / 'FROZEN_READER.json', WORK / 'reader-packs.jsonl']
    original.save(WORK / 'FROZEN_INTERFACE.json', dict(state='new_alias_interface_frozen_no_calls',
        created_at_utc=original.now(), same_queries_rankings_complete20sources=True,
        condition='same-scene engineering regression, not new independent-scene evidence',
        modifications='source citation IDs replaced by stable short aliases; citation-only format instruction appended',
        semantic_prompt_changed=False, answer_text_repair=False, original_reader_calls_replayed=0,
        new_provider_calls_max=72, new_preflight_cny='18', old_real_calls=91,
        aggregate_real_calls_max=163, aggregate_provider_preflight_cny='35.35',
        nominal_guard_preflight_including_original_local_rejection_cny='35.60',
        shared_budget_cny='500', gold_used_for_interface_design=False,
        hashes={p.relative_to(ROOT).as_posix(): original.sha(p) for p in paths}), exclusive=True)
    verify()
    return dict(**result, experiment='memgallery-v7.1', same_source_order_and_questions=True,
                original_reader_calls_preserved=64, new_reader_calls_planned=64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare','reader','judge'))
    parser.add_argument('--live', action='store_true'); parser.add_argument('--max-requests', type=int)
    args = parser.parse_args(); install()
    if args.phase == 'prepare': print(json.dumps(prepare()))
    elif not args.live: raise SystemExit('Explicit --live required')
    else:
        verify()
        original.execute(args.phase, args.max_requests)
