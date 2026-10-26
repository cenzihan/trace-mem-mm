"""V7.1 same-scene citation-interface regression; isolated additive scoring.

The frozen V7 scorer supplies all mathematics, selection and blind judging.
Only new chat paths, the alias-aware reader decoder and provenance are bound.
No retrieval rescoring, provider calls, old-answer repair or receipt replay.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_memgallery_v7 as original

CHAT = 'runtime/memgallery-v7/chat-citation-v71'
INTERFACE = CHAT + '/FROZEN_INTERFACE.json'
PATHS = dict(
    CHAT=CHAT,
    PACKS=CHAT + '/reader-packs.jsonl',
    ANSWERS=CHAT + '/reader-answers.jsonl',
    READER_FROZEN=CHAT + '/FROZEN_READER.json',
    LIVE_FROZEN=CHAT + '/FROZEN_LIVE.json',
    JUDGE_FROZEN=CHAT + '/FROZEN_JUDGE.json',
    JUDGE_REQUESTS=CHAT + '/judge-requests.jsonl',
    JUDGE_MAPPING=CHAT + '/scorer-only/judge-mapping.json',
    JUDGE_RESULTS=CHAT + '/judge-results.jsonl',
    CHAT_PROGRESS='reports/memgallery-v71-chat-progress.json',
    GUARD=CHAT + '/run.sqlite',
    JOURNAL=CHAT + '/receipts/private-embedding.sqlite',
    READER_COMPLETE=CHAT + '/READER_COMPLETED.json',
    JUDGE_COMPLETE=CHAT + '/JUDGE_COMPLETED.json',
    READER_SCORES='reports/memgallery-v71-reader-scores.json',
    READER_DETAILS=CHAT + '/scorer-only/reader-scores.json',
    JUDGE_PREPARATION='reports/memgallery-v71-judge-preparation.json',
)
ORIGINAL_CODE = tuple(original.CODE)
EXTRA_CODE = ('scripts/score_memgallery_v71.py', 'agent_memory/memgallery_v71_reader.py',
              'scripts/run_memgallery_v71_chat.py', 'agent_memory/memgallery_v7_transport.py')
BASE_VERIFY_READER = original.verify_reader_complete
BASE_SAVE_NEW = original.save_new
PROVENANCE = dict(
    experiment='memgallery-v7.1',
    validation_scope='same-scene citation-interface engineering regression, not independent replication',
    independent_replication=False,
    same_original_selection_and_four_rankings=True,
    same_ordered_twenty_source_contexts=True,
    shared_v7_retrieval_results_not_rescored=True,
    new_reader_inferences=64,
    original_v7_reader_receipts_replayed=0,
    original_v7_substantive_answers_repaired=False,
    original_v7_results_preserved=True,
)


def native_reader():
    from agent_memory import memgallery_v71_reader
    return memgallery_v71_reader


def verify_interface(root=ROOT):
    value = original.load(root, INTERFACE)
    if (value.get('state') != 'new_alias_interface_frozen_no_calls'
            or value.get('same_queries_rankings_complete20sources') is not True
            or value.get('new_provider_calls_max') != 72 or value.get('new_preflight_cny') != '18'
            or value.get('original_reader_calls_replayed') != 0
            or value.get('answer_text_repair') is not False
            or value.get('gold_used_for_interface_design') is not False):
        raise ValueError('frozen same-scene interface-regression scope required')
    hashes = value.get('hashes')
    required = {
        'runtime/memgallery-v7/chat-temperature0/FAILURE_COMPLETED.json',
        'runtime/memgallery-v7/chat-temperature0/FROZEN_READER.json',
        'runtime/memgallery-v7/chat-temperature0/FROZEN_LIVE.json',
        'runtime/memgallery-v7/chat-temperature0/reader-packs.jsonl',
        'runtime/memgallery-v7/chat-temperature0/reader-answers.jsonl',
        'runtime/memgallery-v7/chat-temperature0/run.sqlite',
        'runtime/memgallery-v7/chat-temperature0/receipts/private-embedding.sqlite',
        PATHS['READER_FROZEN'], PATHS['PACKS'],
    }
    if not isinstance(hashes, dict) or not required <= set(hashes):
        raise ValueError('old closed run and new reader inputs must remain bound')
    original.check_hashes(root, hashes)
    return value


def verify_reader_complete(root=ROOT, *, final=False):
    verify_interface(root)
    complete = BASE_VERIFY_READER(root, final=final)
    complete['input_hashes'][INTERFACE] = original.sha(root, INTERFACE)
    return complete


def annotated_save(path, value, *, raw=False):
    targets = (PATHS['JUDGE_FROZEN'], PATHS['JUDGE_PREPARATION'], PATHS['READER_SCORES'])
    normalized = Path(path).resolve().as_posix()
    if not raw and isinstance(value, dict) and any(normalized.endswith('/' + name) for name in targets):
        # Add metadata before the original exclusive/fsync write and hash freeze.
        value.update(PROVENANCE)
        value['interface_frozen_file'] = INTERFACE
    return BASE_SAVE_NEW(path, value, raw=raw)


def install():
    native = native_reader()
    for name, value in PATHS.items():
        setattr(original, name, value)
    original.reader_payload = native.reader_payload
    original.decode_reader = native.decode_reader
    original.CODE = tuple(dict.fromkeys((*ORIGINAL_CODE, *EXTRA_CODE)))
    original.verify_reader_complete = verify_reader_complete
    original.save_new = annotated_save


def prepare_judge(root=ROOT, *, count_tokens=None):
    install()
    return original.prepare_judge(root, count_tokens=count_tokens)


def finalize(root=ROOT, *, bootstrap_samples=5000):
    install()
    return original.finalize(root, bootstrap_samples=bootstrap_samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('judge-prepare', 'final'))
    args = parser.parse_args(argv)
    try:
        result = prepare_judge() if args.phase == 'judge-prepare' else finalize()
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    except Exception as exc:
        raise SystemExit('Isolated V7.1 scorer stopped: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
