"""Posthoc citation-format diagnosis; immutable receipts, no gold or provider I/O.

Only aggregate structural counts are published. Primary answers and strict
outcomes remain untouched. JSON-pair serialization is the sole hypothetical
normalization; incomplete labels are diagnosed, never repaired or scored.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import run_memgallery_v7_chat_temperature0 as entry
from scripts import encode_memgallery_v7 as encoding
from agent_memory.embedding_executor import _json
from agent_memory.memgallery_v7_reader import _response, decode_reader, reader_payload

REPORT = ROOT / 'reports/memgallery-v7-citation-format-diagnostic.json'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def analyze(response, packed, source_metadata):
    known = set(packed['source_memory_ids'])
    by_round, sessions = defaultdict(list), set()
    for identity, meta in source_metadata.items():
        by_round[meta['source_round']].append(identity)
        sessions.add(meta['source_session'])
    outcome, citation_types = Counter(responses=1), Counter()
    try:
        value = _response(response)
    except ValueError:
        outcome['response_admission_invalid'] += 1
        return outcome, citation_types
    try:
        decode_reader(response, packed)
        outcome['strict_valid'] += 1
    except ValueError as exc:
        outcome['strict_invalid'] += 1
        reason = str(exc)
        if reason == 'citations must be distinct supplied native source IDs':
            outcome['strict_citation_failure'] += 1
        elif reason == 'inconsistent reader abstention':
            outcome['strict_abstention_failure'] += 1
        else:
            outcome['strict_other_failure'] += 1
    cited = value.get('cited_ids')
    if not isinstance(cited, list):
        outcome['citation_array_invalid'] += 1
        return outcome, citation_types
    fixed, changed, classes = [], False, []
    for cite in cited:
        replacement = cite
        if not isinstance(cite, str):
            category = 'non_string'
        elif cite in known:
            category = 'exact_supplied_native_id'
        else:
            try:
                pair = json.loads(cite)
            except (ValueError, TypeError):
                pair = None
            if isinstance(pair, list) and len(pair) == 2 and all(isinstance(item, str) for item in pair):
                identity = json.dumps(pair, ensure_ascii=False, separators=(',', ':'))
                if identity in known:
                    category = 'json_pair_equivalent_supplied'; replacement = identity
                elif identity in source_metadata:
                    category = 'json_pair_known_corpus_not_supplied'
                else:
                    category = 'json_pair_unknown_native'
            elif cite in by_round:
                retained = [identity for identity in by_round[cite] if identity in known]
                if len(retained) == 1:
                    category = 'single_round_label_unique_supplied'
                elif retained:
                    category = 'single_round_label_ambiguous_supplied'
                else:
                    category = 'single_round_label_not_supplied'
            elif cite in sessions:
                category = 'single_session_label'
            elif cite in set(packed.get('image_ids', [])):
                category = 'image_id_instead_of_source_id'
            else:
                category = 'unrecognized_source_reference'
        citation_types[category] += 1
        classes.append(category)
        fixed.append(replacement)
        changed |= replacement != cite
    if not cited:
        outcome['empty_citation_array'] += 1
    if cited and all(category == 'single_round_label_unique_supplied' for category in classes):
        outcome['all_round_labels_unique_in_context_requires_identity_completion'] += 1
        if all(len(by_round[cite]) == 1 for cite in cited):
            outcome['all_round_labels_globally_unique_requires_identity_completion'] += 1
    if cited and len(cited) % 2 == 0 and all(isinstance(cite, str) for cite in cited):
        pairs = [json.dumps(cited[i:i+2], ensure_ascii=False, separators=(',', ':')) for i in range(0, len(cited), 2)]
        if len(set(pairs)) == len(pairs) and set(pairs) <= known:
            outcome['flattened_adjacent_session_round_pairs_supplied_requires_restructuring'] += 1
    if changed:
        modified = json.loads(json.dumps(response, ensure_ascii=False))
        modified['choices'][0]['message']['content'] = json.dumps(dict(value, cited_ids=fixed), ensure_ascii=False)
        # This is diagnostic memory only; no modified response/answer is saved.
        try:
            decoded = decode_reader(modified, packed)
            if decoded['answer'] != value['answer']:
                raise AssertionError('diagnosis must not alter substantive answer')
            outcome['valid_after_json_pair_serialization_only'] += 1
        except ValueError:
            outcome['still_invalid_after_json_pair_serialization_only'] += 1
    return outcome, citation_types


def verified_snapshot():
    entry.install()
    entry.verify_compatibility()
    frozen, packs = entry.original.verify_reader()
    encoding.verify()
    work = entry.WORK
    answer_file = work / 'reader-answers.jsonl'
    answer_bytes = answer_file.read_bytes()
    # Snapshot only complete JSONL rows while the reader may still be appending.
    boundary = answer_bytes.rfind(b'\n') + 1
    complete_bytes = answer_bytes[:boundary]
    rows = [json.loads(line) for line in complete_bytes.splitlines()]
    expected = {}
    for item in packs:
        request_id = sha_bytes((item['query_id'] + '\0' + item['variant']).encode())
        expected[request_id] = item
    if len(expected) != 64 or [row['request_id'] for row in rows] != list(expected)[:len(rows)]:
        raise ValueError('complete fixed reader request prefix required')
    live = work / 'FROZEN_LIVE.json'
    namespace = 'memgallery-v7:' + encoding.sha(live)
    verified = []
    journal_path, run_path = work / 'receipts/private-embedding.sqlite', work / 'run.sqlite'
    with closing(sqlite3.connect(journal_path.resolve().as_uri() + '?mode=ro', uri=True)) as journal, \
            closing(sqlite3.connect(run_path.resolve().as_uri() + '?mode=ro', uri=True)) as run:
        journal.row_factory = sqlite3.Row
        journal.execute('BEGIN'); run.execute('BEGIN')
        for saved in rows:
            item = expected[saved['request_id']]
            payload = reader_payload(item['pack'])
            manifest = dict(keys=[saved['request_id']], payload_sha256=sha_bytes(_json(payload).encode()),
                            run_path=str(run_path.resolve()))
            fingerprint = sha_bytes(_json(dict(namespace=namespace, manifest=manifest)).encode())
            row = journal.execute('SELECT * FROM operations WHERE fingerprint=?', (fingerprint,)).fetchone()
            if (row is None or row['state'] != 'complete' or not row['transport_completed']
                    or row['manifest'] != _json(manifest) or row['scope'] != namespace
                    or row['run_path'] != str(run_path.resolve()) or row['status'] != 200 or row['redacted']
                    or not isinstance(row['body'], bytes) or sha_bytes(row['body']) != saved['receipt_sha256']
                    or row['body_sha'] != saved['receipt_sha256'] or row['attempt_id'] != saved['attempt_id']
                    or row['run_ordinal'] != saved['run_ordinal']):
                raise ValueError('saved reader output must bind an exact completed receipt')
            if run.execute('SELECT model,state FROM calls WHERE ordinal=?', (row['run_ordinal'],)).fetchone() != ('gpt-4o-mini', 'complete'):
                raise ValueError('same completed reader run claim required')
            claims = journal.execute('SELECT fingerprint FROM claims WHERE scope=? AND chunk_key=?',
                                     (namespace, saved['request_id'])).fetchall()
            if len(claims) != 1 or claims[0][0] != fingerprint:
                raise ValueError('unique request claim required')
            response = json.loads(row['body'])
            result = dict(response=response, attempt_id=row['attempt_id'], receipt_sha256=row['body_sha'],
                          run_ordinal=row['run_ordinal'])
            request = dict(request_id=saved['request_id'], metadata=item)
            if entry.original.saved_row(request, result, 'reader') != saved:
                raise ValueError('primary strict outcome differs from saved receipt')
            verified.append((item, response))
    finished_file = work / 'READER_COMPLETED.json'
    finished = False
    if finished_file.exists():
        complete = encoding.load(finished_file)
        if (complete.get('state') != 'complete' or complete.get('phase') != 'reader' or complete.get('count') != 64
                or len(rows) != 64 or boundary != len(answer_bytes)
                or complete.get('output_sha256') != sha_bytes(answer_bytes)
                or complete.get('frozen_live_sha256') != encoding.sha(live)):
            raise ValueError('reader completion must bind all 64 exact saved outcomes')
        finished = True
    return verified, finished, complete_bytes, answer_file


def audit(*, finalize=False):
    verified, complete, answer_bytes, answer_file = verified_snapshot()
    if finalize and not complete:
        raise ValueError('all 64 completed reader outcomes required for final diagnosis')
    mapping_file = encoding.WORK / 'source-mapping.json'
    metadata = encoding.load(mapping_file)['by_native_id']
    totals, citation_types = Counter(), Counter()
    variants, points = defaultdict(Counter), defaultdict(Counter)
    for item, response in verified:
        outcome, kinds = analyze(response, item['pack'], metadata)
        totals.update(outcome); citation_types.update(kinds)
        variants[item['variant']].update(outcome); points[item['point']].update(outcome)
    provenance_paths = [entry.WORK / name for name in ('FROZEN_COMPATIBILITY.json', 'FROZEN_READER.json',
                         'FROZEN_LIVE.json', 'reader-packs.jsonl')]
    provenance_paths.extend([encoding.WORK / 'FROZEN_ENCODING.json', mapping_file, Path(__file__)])
    if complete:
        provenance_paths.append(entry.WORK / 'READER_COMPLETED.json')
    report = dict(state='final_diagnostic' if complete else 'provisional_diagnostic',
        checked_at_utc=datetime.now(timezone.utc).isoformat(), complete_reader=complete,
        counts=dict(totals), citation_items=dict(citation_types),
        by_variant={key: dict(value) for key, value in variants.items()},
        by_point={key: dict(value) for key, value in points.items()},
        input_hashes={path.relative_to(ROOT).as_posix(): encoding.sha(path) for path in provenance_paths},
        answer_snapshot_file=answer_file.relative_to(ROOT).as_posix(), answer_snapshot_sha256=sha_bytes(answer_bytes),
        primary_strict_outputs_unchanged=True, raw_response_files_unchanged=True,
        gold_loaded=False, provider_requests=0, raw_content_or_ids_in_report=False,
        hypothetical_json_pair_rule='Parse existing citation string as JSON two-string list; compact JSON only; exact supplied ID membership and duplicate checks still required.',
        incomplete_label_policy='Session-only, round-only and flattened pairs require identity completion or restructuring; diagnosed only, never silently normalized.',
        interpretation='Posthoc structural diagnosis, not a new test or accuracy score. Any later secondary interpretation must be independently frozen and retain original strict results.')
    if finalize:
        encoding.save(REPORT, report, exclusive=True)
    return dict(state=report['state'], complete_reader=complete, counts=report['counts'],
                citation_items=report['citation_items'], gold_loaded=False, provider_requests=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(audit(finalize=args.finalize), ensure_ascii=False))
    except Exception as exc:
        raise SystemExit('Citation audit stopped: ' + type(exc).__name__) from None
