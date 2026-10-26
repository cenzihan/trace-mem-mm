"""Reconcile three known successful embedding receipts into original holds.

No reservation, credential or network API is used. Every receipt and all parent
rows validate before changes. Unknown/failed outcomes retain all existing holds.
An interrupted bookkeeping run resumes the same IDs, never a provider request.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.budget import BudgetLedger
from agent_memory.visual_recovery import _strict_json
from scripts import prepare_embedding_smoke_grant as prep


def read_file(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('existing regular input file required')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('input exceeds fixed size limit')
    return path.resolve(), raw


def validate_grant(value):
    fields = {'version', 'grant_id', 'model', 'endpoint', 'dimensions', 'price_source',
              'price_checked_at', 'expires_at', 'slots'}
    if (not isinstance(value, dict) or set(value) != fields or type(value['version']) is not int
            or value['version'] != 1 or value['model'] != prep.MODEL
            or value['endpoint'] != prep.BAILIAN_BEIJING_ENDPOINT
            or type(value['dimensions']) is not int or value['dimensions'] != 256
            or value['price_source'] != prep.PRICE_SOURCE
            or str(uuid.UUID(value['grant_id'])) != value['grant_id']):
        raise ValueError('original embedding smoke grant required')
    checked, expires = prep.timestamp(value['price_checked_at']), prep.timestamp(value['expires_at'])
    # Dispatch expiry is deliberately not a reconciliation deadline: known usage
    # must still be recorded after expiry, without authorizing another request.
    if not 0 < (expires - checked).total_seconds() <= 26 * 3600:
        raise ValueError('original two-hour grant with same-day price check required')
    slots = value['slots']
    if not isinstance(slots, list) or len(slots) != 3:
        raise ValueError('three original parent reservations required')
    ids = set()
    for ordinal, slot in enumerate(slots, 1):
        if (not isinstance(slot, dict) or set(slot) != {'ordinal', 'parent_attempt_id', 'payload_sha256',
                'payload', 'max_input_tokens', 'reserved_micro_cny'}
                or type(slot['ordinal']) is not int or slot['ordinal'] != ordinal
                or type(slot['reserved_micro_cny']) is not int or slot['reserved_micro_cny'] != prep.HOLD
                or not isinstance(slot['parent_attempt_id'], str) or slot['parent_attempt_id'] in ids
                or not slot['parent_attempt_id'].startswith('attempt_')
                or len(slot['parent_attempt_id']) != 40
                or any(c not in '0123456789abcdef' for c in slot['parent_attempt_id'][8:])
                or type(slot['max_input_tokens']) is not int
                or slot['payload_sha256'] != prep.sha(prep._json(slot['payload']).encode('utf-8'))):
            raise ValueError('original unique bounded reservation identity required')
        ids.add(slot['parent_attempt_id'])
    prep.validate_payloads([slot['payload'] for slot in slots])
    if any(slot['max_input_tokens'] != 8192 * len(slot['payload']['input']) for slot in slots):
        raise ValueError('parent context bound differs')
    return value


def receipt_usage(body, slot):
    response = _strict_json(body)
    if not isinstance(response, dict) or response.get('model') != prep.MODEL:
        raise ValueError('receipt model must be text-embedding-v4')
    usage = response.get('usage')
    if (not isinstance(usage, dict) or type(usage.get('total_tokens')) is not int
            or not 0 <= usage['total_tokens'] <= slot['max_input_tokens']
            or ('prompt_tokens' in usage and (type(usage['prompt_tokens']) is not int
                                              or usage['prompt_tokens'] != usage['total_tokens']))
            or ('completion_tokens' in usage and (type(usage['completion_tokens']) is not int
                                                  or usage['completion_tokens'] != 0))):
        raise ValueError('bounded exact integer embedding token usage required')
    data = response.get('data'); count = len(slot['payload']['input'])
    if not isinstance(data, list) or len(data) != count:
        raise ValueError('receipt vector count differs from original batch')
    indices = []
    for row in data:
        if not isinstance(row, dict) or type(row.get('index')) is not int:
            raise ValueError('integer vector indices required')
        indices.append(row['index']); vector = row.get('embedding')
        if (not isinstance(vector, list) or len(vector) != 256
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
                or not any(vector)):
            raise ValueError('finite nonzero256D receipt vectors required')
    if set(indices) != set(range(count)):
        raise ValueError('duplicate or missing original vector index')
    return usage['total_tokens']


def validate_receipts(value, grant, grant_sha):
    if (not isinstance(value, dict) or set(value) != {'version', 'grant_id', 'grant_sha256', 'records', 'memory'}
            or type(value['version']) is not int or value['version'] != 1
            or value['grant_id'] != grant['grant_id'] or value['grant_sha256'] != grant_sha
            or not isinstance(value['records'], list) or len(value['records']) != 3
            or not isinstance(value['memory'], dict)):
        raise ValueError('complete grant-bound private receipt export required')
    expected = {'ordinal', 'parent_attempt_id', 'status', 'body_base64', 'body_sha256', 'redacted',
                'dispatch_started', 'category', 'returned_model', 'input_tokens', 'output_tokens'}
    outcomes = []
    for row, slot in zip(value['records'], grant['slots']):
        if (not isinstance(row, dict) or set(row) != expected or type(row['ordinal']) is not int
                or row['ordinal'] != slot['ordinal'] or row['parent_attempt_id'] != slot['parent_attempt_id']
                or type(row['status']) is not int or row['status'] != 200
                or row['redacted'] is not False or row['dispatch_started'] is not True
                or row['category'] != 'success' or row['returned_model'] != prep.MODEL
                or not isinstance(row['body_base64'], str) or type(row['output_tokens']) is not int
                or row['output_tokens'] != 0):
            raise ValueError('unknown, failed, redacted or mismatched receipt; retain all holds')
        body = base64.b64decode(row['body_base64'], validate=True)
        if len(body) > 1024 * 1024 or not body or prep.sha(body) != row['body_sha256']:
            raise ValueError('raw receipt bytes or SHA differ')
        tokens = receipt_usage(body, slot)
        if type(row['input_tokens']) is not int or row['input_tokens'] != tokens:
            raise ValueError('export usage differs from independently decoded receipt')
        amount = int((Decimal('.5') * tokens).to_integral_value(rounding=ROUND_CEILING))
        outcomes.append(dict(ordinal=slot['ordinal'], parent_attempt_id=slot['parent_attempt_id'],
            input_tokens=tokens, output_tokens=0, accounted_micro_cny=amount,
            receipt_sha256=row['body_sha256']))
    return outcomes


def inputs(ledger_path, grant_path, receipts_path):
    grant_path, raw = read_file(grant_path, 2 * 1024 * 1024)
    grant_sha = prep.sha(raw); grant = validate_grant(_strict_json(raw))
    metadata_path, meta_raw = read_file(grant_path.parent / 'preparation.json', 128 * 1024)
    metadata = _strict_json(meta_raw)
    ledger_file = prep.existing_ledger(ledger_path)
    if (not isinstance(metadata, dict) or metadata.get('state') != 'reserved_not_dispatched' or metadata.get('grant_id') != grant['grant_id']
            or metadata.get('grant_sha256') != grant_sha
            or prep.existing_ledger(metadata.get('ledger_path')) != ledger_file
            or metadata.get('completed_reservations') != 3 or metadata.get('newly_reserved_micro_cny') != 150000
            or type(metadata.get('existing_attempts')) is not int or metadata['existing_attempts'] < 0):
        raise ValueError('original parent preparation metadata must bind grant and exact ledger')
    receipts_path, receipts_raw = read_file(receipts_path, 6 * 1024 * 1024)
    outcomes = validate_receipts(_strict_json(receipts_raw), grant, grant_sha)
    hashes = {str(grant_path): grant_sha, str(metadata_path): prep.sha(meta_raw),
              str(receipts_path): prep.sha(receipts_raw)}
    return dict(grant=grant, grant_sha256=grant_sha, metadata=metadata, ledger=ledger_file,
                outcomes=outcomes, input_hashes=hashes)


def ledger_state(context):
    snapshot = prep.read_ledger(context['ledger'])
    slots = {s['parent_attempt_id']: s for s in context['grant']['slots']}
    outcomes = {s['parent_attempt_id']: s for s in context['outcomes']}
    rows = {r[1]: dict(zip(('rowid', *prep.SCHEMA['attempts']), r)) for r in snapshot['attempts']}
    observations = {r[1]: dict(zip(('rowid', *prep.SCHEMA['provider_observations']), r))
                    for r in snapshot['observations']}
    old_rows = [r for r in snapshot['attempts'] if r[1] not in slots]
    old_obs = [r for r in snapshot['observations'] if r[1] not in slots]
    metadata = context['metadata']
    if (len(snapshot['attempts']) != metadata['existing_attempts'] + 3
            or len(old_rows) != metadata['existing_attempts']
            or prep.sha(prep._json([old_rows, old_obs]).encode('utf-8')) != metadata['existing_history_sha256']):
        raise ValueError('historical rows changed or unexpected extra attempts; never reserve replacements')
    states = {}
    for attempt, slot in slots.items():
        row = rows.get(attempt); observed = observations.get(attempt); outcome = outcomes[attempt]
        if (row is None or row['model'] != prep.MODEL or row['retry_of'] is not None
                or row['max_input'] != slot['max_input_tokens'] or row['max_output'] != 0
                or row['input_rate'] != '.5' or row['output_rate'] != '0'
                or row['price_source'] != prep.PRICE_SOURCE
                or row['price_checked_at'] != context['grant']['price_checked_at']
                or row['reserved'] != prep.HOLD or row['conversion_basis'] != 'verified_fx'):
            raise ValueError('original reserved parent row differs from grant')
        if row['status'] in ('reserved', 'dispatched'):
            if (observed is not None or row['accounted'] != prep.HOLD or row['amount_kind'] != 'fixed_budget_hold'
                    or row['input_tokens'] is not None or row['output_tokens'] is not None):
                raise ValueError('unfinished parent row has conflicting accounting')
        elif row['status'] == 'settled':
            if (row['accounted'] != outcome['accounted_micro_cny'] or row['amount_kind'] != 'provider_usage_estimate'
                    or row['input_tokens'] != outcome['input_tokens'] or row['output_tokens'] != 0 or observed is None):
                raise ValueError('existing reconciliation differs from verified receipt')
            expected = dict(attempt_id=attempt, mode='live', category='success', http_status=200,
                returned_model=prep.MODEL, input_tokens=outcome['input_tokens'], output_tokens=0,
                elapsed_ms=0, estimated_usd=None)
            if any(observed[k] != v for k, v in expected.items()):
                raise ValueError('existing provider observation differs from verified receipt')
            prep.timestamp(observed['observed_at'])
        else:
            raise ValueError('unknown parent state; preserve holds for manual review')
        states[attempt] = row['status']
    return states


@contextmanager
def output_lock(directory):
    path = directory / '.reconcile.lock'
    if path.is_symlink():
        raise ValueError('lock symlink forbidden')
    with path.open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush(); os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def frozen_write(path, value):
    if path.exists():
        if path.is_symlink() or _strict_json(path.read_bytes()) != value:
            raise ValueError('existing reconciliation checkpoint differs')
    else:
        prep.save_exclusive(path, value)


def verify_files(context):
    for name, expected in context['input_hashes'].items():
        if prep.sha(Path(name).read_bytes()) != expected:
            raise ValueError('verified receipt or grant file changed')


def reconcile(ledger_path, grant_path, receipts_path, output, *, apply=False):
    context = inputs(ledger_path, grant_path, receipts_path)
    states = ledger_state(context)  # All three rows and receipts validate first.
    result = dict(state='validated_no_changes', grant_sha256=context['grant_sha256'],
        verified_receipts=3, original_reserved_micro_cny=150000,
        usage_estimate_micro_cny=sum(r['accounted_micro_cny'] for r in context['outcomes']),
        original_attempts_only=True, new_reservations=0, provider_requests=0,
        historical_rows_unchanged=True, actual_bill_reconciled=False)
    if apply is not True:
        return result
    destination = Path(output).absolute()
    if destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError('regular reconciliation output directory required')
    existed = destination.exists()
    if not existed:
        destination.mkdir()
    elif not destination.is_dir() or not (destination / 'PLAN.json').is_file():
        raise ValueError('existing output without matching reconciliation plan cannot be adopted')
    plan = dict(version=1, grant_id=context['grant']['grant_id'], grant_sha256=context['grant_sha256'],
        ledger_path=str(context['ledger']), input_hashes=context['input_hashes'],
        original_history_sha256=context['metadata']['existing_history_sha256'],
        outcomes=context['outcomes'], new_reservations=0, provider_requests=0)
    with output_lock(destination):
        frozen_write(destination / 'PLAN.json', plan)
        verify_files(context); ledger_state(context)
        ledger = BudgetLedger(str(context['ledger']), cap_cny='500')
        for outcome in context['outcomes']:
            attempt = outcome['parent_attempt_id']
            verify_files(context); state = ledger_state(context)[attempt]
            if state == 'reserved':
                ledger.mark_dispatched(attempt)
                state = 'dispatched'
            if state == 'dispatched':
                ledger.observe_provider(attempt, mode='live', category='success', http_status=200,
                    returned_model=prep.MODEL, input_tokens=outcome['input_tokens'], output_tokens=0,
                    elapsed_ms=0, estimated_usd=None)
            if ledger_state(context)[attempt] != 'settled':
                raise ValueError('original attempt accounting did not complete')
            frozen_write(destination / ('slot-' + str(outcome['ordinal']) + '.json'),
                dict(version=1, grant_sha256=context['grant_sha256'], state='usage_recorded', **outcome))
        verify_files(context)
        if set(ledger_state(context).values()) != {'settled'}:
            raise ValueError('all three original attempts must reconcile')
        result.update(state='reconciled_usage_estimate', reconciled_attempts=3,
            model=prep.MODEL, receipt_sha256=[r['receipt_sha256'] for r in context['outcomes']],
            elapsed_ms_policy='zero sentinel; remote export contains no elapsed timing',
            report_file=str(destination / 'reconciliation.json'))
        frozen_write(destination / 'reconciliation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--grant', required=True)
    parser.add_argument('--receipts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = reconcile(args.ledger, args.grant, args.receipts, args.output, apply=args.apply)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        raise SystemExit('Embedding reconciliation stopped: ' + type(exc).__name__ +
                         '; retain receipts and original holds; no provider retry or new reservation.') from None


if __name__ == '__main__':
    main()
