"""Reserve three parent-ledger embedding smoke holds; never dispatch a request.

An existing CNY500 ledger is mandatory. A new output directory becomes a durable
one-shot boundary before any reservation; interrupted preparation retains holds
and refuses automatic replay. Reconciliation is a separate receipt-audit action.
"""
import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.embedding_executor import _json
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT

MODEL = 'text-embedding-v4'
PRICE_SOURCE = 'https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai'
APPLICATION_ID = 0x414D4247
CAP = 500_000_000
HOLD = 50_000
SCHEMA = {
    'budget': ('id', 'cap'),
    'attempts': ('id', 'retry_of', 'model', 'status', 'created_at', 'max_input', 'max_output',
        'input_rate', 'output_rate', 'price_source', 'price_checked_at', 'reserved',
        'input_tokens', 'output_tokens', 'accounted', 'amount_kind', 'conversion_basis'),
    'provider_observations': ('attempt_id', 'observed_at', 'mode', 'category', 'http_status',
        'returned_model', 'input_tokens', 'output_tokens', 'elapsed_ms', 'estimated_usd'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def timestamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError('timezone-aware ISO price check timestamp required')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('invalid ISO price check timestamp') from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('price check timezone required')
    return result.astimezone(timezone.utc)


def existing_ledger(path):
    original = Path(path)
    if not original.is_absolute() or original.is_symlink():
        raise ValueError('absolute existing regular ledger file required')
    resolved = original.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError('existing regular ledger file required')
    return resolved


def read_ledger(path):
    """Read-only admission: never let BudgetLedger create/migrate a wrong DB."""
    path = existing_ledger(path)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        if (db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
                or db.execute('PRAGMA user_version').fetchone()[0] != 2):
            raise ValueError('existing budget application and schema version required')
        objects = db.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        if (set(name for kind, name in objects if kind == 'table') != set(SCHEMA)
                or any(kind in ('trigger', 'view') for kind, _ in objects)):
            raise ValueError('unexpected ledger schema objects')
        for table, expected in SCHEMA.items():
            columns = tuple(row[1] for row in db.execute('PRAGMA table_info(' + table + ')'))
            if columns != expected:
                raise ValueError('existing ledger schema differs')
        if db.execute('SELECT id,cap FROM budget').fetchall() != [(1, CAP)]:
            raise ValueError('existing shared CNY500 cap required')
        attempts = db.execute('SELECT rowid,* FROM attempts ORDER BY rowid').fetchall()
        observations = db.execute('SELECT rowid,* FROM provider_observations ORDER BY rowid').fetchall()
        if db.execute("SELECT 1 FROM attempts WHERE status='needs_reconciliation' LIMIT 1").fetchone():
            raise ValueError('unreconciled ledger anomaly blocks reservations')
        used = db.execute('SELECT coalesce(sum(accounted),0) FROM attempts').fetchone()[0]
        if type(used) is not int or not 0 <= used <= CAP:
            raise ValueError('invalid existing accounted total')
    return dict(path=path, attempts=attempts, observations=observations, accounted=used,
                history_sha256=sha(_json([attempts, observations]).encode('utf-8')))


def expected_payloads():
    from deploy.embedding_smoke_test import expected_embedding_payloads
    return expected_embedding_payloads()


def validate_payloads(payloads):
    if not isinstance(payloads, list) or len(payloads) != 3:
        raise ValueError('exact three fixed smoke payloads required')
    for payload, count in zip(payloads, (2, 1, 1)):
        if (not isinstance(payload, dict) or set(payload) != {'model', 'input', 'dimensions', 'encoding_format'}
                or payload['model'] != MODEL or type(payload['dimensions']) is not int
                or payload['dimensions'] != 256 or payload['encoding_format'] != 'float'
                or not isinstance(payload['input'], list) or len(payload['input']) != count
                or any(not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 8192
                       for text in payload['input'])):
            raise ValueError('fixed source2/query1/query1 embedding256 inputs required')
    if len({_json(payload) for payload in payloads}) != 3:
        raise ValueError('all three smoke request payloads must be distinct')
    # Detach from mutable caller-owned containers and reject nonfinite values.
    return json.loads(_json(payloads))


def save_exclusive(path, value):
    raw = (_json(value) + '\n').encode('utf-8')
    with Path(path).open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    if os.name != 'nt':
        descriptor = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return sha(raw)


def prepare(ledger_path, output, price_checked_at, *, reserve_live=False, now=None, payloads=None):
    if reserve_live is not True:
        raise ValueError('explicit --reserve-live required; no reservations made')
    current = now or datetime.now(timezone.utc)
    if not isinstance(current, datetime) or current.tzinfo is None or current.utcoffset() is None:
        raise ValueError('timezone-aware current time required')
    current = current.astimezone(timezone.utc)
    checked = timestamp(price_checked_at)
    if checked.date() != current.date() or checked > current:
        raise ValueError('official price must have been checked today, not in the future')
    payloads = validate_payloads(expected_payloads() if payloads is None else payloads)
    before = read_ledger(ledger_path)
    if CAP - before['accounted'] < 3 * HOLD:
        raise ValueError('shared ledger lacks the complete three-request hold')
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('output already exists; never replay a preparation')
    if not destination.parent.is_dir():
        raise ValueError('output parent must already exist')
    ledger_file = before['path']
    identity = ledger_file.stat()
    grant_id = str(uuid.uuid4())
    destination.mkdir()  # Exclusive marker boundary, even before marker writing.
    marker = dict(version=1, grant_id=grant_id, state='preparing_reservations_not_dispatched',
        created_at=current.isoformat(), ledger_path=str(ledger_file),
        existing_attempts=len(before['attempts']), existing_history_sha256=before['history_sha256'],
        planned_request_reservations=3, maximum_reserved_micro_cny=3 * HOLD,
        actual_provider_dispatches=0, automatic_retry=False)
    save_exclusive(destination / 'PREPARATION_STARTED.json', marker)
    latest = read_ledger(ledger_file)
    stat = ledger_file.stat()
    if (latest['history_sha256'] != before['history_sha256']
            or (identity.st_dev, identity.st_ino) != (stat.st_dev, stat.st_ino)):
        raise ValueError('ledger changed during preparation; marker retained, no replay')
    rate = VerifiedRate(MODEL, '.5', '0', PRICE_SOURCE, checked.isoformat())
    ledger = BudgetLedger(str(ledger_file), cap_cny='500')
    slots = []
    for ordinal, payload in enumerate(payloads, 1):
        max_input = 8192 * len(payload['input'])
        attempt_id = ledger.reserve(rate, max_input, 0, fixed_reserve_cny='.05')
        slot = dict(ordinal=ordinal, parent_attempt_id=attempt_id,
            payload_sha256=sha(_json(payload).encode('utf-8')), payload=payload,
            max_input_tokens=max_input, reserved_micro_cny=HOLD)
        slots.append(slot)
        save_exclusive(destination / ('reservation-' + str(ordinal) + '.json'),
            dict(version=1, grant_id=grant_id, state='reserved_not_dispatched',
                 completed_reservations=ordinal, actual_provider_dispatches=0, slots=list(slots)))
    after = read_ledger(ledger_file)
    prefix = len(before['attempts'])
    if (after['attempts'][:prefix] != before['attempts'] or after['observations'] != before['observations']
            or len(after['attempts']) != prefix + 3 or after['accounted'] != before['accounted'] + 3 * HOLD):
        raise ValueError('ledger changed beyond three reservations; holds retained for manual review')
    columns = ('rowid', *SCHEMA['attempts'])
    for slot, row in zip(slots, after['attempts'][prefix:]):
        values = dict(zip(columns, row))
        if (values['id'] != slot['parent_attempt_id'] or values['status'] != 'reserved'
                or values['reserved'] != HOLD or values['accounted'] != HOLD
                or values['amount_kind'] != 'fixed_budget_hold' or values['retry_of'] is not None
                or values['input_tokens'] is not None or values['output_tokens'] is not None
                or values['model'] != MODEL or values['max_input'] != slot['max_input_tokens']
                or values['max_output'] != 0 or values['input_rate'] != '.5' or values['output_rate'] != '0'):
            raise ValueError('parent reservation identity or state changed')
    grant = dict(version=1, grant_id=grant_id, model=MODEL, endpoint=BAILIAN_BEIJING_ENDPOINT,
        dimensions=256, price_source=PRICE_SOURCE, price_checked_at=checked.isoformat(),
        expires_at=(current + timedelta(hours=2)).isoformat(), slots=slots)
    grant_hash = save_exclusive(destination / 'grant.json', grant)
    metadata = dict(marker, state='reserved_not_dispatched', grant_sha256=grant_hash,
        completed_reservations=3, newly_reserved_micro_cny=3 * HOLD,
        historical_rows_unchanged=True, actual_remote_dispatches_confirmed=0,
        parent_reconciliation_required=True, remote_budget_created=False,
        remaining_micro_cny=CAP - after['accounted'])
    save_exclusive(destination / 'preparation.json', metadata)
    return dict(state=metadata['state'], grant_file=str(destination / 'grant.json'),
        grant_sha256=grant_hash, reserved_requests=3, reserved_micro_cny=3 * HOLD,
        actual_provider_dispatches=0, parent_reconciliation_required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    command = commands.add_parser('prepare')
    command.add_argument('--ledger', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--price-checked-at', required=True)
    command.add_argument('--reserve-live', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.ledger, args.output, args.price_checked_at, reserve_live=args.reserve_live)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        raise SystemExit('Embedding grant preparation stopped: ' + type(exc).__name__ +
                         '; preserve any existing output and holds, never retry that directory.') from None


if __name__ == '__main__':
    main()
