"""Finite parent-reserved model calls, with receipt-first remote accounting.

This is an execution outbox, not a second budget ledger. A pinned grant contains
one existing parent attempt per allowed request. Unknown calls stop the grant.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import time

from .embedding_executor import _Journal, _ReceiptHTTP, _json
from .provider_http import (BAILIAN_BEIJING_ENDPOINT, OPENAI_ENDPOINT,
                            BudgetedJSONTransport, NativeHTTPS, ProviderFailure)
from .visual_recovery import _strict_json

MODELS = {'gpt-4o-mini': (OPENAI_ENDPOINT, 250000, 128000, 512),
          'text-embedding-v4': (BAILIAN_BEIJING_ENDPOINT, 50000, 81920, 0)}


def validate_grant(value):
    if (not isinstance(value, dict) or value.get('version') != 1
            or not re.fullmatch('[a-f0-9]{32}', value.get('grant_id', ''))):
        raise ValueError('invalid online grant')
    checked = datetime.fromisoformat(value['price_checked_at'])
    expires = datetime.fromisoformat(value['expires_at'])
    now = datetime.now(timezone.utc)
    if (checked.tzinfo is None or expires.tzinfo is None
            or not checked <= now < expires or (expires-checked).total_seconds() > 172800):
        raise ValueError('online grant expired or future dated')
    slots = value['slots']
    if not isinstance(slots, list) or not 1 <= len(slots) <= 1000:
        raise ValueError('bounded parent slots required')
    seen = set()
    for ordinal, slot in enumerate(slots, 1):
        if (slot.get('ordinal') != ordinal or slot.get('model') not in MODELS
                or not re.fullmatch('attempt_[a-f0-9]{32}', slot.get('parent_attempt_id', ''))
                or slot['parent_attempt_id'] in seen
                or slot.get('reserved_micro_cny') != MODELS[slot['model']][1]):
            raise ValueError('invalid parent slot')
        seen.add(slot['parent_attempt_id'])
    return value


class PrepaidOnlineTransport:
    def __init__(self, grant_path, expected_sha256, journal_root, readers, *, http=None, offline_test=False):
        self.path = Path(grant_path)
        raw = self.path.read_bytes()
        if self.path.is_symlink() or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('trusted grant hash mismatch')
        self.hash = expected_sha256
        self.grant = validate_grant(_strict_json(raw))
        self.http = http or NativeHTTPS()
        if bool(getattr(self.http, 'is_fake_http', False)) != offline_test:
            raise ValueError('HTTP execution mode mismatch')
        self.readers = readers
        self.journal = _Journal(journal_root, offline_test)
        binding = self.journal.root/'online-grant-binding.json'
        expected = _json({'grant_sha256': self.hash})
        if binding.exists():
            if binding.read_text() != expected:
                raise ValueError('cannot replace a bound grant')
        else:
            with self.journal.connect() as db:
                if db.execute('SELECT 1 FROM operations LIMIT 1').fetchone():
                    raise ValueError('existing unbound outbox')
            binding.write_text(expected)
        with self.journal.connect() as db:
            columns = {r[1] for r in db.execute('PRAGMA table_info(operations)')}
            if 'online_observation' not in columns:
                db.execute('ALTER TABLE operations ADD COLUMN online_observation TEXT')
            if 'online_payload_sha256' not in columns:
                db.execute('ALTER TABLE operations ADD COLUMN online_payload_sha256 TEXT')
        self.slots = []
        for slot in self.grant['slots']:
            manifest = dict(slot, keys=[slot['parent_attempt_id']], grant_sha256=self.hash)
            fp = hashlib.sha256(_json(manifest).encode()).hexdigest()
            self.journal.install(fp, self.hash, manifest)
            self.slots.append((slot, fp))

    def __call__(self, endpoint, payload):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.hash:
            raise ValueError('grant changed')
        validate_grant(self.grant)
        model = payload.get('model')
        if model not in MODELS or endpoint != MODELS[model][0]:
            raise ValueError('model or endpoint outside grant')
        validator = SimpleNamespace(model=model, output_token_cap=512)
        bounds = BudgetedJSONTransport._bounds(validator, payload)
        if model == 'text-embedding-v4' and payload.get('dimensions') != 256:
            raise ValueError('fixed online embedding dimension required')
        if bounds[0] > MODELS[model][2] or bounds[1] > MODELS[model][3]:
            raise ValueError('parent reservation bounds exceeded')
        chosen = None
        with self.journal.connect() as db:
            states = {row['fingerprint']: dict(row) for row in db.execute('SELECT * FROM operations')}
        for slot, fp in self.slots:
            row = states[fp]
            if row['state'] not in ('ready', 'complete'):
                raise ValueError('unresolved attempt blocks all dispatch')
            if row['state'] == 'complete':
                if (not row['body'] or hashlib.sha256(row['body']).hexdigest() != row['body_sha']
                        or row['redacted'] or not row['transport_completed']):
                    raise ValueError('completed receipt corrupt')
            elif chosen is None and slot['model'] == model:
                chosen = slot, fp
        if chosen is None:
            raise ValueError('parent model grant exhausted')
        slot, fp = chosen
        key = self.readers[model]()
        if not isinstance(key, str) or not 16 <= len(key) <= 16384 or any(not 33 <= ord(c) <= 126 for c in key):
            raise ValueError('credential unavailable')
        raw = _json(payload).encode('utf-8')
        if len(raw) > 44*1024*1024:
            raise ValueError('provider request too large')
        self.journal.start(fp)
        self.journal.attach(fp, slot['parent_attempt_id'], str(self.journal.path.resolve()), slot['ordinal'])
        with self.journal.connect() as db:
            db.execute('UPDATE operations SET online_payload_sha256=? WHERE fingerprint=?',
                       (hashlib.sha256(raw).hexdigest(), fp))
        headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
        wrapped = _ReceiptHTTP(self.http, self.journal, fp, payload, None)
        wrapped.endpoint = endpoint
        tick = time.monotonic()
        observation = dict(category='network_error', http_status=None, returned_model=None,
                           input_tokens=None, output_tokens=None)
        try:
            reply = wrapped.exchange(endpoint, raw, headers, 120, 1024*1024)
            observation['http_status'] = reply.status
            if reply.status != 200:
                raise ProviderFailure('rate_limited' if reply.status == 429 else 'client_error' if reply.status < 500 else 'server_error')
            if self.journal.row(fp)['redacted']:
                raise ProviderFailure('invalid_contract')
            data = _strict_json(reply.body)
            allowed = {model, 'gpt-4o-mini-2024-07-18'} if model == 'gpt-4o-mini' else {model}
            if data.get('model') not in allowed:
                raise ProviderFailure('model_mismatch')
            observation['returned_model'] = data['model']
            inp, out = BudgetedJSONTransport._usage(validator, data)
            observation.update(input_tokens=inp, output_tokens=out)
            if inp > bounds[0] or out > bounds[1]:
                raise ProviderFailure('invalid_usage')
            BudgetedJSONTransport._contract(validator, data, payload)
            observation['category'] = 'success'
            return data
        except BaseException as exc:
            observation['category'] = getattr(exc, 'category', 'network_error')
            raise
        finally:
            observation['elapsed_ms'] = int((time.monotonic()-tick)*1000)
            with self.journal.connect() as db:
                row = db.execute('SELECT state FROM operations WHERE fingerprint=?', (fp,)).fetchone()
                state = 'complete' if observation['category'] == 'success' else 'failed' if row[0] == 'received' else 'unknown'
                db.execute('UPDATE operations SET state=?,online_observation=?,transport_completed=? WHERE fingerprint=?',
                           (state, _json(observation), int(state == 'complete'), fp))

    def summary(self):
        return [dict(slot, **{k: self.journal.row(fp)[k] for k in
                             ('state', 'status', 'body_sha', 'online_payload_sha256', 'online_observation')})
                for slot, fp in self.slots]
