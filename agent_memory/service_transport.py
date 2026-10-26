"""Receipt-first service transport using the migrated original budget ledger.

No second allowance or experiment-grant reset. Successful raw responses are
discarded after validation; accounting keeps only hashes and usage metadata.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import threading
import uuid

from .budget import BudgetLedger, VerifiedRate
from .embedding_executor import _Journal, _ReceiptHTTP, _json
from .provider_http import (BudgetedJSONTransport, NativeHTTPS, SHARED_LEDGER, ProviderFailure,
                            OPENAI_ENDPOINT, BAILIAN_BEIJING_ENDPOINT)


def history_digest(rows):
    return hashlib.sha256(_json([list(row) for row in rows]).encode()).hexdigest()


def omit_nontext_quote_metadata(value):
    """Derived view only: omit unusable auxiliary entries, never rewrite evidence."""
    result = json.loads(json.dumps(value))
    try:
        message = result['choices'][0]['message']
        card = json.loads(message['content'])
        quotes = card['evidence_quotes']
        if not isinstance(quotes,list):
            raise ValueError('quote array required')
        strings = [q for q in quotes if isinstance(q,str)]
        if len(strings) != len(quotes):
            card['evidence_quotes'] = strings
            card['omitted_nontext_quote_count'] = len(quotes)-len(strings)
            message['content'] = _json(card)
    except (KeyError,IndexError,TypeError,ValueError):
        raise ProviderFailure('invalid_contract') from None
    return result


class ServiceBudgetedTransport(BudgetedJSONTransport):
    def _contract(self, data, payload):
        view = omit_nontext_quote_metadata(data) if self.model=='gpt-4o-mini' else data
        return super()._contract(view,payload)


class ServiceTransport:
    mode = 'live'

    def __init__(self, ledger_path, authority_path, prices_path, journal_root, readers,
                 *, http=None, offline_test=False):
        self.ledger_path = Path(ledger_path).resolve()
        self.authority_path = Path(authority_path)
        self.prices_path = Path(prices_path)
        self.offline_test = offline_test
        self.mode = 'offline_fake' if offline_test else 'live'
        self.http = http or NativeHTTPS()
        if bool(getattr(self.http, 'is_fake_http', False)) != offline_test:
            raise ValueError('HTTP execution mode mismatch')
        if not self.ledger_path.is_file() or self.ledger_path.is_symlink():
            raise ValueError('existing original ledger required; never create a service budget')
        if not offline_test and self.ledger_path != SHARED_LEDGER.resolve():
            raise ValueError('service must own the original shared ledger')
        authority = json.loads(self.authority_path.read_text())
        if authority.get('active_owner') != 'server' or authority.get('cap_micro_cny') != 500000000:
            raise ValueError('verified ledger ownership transfer required')
        self.ledger = BudgetLedger(self.ledger_path)
        with self.ledger.connect() as db:
            for table in ('attempts', 'provider_observations'):
                count = authority['history'][table]['rows']
                if type(count) is not int or count < 0:
                    raise ValueError('invalid preserved ledger prefix')
                rows = db.execute('SELECT rowid,* FROM '+table+' ORDER BY rowid LIMIT ?', (count,)).fetchall()
                if len(rows) != count or history_digest(rows) != authority['history'][table]['sha256']:
                    raise ValueError('original budget history changed')
        self.authority_sha = hashlib.sha256(self.authority_path.read_bytes()).hexdigest()
        self.journal = _Journal(journal_root, offline_test)
        with self.journal.connect() as db:
            columns = {row[1] for row in db.execute('PRAGMA table_info(operations)')}
            if 'receipt_policy' not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN receipt_policy TEXT NOT NULL DEFAULT 'retain'")
        self.readers = readers
        self.lock = threading.Lock()

    def prices(self):
        value = json.loads(self.prices_path.read_text())
        if set(value) != {'checked_at', 'models'} or set(value['models']) != {'gpt-4o-mini', 'text-embedding-v4'}:
            raise ValueError('two fixed provider prices required')
        checked = datetime.fromisoformat(value['checked_at'])
        if checked.tzinfo is None or not 0 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 172800:
            raise ValueError('provider price verification must be refreshed within48hours')
        for model, expected in [('gpt-4o-mini', (Decimal('1.5'), Decimal('6'))),
                                ('text-embedding-v4', (Decimal('.5'), Decimal('0')))]:
            rate = VerifiedRate(model=model, checked_at=value['checked_at'], **value['models'][model])
            rate.validate()
            if (Decimal(rate.input_cny_per_million), Decimal(rate.output_cny_per_million)) != expected:
                raise ValueError('price change requires reviewed reservation and USD-accounting update')
        return value

    def ready(self):
        if hashlib.sha256(self.authority_path.read_bytes()).hexdigest() != self.authority_sha:
            raise ValueError('budget ownership changed')
        self.prices()
        with self.journal.connect() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM operations')]
        for row in rows:
            if row['state'] != 'complete':
                raise ValueError('unresolved model attempt blocks further calls; no replay')
            if (row['receipt_policy'] != 'usage_only' or row['body'] is not None
                    or row['redacted'] != 1 or not row['transport_completed']
                    or not isinstance(row['body_sha'], str) or len(row['body_sha']) != 64):
                raise ValueError('invalid minimized receipt')
        if self.ledger.totals()['remaining_micro_cny'] < 5000000:
            raise ValueError('insufficient bounded budget for a complete admitted Add')

    def __call__(self, endpoint, payload):
        with self.lock:
            self.ready()
            model = payload.get('model')
            endpoints = {'gpt-4o-mini': OPENAI_ENDPOINT, 'text-embedding-v4': BAILIAN_BEIJING_ENDPOINT}
            if model not in endpoints or endpoint != endpoints[model]:
                raise ValueError('fixed academic models and endpoints required')
            config = self.prices()
            rate = VerifiedRate(model=model, checked_at=config['checked_at'], **config['models'][model])
            fingerprint = uuid.uuid4().hex
            manifest = {'keys':[fingerprint], 'model':model}
            receipt_http = _ReceiptHTTP(self.http, self.journal, fingerprint, payload, None)
            receipt_http.endpoint = endpoint
            transport = ServiceBudgetedTransport(model=model, endpoint=endpoint, ledger=self.ledger,
                rate=rate, credential_reader=self.readers.get(model), http=receipt_http,
                mode=self.mode, enabled=True, prices_confirmed=True, timeout_seconds=120,
                fixed_reserve_cny='.25' if model == 'gpt-4o-mini' else None,
                usd_rates=('.15','.60') if model == 'gpt-4o-mini' else None,
                attempt_observer=lambda attempt:self.journal.attach(fingerprint, attempt),
                output_token_cap=512)
            transport._bounds(payload)
            if model == 'text-embedding-v4' and payload.get('dimensions') != 256:
                raise ValueError('fixed 256-dimensional vectors required')
            self.journal.install(fingerprint, 'official-service-v1', manifest)
            self.journal.start(fingerprint)
            try:
                value = transport(endpoint, payload)
            except BaseException:
                with self.journal.connect() as db:
                    db.execute("UPDATE operations SET state=CASE WHEN state='received' THEN 'failed' ELSE 'unknown' END WHERE fingerprint=?", (fingerprint,))
                raise
            with self.journal.connect() as db:
                row = db.execute('SELECT * FROM operations WHERE fingerprint=?', (fingerprint,)).fetchone()
                if row['state'] != 'received' or row['redacted'] or not row['body']:
                    raise ValueError('complete unredacted receipt required before minimization')
                if hashlib.sha256(row['body']).hexdigest() != row['body_sha']:
                    raise ValueError('received body hash mismatch')
                # The model response is returned to the memory pipeline in RAM only.
                # Keep no query vectors, source-card response copies, or raw content logs.
                db.execute("UPDATE operations SET state='complete',body=NULL,redacted=1,transport_completed=1,receipt_policy='usage_only' WHERE fingerprint=?", (fingerprint,))
            return omit_nontext_quote_metadata(value) if model=='gpt-4o-mini' else value
