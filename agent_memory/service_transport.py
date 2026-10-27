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
from .embedding_executor import _Journal, _ReceiptHTTP, _json, EmbeddingExecutorBlocked
from .provider_http import (BudgetedJSONTransport, NativeHTTPS, SHARED_LEDGER, ProviderFailure,
                            OPENAI_ENDPOINT, BAILIAN_BEIJING_ENDPOINT)
from .providers import source_card_response_format


def history_digest(rows):
    return hashlib.sha256(_json([list(row) for row in rows]).encode()).hexdigest()


def omit_nontext_quote_metadata(value):
    """Derived view only: omit unusable auxiliary entries, never rewrite evidence."""
    result = json.loads(json.dumps(value))
    try:
        message = result['choices'][0]['message']
        card = json.loads(message['content'])
        quotes = card.get('evidence_quotes')
        if not isinstance(quotes,list):
            # This field is optional ranking metadata, never returned evidence.
            # Do not interpret object values or coerce them into quotations.
            card['evidence_quotes'] = []
            card['omitted_quote_container_type'] = type(quotes).__name__
            message['content'] = _json(card)
            return result
        strings = [q for q in quotes if isinstance(q,str)]
        if len(strings) != len(quotes):
            card['evidence_quotes'] = strings
            card['omitted_nontext_quote_count'] = len(quotes)-len(strings)
            message['content'] = _json(card)
    except (KeyError,IndexError,TypeError,ValueError,AttributeError):
        raise ProviderFailure('invalid_contract') from None
    return result


class ServiceBudgetedTransport(BudgetedJSONTransport):
    def _bounds(self, payload):
        bounds = super()._bounds(payload)
        if self.model=='gpt-4o-mini' and payload.get('response_format') not in (
                {'type':'json_object'},source_card_response_format()):
            raise ValueError('reviewed source-card JSON format required')
        return bounds

    def _contract(self, data, payload):
        view = omit_nontext_quote_metadata(data) if self.model=='gpt-4o-mini' else data
        super()._contract(view,payload)
        if self.model=='gpt-4o-mini':
            card=json.loads(view['choices'][0]['message']['content'])
            if not card['summary'].strip():raise ProviderFailure('invalid_contract')


def acknowledged_failure(row, proof):
    return (proof is not None and row['state']=='failed'
        and row['receipt_policy']=='discarded_known_failure' and row['body'] is None
        and row['redacted']==1 and row['status']==200 and not row['transport_completed']
        and proof['attempt_id']==row['attempt_id'] and proof['body_sha']==row['body_sha'])


def never_dispatched(row):
    return (row['attempt_id'] is None and row['status'] is None and row['body'] is None
        and row['body_sha'] is None and row['redacted'] is None and not row['transport_completed']
        and row['run_path'] is None and row['run_ordinal'] is None)


class _ServiceJournal(_Journal):
    extra_tables = frozenset({'discarded_auxiliary_failures','undispatched_retirements'})

    def start(self, fingerprint):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT * FROM operations WHERE fingerprint=?',(fingerprint,)).fetchone()
            if current is None or current['state']!='ready' or not never_dispatched(current):
                raise EmbeddingExecutorBlocked('operation already attempted; no automatic replay')
            for row in db.execute("SELECT * FROM operations WHERE scope=? AND state NOT IN ('ready','complete')",(current['scope'],)):
                proof=db.execute('SELECT * FROM discarded_auxiliary_failures WHERE fingerprint=?',(row['fingerprint'],)).fetchone()
                retired=db.execute('SELECT 1 FROM undispatched_retirements WHERE fingerprint=?',(row['fingerprint'],)).fetchone()
                if acknowledged_failure(row,proof):continue
                if row['state']=='abandoned_before_dispatch' and retired and never_dispatched(row):continue
                raise EmbeddingExecutorBlocked('unresolved service operation blocks further dispatch')
            db.execute("UPDATE operations SET state='inflight' WHERE fingerprint=?",(fingerprint,))


class ServiceTransport:
    mode = 'live'
    transport_type = ServiceBudgetedTransport
    output_token_cap = 512

    def normalize_result(self, value, model):
        return omit_nontext_quote_metadata(value) if model=='gpt-4o-mini' else value

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
        self.journal = _ServiceJournal(journal_root, offline_test)
        with self.journal.connect() as db:
            columns = {row[1] for row in db.execute('PRAGMA table_info(operations)')}
            if 'receipt_policy' not in columns:
                db.execute("ALTER TABLE operations ADD COLUMN receipt_policy TEXT NOT NULL DEFAULT 'retain'")
            db.execute('''CREATE TABLE IF NOT EXISTS discarded_auxiliary_failures (
                fingerprint TEXT PRIMARY KEY, attempt_id TEXT UNIQUE NOT NULL,
                body_sha TEXT NOT NULL, summary_sha TEXT NOT NULL,
                normalized_sha TEXT NOT NULL, discarded_at TEXT NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS undispatched_retirements (
                fingerprint TEXT PRIMARY KEY, retired_at TEXT NOT NULL)''')
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
            discarded = {row['fingerprint']:dict(row) for row in db.execute('SELECT * FROM discarded_auxiliary_failures')}
            retired = {row[0] for row in db.execute('SELECT fingerprint FROM undispatched_retirements')}
        for row in rows:
            proof = discarded.get(row['fingerprint'])
            if acknowledged_failure(row,proof):continue
            if row['state']=='abandoned_before_dispatch' and row['fingerprint'] in retired and never_dispatched(row):continue
            if row['state'] != 'complete':
                raise ValueError('unresolved model attempt blocks further calls; no replay')
            if (row['receipt_policy'] != 'usage_only' or row['body'] is not None
                    or row['redacted'] != 1 or not row['transport_completed']
                    or not isinstance(row['body_sha'], str) or len(row['body_sha']) != 64):
                raise ValueError('invalid minimized receipt')
        if self.ledger.totals()['remaining_micro_cny'] < 5000000:
            raise ValueError('insufficient bounded budget for a complete admitted Add')

    def retire_undispatched(self, fingerprint):
        """Operator-only: preserve a prepared operation proven never sent to a model."""
        with self.journal.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM operations WHERE fingerprint=?',(fingerprint,)).fetchone()
            if row is None or row['state']!='ready' or not never_dispatched(row):
                raise ValueError('only pristine prepared operations may be retired; never replay unknowns')
            db.execute('INSERT INTO undispatched_retirements VALUES(?,?)',
                       (fingerprint,datetime.now(timezone.utc).isoformat()))
            db.execute("UPDATE operations SET state='abandoned_before_dispatch' WHERE fingerprint=?",(fingerprint,))
        return {'fingerprint':fingerprint,'model_calls':0,'operation_preserved':True}

    def discard_known_auxiliary_failure(self, attempt_id, expected_body_sha):
        """Explicit operator acknowledgement only; never replay or resume an Add.

        Validate a known HTTP200 response under the metadata-only correction,
        retain its failed journal/ledger status and append proof before removing
        the raw copy. Unknown calls, substantive errors and other failures remain
        blocking. This method is never invoked by request handling.
        """
        with self.lock, self.journal.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute('SELECT * FROM operations WHERE attempt_id=?',(attempt_id,)).fetchall()
            if len(rows)!=1:raise ValueError('one exact known attempt required')
            row=dict(rows[0])
            if (row['state']!='failed' or row['status']!=200 or row['redacted']
                    or row['transport_completed'] or not row['body']
                    or row['body_sha']!=expected_body_sha
                    or hashlib.sha256(row['body']).hexdigest()!=expected_body_sha
                    or json.loads(row['manifest']).get('model')!='gpt-4o-mini'):
                raise ValueError('known unredacted GPT response required')
            original=json.loads(row['body'])
            card=json.loads(original['choices'][0]['message']['content'])
            if not isinstance(card,dict) or isinstance(card.get('evidence_quotes'),list):
                raise ValueError('only invalid auxiliary containers can be discarded')
            validator=object.__new__(ServiceBudgetedTransport);validator.model='gpt-4o-mini'
            returned_model=validator._response_model(original)
            usage=validator._usage(original)
            validator._contract(original,{})
            view=omit_nontext_quote_metadata(original)
            normalized=json.loads(view['choices'][0]['message']['content'])
            if card['summary']!=normalized['summary'] or normalized['evidence_quotes']!=[]:
                raise ValueError('substantive content must be unchanged')
            with self.ledger.connect() as ledger:
                observed=ledger.execute('''SELECT category,http_status,returned_model,input_tokens,output_tokens
                    FROM provider_observations WHERE attempt_id=?''',(attempt_id,)).fetchone()
            if observed!=('invalid_contract',200,returned_model,*usage):
                raise ValueError('matching recorded contract failure and usage required')
            proof=dict(fingerprint=row['fingerprint'],attempt_id=attempt_id,body_sha=expected_body_sha,
                summary_sha=hashlib.sha256(card['summary'].encode()).hexdigest(),
                normalized_sha=hashlib.sha256(_json(view).encode()).hexdigest(),
                discarded_at=datetime.now(timezone.utc).isoformat())
            db.execute('INSERT INTO discarded_auxiliary_failures VALUES (:fingerprint,:attempt_id,:body_sha,:summary_sha,:normalized_sha,:discarded_at)',proof)
            db.execute("UPDATE operations SET body=NULL,redacted=1,receipt_policy='discarded_known_failure' WHERE fingerprint=?",(row['fingerprint'],))
            return dict(proof,original_failure_preserved=True,model_calls=0,failed_add_resumed=False)

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
            transport = self.transport_type(model=model, endpoint=endpoint, ledger=self.ledger,
                rate=rate, credential_reader=self.readers.get(model), http=receipt_http,
                mode=self.mode, enabled=True, prices_confirmed=True, timeout_seconds=120,
                fixed_reserve_cny='.25' if model == 'gpt-4o-mini' else None,
                usd_rates=('.15','.60') if model == 'gpt-4o-mini' else None,
                attempt_observer=lambda attempt:self.journal.attach(fingerprint, attempt),
                output_token_cap=self.output_token_cap)
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
            return self.normalize_result(value, model)
