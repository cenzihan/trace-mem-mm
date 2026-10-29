"""Receipt-first service transport using the migrated original budget ledger.

No second allowance or experiment-grant reset. Successful raw responses are
discarded after validation; accounting keeps only hashes and usage metadata.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import errno
import socket
import json
import os
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
                retry_permits=getattr(self,'retry_permits',None)
                if retry_permits and retry_permits(dict(row)):continue
                recovery=getattr(self,'operator_recovery',None)
                if recovery and recovery.permits(dict(row)):continue
                scoped=getattr(self,'scoped_unknown_permits',None)
                if scoped and scoped(dict(row)):continue
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
    receipt_scope = 'official-service-v1'
    scoped_unknown_enabled = False
    retry_failed_requests = False

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
        self.quarantine_path = Path(journal_root)/'operator-quarantines.jsonl'
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
        from .operator_recovery import OperatorRecovery
        self.operator_recovery = OperatorRecovery(Path(journal_root)/'operator-scoped-recovery.json',self.journal,self.ledger)
        self.journal.operator_recovery = self.operator_recovery
        if self.scoped_unknown_enabled:
            self.journal.scoped_unknown_permits = self._valid_scoped_unknown
        if self.retry_failed_requests:
            self.journal.retry_permits = self._accounted_terminal_failure
        self.readers = readers
        self.lock = threading.Lock()

    def quarantines(self):
        """Explicit operator abandonments; never a success or permission to replay."""
        if not self.quarantine_path.exists():return {}
        proofs={}
        for line in self.quarantine_path.read_text().splitlines():
            proof=json.loads(line);fingerprint=proof['fingerprint']
            if fingerprint in proofs:raise ValueError('duplicate quarantine proof')
            row=self.journal.row(fingerprint)
            if (not row or row['state']!='unknown' or row['status'] is not None or row['body'] is not None
                    or row['body_sha'] is not None or row['transport_completed']
                    or row['attempt_id']!=proof['attempt_id'] or row['scope']!=proof['scope']
                    or hashlib.sha256(_json(row).encode()).hexdigest()!=proof['row_sha256']
                    or not proof.get('reason') or len(proof.get('blocked_user_sha256',''))!=64):
                raise ValueError('unknown operation changed; quarantine invalid')
            with self.ledger.connect() as db:
                amount=db.execute('SELECT accounted,reserved FROM attempts WHERE id=?',(row['attempt_id'],)).fetchone()
                observed=db.execute('SELECT category,http_status FROM provider_observations WHERE attempt_id=?',(row['attempt_id'],)).fetchone()
            if not amount or amount[0]<amount[1] or observed not in [('tls_error',None),('timeout',None),('network_error',None)]:
                raise ValueError('unknown call must retain its full money hold and failure')
            proofs[fingerprint]=proof
        return proofs

    def _accounted_terminal_failure(self, row):
        # Retry permission does not settle or rewrite the old attempt.
        if row['state'] not in ('unknown','failed'):return False
        if not row['attempt_id']:return False
        with self.ledger.connect() as db:
            a=db.execute('SELECT accounted,reserved FROM attempts WHERE id=?',(row['attempt_id'],)).fetchone()
            o=db.execute('SELECT category FROM provider_observations WHERE attempt_id=?',(row['attempt_id'],)).fetchone()
        return bool(a and o and a[0]>=a[1] and o[0]!='success')

    def assert_source_allowed(self,payload):
        if self.retry_failed_requests:return
        self.operator_recovery.assert_user(payload['user_id'])
        user_sha=hashlib.sha256(payload['user_id'].encode()).hexdigest()
        if any(p['blocked_user_sha256']==user_sha for p in self.quarantines().values()):
            raise ValueError('retired uncertain input scope cannot be replayed')
        if self.scoped_unknown_enabled and self.has_scoped_unknown(user_sha):
            raise ValueError('user has an unresolved model request')

    def _scoped_unknown(self, row):
        from .scoped_unknown import verify_unknown
        with self.ledger.connect() as db:
            a=db.execute('SELECT id,status,reserved,accounted,model FROM attempts WHERE id=?',(row['attempt_id'],)).fetchone()
            o=db.execute('SELECT attempt_id,http_status,category FROM provider_observations WHERE attempt_id=?',(row['attempt_id'],)).fetchone()
        if a is None or o is None:raise ValueError('missing scoped unknown accounting')
        owner,payload=verify_unknown(row,dict(zip(('id','status','reserved','accounted','model'),a)),
            dict(zip(('attempt_id','http_status','category'),o)))
        return owner,payload,a[4]

    def _valid_scoped_unknown(self, row):
        if not self.scoped_unknown_enabled or row['state']!='unknown':return False
        try:self._scoped_unknown(row)
        except (ValueError,TypeError):return False
        return True

    def has_scoped_unknown(self, owner, *, model=None):
        with self.journal.connect() as db:
            rows=[dict(row) for row in db.execute("SELECT * FROM operations WHERE state='unknown'")]
        for row in rows:
            if not self._valid_scoped_unknown(row):continue
            candidate,_,kind=self._scoped_unknown(row)
            if candidate==owner and (model is None or kind==model):return True
        return False

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
        quarantines=self.quarantines()
        recovery_proofs=self.operator_recovery.proofs()
        with self.journal.connect() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM operations')]
            discarded = {row['fingerprint']:dict(row) for row in db.execute('SELECT * FROM discarded_auxiliary_failures')}
            retired = {row[0] for row in db.execute('SELECT fingerprint FROM undispatched_retirements')}
        for row in rows:
            if self.retry_failed_requests and self._accounted_terminal_failure(row):continue
            recovery=recovery_proofs.get(row['fingerprint'])
            if recovery:
                from .operator_recovery import row_digest
                if row_digest(row)!=recovery['row_sha256']:raise ValueError('recovery row changed')
                continue
            if self._valid_scoped_unknown(row):continue
            proof_unknown=quarantines.get(row['fingerprint'])
            if proof_unknown and self.receipt_scope!=proof_unknown['scope']:continue
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

    def record_transport_failure(self, fingerprint, receipt_http, payload):
        """Append bounded scalar diagnostics for this attempt, never infer dispatch.

        Missing phases stay unknown. This record does not settle money, change
        journal state, permit replay, or add diagnostics to historical attempts.
        """
        phase=getattr(receipt_http,'failure_phase',None)
        phases={'tls_setup','post_started','response_headers','response_body','deadline_unknown'}
        kind=getattr(receipt_http,'failure_kind',None)
        kinds={'SSLCertVerificationError','SSLError','gaierror','TimeoutError','ConnectionResetError',
            'ConnectionAbortedError','BrokenPipeError','RemoteDisconnected','IncompleteRead',
            'BadStatusLine','OSError','OtherTransportError','HTTPXTimeout',
            'HTTPXRemoteProtocolError','HTTPXConnectError','HTTPXReadError','HTTPXWriteError'}
        diagnostic={}
        raw=getattr(receipt_http,'failure_diagnostics',None)
        if isinstance(raw,dict):
            size=raw.get('received_bytes');status=raw.get('response_status')
            if type(size) is int and 0<=size<=8*1024*1024:diagnostic['received_bytes']=size
            if type(status) is int and 100<=status<=599:diagnostic['response_status']=status
            code=raw.get('error_errno')
            known=set(errno.errorcode)
            known.update(getattr(socket,name) for name in dir(socket)
                         if name.startswith('EAI_') and type(getattr(socket,name)) is int)
            if type(code) is int and code in known:diagnostic['error_errno']=code
            for key in ('sse_done_frame','sse_complete','http2'):
                if type(raw.get(key)) is bool:diagnostic[key]=raw[key]
        row={'fingerprint':fingerprint,'phase':phase if isinstance(phase,str) and phase in phases else None,
            'transport_kind':kind if isinstance(kind,str) and kind in kinds else None,
            'request_bytes':len(_json(payload).encode()),'streaming':payload.get('stream') is True,
            'at':datetime.now(timezone.utc).isoformat(),**diagnostic}
        line=(_json(row)+'\n').encode('utf-8')
        path=self.quarantine_path.parent/'transport-failure-phases.jsonl'
        # Called under the service lock after durable failure classification.
        # A full/unwritable diagnostic sink must not mask the original failure.
        try:
            with path.open('ab') as f:
                if os.fstat(f.fileno()).st_size+len(line)>256*1024:return False
                f.write(line);f.flush();os.fsync(f.fileno())
        except OSError:return False
        return True

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
            payload_sha=hashlib.sha256(_json(payload).encode()).hexdigest()
            owner = None
            if self.scoped_unknown_enabled:
                from .scoped_unknown import owner_digest
                owner=owner_digest()
            if self.retry_failed_requests:
                pass  # Human-authorized replay, with a fresh billed attempt.
            elif model == 'text-embedding-v4' and owner is not None:
                self.operator_recovery.assert_embedding_owner(payload_sha, owner)
            else:
                self.operator_recovery.assert_payload(payload_sha)
            if self.scoped_unknown_enabled and not self.retry_failed_requests:
                if self.has_scoped_unknown(owner):
                    from .pre_dispatch import PreDispatchRejection
                    raise PreDispatchRejection('user has an unresolved model request')
                with self.journal.connect() as db:
                    unresolved=[dict(row) for row in db.execute("SELECT * FROM operations WHERE state='unknown'")]
                for row in unresolved:
                    if (self._valid_scoped_unknown(row)
                            and (model != 'text-embedding-v4' or self._scoped_unknown(row)[0]==owner)
                            and self._scoped_unknown(row)[1]==payload_sha):
                        from .pre_dispatch import PreDispatchRejection
                        raise PreDispatchRejection('unknown model request cannot be replayed')
            for proof in ([] if self.retry_failed_requests else self.quarantines().values()):
                original=json.loads(self.journal.row(proof['fingerprint'])['manifest'])
                if original.get('payload_sha256')==payload_sha:
                    from .pre_dispatch import PreDispatchRejection
                    raise PreDispatchRejection('retired unknown model request cannot be replayed')
            manifest = {'keys':[fingerprint], 'model':model,'payload_sha256':payload_sha,
                'request_bytes':len(_json(payload).encode()),'streaming':payload.get('stream') is True}
            if self.scoped_unknown_enabled:manifest['owner_sha256']=owner
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
            self.journal.install(fingerprint, self.receipt_scope, manifest)
            self.journal.start(fingerprint)
            try:
                value = transport(endpoint, payload)
            except BaseException:
                with self.journal.connect() as db:
                    db.execute("UPDATE operations SET state=CASE WHEN state='received' THEN 'failed' ELSE 'unknown' END WHERE fingerprint=?", (fingerprint,))
                self.record_transport_failure(fingerprint,receipt_http,payload)
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
