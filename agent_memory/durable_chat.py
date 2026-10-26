"""One-attempt chat requests using the existing receipt-first journal mechanics.

The private journal implementation is shared with the embedding executor, but
claim keys here identify frozen reader/judge requests, never embedding chunks.
Unknown outcomes or interrupted accounting require inspection, not replay.
"""
import hashlib
from pathlib import Path
from .development_run import ScopedTransport
from .embedding_executor import _Journal, _ReceiptHTTP, _json
from .provider_http import BudgetedJSONTransport, OPENAI_ENDPOINT
from .visual_recovery import _strict_json


class DurableChatBlocked(RuntimeError):
    pass


class DurableChat:
    def __init__(self, root, namespace, http, factory, run, *, offline_test=False, after_received=None):
        if not isinstance(namespace,str) or not namespace:
            raise ValueError('frozen request namespace required')
        self.journal = _Journal(root,offline_test)
        self.namespace,self.http,self.factory,self.run=namespace,http,factory,run
        self.offline_test,self.after_received=offline_test,after_received

    def _identity(self, request_id, payload):
        if not isinstance(request_id,str) or not request_id or payload.get('model')!='gpt-4o-mini':
            raise ValueError('fixed request id and specified chat model required')
        encoded=_json(payload)
        manifest=dict(keys=[request_id],payload_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                      run_path=str(Path(self.run.path).resolve()))
        fingerprint=hashlib.sha256(_json(dict(namespace=self.namespace,manifest=manifest)).encode()).hexdigest()
        return fingerprint,manifest

    def recover(self, request_id, payload):
        """Read-only cache validation: missing receipts can never dispatch a call."""
        fingerprint,manifest=self._identity(request_id,payload)
        return self._complete(fingerprint,manifest,recovered=True)

    def execute(self, request_id, payload):
        fingerprint,manifest=self._identity(request_id,payload)
        self.journal.install(fingerprint,self.namespace,manifest)
        row=self.journal.row(fingerprint)
        if row['state']!='ready':
            return self._complete(fingerprint,manifest,recovered=True)
        self.journal.start(fingerprint)
        wrapped=_ReceiptHTTP(self.http,self.journal,fingerprint,payload,self.after_received)
        def observer(attempt):
            with self.run.connect() as db:
                claims=db.execute("SELECT ordinal,model FROM calls WHERE state='claimed'").fetchall()
            if len(claims)!=1 or claims[0][1]!='gpt-4o-mini':
                raise DurableChatBlocked('exactly one matching run claim required')
            self.journal.attach(fingerprint,attempt,str(Path(self.run.path).resolve()),claims[0][0])
        try:
            transport=self.factory(wrapped,observer)
            if not isinstance(transport,ScopedTransport) or transport.run.path!=self.run.path:
                raise DurableChatBlocked('same scoped guard required')
            base=transport.transport
            if (not isinstance(base,BudgetedJSONTransport) or base.model!='gpt-4o-mini'
                    or base.endpoint!=OPENAI_ENDPOINT or base.http is not wrapped
                    or base.attempt_observer is not observer or base.retry.max_attempts!=1
                    or base.retry.retry_429 or (self.offline_test and base.mode!='offline_fake')):
                raise DurableChatBlocked('one-attempt specified-model transport required')
            wrapped.endpoint=OPENAI_ENDPOINT
            transport(OPENAI_ENDPOINT,payload)
            with self.journal.connect() as db:
                db.execute("UPDATE operations SET transport_completed=1,state='complete' WHERE fingerprint=?",(fingerprint,))
            return self._complete(fingerprint,manifest,recovered=False)
        except BaseException:
            self.journal.unknown(fingerprint)
            raise

    def _complete(self,fingerprint,manifest,*,recovered):
        row=self.journal.row(fingerprint)
        if (row is None or row['state']!='complete' or not row['transport_completed']
                or row['manifest']!=_json(manifest) or row['scope']!=self.namespace
                or row['run_path']!=str(Path(self.run.path).resolve()) or row['status']!=200
                or row['redacted'] or not row['attempt_id'] or row['body'] is None
                or hashlib.sha256(row['body']).hexdigest()!=row['body_sha']):
            raise DurableChatBlocked('receipt/accounting unresolved or changed; never automatically replay')
        with self.run.connect() as db:
            claim=db.execute('SELECT model,state FROM calls WHERE ordinal=?',(row['run_ordinal'],)).fetchone()
        if claim!=('gpt-4o-mini','complete'):
            raise DurableChatBlocked('same completed run claim required')
        response=_strict_json(row['body'])
        if response.get('model') not in ('gpt-4o-mini','gpt-4o-mini-2024-07-18'):
            raise DurableChatBlocked('saved model mismatch')
        return dict(response=response,fingerprint=fingerprint,attempt_id=row['attempt_id'],
                    receipt_sha256=row['body_sha'],run_ordinal=row['run_ordinal'],recovered=recovered)
