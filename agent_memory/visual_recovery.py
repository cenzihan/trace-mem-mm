"""Local private response receipts and original/request blobs, no dispatch defaults.

Capture precedes provider parsing. Unknown operations never auto-replay. No headers,
keys or exception text enter SQLite. Real wiring still needs approved run/transfer.
"""
import base64
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from .protocol import canonical, image_bytes
from .provider_http import HTTPReply, OPENAI_ENDPOINT, BudgetedJSONTransport
from .development_run import ScopedTransport
from .visual_batch8 import GPT4oMiniVisualBatch8 as Batch, VisualInput, VisualBatchFailure

PROJECT_RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
BODY_LIMIT=8*1024*1024


class VisualRecoveryBlocked(RuntimeError):
    pass


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _strict_json(raw):
    """Never repair ambiguous or non-finite JSON while recovering a receipt."""
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('duplicate JSON key')
            value[key]=item
        return value
    def invalid_constant(_value):raise ValueError('non-finite JSON constant')
    if isinstance(raw,bytes):raw=raw.decode('utf-8')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid_constant)


def _credential_variants(secret):
    # Include complete unicode escapes even when a truncated response cannot be
    # parsed. Doubled escapes cover a literal escaped string inside JSON too.
    units=secret.encode('utf-16-be',errors='surrogatepass')
    escaped=''.join('\\u%04x'%int.from_bytes(units[i:i+2],'big') for i in range(0,len(units),2))
    upper=''.join('\\u%04X'%int.from_bytes(units[i:i+2],'big') for i in range(0,len(units),2))
    values={secret,json.dumps(secret,ensure_ascii=True)[1:-1],escaped,upper,
            json.dumps(escaped)[1:-1],json.dumps(upper)[1:-1]}
    return sorted(values,key=lambda item:(-len(item),item))


def _safe_body(raw, headers):
    """Drop header-shaped fields and current credential echoes, never save headers."""
    secrets=[]
    for name,value in headers.items():
        if str(name).lower() in ('authorization','x-api-key','api-key') and isinstance(value,str) and value:
            secrets.append(value)
            if value.startswith('Bearer '):secrets.append(value[7:])
    def replace(text):
        for secret in secrets:
            if secret:
                for variant in _credential_variants(secret):
                    text=text.replace(variant,'[REDACTED_CREDENTIAL]')
        return text
    def walk(value):
        if isinstance(value,dict):
            return {replace(key):('[REDACTED_HEADER_OR_KEY]' if key.lower() in
                        {'authorization','headers','request_headers','api_key','api-key','x-api-key','apikey','access_token','client_secret'}
                        else walk(item)) for key,item in value.items()}
        if isinstance(value,list):return [walk(item) for item in value]
        if isinstance(value,str):return replace(value)
        return value
    try:
        value=_strict_json(raw)
        sanitized=walk(value)
        if sanitized==value:return raw,False
        safe=json.dumps(sanitized,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()
    except (UnicodeError,ValueError,TypeError,RecursionError):
        safe=raw
        for secret in secrets:
            if secret:
                for variant in _credential_variants(secret):
                    safe=safe.replace(variant.encode('utf-8',errors='surrogatepass'),b'[REDACTED_CREDENTIAL]')
    return safe, safe != raw


class PrivateVisualJournal:
    def __init__(self,root,*,offline_test=False):
        self.root=Path(root).resolve()
        allowed=Path(tempfile.gettempdir()).resolve() if offline_test else PROJECT_RUNTIME.resolve()
        if self.root==allowed or not self.root.is_relative_to(allowed):
            raise ValueError('private journal must be beneath runtime, or isolated test temp')
        self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'blobs').mkdir(exist_ok=True)
        self.path=self.root/'private-visual.sqlite'
        if (self.root/'blobs').is_symlink() or self.path.is_symlink():
            raise ValueError('private storage symlinks are not supported')
        with self.connect() as db:
            if db.execute('PRAGMA user_version').fetchone()[0]>1:raise ValueError('future private journal schema')
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables-set(('operations',)):raise ValueError('wrong private journal database')
            db.execute('''CREATE TABLE IF NOT EXISTS operations(
                fingerprint TEXT PRIMARY KEY,namespace TEXT NOT NULL,manifest TEXT NOT NULL,
                state TEXT NOT NULL,attempt_id TEXT, status INTEGER,body BLOB,body_sha TEXT,
                redacted INTEGER,category TEXT,result TEXT)''')
            db.execute('PRAGMA user_version=1')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10);db.row_factory=sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()

    def _blob_path(self,digest):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise VisualRecoveryBlocked('invalid blob identity')
        target=(self.root/'blobs'/digest).resolve()
        if target.parent!=(self.root/'blobs').resolve():raise VisualRecoveryBlocked('blob escaped private root')
        return target

    def put_blob(self,raw):
        digest=_sha(raw);target=self._blob_path(digest)
        if target.exists():
            if _sha(target.read_bytes())!=digest:raise VisualRecoveryBlocked('blob integrity mismatch')
            return digest
        temporary=self.root/'blobs'/('pending-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            # Immutable content-addressed writes; same hash implies identical bytes.
            if target.exists():
                if _sha(target.read_bytes())!=digest:raise VisualRecoveryBlocked('blob integrity mismatch')
            else:os.replace(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
        return digest

    def get_blob(self,digest):
        raw=self._blob_path(digest).read_bytes()
        if _sha(raw)!=digest:raise VisualRecoveryBlocked('blob integrity mismatch')
        return raw

    def install(self,prepared):
        # Rebuild verifies a caller cannot mutate payload/manifests after preparation.
        fresh=Batch.prepare(prepared.original_sources,prepared.namespace)
        if fresh.request_fingerprint!=prepared.request_fingerprint or canonical(fresh.payload)!=canonical(prepared.payload) or fresh.manifest!=prepared.manifest:
            raise VisualRecoveryBlocked('prepared request changed')
        for source,manifest in zip(prepared.original_sources,prepared.manifest):
            original=source.original_bytes if source.original_bytes is not None else image_bytes(source.original_uri)[0]
            request=image_bytes(source.request_uri)[0]
            if self.put_blob(original)!=manifest['original_sha256'] or self.put_blob(request)!=manifest['request_sha256']:
                raise VisualRecoveryBlocked('source variant mismatch')
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO operations(fingerprint,namespace,manifest,state) VALUES(?,?,?,?)',
                       (prepared.request_fingerprint,prepared.namespace,canonical(list(prepared.manifest)),'ready'))
            stored=db.execute('SELECT namespace,manifest FROM operations WHERE fingerprint=?',
                              (prepared.request_fingerprint,)).fetchone()
            if (stored['namespace']!=prepared.namespace or
                stored['manifest']!=canonical(list(prepared.manifest))):
                raise VisualRecoveryBlocked('stored request manifest or namespace changed')
        return prepared.request_fingerprint

    def row(self,fingerprint):
        with self.connect() as db:row=db.execute('SELECT * FROM operations WHERE fingerprint=?',(fingerprint,)).fetchone()
        if row is None:raise VisualRecoveryBlocked('unknown operation')
        return dict(row)

    def prepared(self,fingerprint):
        row=self.row(fingerprint);sources=[]
        for value in json.loads(row['manifest']):
            original=self.get_blob(value['original_sha256']);request=self.get_blob(value['request_sha256'])
            uri='data:image/'+value['request_mime']+';base64,'+base64.b64encode(request).decode()
            sources.append(VisualInput(value['native_id'],value['occurrence_id'],None,uri,value['variant'],original))
        prepared=Batch.prepare(sources,row['namespace'])
        if prepared.request_fingerprint!=fingerprint:raise VisualRecoveryBlocked('journal request integrity mismatch')
        return prepared

    def start(self,fingerprint):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed=db.execute("UPDATE operations SET state='inflight' WHERE fingerprint=? AND state='ready'",(fingerprint,))
            if changed.rowcount!=1:raise VisualRecoveryBlocked('operation already dispatched or unresolved; no automatic replay')

    def attach_attempt(self,fingerprint,attempt_id):
        if not isinstance(attempt_id,str) or not attempt_id or len(attempt_id)>128:raise VisualRecoveryBlocked('budget attempt identity required')
        with self.connect() as db:
            changed=db.execute("UPDATE operations SET attempt_id=? WHERE fingerprint=? AND state='inflight' AND attempt_id IS NULL",(attempt_id,fingerprint))
            if changed.rowcount!=1:raise VisualRecoveryBlocked('duplicate budget dispatch blocked')

    def received(self,fingerprint,reply,headers):
        if (not isinstance(reply,HTTPReply) or type(reply.status) is not int or not 100<=reply.status<=599 or
            not isinstance(reply.body,bytes) or len(reply.body)>BODY_LIMIT):
            raise VisualRecoveryBlocked('response receipt size/type failed')
        safe,redacted=_safe_body(reply.body,headers)
        if len(safe)>BODY_LIMIT:raise VisualRecoveryBlocked('sanitized response exceeds receipt cap')
        with self.connect() as db:
            changed=db.execute("""UPDATE operations SET state='received',status=?,body=?,body_sha=?,redacted=?
                WHERE fingerprint=? AND state='inflight' AND attempt_id IS NOT NULL""",
                (reply.status,safe,_sha(safe),int(redacted),fingerprint))
            if changed.rowcount!=1:raise VisualRecoveryBlocked('receipt requires one budgeted in-flight operation')

    def unknown(self,fingerprint):
        with self.connect() as db:
            db.execute("UPDATE operations SET state='unknown',category='unknown_outcome' WHERE fingerprint=? AND state='inflight'",(fingerprint,))

    def recover(self,fingerprint):
        row=self.row(fingerprint)
        if row['state'] not in ('received','invalid_response','complete') or row['body'] is None:
            raise VisualRecoveryBlocked('no complete saved receipt; reservation retained, no automatic replay')
        if _sha(row['body'])!=row['body_sha']:raise VisualRecoveryBlocked('response receipt integrity mismatch')
        prepared=self.prepared(fingerprint)
        try:
            if row['status']!=200:raise VisualRecoveryBlocked('saved HTTP response is not successful')
            try:
                response=_strict_json(row['body'])
                if isinstance(response,dict) and isinstance(response.get('choices'),list):
                    for choice in response['choices']:
                        message=choice.get('message') if isinstance(choice,dict) else None
                        if isinstance(message,dict) and isinstance(message.get('content'),str):
                            _strict_json(message['content'])
            except (ValueError,UnicodeError):raise VisualRecoveryBlocked('saved response JSON invalid') from None
            if not isinstance(response,dict):raise VisualRecoveryBlocked('saved response object required')
            usage=response.get('usage')
            if (not isinstance(usage,dict) or any(type(usage.get(key)) is not int for key in
                ('prompt_tokens','completion_tokens','total_tokens')) or
                not 0<=usage['prompt_tokens']<=128000 or not 0<=usage['completion_tokens']<=2048 or
                usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']):
                raise VisualRecoveryBlocked('saved usage invalid; reservation retained')
            result=Batch.validate(response,prepared)
        except (VisualBatchFailure,VisualRecoveryBlocked) as error:
            category=error.category if isinstance(error,VisualBatchFailure) else 'http_or_json'
            with self.connect() as db:db.execute("UPDATE operations SET state='invalid_response',category=? WHERE fingerprint=?",(category,fingerprint))
            raise
        with self.connect() as db:
            db.execute("UPDATE operations SET state='complete',category='validated_saved_response',result=? WHERE fingerprint=?",(canonical(result),fingerprint))
        return result


class _ReceiptHTTP:
    def __init__(self,inner,journal,fingerprint,prepared,after_received):
        self.inner=inner;self.journal=journal;self.fingerprint=fingerprint;self.prepared=prepared
        self.after_received=after_received;self.is_fake_http=getattr(inner,'is_fake_http',False)
    def exchange(self,endpoint,body,headers,timeout,max_response_bytes):
        if endpoint!=OPENAI_ENDPOINT or canonical(json.loads(body))!=canonical(self.prepared.payload):
            raise VisualRecoveryBlocked('dispatch changed after preparation')
        operation=self.journal.row(self.fingerprint)
        if operation['state']!='inflight' or not operation['attempt_id']:
            raise VisualRecoveryBlocked('budget attempt must be bound before dispatch')
        reply=self.inner.exchange(endpoint,body,headers,timeout,max_response_bytes)
        self.journal.received(self.fingerprint,reply,headers)
        if self.after_received is not None:self.after_received()
        return reply


class DurableVisualExecutor:
    """Explicit transport factory must retain shared budget and approved run guard.

    Factory signature (wrapped_http, attempt_observer)->BudgetedJSONTransport.
    A cached/recoverable operation never calls the factory or reserves again.
    """
    def __init__(self,journal,http,transport_factory,*,after_received=None):
        self.journal=journal;self.http=http;self.factory=transport_factory;self.after_received=after_received
    def execute(self,prepared):
        fingerprint=self.journal.install(prepared);state=self.journal.row(fingerprint)['state']
        if state in ('received','invalid_response','complete'):return self.journal.recover(fingerprint)
        if state!='ready':raise VisualRecoveryBlocked('unknown/inflight outcome retained; no automatic replay')
        self.journal.start(fingerprint)
        wrapped=_ReceiptHTTP(self.http,self.journal,fingerprint,prepared,self.after_received)
        try:
            observer=lambda attempt:self.journal.attach_attempt(fingerprint,attempt)
            transport=self.factory(wrapped,observer)
            base=transport.transport if isinstance(transport,ScopedTransport) else transport
            if (not isinstance(base,BudgetedJSONTransport) or base.model!=Batch.model or base.endpoint!=Batch.endpoint or
                base.http is not wrapped or base.attempt_observer is not observer or base.retry.max_attempts!=1 or
                (base.mode=='live' and not isinstance(transport,ScopedTransport))):
                raise VisualRecoveryBlocked('one-attempt budgeted transport and approved live run scope required')
            transport(Batch.endpoint,prepared.payload)
            return self.journal.recover(fingerprint)
        except BaseException:
            # Includes process-interruption simulation. A received receipt is never overwritten.
            self.journal.unknown(fingerprint)
            raise
