"""Explicit hosted rerank candidate, distinct from the frozen self-hosted weights.

Only the existing text representations are scored. No image uploads, generated
answers, retries, extra embeddings, or query persistence are introduced here.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import uuid

from .budget import VerifiedRate
from .embedding_executor import _ReceiptHTTP
from .protocol import text_of
from .provider_http import BudgetedJSONTransport, BAILIAN_RERANK_ENDPOINT, ProviderFailure, validate_endpoint
from .service_transport import ServiceTransport

MODEL = 'qwen3-vl-rerank'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query.'
MAX_INPUT = 120000


def payloads(query, documents):
    """Greedy whole-document batches; conservative UTF-8 admission, no truncation."""
    if not isinstance(query, str) or not query.strip() or len(query.encode()) > 4000:
        raise ValueError('hosted rerank query must be1..4000 UTF-8 bytes')
    if not isinstance(documents, list) or not 1 <= len(documents) <= 100:
        raise ValueError('hosted rerank requires1..100 documents')
    if any(not isinstance(d, str) or not d.strip() or len(d.encode()) > 8000 for d in documents):
        raise ValueError('hosted rerank document must be1..8000 UTF-8 bytes')
    batches=[];batch=[];size=0
    for document in documents:
        # Count the repeated query/instruction per pair, plus a framing margin.
        pair_size=len(query.encode())+len(document.encode())+len(INSTRUCTION.encode())+256
        if batch and size+pair_size > MAX_INPUT:
            batches.append(batch);batch=[];size=0
        batch.append(document);size+=pair_size
    if batch:batches.append(batch)
    return [dict(model=MODEL,input=dict(query=query,documents=batch),
                 parameters=dict(top_n=len(batch),return_documents=False,instruct=INSTRUCTION))
            for batch in batches]


class RerankJSONTransport(BudgetedJSONTransport):
    def _bounds(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'model','input','parameters'} or payload['model'] != MODEL:
            raise ValueError('explicit hosted scoring model required')
        value=payload['input']
        if not isinstance(value, dict) or set(value) != {'query','documents'}:
            raise ValueError('text-only hosted rerank input required')
        expected=payloads(value['query'],value['documents'])
        if len(expected) != 1 or payload != expected[0]:
            raise ValueError('hosted rerank batch or parameters exceed reviewed contract')
        return MAX_INPUT,0

    def _usage(self, data):
        usage=data.get('usage')
        if not isinstance(usage,dict):raise ProviderFailure('invalid_usage')
        total=usage.get('total_tokens')
        if type(total) is not int or total < 0 or usage.get('prompt_tokens',total) != total:
            raise ProviderFailure('invalid_usage')
        return total,0

    def _response_model(self, data):
        # DashScope's documented response has no model/version field. Preserve
        # this absence in the ledger; never invent an observed weight identity.
        observed=data.get('model')
        if observed is not None and observed != MODEL:raise ProviderFailure('model_mismatch')
        return observed

    def _contract(self, data, payload):
        output=data.get('output');count=len(payload['input']['documents'])
        rows=output.get('results') if isinstance(output,dict) else None
        if not isinstance(rows,list) or len(rows) != count:
            raise ProviderFailure('invalid_contract')
        indices=[]
        for row in rows:
            if not isinstance(row,dict):raise ProviderFailure('invalid_contract')
            index=row.get('index');score=row.get('relevance_score')
            if (type(index) is not int or not 0 <= index < count
                    or type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 1
                    or 'document' in row):
                raise ProviderFailure('invalid_contract')
            indices.append(index)
        if len(set(indices)) != count:raise ProviderFailure('invalid_contract')


class HostedRerankTransport(ServiceTransport):
    """Same original ledger, separate minimized journal and immutable call ceiling."""
    def __init__(self, *args, call_limit, **kwargs):
        if type(call_limit) is not int or not 1 <= call_limit <= 4096:
            raise ValueError('explicit bounded hosted call limit required')
        super().__init__(*args,**kwargs)
        self.call_limit=call_limit
        path=self.journal.root/'hosted-call-limit.json'
        expected=dict(model=MODEL,max_calls=call_limit)
        if not path.exists():
            with self.journal.connect() as db:
                if db.execute('SELECT count(*) FROM operations').fetchone()[0]:
                    raise ValueError('missing hosted call limit after activity')
            with path.open('x',encoding='utf-8') as target:
                json.dump(expected,target,sort_keys=True)
        if json.loads(path.read_text()) != expected:
            raise ValueError('hosted lifetime call ceiling cannot change')
        self.limit_path=path;self.limit_sha=hashlib.sha256(path.read_bytes()).hexdigest()

    def prices(self):
        value=json.loads(self.prices_path.read_text())
        if set(value) != {'checked_at','models'} or set(value['models']) != {MODEL}:
            raise ValueError('separate verified hosted rerank price required')
        checked=datetime.fromisoformat(value['checked_at'])
        if checked.tzinfo is None or not 0 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 172800:
            raise ValueError('hosted price verification expired')
        rate=VerifiedRate(model=MODEL,checked_at=value['checked_at'],**value['models'][MODEL]);rate.validate()
        if (Decimal(rate.input_cny_per_million),Decimal(rate.output_cny_per_million)) != (Decimal('.5'),Decimal('0')):
            raise ValueError('hosted price change needs reservation review')
        if rate.conversion_basis != 'verified_fx':raise ValueError('native CNY hosted price required')
        return value

    def ready(self):
        super().ready()
        if hashlib.sha256(self.limit_path.read_bytes()).hexdigest() != self.limit_sha:
            raise ValueError('hosted lifetime call ceiling changed')
        with self.journal.connect() as db:
            if db.execute('SELECT count(*) FROM operations').fetchone()[0] >= self.call_limit:
                raise ValueError('hosted lifetime call ceiling exhausted')

    def __call__(self, endpoint, payload):
        with self.lock:
            self.ready();validate_endpoint(MODEL,endpoint)
            config=self.prices()
            rate=VerifiedRate(model=MODEL,checked_at=config['checked_at'],**config['models'][MODEL])
            fingerprint=uuid.uuid4().hex
            receipt_http=_ReceiptHTTP(self.http,self.journal,fingerprint,payload,None)
            receipt_http.endpoint=endpoint
            transport=RerankJSONTransport(model=MODEL,endpoint=endpoint,ledger=self.ledger,rate=rate,
                credential_reader=self.readers.get(MODEL),http=receipt_http,mode=self.mode,
                enabled=True,prices_confirmed=True,timeout_seconds=120,max_response_bytes=65536,
                attempt_observer=lambda attempt:self.journal.attach(fingerprint,attempt))
            transport._bounds(payload)
            with self.journal.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                if (db.execute('SELECT count(*) FROM operations').fetchone()[0]>=self.call_limit
                        or db.execute("SELECT 1 FROM operations WHERE state!='complete' LIMIT 1").fetchone()):
                    raise ValueError('hosted call ceiling or unresolved attempt prevents dispatch')
                manifest=json.dumps({'keys':[fingerprint],'model':MODEL},sort_keys=True)
                db.execute('INSERT INTO operations(fingerprint,scope,manifest,state) VALUES(?,?,?,?)',
                           (fingerprint,'hosted-rerank-v2',manifest,'inflight'))
                db.execute('INSERT INTO claims(scope,chunk_key,fingerprint) VALUES(?,?,?)',
                           ('hosted-rerank-v2',fingerprint,fingerprint))
            try:value=transport(endpoint,payload)
            except BaseException:
                with self.journal.connect() as db:
                    db.execute("UPDATE operations SET state=CASE WHEN state='received' THEN 'failed' ELSE 'unknown' END WHERE fingerprint=?",(fingerprint,))
                raise
            with self.journal.connect() as db:
                row=db.execute('SELECT * FROM operations WHERE fingerprint=?',(fingerprint,)).fetchone()
                if row['state'] != 'received' or row['redacted'] or not row['body'] or hashlib.sha256(row['body']).hexdigest() != row['body_sha']:
                    raise ValueError('complete original hosted receipt required')
                db.execute("UPDATE operations SET state='complete',body=NULL,redacted=1,transport_completed=1,receipt_policy='usage_only' WHERE fingerprint=?",(fingerprint,))
            return value


class BailianScorer:
    model=MODEL
    source_text_only=True
    weight_identity_verified=False

    def __init__(self, transport, endpoint=BAILIAN_RERANK_ENDPOINT):
        validate_endpoint(MODEL,endpoint)
        self.transport,self.endpoint=transport,endpoint

    def score(self, parts, candidates, base_scores):
        batches=payloads(text_of(parts),[c['retrieval_text'] for c in candidates])
        scores=[]
        for payload in batches:
            response=self.transport(self.endpoint,payload)
            # Recheck even injected transports; scores must retain source order.
            RerankJSONTransport._contract(None,response,payload)
            batch=[None]*len(payload['input']['documents'])
            for item in response['output']['results']:batch[item['index']]=item['relevance_score']
            scores.extend(batch)
        return scores
