"""Public API profile: same retrieval, one active request, bounded original ledger."""
import http.client
import json
import os
from pathlib import Path
import threading
import time

from .app import Application
from .container_app import read_service_token
from .embedding_cache import RUNTIME
from .embedding_smoke_app import read_embedding_token
from .online_app import PrivateQwenScorer
from .online_candidate import OnlineDenseRerankSystem
from .online_cards import VerifiedQuoteCards
from .provider_http import BAILIAN_BEIJING_ENDPOINT, SHARED_LEDGER
from .providers import TextEmbeddingV4
from .protocol import ContractError, content_parts, text_of, validate_add, validate_search
from .cards import derive
from .service_transport import ServiceTransport
from .readiness_error import ReadinessFailure

VERSION = 'tracemem-mm-online-v1.0.0'


class ServingSystem(OnlineDenseRerankSystem):
    def add(self, payload):
        validate_add(payload)
        for message in payload['messages']:
            parts = content_parts(message['content'])
            text = text_of(parts)
            facts = derive(text)['facts']
            indexed = text + ' ' + ' '.join(' '.join(f[k] for k in ('entity','attribute','value')) for f in facts)
            if len(indexed.encode()) > 6000 or sum(p['type']=='image_url' for p in parts) > 8:
                raise ContractError('source capacity: 6000 UTF-8 bytes including extracted facts and 8 images per message',413)
        return super().add(payload)

    def search(self, payload):
        parts = validate_search(payload)
        if len(text_of(parts).encode()) > 8192:
            raise ContractError('query capacity: 8192 UTF-8 text bytes',413)
        return super().search(payload)


class ServiceApplication(Application):
    def __init__(self, store, transport, key, *, scorer_health=None, expires_at=None,
                 admission_capacity=1, queue_wait_seconds=0):
        if type(admission_capacity) is not int or not 1 <= admission_capacity <= 16:
            raise ValueError('admission capacity must be 1..16')
        if not isinstance(queue_wait_seconds, (int, float)) or not 0 <= queue_wait_seconds <= 30:
            raise ValueError('queue wait must be 0..30 seconds')
        super().__init__(store, auth_mode='Bearer', key=key)
        self.transport = transport
        self.active = threading.Lock()
        self.admissions = threading.BoundedSemaphore(admission_capacity)
        self.queue_wait_seconds = queue_wait_seconds
        self.scorer_health = scorer_health or self.check_scorer
        self.expires_at = expires_at

    @staticmethod
    def reply(start_response, status, error, extra=()):
        body = json.dumps({'error':error}).encode()
        start_response(status, [('Content-Type','application/json'),('Content-Length',str(len(body))),('Cache-Control','no-store'),*extra])
        return [body]

    @staticmethod
    def check_scorer():
        conn = http.client.HTTPConnection('127.0.0.1',18978,timeout=3)
        try:
            conn.request('GET','/health');response=conn.getresponse();value=json.loads(response.read(65537))
            if response.status != 200 or any(value.get(k) != getattr(PrivateQwenScorer,k) for k in ('model','revision','weights_sha256')):
                raise ValueError('pinned scorer unavailable')
        finally:conn.close()

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO')
        if self.expires_at is not None and time.time() >= self.expires_at:
            return self.reply(start_response,'503 Service Unavailable','deployment lease expired')
        if path == '/health' and environ.get('REQUEST_METHOD') == 'GET':
            try:self.transport.ready();self.scorer_health()
            except ReadinessFailure as exc:
                return self.reply(start_response,'503 Service Unavailable','readiness check failed',[('X-TraceMem-Readiness',exc.code)])
            except Exception:return self.reply(start_response,'503 Service Unavailable','readiness check failed')
        if path in ('/add','/search') and self.authorized(environ):
            if not self.admissions.acquire(blocking=False):
                return self.reply(start_response,'429 Too Many Requests','request admission capacity reached',[('Retry-After','5')])
            try:
                if not self.active.acquire(timeout=self.queue_wait_seconds):
                    return self.reply(start_response,'429 Too Many Requests','request queue wait exceeded',[('Retry-After','5')])
                try:
                    # Waiting must never extend the original deployment lease.
                    if self.expires_at is not None and time.time() >= self.expires_at:
                        return self.reply(start_response,'503 Service Unavailable','deployment lease expired')
                    try:self.transport.ready()
                    except ReadinessFailure as exc:
                        return self.reply(start_response,'503 Service Unavailable','model budget or unresolved attempt requires operator attention',[('X-TraceMem-Readiness',exc.code)])
                    except Exception:return self.reply(start_response,'503 Service Unavailable','model budget or unresolved attempt requires operator attention')
                    return super().__call__(environ,start_response)
                finally:self.active.release()
            finally:self.admissions.release()
        return super().__call__(environ,start_response)


def create_app(environ=None):
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_SERVICE_ENABLED') != 'original-ledger-owned-by-server':
        raise ValueError('explicit original-ledger service activation required')
    state = Path(env['TRACEMEM_SERVICE_STATE']).resolve()
    if state == RUNTIME.resolve() or not state.is_relative_to(RUNTIME.resolve()):
        raise ValueError('isolated official service runtime required')
    readers = {model:(lambda path=env[key]:read_embedding_token(path)) for model,key in
               [('gpt-4o-mini','TRACEMEM_OPENAI_TOKEN_FILE'),('text-embedding-v4','TRACEMEM_BAILIAN_TOKEN_FILE')]}
    transport = ServiceTransport(SHARED_LEDGER, env['TRACEMEM_LEDGER_AUTHORITY_FILE'],
        env['TRACEMEM_VERIFIED_PRICES_FILE'], state/'receipts', readers)
    expires_at = int(env['TRACEMEM_SERVICE_EXPIRES_AT'])
    if not time.time() < expires_at <= time.time()+28*86400:
        raise ValueError('explicit service lease within 28 days required')
    store = ServingSystem(state/'memory.sqlite',
        embedding=TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT,transport,dimensions=256),
        card_adapter=VerifiedQuoteCards(transport,image_detail='low'),
        scoring_adapter=PrivateQwenScorer(read_service_token(env['TRACEMEM_RERANKER_TOKEN_FILE'])))
    capabilities = store.capabilities
    store.capabilities = lambda:dict(capabilities(),version=VERSION,release_stage='submitted-api-candidate',
        deployment_profile='official_access_v1',concurrency=1,receipt_content_logging=False)
    return ServiceApplication(store,transport,read_service_token(env['TRACEMEM_AUTH_TOKEN_FILE']),expires_at=expires_at)
