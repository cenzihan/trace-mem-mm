"""Explicit native Mem0 deployment; original ledger/receipts, separate memories."""
import os
from pathlib import Path
import time

from .container_app import read_service_token
from .embedding_cache import RUNTIME
from .embedding_smoke_app import read_embedding_token
from .mem0_bridge import Mem0System
from .mem0_transport import Mem0ServiceTransport,Mem0EvidenceTransport
from .online_cards import VerifiedQuoteCards
from .provider_http import SHARED_LEDGER,BAILIAN_BEIJING_ENDPOINT,NativeHTTPS
from .providers import TextEmbeddingV4
from .service_app import ServiceApplication


def create_app(environ=None, *, _system_type=Mem0System,
               _transport_type=Mem0ServiceTransport,_stop_at_chat_done=False,_http_type=None):
    env=os.environ if environ is None else environ
    if env.get('TRACEMEM_MEM0_ENABLED')!='approved-native-mem0-night':raise ValueError('explicit native activation required')
    state=Path(env['TRACEMEM_SERVICE_STATE']).resolve()
    if not state.is_relative_to(RUNTIME.resolve()) or not state.name.startswith('mem0-night-'):
        raise ValueError('isolated native memory state required')
    readers={model:(lambda path=env[key]:read_embedding_token(path)) for model,key in
        [('gpt-4o-mini','TRACEMEM_OPENAI_TOKEN_FILE'),('text-embedding-v4','TRACEMEM_BAILIAN_TOKEN_FILE')]}
    # Reuse the original MM journal, including all prior failure acknowledgements.
    journal=RUNTIME/'bailian-v2-service'/'provider-receipts'
    base=_transport_type(SHARED_LEDGER,env['TRACEMEM_LEDGER_AUTHORITY_FILE'],
        env['TRACEMEM_VERIFIED_PRICES_FILE'],journal,readers,
        http=(_http_type() if _http_type else NativeHTTPS(handshake_attempts=3,stop_at_chat_done=_stop_at_chat_done)))
    expires=int(env['TRACEMEM_SERVICE_EXPIRES_AT'])
    if not time.time()<expires<=1793896200:raise ValueError('original lease must not be extended')
    store=_system_type(state/'memory.sqlite',transport=base,
        embedding=TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT,base,dimensions=256),
        card_adapter=VerifiedQuoteCards(base,image_detail='low',structured_output=True))
    return ServiceApplication(store,store,read_service_token(env['TRACEMEM_AUTH_TOKEN_FILE']),
        scorer_health=lambda:None,expires_at=expires,admission_capacity=16,queue_wait_seconds=30)


def create_evidence_app(environ=None):
    env=os.environ if environ is None else environ
    if env.get('TRACEMEM_MEM0_EVIDENCE_ENABLED')!='approved-native-mem0-evidence':
        raise ValueError('explicit evidence candidate activation required')
    from .mem0_evidence import Mem0EvidenceSystem
    from .http2_transport import HTTP2AcademicHTTPS
    return create_app(env,_system_type=Mem0EvidenceSystem,
                      _transport_type=Mem0EvidenceTransport,_http_type=HTTP2AcademicHTTPS)


def create_raw_app(environ=None):
    env=os.environ if environ is None else environ
    if env.get('TRACEMEM_MEM0_RAW_ENABLED')!='approved-native-mem0-raw':
        raise ValueError('explicit raw-memory ablation activation required')
    from .mem0_raw import Mem0RawSystem,Mem0RawTransport
    return create_app(env,_system_type=Mem0RawSystem,_transport_type=Mem0RawTransport)


def create_visual_app(environ=None):
    env=os.environ if environ is None else environ
    if env.get('TRACEMEM_MEM0_VISUAL_ENABLED')!='approved-native-mem0-visual':
        raise ValueError('explicit visual-rerank candidate activation required')
    from .visual_rerank import VisualHostedTransport,VisualScorer,MODEL
    from .mem0_visual import Mem0VisualSystem
    from .mem0_raw import Mem0RawTransport
    rerank=VisualHostedTransport(SHARED_LEDGER,env['TRACEMEM_LEDGER_AUTHORITY_FILE'],
        env['TRACEMEM_RERANK_PRICES_FILE'],RUNTIME/'bailian-v2-service'/'rerank-receipts',
        {MODEL:lambda:read_embedding_token(env['TRACEMEM_BAILIAN_TOKEN_FILE'])},call_limit=512)
    def make_store(*args,**kwargs):
        return Mem0VisualSystem(*args,scoring_adapter=VisualScorer(rerank),**kwargs)
    return create_app(env,_system_type=make_store,_transport_type=Mem0RawTransport)
