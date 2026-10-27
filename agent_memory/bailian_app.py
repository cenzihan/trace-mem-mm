"""Versioned hosted-rerank service; no dependency on the GPU SSH bridge."""
import os
from pathlib import Path
import time

from .bailian_rerank import BailianScorer, HostedRerankTransport, MODEL
from .container_app import read_service_token
from .embedding_cache import RUNTIME
from .embedding_smoke_app import read_embedding_token
from .online_cards import VerifiedQuoteCards
from .provider_http import BAILIAN_BEIJING_ENDPOINT, BAILIAN_RERANK_ENDPOINT, SHARED_LEDGER
from .providers import TextEmbeddingV4
from .protocol import ContractError,text_of,validate_search
from .service_app import ServingSystem,ServiceApplication
from .service_transport import ServiceTransport

VERSION='tracemem-mm-bailian-v2.0.3'


class BailianServingSystem(ServingSystem):
    reranker_model=MODEL
    model_add_message_limit=20

    def search(self,payload):
        parts=validate_search(payload)
        if len(text_of(parts).encode())>4000:
            raise ContractError('hosted query capacity:4000 UTF-8 bytes',413)
        return super().search(payload)


class CombinedReadiness:
    def __init__(self,base,rerank):self.base,self.rerank=base,rerank
    def ready(self):self.base.ready();self.rerank.ready()


def create_app(environ=None):
    env=os.environ if environ is None else environ
    if env.get('TRACEMEM_HOSTED_RERANK_ENABLED')!='approved-versioned-bailian-v2':
        raise ValueError('explicit new hosted model/version activation required')
    state=Path(env['TRACEMEM_SERVICE_STATE']).resolve()
    if not state.is_relative_to(RUNTIME.resolve()) or not state.name.startswith('bailian-v2'):
        raise ValueError('isolated hosted candidate runtime required')
    readers={model:(lambda path=env[key]:read_embedding_token(path)) for model,key in
             [('gpt-4o-mini','TRACEMEM_OPENAI_TOKEN_FILE'),('text-embedding-v4','TRACEMEM_BAILIAN_TOKEN_FILE'),(MODEL,'TRACEMEM_BAILIAN_TOKEN_FILE')]}
    base=ServiceTransport(SHARED_LEDGER,env['TRACEMEM_LEDGER_AUTHORITY_FILE'],
                          env['TRACEMEM_VERIFIED_PRICES_FILE'],state/'provider-receipts',readers)
    rerank=HostedRerankTransport(SHARED_LEDGER,env['TRACEMEM_LEDGER_AUTHORITY_FILE'],
        env['TRACEMEM_RERANK_PRICES_FILE'],state/'rerank-receipts',readers,
        call_limit=int(env['TRACEMEM_RERANK_CALL_LIMIT']))
    expires=int(env['TRACEMEM_SERVICE_EXPIRES_AT'])
    if not time.time()<expires<=time.time()+28*86400:raise ValueError('explicit hosted service lease required')
    store=BailianServingSystem(state/'memory.sqlite',
        embedding=TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT,base,dimensions=256),
        card_adapter=VerifiedQuoteCards(base,image_detail='low',structured_output=True),
        scoring_adapter=BailianScorer(rerank,env.get('TRACEMEM_RERANK_ENDPOINT',BAILIAN_RERANK_ENDPOINT)))
    capabilities=store.capabilities
    store.capabilities=lambda:dict(capabilities(),version=VERSION,deployment_profile='hosted_bailian_v2',
        release_stage='versioned-hosted-candidate',concurrency=1,admission_capacity=16,
        queue_wait_seconds=30,receipt_content_logging=False,
        source_card_format='strict-json-schema',max_add_messages=20,
        reranker_weights_verified=False,reranker_source_text_only=True,
        readiness_scope='local-config-budget-and-journals; upstream not probed')
    readiness=CombinedReadiness(base,rerank)
    return ServiceApplication(store,readiness,read_service_token(env['TRACEMEM_AUTH_TOKEN_FILE']),
                              scorer_health=rerank.ready,expires_at=expires,
                              admission_capacity=16,queue_wait_seconds=30)
