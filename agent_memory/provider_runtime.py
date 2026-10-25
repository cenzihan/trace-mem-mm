"""Explicit composition for reviewed provider settings. Construction makes no requests."""
from pathlib import Path

from .budget import BudgetLedger
from .provider_http import BudgetedJSONTransport,OPENAI_ENDPOINT,SHARED_LEDGER
from .providers import GPT4oMiniCards,TextEmbeddingV4,ProviderBlocked
from .system import AgentMemorySystem,RetrievalConfig


def build_provider_system(database,*,settings,credential_readers,rates,http=None,mode='live',ledger=None):
    """Only safe readers, never secret values, are passed in. Defaults stay disabled.

    No env/file/key lookup, regional fallback, image embedding substitution, or model download.
    Offline fake mode needs an explicitly fake HTTP exchange and a temporary test ledger.
    """
    allowed={'embedding_model','llm_model','embedding_endpoint','dense','cards','dimensions',
             'real_calls_enabled','prices_confirmed','timeout_seconds'}
    if not isinstance(settings,dict) or set(settings)-allowed:raise ValueError('unknown settings; secrets do not belong in configuration')
    if settings.get('embedding_model','text-embedding-v4')!='text-embedding-v4' or settings.get('llm_model','gpt-4o-mini')!='gpt-4o-mini':
        raise ValueError('Academic model substitution is forbidden')
    dense=settings.get('dense',False);cards=settings.get('cards',False)
    if type(dense) is not bool or type(cards) is not bool:raise ValueError('explicit feature flags required')
    if mode not in ('live','offline_fake'):raise ValueError('unknown provider mode')
    if mode=='offline_fake' and (http is None or ledger is None):raise ValueError('fake exchange and isolated ledger required')
    if ledger is None:ledger=BudgetLedger(SHARED_LEDGER)
    if mode=='live' and Path(ledger.path).resolve()!=SHARED_LEDGER.resolve():raise ValueError('shared live ledger required')
    def transport(model,endpoint):
        if not endpoint:raise ProviderBlocked('explicit Bailian workspace endpoint required')
        if model not in rates:raise ProviderBlocked('verified model prices/FX missing')
        return BudgetedJSONTransport(model=model,endpoint=endpoint,ledger=ledger,rate=rates[model],
            credential_reader=credential_readers.get(model),http=http,mode=mode,
            enabled=settings.get('real_calls_enabled',False),prices_confirmed=settings.get('prices_confirmed',False),
            timeout_seconds=settings.get('timeout_seconds',30))
    embedding=None;card_adapter=None
    if dense:
        endpoint=settings.get('embedding_endpoint')
        embedding=TextEmbeddingV4(endpoint,transport('text-embedding-v4',endpoint),settings.get('dimensions',1024))
    if cards:card_adapter=GPT4oMiniCards(transport('gpt-4o-mini',OPENAI_ENDPOINT))
    return AgentMemorySystem(database,RetrievalConfig(dense=dense),embedding=embedding,card_adapter=card_adapter)
