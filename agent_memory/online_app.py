"""Opt-in online candidate; a finite reviewed grant is required at startup."""
import os
import http.client
import json
import math
from pathlib import Path

from .app import Application
from .container_app import read_service_token
from .embedding_cache import RUNTIME
from .embedding_smoke_app import read_embedding_token
from .online_candidate import OnlineDenseRerankSystem
from .prepaid_online import PrepaidOnlineTransport
from .protocol import text_of
from .provider_http import BAILIAN_BEIJING_ENDPOINT
from .providers import TextEmbeddingV4
from .online_cards import VerifiedQuoteCards


class PrivateQwenScorer:
    model = 'Qwen/Qwen3-VL-Reranker-2B'
    revision = '4bd860ac4f15ad1897a214615cccc700f8f71818'
    weights_sha256 = '466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9'

    def __init__(self, token, port=18978):
        self.token, self.port = token, port

    def score(self, parts, candidates, base_scores):
        body = json.dumps({'query': text_of(parts), 'documents': [c['retrieval_text'] for c in candidates]}, ensure_ascii=False, allow_nan=False).encode()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=180)
        try:
            conn.request('POST', '/score', body, {'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/json'})
            response = conn.getresponse(); raw = response.read(65537)
            if response.status != 200 or len(raw) > 65536:
                raise ValueError('private reranker unavailable')
            value = json.loads(raw); scores = value.get('scores')
            if (value.get('model') != self.model or value.get('revision') != self.revision
                    or value.get('weights_sha256') != self.weights_sha256
                    or not isinstance(scores, list) or len(scores) != len(candidates)
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in scores)):
                raise ValueError('reranker identity or scores mismatch')
            return scores
        finally: conn.close()


def create_app(environ=None):
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_ONLINE_ENABLED') != 'approved-parent-grant':
        raise ValueError('explicit online activation required')
    state = Path(env['TRACEMEM_ONLINE_STATE']).resolve()
    if state == RUNTIME.resolve() or not state.is_relative_to(RUNTIME.resolve()):
        raise ValueError('isolated private runtime required')
    token = read_service_token(env['TRACEMEM_AUTH_TOKEN_FILE'])
    readers = {model: (lambda path=env[key]: read_embedding_token(path)) for model, key in
               [('gpt-4o-mini', 'TRACEMEM_OPENAI_TOKEN_FILE'),
                ('text-embedding-v4', 'TRACEMEM_BAILIAN_TOKEN_FILE')]}
    transport = PrepaidOnlineTransport(env['TRACEMEM_GRANT_FILE'], env['TRACEMEM_GRANT_SHA256'],
                                      state/'receipts', readers)
    store = OnlineDenseRerankSystem(state/'memory.sqlite',
        embedding=TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT, transport, dimensions=256),
        card_adapter=VerifiedQuoteCards(transport, image_detail='low'),
        scoring_adapter=PrivateQwenScorer(read_service_token(env['TRACEMEM_RERANKER_TOKEN_FILE'])))
    return Application(store, auth_mode='Bearer', key=token)
