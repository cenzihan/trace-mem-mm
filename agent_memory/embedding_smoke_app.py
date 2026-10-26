"""Explicit, finite, real-embedding smoke profile; baseline factory is unchanged."""
import hashlib
import os
from pathlib import Path
import re

from .app import Application
from .container_app import read_service_token
from .embedding_cache import RUNTIME
from .granted_embedding_smoke import GrantedEmbeddingTransport
from .provider_http import BAILIAN_BEIJING_ENDPOINT
from .providers import TextEmbeddingV4
from .system import AgentMemorySystem, RetrievalConfig


class EmbeddingSmokeSystem(AgentMemorySystem):
    def capabilities(self):
        return dict(super().capabilities(), mode='live_prepaid_embedding_smoke',
                    retrieval='BM25+dense+RRF', deployment_profile='bounded_embedding_smoke',
                    release_stage='functional_smoke', experiment_v6_online=False,
                    max_provider_requests=3, llm_enabled=False,
                    official_evaluation=False)


def read_embedding_token(filename):
    if not filename:
        raise ValueError('explicit embedding token file required')
    try:
        with Path(filename).open('rb') as source:
            raw = source.read(4097)
        key = raw.decode('ascii').rstrip('\r\n')
    except (OSError, UnicodeError):
        raise ValueError('embedding token is unavailable') from None
    if len(raw) > 4096 or not 16 <= len(key) <= 4096 or any(not 33 <= ord(c) <= 126 for c in key):
        raise ValueError('embedding token has invalid format')
    return key


def create_app(environ=None):
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_SMOKE_ENABLED') != 'embedding-only-approved-grant':
        raise ValueError('explicit finite embedding smoke profile required')
    token = read_service_token(env.get('TRACEMEM_AUTH_TOKEN_FILE'))
    grant = Path(env.get('TRACEMEM_GRANT_FILE', ''))
    expected = env.get('TRACEMEM_GRANT_SHA256', '')
    if not grant.is_absolute() or not re.fullmatch('[a-f0-9]{64}', expected):
        raise ValueError('absolute grant file and separately pinned SHA256 required')
    if grant.stat().st_size > 65536 or hashlib.sha256(grant.read_bytes()).hexdigest() != expected:
        raise ValueError('grant does not match the approved hash')
    state = Path(env.get('TRACEMEM_SMOKE_STATE', str(RUNTIME / 'deployment-embedding-smoke'))).resolve()
    if state == RUNTIME.resolve() or not state.is_relative_to(RUNTIME.resolve()):
        raise ValueError('smoke state must be inside its private runtime volume')
    key_file = env.get('TRACEMEM_BAILIAN_TOKEN_FILE')
    if not key_file or not Path(key_file).is_absolute():
        raise ValueError('absolute embedding secret path required')
    transport = GrantedEmbeddingTransport(grant, state / 'receipts',
                                          lambda: read_embedding_token(key_file),
                                          expected_grant_sha256=expected)
    embedding = TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT, transport, dimensions=256)
    store = EmbeddingSmokeSystem(state / 'memory.sqlite', RetrievalConfig(dense=True),
                                 embedding=embedding)
    return Application(store, auth_mode='Bearer', key=token)
