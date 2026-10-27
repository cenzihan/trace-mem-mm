"""Explicitly activated fixed candidate for an authorized official Smoke."""
import os

from .bailian_app import _create_app
from .context_candidate import EvidenceContextCandidate, VERSION


def create_app(environ=None):
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_CONTEXT_RETRIEVAL_ENABLED') != 'approved-context-v2.1.0-rc1':
        raise ValueError('explicit context candidate activation required')
    return _create_app(env, store_type=EvidenceContextCandidate, version=VERSION,
                       profile='hosted_context_v2_1', release_stage='official-smoke-candidate')
