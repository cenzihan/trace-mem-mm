"""Explicit Linux-container application factory for the deterministic service.

Importing this module does not read a secret, open a database or start a server.
V6 research inference and provider credentials are intentionally not wired here.
"""
import os
from pathlib import Path

from .app import Application
from .system import AgentMemorySystem, RetrievalConfig


class ContainerMemorySystem(AgentMemorySystem):
    def capabilities(self):
        return dict(super().capabilities(), deployment_profile='deterministic_container',
                    release_stage='deployment_preview', experiment_v6_online=False)


def read_service_token(filename):
    if not filename:
        raise ValueError('TRACEMEM_AUTH_TOKEN_FILE is required')
    try:
        with Path(filename).open('rb') as source:
            raw = source.read(4097)
        token = raw.decode('ascii').rstrip('\r\n')
    except (OSError, UnicodeError):
        raise ValueError('service token file is unreadable or invalid') from None
    if len(raw) > 4096 or not 32 <= len(token) <= 4096 or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError('service token must contain 32..4096 printable non-whitespace ASCII characters')
    return token


def create_app(environ=None):
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_MODE', 'local_deterministic') != 'local_deterministic':
        raise ValueError('this container supports local_deterministic only; V6 service integration is pending')
    auth_mode = env.get('TRACEMEM_AUTH_MODE', 'Bearer')
    if auth_mode not in ('Bearer', 'Token', 'X-Api-Key'):
        raise ValueError('container Add/Search require Bearer, Token or X-Api-Key authentication')
    token = read_service_token(env.get('TRACEMEM_AUTH_TOKEN_FILE'))
    path = Path(env.get('TRACEMEM_DB', '/data/system.sqlite'))
    if not path.is_absolute() or path.name in ('', '.', '..'):
        raise ValueError('TRACEMEM_DB must name an absolute persistent database file')
    store = ContainerMemorySystem(path, RetrievalConfig())
    return Application(store, auth_mode=auth_mode, key=token)
