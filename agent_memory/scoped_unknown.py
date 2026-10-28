"""Proof primitives for isolating future unknown calls to one user.

This module is not wired into production. A caller must persist the owner hash
before dispatch and preserve the original attempt, observation and reservation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from functools import wraps

_owner = ContextVar('scoped_unknown_owner', default=None)


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@contextmanager
def request_owner(user_id):
    if not isinstance(user_id, str) or not user_id:
        raise ValueError('explicit request owner required')
    token = _owner.set(hashlib.sha256(user_id.encode('utf-8')).hexdigest())
    try:
        yield
    finally:
        _owner.reset(token)


def owner_digest():
    owner = _owner.get()
    if not _is_digest(owner):
        raise ValueError('request owner not bound')
    return owner


def bound_request(method):
    """Bind the API owner for a single opt-in store call; never log the ID."""
    @wraps(method)
    def wrapped(self, payload):
        transport = self.transport
        if not getattr(transport, 'scoped_unknown_enabled', False):
            return method(self, payload)
        # Preserve the existing validation path for malformed requests.
        if not isinstance(payload, dict) or not isinstance(payload.get('user_id'), str) or not payload['user_id']:
            return method(self, payload)
        with request_owner(payload['user_id']):
            try:
                return method(self, payload)
            except BaseException:
                # Clear only the SDK's transient latch after this owner has a
                # durable scoped unknown and all other readiness gates pass.
                try:
                    if transport.has_scoped_unknown(owner_digest(), model='text-embedding-v4'):
                        transport.ready()
                        self.native.embedding_model.failed = False
                except Exception:
                    pass
                raise
    return wrapped


def verify_unknown(row, attempt, observation):
    """Validate durable, pre-dispatch attribution and full retained reserve.

    Return only owner and payload digests. Missing or ambiguous evidence fails
    closed; historical records cannot be upgraded by inventing an owner.
    """
    try:
        manifest = json.loads(row['manifest'])
        owner = manifest['owner_sha256']
        payload = manifest['payload_sha256']
        if (row['state'] != 'unknown' or row['status'] is not None or
                row['body'] is not None or row['transport_completed'] or
                row['attempt_id'] != attempt['id'] or not _is_digest(owner) or
                not _is_digest(payload) or attempt['status'] != 'uncertain' or
                type(attempt['reserved']) is not int or attempt['reserved'] <= 0 or
                attempt['reserved'] != attempt['accounted'] or
                observation['attempt_id'] != attempt['id'] or
                observation['http_status'] is not None or
                observation['category'] not in ('network_error', 'timeout', 'tls_error')):
            raise ValueError('unknown call cannot be scoped safely')
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError('unknown call cannot be scoped safely') from None
    return owner, payload
