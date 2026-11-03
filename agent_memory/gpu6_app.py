"""Explicit single-writer GPU6 migration; original evidence and accounts retained."""
import json
import http.client
import os
from pathlib import Path
import threading
import time
import ssl
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar

from .budget import BudgetLedger
from .disk_options import DiskOptionsSystem
from .joint_selection import SelectionBudgetedTransport
from .mem0_app import create_app
from .two_lane_options import TwoLaneRetryTransport
from .provider_http import NativeHTTPS
from .indexed_readiness import install_readiness_indexes, readiness_rows


class GPU6HTTPS(NativeHTTPS):
    def connection(self, hostname, timeout):
        if hostname != 'api.openai.com':return super().connection(hostname, timeout)
        # Same local iKuu route as before, extended over the private SSH channel.
        # HTTPSConnection verifies TLS against the tunnel destination hostname.
        conn = http.client.HTTPSConnection('127.0.0.1', 28979, timeout=timeout,
                                           context=ssl.create_default_context())
        conn.set_tunnel('api.openai.com', 443)
        return conn


class RecordingLedger(BudgetLedger):
    # Human instruction 2026-11-01: no aggregate spending ceiling for this Full.
    # Historical cap/consents/attempts remain unchanged; all usage still recorded.
    enforce_cap = False

    def __init__(self, *args, **kwargs):
        self._validation_db = ContextVar('gpu6_validation_db', default=None)
        super().__init__(*args, **kwargs)

    @contextmanager
    def validation_scope(self):
        """Reuse a read-only connection for one locked validation, never its results.

        No explicit transaction: each SELECT observes committed accounting as
        before. No connection survives a validation or crosses threads. Model
        dispatch and ledger writes always use the original connect() method.
        """
        if self._validation_db.get() is not None:
            yield
            return
        db = sqlite3.connect(Path(self.path).resolve().as_uri()+'?mode=ro', uri=True, timeout=30)
        token = self._validation_db.set(db)
        try:
            yield
        finally:
            self._validation_db.reset(token)
            db.close()

    @contextmanager
    def validation_connect(self):
        db = self._validation_db.get()
        if db is None:
            with super().validation_connect() as connection:
                yield connection
        else:
            yield db


class HistoricalEstimateTransport(SelectionBudgetedTransport):
    # Keep the true old checked_at and rates; do not fabricate fresh verification.
    price_max_age_seconds = None


class GPU6Transport(TwoLaneRetryTransport):
    lane_count = 16
    max_provider_lanes = 16
    price_max_age_seconds = None
    ledger_type = RecordingLedger
    transport_type = HistoricalEstimateTransport

    def __init__(self, *args, **kwargs):
        self.frozen_unbound_failures = None
        if not kwargs.get('offline_test'):
            kwargs.setdefault('http_factory', lambda:GPU6HTTPS(6, stop_at_chat_done=True))
        super().__init__(*args, **kwargs)
        from .frozen_unbound_failure import FrozenUnboundFailures
        self.frozen_unbound_failures = FrozenUnboundFailures(
            self.journal.root / 'gpu6-frozen-unbound-failures.json', self.ledger, self.receipt_scope)
        with self.journal.connect() as db:
            install_readiness_indexes(db)
        self.journal.validation_scope = self.ledger.validation_scope

    def ready(self):
        # Acquire the existing ownership lock first; preserve journal/ledger
        # lock ordering and revalidate every exceptional row on every call.
        with self._live_lock, self.ledger.validation_scope():
            return super().ready()

    def _readiness_rows(self, db):
        return readiness_rows(db)

    def _accounted_terminal_failure(self, row):
        return (super()._accounted_terminal_failure(row) or bool(
            self.frozen_unbound_failures and self.frozen_unbound_failures.permits(row)))


class FileConcurrency:
    """Hot adjustment without restarting workers or interrupting admitted work."""
    def __init__(self, path):
        self.path = Path(path)
        self.condition = threading.Condition()
        self.inflight = 0
        self.limit()  # Reject a missing/invalid deployment control before serving.

    def limit(self):
        value = json.loads(self.path.read_text())
        if set(value) != {'concurrency'} or type(value['concurrency']) is not int or value['concurrency'] not in (2,4,8,16):
            raise ValueError('concurrency must be one of 2,4,8,16')
        return value['concurrency']

    def acquire(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                # A bad operator edit admits no new work; existing work still exits.
                limit = self.limit()
                if self.inflight < limit:
                    self.inflight += 1
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:return False
                self.condition.wait(min(.2, remaining))

    def release(self):
        with self.condition:
            if self.inflight <= 0:raise ValueError('unbalanced concurrency release')
            self.inflight -= 1
            self.condition.notify_all()


def create_gpu6_app(environ=None):
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_GPU6_MIGRATION') != 'original-full-single-writer':
        raise ValueError('explicit original Full migration required')
    if env.get('TRACEMEM_ACCOUNTING_POLICY') != 'record-historical-estimates-no-cap-no-expiry':
        raise ValueError('explicit accounting policy required')
    if env.get('TRACEMEM_DISK_VECTORS_ENABLED') != 'verified-preserving-migration':
        raise ValueError('verified disk migration required')
    limiter = FileConcurrency(env['TRACEMEM_CONCURRENCY_FILE'])
    class GPU6System(DiskOptionsSystem):
        embedding_preparation_batch_size = 10
        def add(self, payload):
            from .request_media import media_request_cache
            with media_request_cache():
                return super().add(payload)
        def capabilities(self):
            return dict(super().capabilities(), version='tracemem-mm-gpu6-v4.3.15-rc7',
                        concurrency=limiter.limit(), provider_concurrency=limiter.limit(),
                        provider_pool_capacity=16, embedding_message_batch_size=10,
                        accounting='historical-estimates-no-cap-no-expiry')
    app = create_app(env, _system_type=GPU6System, _transport_type=GPU6Transport,
                     _stop_at_chat_done=True, _execution_capacity=16,
                     _http_type=lambda:GPU6HTTPS(6, stop_at_chat_done=True))
    app.active = limiter
    return app
