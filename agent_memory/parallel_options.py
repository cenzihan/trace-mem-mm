"""Bounded network concurrency; original Mem0 writes remain atomic and serialized.

Embedding preparation retains the exact original chunking, calls and vectors.
No source content is cached across requests. The local Qdrant instance is never
written concurrently. Search selects from a committed, immutable source snapshot.
"""
from collections import deque
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import copy
import hashlib
import queue
import threading

from .caption_image import caption_part
from .mem0_bridge import BudgetEmbedding
from .protocol import ContractError, canonical, content_parts, text_of, validate_add
from .provider_http import NativeHTTPS
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport
from .scoped_unknown import bound_request


class ParallelRetryTransport(RetrySelectionTransport):
    """Four independent HTTP lanes, one original transactional ledger/journal.

Only fingerprints currently owned by this process may coexist in-flight. After
restart an unaccounted orphan is still rejected. A timed-out OS worker retains
its lane until it really exits, including when all bounded retries are exhausted.
"""
    lane_wait_seconds = 120

    def __init__(self, *args, http_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        template = self.http
        if http_factory is None:
            if type(template) is not NativeHTTPS:
                raise ValueError('parallel transport requires isolated native HTTP lanes')
            http_factory = lambda: NativeHTTPS(template.handshake_attempts,
                                               stop_at_chat_done=template.stop_at_chat_done)
        self._lanes = queue.Queue(maxsize=4)
        for _ in range(4):
            http = http_factory()
            if bool(getattr(http, 'is_fake_http', False)) != self.offline_test:
                raise ValueError('parallel HTTP execution mode mismatch')
            if any(http is item for item in self._lanes.queue):
                raise ValueError('each lane must own a distinct HTTP client')
            self._lanes.put(http)
        self._live = set()
        self._live_lock = threading.RLock()
        self.journal.runtime_active = self._runtime_active

    def ready(self):
        # Keep ownership stable across the journal snapshot and validation. A
        # completed worker must not disappear between reading its in-flight row
        # and checking ownership, which would misclassify normal completion.
        with self._live_lock:super().ready()

    def _runtime_active(self, row):
        with self._live_lock:
            return (row['scope'] == self.receipt_scope and
                    row['state'] in ('ready', 'inflight', 'received', 'complete', 'unknown', 'failed') and
                    row['fingerprint'] in self._live)

    @contextmanager
    def _attempt_scope(self, fingerprint):
        with self._live_lock:self._live.add(fingerprint)
        try:yield
        finally:
            with self._live_lock:self._live.remove(fingerprint)

    def __call__(self, endpoint, payload):
        # Admission is already bounded by HTTP request execution slots. Waiting
        # here makes no provider attempt and cannot consume a paid retry.
        try:http = self._lanes.get(timeout=self.lane_wait_seconds)
        except queue.Empty:raise ContractError('provider execution capacity busy', 429) from None
        lane = copy.copy(self)
        lane.http = http
        lane.lock = nullcontext()
        try:
            return RetrySelectionTransport.__call__(lane, endpoint, payload)
        finally:
            slot = getattr(http, '_slot', None)
            if slot is None or slot.acquire(blocking=False):
                if slot is not None:slot.release()
                self._lanes.put(http)
            else:
                def release_when_finished():
                    with slot:pass
                    self._lanes.put(http)
                threading.Thread(target=release_when_finished, daemon=True).start()


class RequestEmbedding(BudgetEmbedding):
    """Request-local SDK error flag and one-use prepared vectors, never fallback."""
    def __init__(self, original):
        self.config = original.config
        self.embedding = original.embedding
        self._failed = ContextVar('parallel_embedding_failed', default=False)
        self._prepared = ContextVar('parallel_prepared_vectors', default=None)

    @property
    def failed(self):return self._failed.get()

    @failed.setter
    def failed(self, value):self._failed.set(value)

    @contextmanager
    def prepared(self, pairs):
        pending = deque(pairs)
        token = self._prepared.set(pending)
        try:
            yield
            if pending:raise ValueError('native write did not consume prepared vectors')
        finally:self._prepared.reset(token)

    def embed(self, text, memory_action=None):
        pending = self._prepared.get()
        if pending is None:return super().embed(text, memory_action)
        if memory_action != 'add' or not pending or pending[0][0] != text:
            raise ValueError('native embedding preparation mismatch')
        return pending.popleft()[1]


class KeyedLocks:
    """Duplicate-operation gates are removed once the last waiter leaves."""
    def __init__(self):
        self.guard = threading.Lock()
        self.entries = {}

    @contextmanager
    def hold(self, key):
        with self.guard:
            entry = self.entries.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:yield
        finally:
            with self.guard:
                entry[1] -= 1
                if not entry[1]:del self.entries[key]


class ParallelOptionsSystem(RecoverableOptionsSystem):
    parallel_selection = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native.embedding_model = RequestEmbedding(self.native.embedding_model)
        self._operations = KeyedLocks()

    def capabilities(self):
        return dict(super().capabilities(), version='tracemem-mm-parallel-v4.3.11-rc1',
                    concurrency=4, provider_concurrency=4, queue_wait_seconds=120,
                    writes='parallel-embedding-serial-atomic-commit',
                    search_concurrency='committed-snapshot-selection')

    @bound_request
    def add(self, payload):
        validate_add(payload)
        self._assert_user_allowed(payload)
        # All image/capacity checks still precede any model dispatch.
        for message in payload['messages']:
            for part in content_parts(message['content']):
                if part['type'] == 'image_url':caption_part(part)
        uid, req = payload['user_id'], payload['request_id']
        digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
        with self._operations.hold((uid, req)):
            with self.lock, self.connect() as db:
                old = db.execute('SELECT digest,state FROM add_operations WHERE user_id=? AND request_id=?',
                                 (uid, req)).fetchone()
                if old and old[0] != digest:raise ContractError('request_id conflict', 409)
                if old and old[1] == 'complete':return self._persist_add(db, payload)
            adapter = self.native.embedding_model
            adapter.failed = False
            try:
                # No writes occur until every vector is ready. Different Adds,
                # including those for the SAME user, can prepare simultaneously.
                pairs = []
                for message in payload['messages']:
                    text = text_of(content_parts(message['content']))
                    if text.strip():pairs.append((text, adapter.embed(text, 'add')))
                with adapter.prepared(pairs):
                    return super().add(payload)
            finally:adapter.failed = False
