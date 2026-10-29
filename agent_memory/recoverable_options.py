"""Explicit human-authorized bounded retries; every attempt retains its charge."""
import hashlib
import time
from .joint_selection import ScopedSelectionServiceTransport
from .mem0_options import Mem0OptionsSystem
from .mem0_bridge import native_scope
from .protocol import canonical
from .provider_http import ProviderFailure


class RetrySelectionTransport(ScopedSelectionServiceTransport):
    retry_failed_requests = True
    retry_delays = (1, 2, 4, 8, 16)

    def __call__(self, endpoint, payload):
        for index in range(6):
            try:
                return super().__call__(endpoint, payload)
            except ProviderFailure as exc:
                if index == 5 or exc.category not in {'timeout','network_error','tls_error','rate_limited','server_error'}:
                    raise
                # Never overlap an unfinished worker with the next attempt.
                slot = getattr(self.http, '_slot', None)
                if slot is not None and not slot.acquire(timeout=120):
                    raise
                if slot is not None:slot.release()
                time.sleep(self.retry_delays[index])


class RecoverableOptionsSystem(Mem0OptionsSystem):
    recoverable_add = True

    def capabilities(self):
        return dict(super().capabilities(), version='tracemem-mm-mem0-options-v4.3.10-rc1',
                    provider_attempts_max=6, uncertain_retry='explicit-human-authorized-billed',
                    failed_add='remove-uncommitted-native-vectors-then-retry')

    def _clear_partial_native(self, payload):
        # Only deterministic sources for this uncommitted Add, under the store lock.
        from qdrant_client import models
        uid, req = payload['user_id'], payload['request_id']
        ids = ['mem_' + hashlib.sha256(canonical([uid,payload['session_id'],req,i]).encode()).hexdigest()
               for i in range(len(payload['messages']))]
        with self.connect() as db:
            if db.execute('SELECT 1 FROM requests WHERE user_id=? AND request_id=?',(uid,req)).fetchone():
                raise ValueError('cannot clean a committed Add')
        store = self.native.vector_store
        store.client.delete(collection_name=store.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key='user_id',match=models.MatchValue(value=native_scope(uid))),
                models.FieldCondition(key='tracemem_source_id',match=models.MatchAny(any=ids))])), wait=True)

    def search(self, payload):
        # A failed query must not poison all later requests through SDK flags.
        with self.lock:
            self.native.embedding_model.failed=False
            self.native.llm.failed=False
            try:return super().search(payload)
            finally:
                self.native.embedding_model.failed=False
                self.native.llm.failed=False
