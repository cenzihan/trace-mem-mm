"""Synthetic failures only; no provider calls or protected inputs."""
import unittest

from agent_memory.mem0_bridge import BudgetEmbedding, _adapters
from agent_memory.pre_dispatch import PreDispatchRejection
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, ProviderFailure
from agent_memory.providers import TextEmbeddingV4
from agent_memory.scoped_unknown import request_owner
from tests.test_scoped_transport_integration import ScopedTransportTests


class FakeEmbedding:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return [[1.0] * 256 for _ in texts]


class PreDispatchLatchTests(unittest.TestCase):
    def make(self, error):
        fake = FakeEmbedding(error)
        token = _adapters.set({'embedding': fake})
        self.addCleanup(_adapters.reset, token)
        return BudgetEmbedding(None), fake

    def test_known_rejection_does_not_block_another_request(self):
        adapter, fake = self.make(PreDispatchRejection('retired request'))
        with self.assertRaises(PreDispatchRejection):
            adapter.embed('synthetic first input')
        self.assertFalse(adapter.failed)
        self.assertEqual(len(adapter.embed('different synthetic input')), 256)
        self.assertEqual(fake.calls, 2)

    def test_unknown_failure_still_blocks_further_calls(self):
        adapter, fake = self.make(TimeoutError('synthetic uncertainty'))
        with self.assertRaises(TimeoutError):
            adapter.embed('synthetic first input')
        self.assertTrue(adapter.failed)
        with self.assertRaises(RuntimeError):
            adapter.embed('different synthetic input')
        self.assertEqual(fake.calls, 1)


class GuardIntegrationTests(ScopedTransportTests):
    def test_retired_payload_rejection_does_not_poison_other_users(self):
        transport = self.make()
        self.http.fail = True
        with request_owner('synthetic-a'):
            with self.assertRaises(ProviderFailure):
                transport(BAILIAN_BEIJING_ENDPOINT, self.payload)
        self.http.fail = False
        embed = TextEmbeddingV4(endpoint=BAILIAN_BEIJING_ENDPOINT,
                                transport=transport, dimensions=256)
        token = _adapters.set({'embedding': embed})
        self.addCleanup(_adapters.reset, token)
        adapter = BudgetEmbedding(None)
        with request_owner('synthetic-b'):
            with self.assertRaises(PreDispatchRejection):
                adapter.embed('fixture-one')
            self.assertFalse(adapter.failed)
            adapter.embed('different synthetic input')
        self.assertEqual(self.http.calls, 2)
        with self.ledger.connect() as db:
            rows = db.execute('select status,reserved,accounted from attempts order by rowid').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 'uncertain')
        self.assertEqual(rows[0][1], rows[0][2])
        self.assertEqual(rows[1][0], 'settled')
