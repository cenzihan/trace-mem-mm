"""Retired users cannot re-enter through Search; real SDK, fake providers only."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from agent_memory.mem0_bridge import Mem0System
from agent_memory.mem0_evidence import Mem0EvidenceSystem
from agent_memory.mem0_raw import Mem0RawSystem
from agent_memory.mem0_options import Mem0OptionsSystem
from agent_memory.protocol import ContractError
from test_mem0_native import FakeEmbedding, FakeCards, FakeTransport
from test_mem0_joint import SelectFake


@unittest.skipUnless(importlib.util.find_spec('mem0'), 'optional pinned Mem0 dependency')
class RetiredQueryTests(unittest.TestCase):
    variants = (Mem0System, Mem0EvidenceSystem, Mem0RawSystem, Mem0OptionsSystem)

    def open_store(self, cls, path, transport, embedding, cards):
        return cls(path, transport=transport, embedding=embedding, card_adapter=cards)

    @staticmethod
    def close_store(store):
        store.close()
        connection = getattr(store.native.db, 'connection', None)
        if connection:
            connection.close()

    def exercise(self, cls, *, existing, restart):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'memory.sqlite'
            transport = SelectFake() if issubclass(cls, Mem0RawSystem) else FakeTransport()
            embedding, cards = FakeEmbedding(), FakeCards()
            retired = set()

            def check(payload):
                if payload['user_id'] in retired:
                    raise ValueError('retired uncertain input scope cannot be replayed')

            transport.assert_source_allowed = check
            store = self.open_store(cls, path, transport, embedding, cards)
            try:
                if existing:
                    store.add(dict(user_id='retired-user', request_id='r', session_id='s',
                                   messages=[dict(role='user', content='A teal notebook in a workshop.')]))
                retired.add('retired-user')
                if restart:
                    self.close_store(store)
                    store = self.open_store(cls, path, transport, embedding, cards)
                before = (len(embedding.calls), len(transport.calls), cards.calls)
                with self.assertRaises(ContractError) as error:
                    store.search(dict(user_id='retired-user', query='Where is the notebook?', top_k=100))
                self.assertEqual(error.exception.status, 503)
                self.assertEqual((len(embedding.calls), len(transport.calls), cards.calls), before)
                # An unrelated empty user still has the ordinary empty result.
                self.assertEqual(store.search(dict(user_id='unrelated', query='notebook', top_k=100)), {'data': []})
                self.assertEqual((len(embedding.calls), len(transport.calls), cards.calls), before)
            finally:
                self.close_store(store)

    def test_existing_retired_user_is_rejected_before_any_provider(self):
        for cls in self.variants:
            with self.subTest(system=cls.__name__):self.exercise(cls, existing=True, restart=False)

    def test_retired_user_cannot_get_empty_success_in_fresh_store(self):
        for cls in self.variants:
            with self.subTest(system=cls.__name__):self.exercise(cls, existing=False, restart=False)

    def test_retirement_is_enforced_after_restart(self):
        for cls in self.variants:
            with self.subTest(system=cls.__name__):self.exercise(cls, existing=True, restart=True)


if __name__ == '__main__':unittest.main()
