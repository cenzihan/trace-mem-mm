"""Factory guards and shared original accounting; synthetic adapters only."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory import bailian_app, context_app
from agent_memory.bailian_rerank import MODEL
from agent_memory.context_candidate import EvidenceContextCandidate, VERSION


class ContextApplicationTests(unittest.TestCase):
    def test_context_activation_is_explicit(self):
        for value in (None, '', 'true', 'approved-versioned-bailian-v2'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'context candidate activation'):
                context_app.create_app({'TRACEMEM_CONTEXT_RETRIEVAL_ENABLED': value})

    def test_context_gate_does_not_bypass_hosted_gate(self):
        with self.assertRaisesRegex(ValueError, 'hosted model/version activation'):
            context_app.create_app({'TRACEMEM_CONTEXT_RETRIEVAL_ENABLED': 'approved-context-v2.1.0-rc1'})

    def factory(self, new):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runtime = Path(directory.name)
        state = runtime/'bailian-v2-service'
        state.mkdir()
        env = dict(TRACEMEM_CONTEXT_RETRIEVAL_ENABLED='approved-context-v2.1.0-rc1',
            TRACEMEM_HOSTED_RERANK_ENABLED='approved-versioned-bailian-v2',
            TRACEMEM_SERVICE_STATE=str(state), TRACEMEM_SERVICE_EXPIRES_AT=str(int(time.time())+3600),
            TRACEMEM_OPENAI_TOKEN_FILE='unused-openai-test-path',
            TRACEMEM_BAILIAN_TOKEN_FILE='unused-bailian-test-path',
            TRACEMEM_AUTH_TOKEN_FILE='unused-service-test-path',
            TRACEMEM_LEDGER_AUTHORITY_FILE='unchanged-authority',
            TRACEMEM_VERIFIED_PRICES_FILE='unchanged-base-prices',
            TRACEMEM_RERANK_PRICES_FILE='unchanged-rerank-prices',
            TRACEMEM_RERANK_CALL_LIMIT='512')
        with patch.object(bailian_app, 'RUNTIME', runtime), \
             patch.object(bailian_app, 'ServiceTransport') as base, \
             patch.object(bailian_app, 'HostedRerankTransport') as rerank, \
             patch.object(bailian_app, 'read_service_token', return_value='synthetic-test-token'):
            app = (context_app if new else bailian_app).create_app(env)
        return app, base, rerank, state

    def test_candidate_factory_preserves_journal_budget_and_lifetime_paths(self):
        app, base, rerank, state = self.factory(True)
        self.assertIsInstance(app.store, EvidenceContextCandidate)
        self.assertEqual(base.call_args.args[:4], (bailian_app.SHARED_LEDGER,
            'unchanged-authority', 'unchanged-base-prices', state/'provider-receipts'))
        self.assertEqual(rerank.call_args.args[:4], (bailian_app.SHARED_LEDGER,
            'unchanged-authority', 'unchanged-rerank-prices', state/'rerank-receipts'))
        self.assertEqual(rerank.call_args.kwargs, {'call_limit': 512})
        self.assertEqual(app.store.embedding.model, 'text-embedding-v4')
        self.assertEqual(app.store.embedding.dimensions, 256)
        self.assertEqual(app.store.card_adapter.model, 'gpt-4o-mini')
        self.assertEqual(app.store.scoring_adapter.model, MODEL)
        capabilities = app.store.capabilities()
        self.assertEqual(capabilities['version'], VERSION)
        self.assertEqual(capabilities['deployment_profile'], 'hosted_context_v2_1')
        self.assertEqual(capabilities['release_stage'], 'official-smoke-candidate')
        self.assertEqual(capabilities['admission_capacity'], 16)
        self.assertEqual(capabilities['queue_wait_seconds'], 30)
        self.assertTrue(capabilities['reranker_source_text_only'])
        self.assertFalse(capabilities['calibrated_abstention'])
        base.return_value.assert_not_called()
        rerank.return_value.assert_not_called()
        base.return_value.post.assert_not_called()
        rerank.return_value.post.assert_not_called()

    def test_original_factory_stays_on_fixed_v2_0_4(self):
        app, _, _, _ = self.factory(False)
        self.assertIs(type(app.store), bailian_app.BailianServingSystem)
        self.assertEqual(app.store.capabilities()['version'], 'tracemem-mm-bailian-v2.0.4')
        self.assertEqual(app.store.capabilities()['deployment_profile'], 'hosted_bailian_v2')


if __name__ == '__main__':
    unittest.main()
