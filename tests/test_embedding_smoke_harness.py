"""Real loopback harness checks with fake vectors; no provider or real keys."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from wsgiref.simple_server import WSGIRequestHandler, make_server

from agent_memory.app import Application
from agent_memory.providers import TextEmbeddingV4
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT
from agent_memory.system import AgentMemorySystem, RetrievalConfig
from deploy import embedding_smoke_test as harness


ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'synthetic-embedding-smoke-token-not-a-provider-key'


class CaptureTransport:
    mode = 'offline_fake'

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, payload):
        if endpoint != BAILIAN_BEIJING_ENDPOINT:
            raise AssertionError('unexpected embedding endpoint')
        self.calls.append(copy.deepcopy(payload))
        vectors = []
        for index, text in enumerate(payload['input']):
            # Fake orthogonal vectors make keys and meeting records distinct.
            first, second = (0.0, 1.0) if 'meeting' in text else (1.0, 0.0)
            vectors.append(dict(index=index, embedding=[first, second] + [0.0] * 254))
        return dict(model='text-embedding-v4', data=list(reversed(vectors)))


class CaptureEmbedding(TextEmbeddingV4):
    def __init__(self, transport):
        super().__init__(BAILIAN_BEIJING_ENDPOINT, transport, dimensions=256)


class HarnessSystem(AgentMemorySystem):
    def capabilities(self):
        # Synthetic metadata is needed to exercise the live-profile HTTP client;
        # CaptureTransport is always fake and no live success is claimed here.
        return dict(super().capabilities(), mode=harness.PROFILE)


@contextmanager
def loopback(app):
    class QuietHandler(WSGIRequestHandler):
        def log_message(self, *args):
            pass
    server = make_server('127.0.0.1', 0, app, handler_class=QuietHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
        if worker.is_alive():
            raise AssertionError('test HTTP server did not stop')


class EmbeddingSmokeHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='embedding-smoke-harness-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / 'memory.sqlite'
        self.token = self.root / 'synthetic-service-token'
        self.token.write_text(TOKEN + '\n', encoding='ascii')
        self.capture = CaptureTransport()
        self.provider = self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',
                                               side_effect=AssertionError('provider I/O forbidden')))
        self.keys = self.enterContext(patch('agent_memory.windows_credentials.read_key',
                                           side_effect=AssertionError('real credential access forbidden')))

    def tearDown(self):
        self.provider.assert_not_called()
        self.keys.assert_not_called()

    def invoke(self, phase, base_url):
        env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'PATH') if key in os.environ}
        env.update(TRACEMEM_AUTH_TOKEN_FILE=str(self.token), TRACEMEM_AUTH_MODE='Bearer',
                   PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'deploy/embedding_smoke_test.py'),
             phase, '--base-url', base_url], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.stderr, '')
        self.assertNotIn(TOKEN, result.stdout)
        for text in harness.SOURCE_TEXTS:
            self.assertNotIn(text, result.stdout)
        return result.returncode, json.loads(result.stdout)

    def test_seed_and_recreated_service_use_exactly_three_frozen_requests(self):
        for phase, expected_count in (('seed', 2), ('verify', 3)):
            service = HarnessSystem(self.database, RetrievalConfig(dense=True),
                                    embedding=CaptureEmbedding(self.capture))
            with loopback(Application(service, auth_mode='Bearer', key=TOKEN)) as url:
                status, report = self.invoke(phase, url)
            self.assertEqual(status, 0, report)
            self.assertEqual(report['state'], 'passed')
            self.assertFalse(report['provider_receipts_verified'])
            self.assertEqual(report['embedding_requests_expected'], 2 if phase == 'seed' else 1)
            self.assertEqual(len(self.capture.calls), expected_count)
            self.assertEqual(service.inspect_user(harness.USER_ID)['memories'], 2)
            self.assertEqual(service.inspect_user(harness.OTHER_USER_ID)['memories'], 0)
            with service.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM requests').fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM add_operations WHERE state='complete'").fetchone()[0], 1)
                vectors = db.execute('SELECT vector,vector_model FROM cards').fetchall()
                self.assertEqual(len(vectors), 2)
                self.assertTrue(all(len(json.loads(vector)) == 256 and model == 'text-embedding-v4'
                                    for vector, model in vectors))
        # This compares the native service's real text derivation and adapter
        # payload building with the separate frozen grant specification.
        self.assertEqual(self.capture.calls, harness.expected_embedding_payloads())
        self.assertNotEqual(self.capture.calls[1]['input'], self.capture.calls[2]['input'])

    def test_deterministic_baseline_is_rejected_before_any_add_or_provider_call(self):
        baseline = AgentMemorySystem(self.database)
        with loopback(Application(baseline, auth_mode='Bearer', key=TOKEN)) as url:
            status, report = self.invoke('seed', url)
        self.assertEqual(status, 1)
        self.assertEqual(report['error'], 'expected_embedding_only_profile_required')
        self.assertEqual(baseline.inspect_user(harness.USER_ID)['memories'], 0)
        self.assertEqual(self.capture.calls, [])

    def test_fixture_and_expected_payloads_return_independent_values(self):
        add = harness.add_payload()
        add['messages'][0]['content'] = 'mutated'
        self.assertEqual(harness.add_payload()['messages'][0]['content'], harness.SOURCE_TEXTS[0])
        first = harness.expected_embedding_payloads()
        first[0]['input'][0] = 'mutated'
        self.assertEqual(harness.expected_embedding_payloads()[0]['input'][0], harness.SOURCE_TEXTS[0] + ' ')
        with self.assertRaises(ValueError):
            harness.query_payload('unknown')

    def test_remote_or_credential_bearing_urls_are_rejected(self):
        invalid = ('https://127.0.0.1:8000', 'http://example.org:8000',
                   'http://user:secret@127.0.0.1:8000', 'http://127.0.0.1:8000/path',
                   'http://127.0.0.1:8000?token=synthetic', 'http://127.0.0.1:8000#fragment',
                   'http://127.0.0.1:0', 'http://127.0.0.1:99999')
        for value in invalid:
            with self.subTest(url=value), self.assertRaises(harness.SmokeFailure):
                harness.local_target(value)
        self.assertEqual(harness.local_target('http://127.0.0.1:8000'), ('127.0.0.1', 8000))


if __name__ == '__main__':
    unittest.main()
