"""Factory admission and secret-loading contracts; no real ledger or provider I/O."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_memory import embedding_smoke_app as factory
from agent_memory.granted_embedding_smoke import GrantedEmbeddingBlocked, GrantedEmbeddingTransport
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT
from deploy.embedding_smoke_test import add_payload


SERVICE_TOKEN = 'synthetic-smoke-service-auth-token-not-provider-key'
EMBEDDING_TOKEN = 'synthetic-embedding-key-no-provider-access'


def call(app, path, payload=None, method='POST', authenticated=False):
    body = b'' if payload is None else json.dumps(payload).encode('utf-8')
    environment = dict(PATH_INFO=path, REQUEST_METHOD=method, CONTENT_TYPE='application/json',
                       CONTENT_LENGTH=str(len(body)), **{'wsgi.input': io.BytesIO(body)})
    if authenticated:
        environment['HTTP_AUTHORIZATION'] = 'Bearer ' + SERVICE_TOKEN
    statuses = []
    response = b''.join(app(environment, lambda status, headers: statuses.append(status)))
    return int(statuses[0].split()[0]), json.loads(response)


class SyntheticGrantedTransport:
    mode = 'offline_fake'

    def __init__(self, grant, journal, reader, *, expected_grant_sha256):
        self.grant, self.journal, self.reader = grant, journal, reader
        self.expected_grant_sha256 = expected_grant_sha256
        self.calls = 0

    def __call__(self, endpoint, payload):
        if endpoint != BAILIAN_BEIJING_ENDPOINT:
            raise AssertionError('wrong embedding endpoint')
        if self.reader() != EMBEDDING_TOKEN:
            raise AssertionError('wrong synthetic credential')
        self.calls += 1
        return dict(model='text-embedding-v4',
                    data=[dict(index=index, embedding=[1.0] + [0.0] * 255)
                          for index, _ in enumerate(payload['input'])])


class EmbeddingSmokeAppTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='embedding-smoke-app-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.runtime = self.root / 'runtime'
        self.state = self.runtime / 'synthetic-live-smoke'
        self.grant = self.root / 'synthetic-grant.json'
        # Real grant parsing belongs to the independently tested transport; this
        # factory fixture only tests the separately pinned exact file bytes.
        self.grant.write_text('{"factory_fixture":true}\n', encoding='ascii')
        self.digest = hashlib.sha256(self.grant.read_bytes()).hexdigest()
        self.auth = self.root / 'synthetic-service-token'
        self.auth.write_text(SERVICE_TOKEN + '\n', encoding='ascii')
        self.env = dict(TRACEMEM_SMOKE_ENABLED='embedding-only-approved-grant',
                        TRACEMEM_AUTH_TOKEN_FILE=str(self.auth),
                        TRACEMEM_GRANT_FILE=str(self.grant), TRACEMEM_GRANT_SHA256=self.digest,
                        TRACEMEM_SMOKE_STATE=str(self.state),
                        TRACEMEM_BAILIAN_TOKEN_FILE=str(self.root / 'not-read-on-startup'))
        self.enterContext(patch.object(factory, 'RUNTIME', self.runtime))
        self.construct = self.enterContext(patch.object(factory, 'GrantedEmbeddingTransport',
                                                        side_effect=SyntheticGrantedTransport))
        self.reader = self.enterContext(patch.object(factory, 'read_embedding_token',
                                                     return_value=EMBEDDING_TOKEN))
        self.http = self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',
                                           side_effect=AssertionError('provider I/O forbidden')))
        self.windows_keys = self.enterContext(patch('agent_memory.windows_credentials.read_key',
                                                   side_effect=AssertionError('real keys forbidden')))

    def tearDown(self):
        self.http.assert_not_called()
        self.windows_keys.assert_not_called()

    def test_factory_passes_trusted_hash_and_reads_embedding_secret_only_on_call(self):
        app = factory.create_app(self.env)
        self.construct.assert_called_once()
        self.assertEqual(self.construct.call_args.kwargs['expected_grant_sha256'], self.digest)
        self.assertEqual(self.construct.call_args.args[:2], (self.grant, self.state / 'receipts'))
        self.reader.assert_not_called()
        status, health = call(app, '/health', method='GET')
        self.assertEqual(status, 200)
        self.assertEqual(health['mode'], 'live_prepaid_embedding_smoke')
        self.assertEqual(health['retrieval'], 'BM25+dense+RRF')
        self.assertEqual(health['dense'], 'text-embedding-v4')
        self.assertEqual(health['cards'], 'explicit-source-cards-no-LLM')
        self.assertEqual(health['reranker'], 'disabled')
        self.assertFalse(health['llm_enabled'])
        self.assertFalse(health['online_verified'])
        self.assertFalse(health['official_evaluation'])
        self.assertEqual(health['max_provider_requests'], 3)
        self.assertEqual(call(app, '/add', add_payload())[0], 401)
        self.reader.assert_not_called()
        self.assertEqual(call(app, '/add', add_payload(), authenticated=True)[0], 200)
        self.reader.assert_called_once_with(self.env['TRACEMEM_BAILIAN_TOKEN_FILE'])
        self.assertEqual(call(app, '/add', add_payload(), authenticated=True)[0], 200)
        self.assertEqual(app.store.embedding.transport.calls, 1)
        self.assertEqual(self.reader.call_count, 1)
        self.assertFalse((self.root / 'not-read-on-startup').exists())

    def test_unapproved_profile_and_bad_paths_fail_before_transport_or_database(self):
        changes = (
            {'TRACEMEM_SMOKE_ENABLED': ''},
            {'TRACEMEM_SMOKE_ENABLED': 'local_deterministic'},
            {'TRACEMEM_GRANT_FILE': 'relative-grant.json'},
            {'TRACEMEM_GRANT_SHA256': ''},
            {'TRACEMEM_GRANT_SHA256': 'z' * 64},
            {'TRACEMEM_GRANT_SHA256': '0' * 64},
            {'TRACEMEM_SMOKE_STATE': str(self.runtime)},
            {'TRACEMEM_SMOKE_STATE': str(self.root / 'outside-runtime')},
            {'TRACEMEM_BAILIAN_TOKEN_FILE': ''},
            {'TRACEMEM_BAILIAN_TOKEN_FILE': 'relative-secret'},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                factory.create_app(dict(self.env, **change))
            self.construct.assert_not_called()
            self.reader.assert_not_called()
            self.assertFalse(self.state.exists())

    def test_missing_invalid_auth_and_oversized_grant_have_no_state_side_effects(self):
        for contents in (b'', b'short', b'x' * 32 + b' whitespace'):
            self.auth.write_bytes(contents)
            with self.subTest(auth_length=len(contents)), self.assertRaises(ValueError):
                factory.create_app(self.env)
        self.auth.unlink()
        with self.assertRaises(ValueError):
            factory.create_app(self.env)
        self.auth.write_text(SERVICE_TOKEN, encoding='ascii')
        self.grant.write_bytes(b'x' * 65537)
        env = dict(self.env, TRACEMEM_GRANT_SHA256=hashlib.sha256(self.grant.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            factory.create_app(env)
        self.construct.assert_not_called()
        self.reader.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_changed_grant_between_factory_check_and_transport_read_is_rejected(self):
        def changed_grant(grant, journal, reader, **options):
            grant.write_text('{"changed_after_factory_check":true}', encoding='ascii')
            return GrantedEmbeddingTransport(grant, journal, reader, **options)
        with patch.object(factory, 'GrantedEmbeddingTransport', side_effect=changed_grant):
            with self.assertRaisesRegex(GrantedEmbeddingBlocked, 'grant_hash_mismatch'):
                factory.create_app(self.env)
        self.reader.assert_not_called()
        self.assertFalse(self.state.exists())


if __name__ == '__main__':
    unittest.main()
