"""Synthetic factory and real loopback HTTP checks, not Docker-engine tests."""
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch
from wsgiref.simple_server import WSGIRequestHandler, make_server

from agent_memory.container_app import create_app


ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'synthetic-container-test-token-no-provider-credentials'
PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII='
CONTENT = [
    {'type': 'text', 'text': 'Synthetic indigo source before image.'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + PNG}},
    {'type': 'text', 'text': 'Synthetic indigo source after image.'},
]
ADD = dict(request_id='same-request', user_id='synthetic-owner', session_id='session-a',
           messages=[dict(role='user', content=CONTENT)])
QUERY = dict(query='Synthetic indigo source', user_id=ADD['user_id'], top_k=100)


def call(app, path, payload=None, method='POST', **headers):
    raw = b'' if payload is None else json.dumps(payload).encode('utf-8')
    environ = dict(PATH_INFO=path, REQUEST_METHOD=method, CONTENT_TYPE='application/json',
                   CONTENT_LENGTH=str(len(raw)), **{'wsgi.input': io.BytesIO(raw)}, **headers)
    statuses = []
    body = b''.join(app(environ, lambda status, response_headers: statuses.append(status)))
    return int(statuses[0].split()[0]), json.loads(body)


@contextmanager
def loopback_server(app):
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
            raise AssertionError('synthetic loopback server did not stop')


class ContainerAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tracemem-container-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.token_file = self.root / 'synthetic-token'
        self.token_file.write_text(TOKEN + '\n', encoding='ascii')
        self.database = self.root / 'new-volume' / 'memory.sqlite'
        self.env = dict(TRACEMEM_DB=str(self.database),
                        TRACEMEM_AUTH_TOKEN_FILE=str(self.token_file))
        self.authorized = {'HTTP_AUTHORIZATION': 'Bearer ' + TOKEN}
        # A regression that wires provider I/O must fail, even if local credentials exist.
        self.provider = self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',
                                               side_effect=AssertionError('provider I/O forbidden')))
        self.credentials = self.enterContext(patch('agent_memory.windows_credentials.read_key',
                                                  side_effect=AssertionError('provider key lookup forbidden')))

    def tearDown(self):
        self.provider.assert_not_called()
        self.credentials.assert_not_called()

    def test_missing_or_unreadable_key_fails_before_creating_database(self):
        for filename in (None, str(self.root / 'missing-token')):
            env = dict(self.env)
            if filename is None:
                env.pop('TRACEMEM_AUTH_TOKEN_FILE')
            else:
                env['TRACEMEM_AUTH_TOKEN_FILE'] = filename
            with self.subTest(filename_present=filename is not None), self.assertRaises(ValueError):
                create_app(env)
            self.assertFalse(self.database.parent.exists())

    def test_invalid_key_rejected_before_creating_database(self):
        for raw in (b'', b'too-short', b'x' * 31, b'x' * 4097,
                    b'x' * 32 + b' space', b'x' * 32 + b'\x00', b'\xff' * 32):
            self.token_file.write_bytes(raw)
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                create_app(self.env)
            self.assertFalse(self.database.parent.exists())

    def test_unsupported_mode_and_auth_fail_before_creating_database(self):
        for override in ({'TRACEMEM_MODE': 'v6'}, {'TRACEMEM_MODE': 'hybrid_rerank'},
                         {'TRACEMEM_AUTH_MODE': 'none'}, {'TRACEMEM_AUTH_MODE': 'Basic'},
                         {'TRACEMEM_DB': 'relative.sqlite'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                create_app(dict(self.env, **override))
            self.assertFalse(self.database.parent.exists())

    def test_health_and_all_authenticated_modes(self):
        for mode in ('Bearer', 'Token', 'X-Api-Key'):
            with self.subTest(mode=mode):
                app = create_app(dict(self.env, TRACEMEM_AUTH_MODE=mode))
                status, health = call(app, '/health', method='GET')
                self.assertEqual(status, 200)
                self.assertEqual(health['mode'], 'local_deterministic')
                self.assertEqual(health['deployment_profile'], 'deterministic_container')
                self.assertIs(health['experiment_v6_online'], False)
                self.assertEqual(health['dense'], 'disabled_not_simulated')
                self.assertEqual(health['reranker'], 'disabled')
                self.assertEqual(call(app, '/search', QUERY)[0], 401)
                self.assertEqual(call(app, '/add', ADD)[0], 401)
                header = 'HTTP_X_API_KEY' if mode == 'X-Api-Key' else 'HTTP_AUTHORIZATION'
                value = TOKEN if mode == 'X-Api-Key' else mode + ' ' + TOKEN
                self.assertEqual(call(app, '/search', QUERY, **{header: value + '-wrong'})[0], 401)
                self.assertEqual(call(app, '/search', QUERY, **{header: value}), (200, {'data': []}))

    def test_recreated_app_preserves_multimodal_order_scope_and_idempotency(self):
        first = create_app(self.env)
        expected = dict(success=True, **{key: ADD[key] for key in ('request_id', 'user_id', 'session_id')})
        self.assertEqual(call(first, '/add', ADD, **self.authorized), (200, expected))
        original_result = call(first, '/search', QUERY, **self.authorized)
        self.assertEqual(original_result[0], 200)
        self.assertEqual(len(original_result[1]['data']), 1)
        self.assertEqual(original_result[1]['data'][0]['content'], CONTENT)
        second = create_app(self.env)
        self.assertIsNot(first.store, second.store)
        self.assertEqual(call(second, '/search', QUERY, **self.authorized), original_result)
        self.assertEqual(call(second, '/add', copy.deepcopy(ADD), **self.authorized), (200, expected))
        conflict = copy.deepcopy(ADD)
        conflict['messages'][0]['content'][0]['text'] = 'changed synthetic evidence'
        self.assertEqual(call(second, '/add', conflict, **self.authorized)[0], 409)
        self.assertEqual(call(second, '/search', QUERY, **self.authorized), original_result)
        other_query = dict(QUERY, user_id='different-synthetic-owner')
        self.assertEqual(call(second, '/search', other_query, **self.authorized), (200, {'data': []}))
        other_add = dict(ADD, user_id=other_query['user_id'])
        self.assertEqual(call(second, '/add', other_add, **self.authorized)[0], 200)
        other_result = call(second, '/search', other_query, **self.authorized)[1]
        self.assertEqual(len(other_result['data']), 1)
        self.assertNotEqual(other_result['data'][0]['id'], original_result[1]['data'][0]['id'])
        self.assertEqual(call(second, '/search', QUERY, **self.authorized), original_result)

    def test_smoke_script_seed_and_verify_over_real_http_after_app_recreation(self):
        child_env = {name: os.environ[name] for name in ('SYSTEMROOT', 'WINDIR', 'PATH') if name in os.environ}
        child_env.update(TRACEMEM_AUTH_TOKEN_FILE=str(self.token_file), PYTHONDONTWRITEBYTECODE='1')
        request_logs = io.StringIO()
        with redirect_stderr(request_logs):
            for mode in ('Bearer', 'Token', 'X-Api-Key'):
                env = dict(self.env, TRACEMEM_AUTH_MODE=mode,
                           TRACEMEM_DB=str(self.root / (mode + '.sqlite')))
                child_env['TRACEMEM_AUTH_MODE'] = mode
                for phase in ('seed', 'verify'):
                    # Each phase gets a newly constructed app and a newly bound ephemeral port.
                    with self.subTest(mode=mode, phase=phase), loopback_server(create_app(env)) as base_url:
                        completed = subprocess.run(
                            [sys.executable, '-B', str(ROOT / 'deploy' / 'smoke_test.py'), phase,
                             '--base-url', base_url], cwd=str(ROOT), env=child_env,
                            text=True, capture_output=True, timeout=20, check=False)
                        self.assertEqual(completed.returncode, 0, 'synthetic HTTP smoke failed: ' + completed.stderr)
                        self.assertEqual(completed.stderr, '')
                        result = json.loads(completed.stdout)
                        self.assertEqual(result['state'], 'passed')
                        self.assertEqual(result['phase'], phase)
                        self.assertIs(result['multimodal_order_preserved'], True)
                        self.assertIs(result['user_isolation'], True)
                        self.assertNotIn(TOKEN, completed.stdout)
        self.assertEqual(request_logs.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
