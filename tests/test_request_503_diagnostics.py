import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.protocol import ContractError
from agent_memory.service_app import ServiceApplication


class RequestDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'request-503-reasons.jsonl'
        class Transport:
            quarantine_path = Path(self.temp.name) / 'quarantine.sqlite'
            def ready(self): pass
        class Store:
            error = 'user has unresolved Add'
            def add(self, payload): raise ContractError(self.error, 503)
        self.store = Store()
        self.app = ServiceApplication(self.store, Transport(), 'test', scorer_health=lambda: None)

    def request(self):
        captured = []
        env = {'PATH_INFO':'/add','REQUEST_METHOD':'POST', 'HTTP_AUTHORIZATION':'Bearer test',
               'CONTENT_TYPE':'application/json','CONTENT_LENGTH':'2','wsgi.input':io.BytesIO(b'{}')}
        body = self.app(env, lambda status, headers: captured.append(status))
        return captured[0], b''.join(body)

    def test_fixed_reason_and_no_content(self):
        status, body = self.request()
        self.assertTrue(status.startswith('503 '))
        self.assertEqual(json.loads(body)['error'], 'user has unresolved Add')
        entry = json.loads(self.path.read_text())
        self.assertEqual(entry['reason'], 'unresolved_user_add')
        self.assertEqual(set(entry), {'at', 'reason'})

    def test_unknown_error_is_not_logged(self):
        self.store.error = 'private text and private identifier'
        self.request()
        text = self.path.read_text()
        self.assertEqual(json.loads(text)['reason'], 'other')
        self.assertNotIn('private', text)

    def test_full_or_unwritable_sink_does_not_change_response(self):
        self.path.write_bytes(b'x' * (256 * 1024))
        status, _ = self.request()
        self.assertTrue(status.startswith('503 '))
        self.assertEqual(self.path.stat().st_size, 256 * 1024)
        self.path.unlink()
        self.path.mkdir()
        status, _ = self.request()
        self.assertTrue(status.startswith('503 '))

