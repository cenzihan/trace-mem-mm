import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.protocol import ContractError
from agent_memory.service_app import ServiceApplication
from agent_memory.readiness_error import ReadinessFailure


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

    def request(self, payload=None):
        captured = []
        raw = json.dumps({} if payload is None else payload).encode()
        env = {'PATH_INFO':'/add','REQUEST_METHOD':'POST', 'HTTP_AUTHORIZATION':'Bearer test',
               'CONTENT_TYPE':'application/json','CONTENT_LENGTH':str(len(raw)),'wsgi.input':io.BytesIO(raw)}
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

    def test_exception_details_never_include_message_or_locals(self):
        def fail(payload):
            try:
                raise OSError(111, 'secret question and secret key')
            except OSError as exc:
                raise RuntimeError('private response body') from exc
        self.store.add = fail
        status, body = self.request()
        self.assertTrue(status.startswith('500 '))
        self.assertEqual(json.loads(body)['error'], 'internal service error')
        text = (Path(self.temp.name) / 'request-errors.jsonl').read_text()
        event = json.loads(text)
        self.assertEqual(event['stage'], '/add')
        self.assertEqual(event['http_status'], 500)
        self.assertEqual([x['type'] for x in event['exceptions']], ['RuntimeError', 'OSError'])
        self.assertEqual(event['exceptions'][1]['errno'], 111)
        for word in ('secret', 'question', 'key', 'payload', 'response body'):
            self.assertNotIn(word, text)

    def test_readiness_reason_and_sink_failure(self):
        def fail(): raise ReadinessFailure('embedding_latch')
        self.app.transport.ready = fail
        status, _ = self.request()
        self.assertTrue(status.startswith('503 '))
        path = Path(self.temp.name) / 'request-errors.jsonl'
        entry = json.loads(path.read_text())
        self.assertEqual(entry['exceptions'][0]['readiness_code'], 'embedding_latch')
        path.unlink()
        path.mkdir()
        status, _ = self.request()
        self.assertTrue(status.startswith('503 '))

    def test_original_input_separately_private_and_no_auth_headers(self):
        payload = {'query':'synthetic question', 'messages':['synthetic source']}
        status, _ = self.request(payload)
        self.assertTrue(status.startswith('503 '))
        event = json.loads((Path(self.temp.name) / 'request-errors.jsonl').read_text())
        body_path = Path(self.temp.name) / 'private-error-inputs' / (event['private_input'] + '.json')
        body = json.loads(body_path.read_text(encoding='utf-8'))
        self.assertEqual(body, {'request':payload, 'response':{'error':'user has unresolved Add'}})
        self.assertNotIn('Bearer test', body_path.read_text(encoding='utf-8'))
        self.assertNotIn('synthetic question', json.dumps(event))

