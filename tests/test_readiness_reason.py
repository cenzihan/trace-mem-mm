import io
import json
import unittest
from types import SimpleNamespace

from agent_memory.mem0_bridge import Mem0System
from agent_memory.readiness_error import ReadinessFailure
from agent_memory.service_app import ServiceApplication


class ReadinessReasonTests(unittest.TestCase):
    def test_sdk_latches_are_distinct_and_do_not_call_transport(self):
        for embedding, llm, expected in [(True, False, 'embedding_latch'), (False, True, 'llm_latch')]:
            transport = SimpleNamespace(ready=lambda: self.fail('transport must not be called'))
            store = SimpleNamespace(native=SimpleNamespace(embedding_model=SimpleNamespace(failed=embedding),
                                                        llm=SimpleNamespace(failed=llm)), transport=transport)
            with self.assertRaises(ReadinessFailure) as error:
                Mem0System.ready(store)
            self.assertEqual(error.exception.code, expected)

    def test_only_allowlisted_reason_is_returned(self):
        class Broken:
            def ready(self):
                raise ReadinessFailure('embedding_latch')
        app = ServiceApplication(object(), Broken(), 'token', scorer_health=lambda: None)
        values = []
        body = b''.join(app({'PATH_INFO': '/health', 'REQUEST_METHOD': 'GET', 'wsgi.input': io.BytesIO()},
                            lambda status, headers: values.append((status, dict(headers)))))
        self.assertEqual(values[0][0], '503 Service Unavailable')
        self.assertEqual(values[0][1]['X-TraceMem-Readiness'], 'embedding_latch')
        self.assertEqual(json.loads(body), {'error': 'readiness check failed'})
        with self.assertRaises(ValueError):
            ReadinessFailure('private error or request content')
