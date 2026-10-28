"""Failure diagnostics never include provider messages or request data."""
import errno
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, NativeHTTPS, ProviderFailure, safe_failure_errno, safe_failure_kind
from agent_memory.service_transport import ServiceTransport


class TransportErrnoDiagnosticsTests(unittest.TestCase):
    def test_dns_and_os_codes_are_bounded_and_message_free(self):
        dns = socket.gaierror(socket.EAI_AGAIN, 'SECRET_DNS')
        self.assertEqual(safe_failure_kind(dns), 'gaierror')
        self.assertEqual(safe_failure_errno(dns), socket.EAI_AGAIN)
        self.assertIsNone(safe_failure_errno(OSError(999999, 'SECRET')))
        self.assertIsNone(safe_failure_errno(OSError('SECRET')))

    def test_tls_failure_preserves_only_numeric_errno_without_http_dispatch(self):
        conn = SimpleNamespace(connect=lambda: (_ for _ in ()).throw(OSError(errno.ENETUNREACH, 'SECRET')), close=lambda: None)
        with patch('agent_memory.provider_http.http.client.HTTPSConnection', return_value=conn):
            with self.assertRaises(ProviderFailure) as caught:
                NativeHTTPS().exchange(BAILIAN_BEIJING_ENDPOINT, b'SECRET_BODY', {'Authorization':'SECRET_KEY'}, 5, 1024)
        failure = caught.exception
        self.assertEqual(failure.phase, 'tls_setup')
        self.assertEqual(failure.response_diagnostics['error_errno'], errno.ENETUNREACH)
        self.assertNotIn('SECRET', str(failure.response_diagnostics))

    def test_service_persists_allowed_code_and_rejects_unrecognized_values(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = object.__new__(ServiceTransport)
            transport.quarantine_path = Path(directory)/'unused.jsonl'
            for code in (errno.ECONNRESET, 999999, 'SECRET'):
                http = SimpleNamespace(failure_phase='tls_setup', failure_kind='OSError',
                    failure_diagnostics={'error_errno':code,'received_bytes':0,'raw':'SECRET'})
                transport.record_transport_failure('synthetic', http, {'model':'text-embedding-v4'})
            records=[json.loads(x) for x in (Path(directory)/'transport-failure-phases.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['error_errno'],errno.ECONNRESET)
            self.assertNotIn('error_errno',records[1]);self.assertNotIn('error_errno',records[2])
            self.assertNotIn('SECRET',json.dumps(records))


if __name__ == '__main__':
    unittest.main()
