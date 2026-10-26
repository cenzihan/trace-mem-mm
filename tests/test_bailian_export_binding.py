"""Synthetic Credential Manager boundary; never reads a real credential."""
import ctypes
import importlib
import unittest
from unittest.mock import Mock, patch

from scripts import bailian_save_and_check as binding


class FakeCredentialAPI:
    def __init__(self, value, *, available=True):
        self.available = available
        raw = value.encode('utf-16-le')
        self.buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        self.credential = binding.CREDENTIALW()
        self.credential.CredentialBlobSize = len(raw)
        self.credential.CredentialBlob = ctypes.cast(self.buffer, ctypes.POINTER(ctypes.c_ubyte))
        self.read_calls = []
        self.freed = 0

    def CredReadW(self, target, credential_type, flags, output):
        self.read_calls.append((target, credential_type, flags))
        if not self.available:
            return False
        pointer = ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(binding.CREDENTIALW)))
        pointer[0] = ctypes.pointer(self.credential)
        return True

    def CredFree(self, pointer):
        self.freed += 1


class BailianExportBindingTests(unittest.TestCase):
    def test_import_does_not_access_windows_api_or_expose_setup_functions(self):
        blocked = Mock(side_effect=AssertionError('real credentials forbidden'))
        try:
            with patch('agent_memory.windows_credentials.api', blocked):
                importlib.reload(binding)
                blocked.assert_not_called()
        finally:
            importlib.reload(binding)
        for name in ('save_bailian', 'post_once', 'run_once', 'main', 'SOURCE'):
            self.assertFalse(hasattr(binding, name))

    def test_explicit_read_uses_exact_target_and_frees_native_buffer(self):
        value = 'synthetic-bailian-credential'
        api = FakeCredentialAPI(value)
        with patch.object(binding, 'api', return_value=api):
            self.assertEqual(binding.read_bailian(), value)
        self.assertEqual(api.read_calls, [(binding.TARGET, binding.CRED_TYPE_GENERIC, 0)])
        self.assertEqual(api.freed, 1)

    def test_invalid_value_is_not_echoed_and_buffer_is_freed(self):
        value = 'synthetic-secret\nunsafe'
        api = FakeCredentialAPI(value)
        with patch.object(binding, 'api', return_value=api), self.assertRaises(ValueError) as failure:
            binding.read_bailian()
        self.assertNotIn(value, str(failure.exception))
        self.assertEqual(api.freed, 1)

    def test_missing_target_has_no_fallback_or_write(self):
        api = FakeCredentialAPI('synthetic', available=False)
        with patch.object(binding, 'api', return_value=api), self.assertRaisesRegex(ValueError, '^bailian_credential_unavailable$'):
            binding.read_bailian()
        self.assertEqual(len(api.read_calls), 1)
        self.assertEqual(api.freed, 0)

    def test_header_storage_limits_match_source_validator(self):
        for value in (None, '', 'with space', 'line\nfeed', 'hidden\u200bcharacter', 'x' * 1281, '\u4e2d'):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                binding.validate_bailian(value)
        self.assertIsNone(binding.validate_bailian('synthetic-valid-value'))


if __name__ == '__main__':
    unittest.main()
