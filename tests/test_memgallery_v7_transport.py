"""Independent synthetic temperature-zero compatibility/receipt-chain tests."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.development_run import DevelopmentRun, ScopedTransport
from agent_memory.durable_chat import DurableChat
from agent_memory.memgallery_v7_transport import MemGalleryV7TemperatureZeroTransport
from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply, OPENAI_ENDPOINT


def payload():
    return dict(model='gpt-4o-mini', temperature=0, max_completion_tokens=768,
                store=False, n=1, response_format=dict(type='json_object'),
                messages=[dict(role='system', content='Synthetic fixed reader contract.'),
                          dict(role='user', content=[dict(type='text', text='Synthetic source only.')])])


class FakeHTTP:
    is_fake_http = True

    def __init__(self):
        self.requests = []

    def exchange(self, endpoint, body, headers, timeout, max_response_bytes):
        self.requests.append(json.loads(body))
        value = dict(summary='Free auxiliary summary, not a fixed constant', evidence_quotes=['Synthetic source'],
                     answer='Synthetic answer', cited_ids=[], confidence='supported')
        response = dict(model='gpt-4o-mini-2024-07-18',
                        usage=dict(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                        choices=[dict(finish_reason='stop', message=dict(content=json.dumps(value)))])
        return HTTPReply(200, json.dumps(response).encode('utf-8'), {})


class MemGalleryV7TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v7-temperature-transport-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = BudgetLedger(self.root / 'budget.sqlite')
        self.rate = VerifiedRate('gpt-4o-mini', '1.5', '6',
            'https://developers.openai.com/api/docs/models/gpt-4o-mini',
            datetime.now(timezone.utc).isoformat())
        self.no_real_http = self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',
                                                   side_effect=AssertionError('real network forbidden')))

    def test_exact_zero_survives_full_durable_scoped_transport_and_receipt_reuse(self):
        http = FakeHTTP()
        credential = Mock(side_effect=AssertionError('fake HTTP must not read any credential'))
        run = DevelopmentRun(self.root / 'run.sqlite', create=True,
            max_requests=1, max_reservation_cny='.25', approval_scope='native_100_20')
        def factory(wrapped_http, observer):
            base = MemGalleryV7TemperatureZeroTransport(model='gpt-4o-mini', endpoint=OPENAI_ENDPOINT,
                ledger=self.ledger, rate=self.rate, credential_reader=credential, http=wrapped_http,
                mode='offline_fake', fixed_reserve_cny='.25', output_token_cap=1536,
                attempt_observer=observer)
            self.assertIsInstance(base, BudgetedJSONTransport)
            return ScopedTransport(base, run, '.25')
        received = []
        def after_received():
            with executor.journal.connect() as db:
                row = db.execute('SELECT state,body,attempt_id FROM operations').fetchone()
            self.assertEqual(row[0], 'received')
            self.assertIsInstance(row[1], bytes)
            self.assertTrue(row[2])
            received.append(True)
        executor = DurableChat(self.root / 'receipts', 'synthetic-v7-temperature0', http, factory, run,
                               offline_test=True, after_received=after_received)
        original = payload()
        before = copy.deepcopy(original)
        first = executor.execute('synthetic-fixed-request', original)
        reused = executor.recover('synthetic-fixed-request', original)
        self.assertEqual(original, before)
        self.assertEqual(http.requests, [before])
        self.assertIs(type(http.requests[0]['temperature']), int)
        self.assertEqual(http.requests[0]['temperature'], 0)
        self.assertEqual(received, [True])
        self.assertEqual(first['receipt_sha256'], reused['receipt_sha256'])
        self.assertTrue(reused['recovered'])
        self.assertEqual(self.ledger.totals()['attempts'], 1)
        credential.assert_not_called()
        self.no_real_http.assert_not_called()

    def test_invalid_temperature_and_extra_parameters_fail_before_live_credential_or_budget(self):
        credential = Mock(side_effect=AssertionError('invalid payload reached credential reader'))
        class NoNetwork:
            def __init__(self):
                self.exchange = Mock(side_effect=AssertionError('invalid payload reached HTTP'))
        http = NoNetwork()
        # Construction only: the temporary ledger stands in for the live ledger.
        # Every call below must fail during _bounds before it can enter live I/O.
        with patch('agent_memory.provider_http.SHARED_LEDGER', Path(self.ledger.path)):
            client = MemGalleryV7TemperatureZeroTransport(model='gpt-4o-mini', endpoint=OPENAI_ENDPOINT,
                ledger=self.ledger, rate=self.rate, credential_reader=credential, http=http,
                mode='live', enabled=True, prices_confirmed=True, fixed_reserve_cny='.25', output_token_cap=1536)
        invalid = []
        for value in (1, -1, True, False, 0.0, '0', None):
            invalid.append(dict(payload(), temperature=value))
        missing = payload(); missing.pop('temperature'); invalid.append(missing)
        for name, value in (('top_p', 1), ('seed', 0), ('tools', []), ('stream', True)):
            invalid.append(dict(payload(), **{name: value}))
        invalid.extend([dict(payload(), n=True), dict(payload(), store=True),
                        dict(payload(), max_completion_tokens=1537), dict(payload(), model='other-model')])
        for number, body in enumerate(invalid):
            before = copy.deepcopy(body)
            with self.subTest(case=number), self.assertRaises(ValueError):
                client(OPENAI_ENDPOINT, body)
            self.assertEqual(body, before)
        credential.assert_not_called()
        http.exchange.assert_not_called()
        self.assertEqual(self.ledger.totals()['attempts'], 0)

    def test_existing_legacy_transport_still_rejects_temperature(self):
        http = FakeHTTP()
        old = BudgetedJSONTransport(model='gpt-4o-mini', endpoint=OPENAI_ENDPOINT,
            ledger=self.ledger, rate=self.rate, http=http, mode='offline_fake', output_token_cap=1536)
        with self.assertRaises(ValueError):
            old(OPENAI_ENDPOINT, payload())
        self.assertEqual(http.requests, [])
        self.assertEqual(self.ledger.totals()['attempts'], 0)

    def test_wrapper_install_changes_only_explicit_execution_bindings(self):
        from scripts import run_memgallery_v7_chat_temperature0 as entry
        original = entry.original
        function_identity = (original.prepare, original.execute, original.requests_for, original.verify_reader)
        previous_codes = original.CODE
        with patch.multiple(original, WORK=original.WORK, CODE=original.CODE,
                            BudgetedJSONTransport=original.BudgetedJSONTransport):
            entry.install()
            self.assertEqual(original.WORK.name, 'chat-temperature0')
            self.assertEqual(entry.OLD.name, 'chat')
            self.assertEqual(original.CODE, (*previous_codes, *entry.EXTRA_CODE))
            self.assertIs(original.BudgetedJSONTransport, MemGalleryV7TemperatureZeroTransport)
            self.assertEqual((original.prepare, original.execute, original.requests_for, original.verify_reader),
                             function_identity)


if __name__ == '__main__':
    unittest.main()
