"""Prepaid embedding transport contracts; only synthetic keys and fake HTTP."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone
import uuid

from agent_memory.embedding_executor import _json
from agent_memory.granted_embedding_smoke import GrantedEmbeddingBlocked, GrantedEmbeddingTransport
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, HTTPReply, ProviderFailure


def payload(texts):
    return dict(model="text-embedding-v4", input=texts, dimensions=256, encoding_format="float")


class FakeHTTP:
    is_fake_http = True

    def __init__(self, result=None, entered=None, release=None):
        self.result, self.entered, self.release = result, entered, release
        self.calls = 0

    def exchange(self, endpoint, body, headers, timeout, limit):
        self.calls += 1
        if self.entered:
            self.entered.set()
        if self.release and not self.release.wait(5):
            raise AssertionError("test failed to release the fake response")
        if isinstance(self.result, BaseException):
            raise self.result
        if self.result is not None:
            return self.result
        request = json.loads(body)
        count = len(request["input"])
        response = dict(model="text-embedding-v4", usage=dict(total_tokens=3 * count),
                        data=[dict(index=i, embedding=[float(i + 1)] + [0.0] * 255)
                              for i in reversed(range(count))])
        return HTTPReply(200, json.dumps(response).encode())


class GrantedEmbeddingSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="granted-smoke-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "grant.json"
        self.outbox = self.root / "outbox"
        now = datetime.now(timezone.utc)
        self.requests = [payload(["Synthetic cobalt source", "Synthetic amber source"]),
                         payload(["Find the cobalt source"]), payload(["Locate the amber source"])]
        slots = [dict(ordinal=i, parent_attempt_id="attempt_" + uuid.uuid4().hex,
                      payload_sha256=hashlib.sha256(_json(value).encode()).hexdigest(), payload=value,
                      max_input_tokens=8192 * len(value["input"]), reserved_micro_cny=50000)
                 for i, value in enumerate(self.requests, 1)]
        self.grant = dict(version=1, grant_id=str(uuid.uuid4()), model="text-embedding-v4",
                          endpoint=BAILIAN_BEIJING_ENDPOINT, dimensions=256,
                          price_source="https://help.aliyun.com/zh/model-studio/text-embedding-v4",
                          price_checked_at=(now - timedelta(seconds=5)).isoformat(),
                          expires_at=(now + timedelta(hours=1)).isoformat(), slots=slots)
        self.save(self.grant)
        self.reader = Mock(side_effect=AssertionError("offline tests must not read credentials"))

    def save(self, value):
        self.path.write_text(_json(value), encoding="utf-8")
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def transport(self, http=None, **options):
        return GrantedEmbeddingTransport(self.path, self.outbox, self.reader,
                                         http=http or FakeHTTP(), offline_test=True, **options)

    def call(self, transport, index=0):
        return transport(BAILIAN_BEIJING_ENDPOINT, self.requests[index])

    def test_complete_three_prepaid_calls_and_no_remote_budget(self):
        http = FakeHTTP()
        transport = self.transport(http)
        for i in range(3):
            result = self.call(transport, i)
            self.assertEqual(result["model"], "text-embedding-v4")
            self.assertEqual(len(result["data"]), 2 if i == 0 else 1)
        summary = transport.summary()
        self.assertEqual(http.calls, 3)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["maximum_reserved_micro_cny"], 150000)
        self.assertFalse(summary["remote_budget_created"])
        self.assertTrue(summary["parent_reconciliation_required"])
        self.assertEqual([slot["input_tokens"] for slot in summary["slots"]], [6, 3, 3])
        self.assertEqual([slot["parent_attempt_id"] for slot in summary["slots"]],
                         [slot["parent_attempt_id"] for slot in self.grant["slots"]])
        with transport.journal.connect() as database:
            tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables, {"operations", "claims"})
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "grant_exhausted"):
            self.call(transport, 2)
        self.assertEqual(http.calls, 3)
        self.reader.assert_not_called()
        self.assertNotIn('"embedding":', json.dumps(summary["slots"]))
        self.assertNotIn("cobalt", json.dumps(summary))

    def test_restart_resumes_next_slot_without_replaying_completed(self):
        first_http = FakeHTTP()
        first = self.transport(first_http)
        self.call(first)
        second_http = FakeHTTP()
        restarted = self.transport(second_http)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "next_slot"):
            self.call(restarted, 0)
        self.assertEqual(second_http.calls, 0)
        self.call(restarted, 1)
        self.assertEqual(first_http.calls, 1)
        self.assertEqual(second_http.calls, 1)
        self.assertEqual(restarted.summary()["dispatches_started"], 2)

    def test_out_of_order_payload_and_wrong_endpoint_block_before_http(self):
        http = FakeHTTP()
        transport = self.transport(http)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "next_slot"):
            self.call(transport, 1)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "endpoint_mismatch"):
            transport("https://api.openai.com/v1/embeddings", self.requests[0])
        self.assertEqual(http.calls, 0)
        self.assertEqual(transport.summary()["dispatches_started"], 0)
        self.reader.assert_not_called()

    def test_unknown_request_freezes_across_restart_without_new_call(self):
        http = FakeHTTP(ProviderFailure("timeout"))
        transport = self.transport(http)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "timeout"):
            self.call(transport)
        slot = transport.summary()["slots"][0]
        self.assertEqual(slot["state"], "unknown")
        self.assertTrue(slot["dispatch_started"])
        self.assertIsNone(slot["receipt_sha256"])
        next_http = FakeHTTP()
        restarted = self.transport(next_http)
        for i in (0, 1):
            with self.assertRaisesRegex(GrantedEmbeddingBlocked, "unfinished"):
                self.call(restarted, i)
        self.assertEqual(http.calls, 1)
        self.assertEqual(next_http.calls, 0)

    def test_bad_json_receipt_is_saved_before_parser_and_freezes(self):
        body = b"{bad synthetic JSON"
        http = FakeHTTP(HTTPReply(200, body))
        transport = self.transport(http)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "invalid_json"):
            self.call(transport)
        row = transport.journal.row(transport.slots[0][1])
        self.assertEqual(row["body"], body)
        self.assertEqual(row["body_sha"], hashlib.sha256(body).hexdigest())
        self.assertEqual(row["state"], "failed")
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "unfinished"):
            self.call(transport)
        self.assertEqual(http.calls, 1)

    def test_response_receipt_exists_before_success_parser(self):
        transport = self.transport()
        original = transport._parse_reply
        observed = []
        def inspected(reply, slot, fingerprint):
            row = transport.journal.row(fingerprint)
            observed.append(row["state"])
            self.assertEqual(row["body_sha"], hashlib.sha256(row["body"]).hexdigest())
            return original(reply, slot, fingerprint)
        with patch.object(transport, "_parse_reply", side_effect=inspected):
            self.call(transport)
        self.assertEqual(observed, ["received"])

    def test_http_rejection_keeps_receipt_and_no_retry(self):
        http = FakeHTTP(HTTPReply(403, b'{"error":"region rejection"}'))
        transport = self.transport(http)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "permission_or_region"):
            self.call(transport)
        summary = transport.summary()["slots"][0]
        self.assertEqual(summary["http_status"], 403)
        self.assertIsNotNone(summary["receipt_sha256"])
        self.assertIsNone(summary["input_tokens"])
        with self.assertRaises(GrantedEmbeddingBlocked):
            self.call(transport, 1)
        self.assertEqual(http.calls, 1)

    def test_usage_is_preserved_when_vector_contract_fails(self):
        value = dict(model="text-embedding-v4", usage=dict(total_tokens=17), data=[])
        transport = self.transport(FakeHTTP(HTTPReply(200, json.dumps(value).encode())))
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "invalid_contract"):
            self.call(transport)
        slot = transport.summary()["slots"][0]
        self.assertEqual(slot["input_tokens"], 17)
        self.assertEqual(slot["returned_model"], "text-embedding-v4")
        self.assertEqual(slot["output_tokens"], 0)

    def test_mismatched_model_and_usage_overflow_do_not_complete(self):
        for model, tokens, category in (("other-model", 3, "model_mismatch"),
                                        ("text-embedding-v4", 16385, "invalid_usage")):
            with self.subTest(category=category):
                root = self.root / category
                value = dict(model=model, usage=dict(total_tokens=tokens), data=[])
                transport = GrantedEmbeddingTransport(self.path, root, self.reader,
                    http=FakeHTTP(HTTPReply(200, json.dumps(value).encode())), offline_test=True)
                with self.assertRaisesRegex(GrantedEmbeddingBlocked, category):
                    self.call(transport)
                self.assertEqual(transport.summary()["slots"][0]["state"], "failed")

    def test_only_one_concurrent_claim_dispatches(self):
        entered, release = threading.Event(), threading.Event()
        http = FakeHTTP(entered=entered, release=release)
        transport = self.transport(http)
        errors = []
        def first():
            try:
                self.call(transport)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=first)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            for i in (0, 1):
                with self.assertRaises(GrantedEmbeddingBlocked):
                    self.call(transport, i)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(http.calls, 1)

    def test_changed_grant_blocks_in_process_and_after_restart(self):
        transport = self.transport()
        changed = copy.deepcopy(self.grant)
        changed["grant_id"] = str(uuid.uuid4())
        self.save(changed)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "grant_file_changed"):
            self.call(transport)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "changed_since"):
            self.transport()
        self.reader.assert_not_called()

    def test_completed_receipt_tampering_blocks_next_slot(self):
        transport = self.transport()
        self.call(transport)
        with transport.journal.connect() as database:
            database.execute("UPDATE operations SET body=? WHERE fingerprint=?",
                             (b"tampered", transport.slots[0][1]))
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "completed_receipt_changed"):
            self.call(transport, 1)

    def test_grant_validation_rejects_before_journal_or_credentials(self):
        invalid = []
        value = copy.deepcopy(self.grant); value["model"] = "gpt-4o-mini"; invalid.append(value)
        value = copy.deepcopy(self.grant); value["slots"] = value["slots"][:2]; invalid.append(value)
        value = copy.deepcopy(self.grant); value["slots"][0]["reserved_micro_cny"] = 5000000; invalid.append(value)
        value = copy.deepcopy(self.grant); value["slots"][1]["parent_attempt_id"] = value["slots"][0]["parent_attempt_id"]; invalid.append(value)
        value = copy.deepcopy(self.grant); value["slots"][0]["payload_sha256"] = "0" * 64; invalid.append(value)
        value = copy.deepcopy(self.grant); value["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(); invalid.append(value)
        value = copy.deepcopy(self.grant); value["price_source"] = "https://example.com/pricing"; invalid.append(value)
        for i, value in enumerate(invalid):
            with self.subTest(index=i):
                self.save(value)
                with self.assertRaises(GrantedEmbeddingBlocked):
                    self.transport()
                self.assertFalse(self.outbox.exists())
        self.reader.assert_not_called()

    def test_trusted_hash_and_http_mode_are_enforced(self):
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "grant_hash_mismatch"):
            self.transport(expected_grant_sha256="0" * 64)
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "http_mode_mismatch"):
            GrantedEmbeddingTransport(self.path, self.outbox, self.reader, http=FakeHTTP())
        with self.assertRaisesRegex(GrantedEmbeddingBlocked, "trusted_grant_hash_required"):
            GrantedEmbeddingTransport(self.path, self.outbox, self.reader)
        self.reader.assert_not_called()

    def test_credential_echo_is_not_saved_or_returned_in_summary(self):
        body = b'{"error":"SYNTHETIC_GRANT_KEY_NEVER_LOG"}'
        transport = self.transport(FakeHTTP(HTTPReply(401, body)))
        with self.assertRaises(GrantedEmbeddingBlocked):
            self.call(transport)
        row = transport.journal.row(transport.slots[0][1])
        self.assertNotIn(b"SYNTHETIC_GRANT_KEY_NEVER_LOG", row["body"])
        self.assertTrue(row["redacted"])
        self.assertNotIn("SYNTHETIC_GRANT_KEY_NEVER_LOG", json.dumps(transport.summary()))


if __name__ == "__main__":
    unittest.main()
