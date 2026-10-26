"""Read-only audit of real synthetic SQLite files and fake-provider receipts."""
import base64
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from agent_memory.embedding_executor import _json
from agent_memory.granted_embedding_smoke import GrantedEmbeddingTransport
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, HTTPReply
from agent_memory.providers import TextEmbeddingV4
from agent_memory.system import AgentMemorySystem, RetrievalConfig
from deploy import audit_embedding_smoke as auditor
from deploy.embedding_smoke_test import add_payload, expected_embedding_payloads, query_payload


class FakeHTTP:
    is_fake_http = True

    def exchange(self, endpoint, body, headers, timeout, limit):
        request = json.loads(body)
        count = len(request["input"])
        value = dict(model="text-embedding-v4", usage=dict(total_tokens=count * 7),
                     data=[dict(index=i, embedding=[1.0, float(i)] + [0.0] * 254) for i in range(count)])
        return HTTPReply(200, json.dumps(value).encode())


class EmbeddingSmokeAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="embedding-audit-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "state"
        self.path = self.root / "grant.json"
        now = datetime.now(timezone.utc)
        slots = [dict(ordinal=i, parent_attempt_id="attempt_" + uuid.uuid4().hex,
                      payload_sha256=hashlib.sha256(_json(payload).encode()).hexdigest(), payload=payload,
                      max_input_tokens=8192 * len(payload["input"]), reserved_micro_cny=50000)
                 for i, payload in enumerate(expected_embedding_payloads(), 1)]
        self.grant = dict(version=1, grant_id=str(uuid.uuid4()), model="text-embedding-v4",
                          endpoint=BAILIAN_BEIJING_ENDPOINT, dimensions=256,
                          price_source="https://help.aliyun.com/zh/model-studio/text-embedding-v4",
                          price_checked_at=(now - timedelta(seconds=5)).isoformat(),
                          expires_at=(now + timedelta(hours=1)).isoformat(), slots=slots)
        self.path.write_text(_json(self.grant), encoding="utf-8")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        reader = Mock(side_effect=AssertionError("No credentials in audit fixtures"))
        self.transport = GrantedEmbeddingTransport(self.path, self.state / "receipts", reader,
                                                    http=FakeHTTP(), offline_test=True)
        embedding = TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT, self.transport, dimensions=256)
        service = AgentMemorySystem(self.state / "memory.sqlite", RetrievalConfig(dense=True), embedding=embedding)
        service.add(add_payload())
        service.search(query_payload("seed"))
        recreated = AgentMemorySystem(self.state / "memory.sqlite", RetrievalConfig(dense=True), embedding=embedding)
        recreated.search(query_payload("verify"))
        reader.assert_not_called()
        self.journal = self.state / "receipts" / "private-embedding.sqlite"

    def audit(self):
        return auditor.audit(self.path, self.digest, self.state)

    def change(self, path, sql, parameters=()):
        with contextlib.closing(sqlite3.connect(path)) as connection:
            connection.execute(sql, parameters)
            connection.commit()

    def test_success_verifies_receipts_vectors_and_never_mutates_inputs(self):
        files = [self.path, self.journal, self.state / "memory.sqlite", self.state / "receipts" / "grant-binding.json"]
        before = {path: path.read_bytes() for path in files}
        with patch("agent_memory.provider_http.NativeHTTPS.exchange", side_effect=AssertionError("HTTP forbidden")), \
             patch("agent_memory.granted_embedding_smoke.GrantedEmbeddingTransport.__init__", side_effect=AssertionError("transport construction forbidden")):
            public, private = self.audit()
        self.assertEqual(public["state"], "passed")
        self.assertEqual(public["provider_receipts_verified"], 3)
        self.assertEqual(public["input_tokens_total"], 28)
        self.assertTrue(public["memory"]["source_vectors_match_receipt"])
        self.assertEqual(public["memory"]["records"], 2)
        self.assertNotIn("body_base64", json.dumps(public))
        self.assertNotIn("spare door keys", json.dumps(public))
        self.assertNotIn('"embedding":', json.dumps(public))
        for record in private["records"]:
            body = base64.b64decode(record["body_base64"], validate=True)
            self.assertEqual(hashlib.sha256(body).hexdigest(), record["body_sha256"])
        self.assertEqual(before, {path: path.read_bytes() for path in files})

    def test_private_export_is_exclusive_and_private(self):
        public, private = self.audit()
        destination = self.root / "private-reconciliation.json"
        auditor.export_private(destination, private)
        self.assertEqual(json.loads(destination.read_text()), private)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        original = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            auditor.export_private(destination, private)
        self.assertEqual(original, destination.read_bytes())

    def test_incomplete_slot_refuses_proof(self):
        self.change(self.journal, "UPDATE operations SET state='unknown' WHERE run_ordinal=3")
        with self.assertRaisesRegex(auditor.AuditFailure, "attempt_not_complete"):
            self.audit()

    def test_extra_operation_or_claim_refuses_proof(self):
        self.change(self.journal, "INSERT INTO operations(fingerprint,scope,manifest,state) VALUES('extra','extra','{}','ready')")
        with self.assertRaisesRegex(auditor.AuditFailure, "attempt_count"):
            self.audit()

    def test_receipt_hash_tampering_refuses_proof(self):
        self.change(self.journal, "UPDATE operations SET body=? WHERE run_ordinal=2", (b"tampered",))
        with self.assertRaisesRegex(auditor.AuditFailure, "receipt_hash"):
            self.audit()

    def test_rehashed_wrong_dimension_receipt_still_refused(self):
        with contextlib.closing(sqlite3.connect(self.journal)) as connection:
            raw = connection.execute("SELECT body FROM operations WHERE run_ordinal=2").fetchone()[0]
        value = json.loads(raw)
        value["data"][0]["embedding"] = [1.0]
        replacement = json.dumps(value).encode()
        self.change(self.journal, "UPDATE operations SET body=?,body_sha=? WHERE run_ordinal=2",
                    (replacement, hashlib.sha256(replacement).hexdigest()))
        with self.assertRaisesRegex(auditor.AuditFailure, "receipt_vector_invalid"):
            self.audit()

    def test_rehashed_model_substitution_refused(self):
        with contextlib.closing(sqlite3.connect(self.journal)) as connection:
            raw = connection.execute("SELECT body FROM operations WHERE run_ordinal=3").fetchone()[0]
        value = json.loads(raw)
        value["model"] = "different-model"
        replacement = json.dumps(value).encode()
        self.change(self.journal, "UPDATE operations SET body=?,body_sha=? WHERE run_ordinal=3",
                    (replacement, hashlib.sha256(replacement).hexdigest()))
        with self.assertRaisesRegex(auditor.AuditFailure, "receipt_model_mismatch"):
            self.audit()

    def test_usage_summary_must_match_raw_receipt(self):
        self.change(self.journal, "UPDATE operations SET grant_input_tokens=999 WHERE run_ordinal=1")
        with self.assertRaisesRegex(auditor.AuditFailure, "journal_usage_mismatch"):
            self.audit()

    def test_memory_vector_must_equal_add_receipt(self):
        vector = json.dumps([0.0, 1.0] + [0.0] * 254)
        self.change(self.state / "memory.sqlite", "UPDATE cards SET vector=?", (vector,))
        with self.assertRaisesRegex(auditor.AuditFailure, "vector_receipt_mismatch"):
            self.audit()

    def test_memory_user_isolation_mismatch_refused(self):
        self.change(self.state / "memory.sqlite", "UPDATE cards SET user_id='unexpected-user'")
        with self.assertRaisesRegex(auditor.AuditFailure, "index_identity_mismatch"):
            self.audit()

    def test_grant_pin_mismatch_blocks_before_any_database_read(self):
        with patch.object(auditor, "readonly_database", side_effect=AssertionError("must not read database")):
            with self.assertRaisesRegex(auditor.AuditFailure, "grant_hash_mismatch"):
                auditor.audit(self.path, "0" * 64, self.state)

    def test_completed_run_can_be_audited_after_expiry(self):
        with patch.object(auditor, "datetime", wraps=datetime) as time:
            time.now.return_value = datetime.now(timezone.utc) + timedelta(days=4)
            public, _ = self.audit()
        self.assertEqual(public["state"], "passed")

    def test_cli_failure_is_sanitized_and_does_not_write_private_export(self):
        self.change(self.journal, "UPDATE operations SET state='unknown' WHERE run_ordinal=1")
        destination = self.root / "must-not-exist.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = auditor.main(["--grant", str(self.path), "--expected-grant-sha256", self.digest,
                                   "--state", str(self.state), "--private-output", str(destination)])
        self.assertEqual(result, 1)
        value = json.loads(output.getvalue())
        self.assertEqual(value["state"], "failed")
        self.assertFalse(destination.exists())
        self.assertNotIn("spare door keys", output.getvalue())
        self.assertNotIn("body_base64", output.getvalue())

    def test_cli_public_output_and_private_receipt_export_are_separate(self):
        destination = self.root / "reconciliation.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = auditor.main(["--grant", str(self.path), "--expected-grant-sha256", self.digest,
                                   "--state", str(self.state), "--private-output", str(destination)])
        self.assertEqual(result, 0)
        value = json.loads(output.getvalue())
        self.assertTrue(value["private_export_written"])
        self.assertNotIn("body_base64", output.getvalue())
        self.assertIn("body_base64", destination.read_text())


if __name__ == "__main__":
    unittest.main()
