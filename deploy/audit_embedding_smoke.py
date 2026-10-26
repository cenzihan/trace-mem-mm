"""Read-only proof of three granted embedding receipts and persisted source vectors.

No HTTP, credential reader, transport constructor, or budget mutation is used.
Optional private export contains complete verified response receipts; stdout
contains only aggregate metadata. An expired grant may be audited after the run.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_memory.embedding_executor import _json
from agent_memory.granted_embedding_smoke import BODY_LIMIT, _payload_bounds, _time
from agent_memory.protocol import canonical, content_parts
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT
from agent_memory.visual_recovery import _strict_json
from deploy.embedding_smoke_test import add_payload, expected_embedding_payloads


class AuditFailure(RuntimeError):
    pass


def require(condition, category):
    if not condition:
        raise AuditFailure(category)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path, maximum):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "regular_file_required")
    with path.open("rb") as source:
        raw = source.read(maximum + 1)
    require(len(raw) <= maximum, "file_size_limit")
    return raw


@contextmanager
def readonly_database(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "database_file_required")
    require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
            "database_has_active_or_unverified_sidecars")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("BEGIN")
        require([tuple(row) for row in connection.execute("PRAGMA integrity_check")] == [("ok",)], "sqlite_integrity_failed")
        require(connection.execute("PRAGMA foreign_key_check").fetchone() is None, "sqlite_foreign_key_failed")
        yield connection
    finally:
        connection.close()


def verify_grant(path, expected_hash):
    require(isinstance(expected_hash, str) and re.fullmatch(r"[a-f0-9]{64}", expected_hash), "trusted_grant_hash_required")
    raw = read_regular(path, 2 * 1024 * 1024)
    require(sha(raw) == expected_hash, "grant_hash_mismatch")
    grant = _strict_json(raw)
    fields = {"version", "grant_id", "model", "endpoint", "dimensions", "price_source",
              "price_checked_at", "expires_at", "slots"}
    require(isinstance(grant, dict) and set(grant) == fields, "grant_schema_invalid")
    require(type(grant["version"]) is int and grant["version"] == 1
            and grant["model"] == "text-embedding-v4" and grant["endpoint"] == BAILIAN_BEIJING_ENDPOINT
            and type(grant["dimensions"]) is int and grant["dimensions"] == 256, "grant_profile_invalid")
    require(isinstance(grant["grant_id"], str) and str(uuid.UUID(grant["grant_id"])) == grant["grant_id"], "grant_identity_invalid")
    source = urlsplit(grant["price_source"])
    require(source.scheme == "https" and source.hostname == "help.aliyun.com"
            and source.port in (None, 443) and not source.username and not source.password
            and not source.query and not source.fragment and source.path.startswith("/zh/model-studio/"),
            "grant_price_source_invalid")
    checked, expires = _time(grant["price_checked_at"]), _time(grant["expires_at"])
    require(0 < (expires - checked).total_seconds() <= 172800, "grant_validity_invalid")
    slots = grant["slots"]
    require(isinstance(slots, list) and len(slots) == 3, "exactly_three_slots_required")
    expected_payloads = expected_embedding_payloads()
    seen = set()
    for ordinal, (slot, expected) in enumerate(zip(slots, expected_payloads), 1):
        require(isinstance(slot, dict) and set(slot) == {"ordinal", "parent_attempt_id", "payload_sha256",
                "payload", "max_input_tokens", "reserved_micro_cny"}, "grant_slot_schema_invalid")
        require(type(slot["ordinal"]) is int and slot["ordinal"] == ordinal
                and type(slot["reserved_micro_cny"]) is int and slot["reserved_micro_cny"] == 50000,
                "grant_slot_bound_invalid")
        attempt = slot["parent_attempt_id"]
        require(isinstance(attempt, str) and re.fullmatch(r"attempt_[a-f0-9]{32}", attempt)
                and attempt not in seen, "grant_parent_identity_invalid")
        seen.add(attempt)
        bounds = _payload_bounds(slot["payload"])
        require(type(slot["max_input_tokens"]) is int and slot["max_input_tokens"] == bounds
                and slot["payload"] == expected
                and slot["payload_sha256"] == sha(_json(expected).encode("utf-8")), "grant_payload_mismatch")
    return grant


def verify_receipt(body, count, max_tokens):
    require(isinstance(body, bytes) and len(body) <= BODY_LIMIT, "receipt_body_invalid")
    value = _strict_json(body)
    require(isinstance(value, dict) and value.get("model") == "text-embedding-v4", "receipt_model_mismatch")
    usage = value.get("usage")
    require(isinstance(usage, dict) and type(usage.get("total_tokens")) is int, "receipt_usage_invalid")
    tokens = usage["total_tokens"]
    require(0 <= tokens <= max_tokens and ("prompt_tokens" not in usage
            or type(usage["prompt_tokens"]) is int and usage["prompt_tokens"] == tokens), "receipt_usage_invalid")
    rows = value.get("data")
    require(isinstance(rows, list) and len(rows) == count, "receipt_vector_count_invalid")
    vectors = {}
    for row in rows:
        require(isinstance(row, dict) and type(row.get("index")) is int, "receipt_vector_index_invalid")
        index, vector = row["index"], row.get("embedding")
        require(index not in vectors and 0 <= index < count, "receipt_vector_index_invalid")
        require(isinstance(vector, list) and len(vector) == 256
                and all(type(item) in (int, float) and math.isfinite(item) for item in vector)
                and any(vector), "receipt_vector_invalid")
        vectors[index] = vector
    return tokens, vectors


def verify_memory(path, grant, source_vectors):
    fixture = add_payload()
    with readonly_database(path) as database:
        require(database.execute("PRAGMA application_id").fetchone()[0] == 0x414D4D32
                and database.execute("PRAGMA user_version").fetchone()[0] == 3, "memory_schema_invalid")
        requests = database.execute("SELECT * FROM requests").fetchall()
        require(len(requests) == 1 and requests[0]["user_id"] == fixture["user_id"]
                and requests[0]["request_id"] == fixture["request_id"]
                and requests[0]["session_id"] == fixture["session_id"]
                and requests[0]["digest"] == sha(canonical(fixture).encode()), "memory_request_mismatch")
        memories = database.execute("SELECT * FROM memories ORDER BY ordinal").fetchall()
        cards = database.execute("SELECT * FROM cards").fetchall()
        require(len(memories) == 2 and len(cards) == 2, "memory_record_count_mismatch")
        by_memory = {row["memory_id"]: row for row in cards}
        require(len(by_memory) == 2, "memory_card_mapping_invalid")
        for ordinal, row in enumerate(memories):
            expected_id = "mem_" + sha(canonical([fixture["user_id"], fixture["session_id"],
                                                  fixture["request_id"], ordinal]).encode())
            require(row["id"] == expected_id and row["ordinal"] == ordinal
                    and row["user_id"] == fixture["user_id"] and row["session_id"] == fixture["session_id"]
                    and row["request_id"] == fixture["request_id"] and row["role"] == "user"
                    and _strict_json(row["content"]) == content_parts(fixture["messages"][ordinal]["content"]),
                    "memory_original_evidence_mismatch")
            card = by_memory.get(expected_id)
            require(card is not None and card["user_id"] == fixture["user_id"]
                    and card["session_id"] == fixture["session_id"] and card["vector_model"] == "text-embedding-v4"
                    and card["indexed_text"] == grant["slots"][0]["payload"]["input"][ordinal],
                    "memory_index_identity_mismatch")
            require(_strict_json(card["vector"]) == source_vectors[ordinal], "memory_vector_receipt_mismatch")
            derived = _strict_json(card["content"])
            require(isinstance(derived, dict) and "generated" not in derived and "generator" not in derived,
                    "unexpected_generated_card")
        operations = database.execute("SELECT user_id,request_id,state FROM add_operations").fetchall()
        require(len(operations) == 1 and operations[0]["state"] == "complete"
                and operations[0]["user_id"] == fixture["user_id"]
                and operations[0]["request_id"] == fixture["request_id"], "memory_add_not_complete")
        require(database.execute("SELECT count(*) FROM evidence").fetchone()[0] == 0, "unexpected_image_evidence")
    return {"records": 2, "vector_count": 2, "dimensions": 256, "source_vectors_match_receipt": True,
            "original_text_preserved": True, "expected_user_scope_only": True, "add_complete": True,
            "generated_cards": 0, "image_evidence": 0}


def audit(grant_path, expected_hash, state_path):
    grant = verify_grant(grant_path, expected_hash)
    state = Path(state_path).resolve()
    journal_path = state / "receipts" / "private-embedding.sqlite"
    binding = _strict_json(read_regular(state / "receipts" / "grant-binding.json", 4096))
    require(binding == {"grant_id": grant["grant_id"], "grant_sha256": expected_hash}, "journal_grant_binding_mismatch")
    records, source_vectors = [], None
    with readonly_database(journal_path) as database:
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require(tables == {"operations", "claims"}, "unexpected_journal_tables")
        rows = database.execute("SELECT * FROM operations").fetchall()
        claims = database.execute("SELECT scope,chunk_key,fingerprint FROM claims").fetchall()
        require(len(rows) == 3 and len(claims) == 3, "unexpected_attempt_count")
        by_fingerprint = {row["fingerprint"]: row for row in rows}
        expected_claims = set()
        for slot in grant["slots"]:
            manifest = {"keys": [slot["parent_attempt_id"]], "ordinal": slot["ordinal"],
                        "payload_sha256": slot["payload_sha256"], "grant_sha256": expected_hash}
            fingerprint = sha(_json(manifest).encode())
            row = by_fingerprint.get(fingerprint)
            require(row is not None and row["scope"] == expected_hash and row["manifest"] == _json(manifest),
                    "journal_request_binding_mismatch")
            require(row["state"] == "complete" and row["transport_completed"] == 1
                    and row["attempt_id"] == slot["parent_attempt_id"] and row["status"] == 200
                    and row["run_path"] == str(journal_path) and row["run_ordinal"] == slot["ordinal"]
                    and row["grant_dispatch_started"] == 1 and row["grant_category"] == "success"
                    and row["grant_returned_model"] == "text-embedding-v4" and row["redacted"] == 0,
                    "attempt_not_complete_and_bound")
            require(isinstance(row["body"], bytes) and sha(row["body"]) == row["body_sha"], "receipt_hash_mismatch")
            tokens, vectors = verify_receipt(row["body"], len(slot["payload"]["input"]), slot["max_input_tokens"])
            require(row["grant_input_tokens"] == tokens, "journal_usage_mismatch")
            if slot["ordinal"] == 1:
                source_vectors = vectors
            expected_claims.add((expected_hash, slot["parent_attempt_id"], fingerprint))
            records.append({"ordinal": slot["ordinal"], "parent_attempt_id": slot["parent_attempt_id"],
                            "status": 200, "body_base64": base64.b64encode(row["body"]).decode("ascii"),
                            "body_sha256": row["body_sha"], "redacted": False, "dispatch_started": True,
                            "category": "success", "returned_model": "text-embedding-v4",
                            "input_tokens": tokens, "output_tokens": 0})
        require({tuple(row) for row in claims} == expected_claims, "journal_claims_mismatch")
    memory = verify_memory(state / "memory.sqlite", grant, source_vectors)
    private = {"version": 1, "grant_id": grant["grant_id"], "grant_sha256": expected_hash,
               "records": records, "memory": memory}
    public = {"state": "passed", "grant_id": grant["grant_id"], "grant_sha256": expected_hash,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(), "read_only": True,
              "model": "text-embedding-v4", "provider_receipts_verified": 3,
              "journal_operations": 3, "journal_claims": 3,
              "slots": [{key: value for key, value in record.items() if key != "body_base64"} for record in records],
              "input_tokens_total": sum(record["input_tokens"] for record in records),
              "memory": memory, "parent_reconciliation_required": True,
              "llm_verified": False, "reranker_verified": False, "official_evaluation": False}
    return public, private


def export_private(path, value):
    path = Path(path).absolute()
    require(path.parent.is_dir() and not path.is_symlink(), "private_output_parent_required")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grant", required=True, type=Path)
    parser.add_argument("--expected-grant-sha256", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--private-output", type=Path)
    args = parser.parse_args(argv)
    try:
        public, private = audit(args.grant, args.expected_grant_sha256, args.state)
        if args.private_output is not None:
            export_private(args.private_output, private)
        public["private_export_written"] = args.private_output is not None
    except (AuditFailure, OSError, ValueError, TypeError, KeyError, OverflowError, sqlite3.Error, RuntimeError) as error:
        category = str(error) if isinstance(error, AuditFailure) else "audit_input_or_storage_invalid"
        print(json.dumps({"state": "failed", "error": category, "read_only": True,
                          "provider_receipts_verified": False, "parent_reconciled": False}))
        return 1
    print(json.dumps(public, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
