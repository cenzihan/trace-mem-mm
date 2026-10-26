"""Three pre-reserved parent embedding attempts; no remote budget is created.

The grant is pinned by its trusted file SHA256. Its private receipt journal is an
outbox for later parent-ledger reconciliation, never proof that reconciliation
has happened. Unknown or failed slots freeze the grant without automatic replay.
"""
import hashlib
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone
import uuid
from urllib.parse import urlsplit

from .embedding_executor import _Journal, _ReceiptHTTP, _json
from .provider_http import BAILIAN_BEIJING_ENDPOINT, HTTPReply, NativeHTTPS, ProviderFailure
from .visual_recovery import _strict_json


MODEL = "text-embedding-v4"
BODY_LIMIT = 1024 * 1024
SLOT_HOLD = 50000


class GrantedEmbeddingBlocked(RuntimeError):
    def __init__(self, category):
        self.category = category
        super().__init__("granted embedding: " + category)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError()
        return result
    except (AttributeError, TypeError, ValueError):
        raise GrantedEmbeddingBlocked("invalid_grant_time") from None


def _payload_bounds(payload):
    if (not isinstance(payload, dict) or set(payload) != {"model", "input", "dimensions", "encoding_format"}
            or payload.get("model") != MODEL or type(payload.get("dimensions")) is not int
            or payload["dimensions"] != 256 or payload.get("encoding_format") != "float"):
        raise GrantedEmbeddingBlocked("invalid_embedding_payload")
    texts = payload["input"]
    if not isinstance(texts, list) or not 1 <= len(texts) <= 10:
        raise GrantedEmbeddingBlocked("invalid_embedding_payload")
    try:
        if any(not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 8192 for text in texts):
            raise GrantedEmbeddingBlocked("invalid_embedding_payload")
    except UnicodeError:
        raise GrantedEmbeddingBlocked("invalid_embedding_payload") from None
    return 8192 * len(texts)


def _validate_grant(value):
    fields = {"version", "grant_id", "model", "endpoint", "dimensions", "price_source",
              "price_checked_at", "expires_at", "slots"}
    if (not isinstance(value, dict) or set(value) != fields or type(value.get("version")) is not int
            or value["version"] != 1 or value.get("model") != MODEL
            or value.get("endpoint") != BAILIAN_BEIJING_ENDPOINT
            or type(value.get("dimensions")) is not int or value["dimensions"] != 256):
        raise GrantedEmbeddingBlocked("invalid_grant")
    try:
        if str(uuid.UUID(value["grant_id"])) != value["grant_id"]:
            raise ValueError()
        source = urlsplit(value["price_source"])
        if (source.scheme != "https" or source.hostname != "help.aliyun.com"
                or source.username or source.password or source.query or source.fragment
                or source.port not in (None, 443) or not source.path.startswith("/zh/model-studio/")):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise GrantedEmbeddingBlocked("invalid_grant_identity_or_price") from None
    checked, expires = _time(value["price_checked_at"]), _time(value["expires_at"])
    now = datetime.now(timezone.utc)
    if not 0 <= (now - checked).total_seconds() <= 172800 or not checked < expires or not now < expires:
        raise GrantedEmbeddingBlocked("grant_expired_or_price_stale")
    if (expires - checked).total_seconds() > 172800:
        raise GrantedEmbeddingBlocked("grant_validity_too_long")
    slots = value["slots"]
    if not isinstance(slots, list) or len(slots) != 3:
        raise GrantedEmbeddingBlocked("exactly_three_parent_attempts_required")
    attempts, hashes = set(), set()
    for ordinal, slot in enumerate(slots, 1):
        if (not isinstance(slot, dict) or set(slot) != {"ordinal", "parent_attempt_id", "payload_sha256",
                                                       "payload", "max_input_tokens", "reserved_micro_cny"}
                or type(slot.get("ordinal")) is not int or slot["ordinal"] != ordinal
                or type(slot.get("reserved_micro_cny")) is not int or slot["reserved_micro_cny"] != SLOT_HOLD
                or type(slot.get("max_input_tokens")) is not int):
            raise GrantedEmbeddingBlocked("invalid_grant_slot")
        attempt = slot["parent_attempt_id"]
        if not isinstance(attempt, str) or not re.fullmatch(r"attempt_[a-f0-9]{32}", attempt) or attempt in attempts:
            raise GrantedEmbeddingBlocked("invalid_parent_attempt_identity")
        bounds = _payload_bounds(slot["payload"])
        payload_hash = _digest(_json(slot["payload"]).encode("utf-8"))
        if (slot["max_input_tokens"] != bounds or slot["payload_sha256"] != payload_hash
                or payload_hash in hashes):
            raise GrantedEmbeddingBlocked("invalid_or_repeated_grant_payload")
        attempts.add(attempt)
        hashes.add(payload_hash)
    return value


class GrantedEmbeddingTransport:
    mode = "live_prepaid_embedding_smoke"

    def __init__(self, grant_path, journal_root, credential_reader, http=None, offline_test=False,
                 *, expected_grant_sha256=None):
        if type(offline_test) is not bool or not callable(credential_reader):
            raise GrantedEmbeddingBlocked("invalid_configuration")
        self.offline_test = offline_test
        self.mode = "offline_fake" if offline_test else type(self).mode
        self.http = http if http is not None else NativeHTTPS()
        if bool(getattr(self.http, "is_fake_http", False)) != offline_test:
            raise GrantedEmbeddingBlocked("http_mode_mismatch")
        self.credential_reader = credential_reader
        self.grant_path = Path(grant_path)
        if self.grant_path.is_symlink() or not self.grant_path.is_file():
            raise GrantedEmbeddingBlocked("grant_file_unavailable")
        try:
            with self.grant_path.open("rb") as source:
                raw = source.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise GrantedEmbeddingBlocked("grant_file_too_large")
            self.grant_sha256 = _digest(raw)
            if not offline_test and expected_grant_sha256 is None:
                raise GrantedEmbeddingBlocked("trusted_grant_hash_required")
            if expected_grant_sha256 is not None and expected_grant_sha256 != self.grant_sha256:
                raise GrantedEmbeddingBlocked("grant_hash_mismatch")
            self.grant = _validate_grant(_strict_json(raw))
        except (OSError, UnicodeError, ValueError):
            raise GrantedEmbeddingBlocked("grant_file_invalid") from None
        root = Path(journal_root).resolve()
        binding = root / "grant-binding.json"
        # An existing foreign journal may not be adopted into a new grant.
        if not binding.exists() and (root / "private-embedding.sqlite").exists():
            raise GrantedEmbeddingBlocked("unbound_existing_journal")
        self.journal = _Journal(root, offline_test)
        if os.name == "posix":
            root.chmod(0o700)
        expected = _json({"grant_id": self.grant["grant_id"], "grant_sha256": self.grant_sha256})
        if binding.is_symlink():
            raise GrantedEmbeddingBlocked("invalid_grant_binding")
        try:
            with binding.open("x", encoding="utf-8") as output:
                output.write(expected)
                output.flush()
                os.fsync(output.fileno())
            if os.name == "posix":
                binding.chmod(0o600)
        except FileExistsError:
            if binding.read_text(encoding="utf-8") != expected:
                raise GrantedEmbeddingBlocked("grant_changed_since_initialization") from None
        with self.journal.connect() as database:
            columns = {row[1] for row in database.execute("PRAGMA table_info(operations)")}
            for name, definition in (("grant_category", "TEXT"), ("grant_input_tokens", "INTEGER"),
                                     ("grant_returned_model", "TEXT"),
                                     ("grant_dispatch_started", "INTEGER NOT NULL DEFAULT 0")):
                if name not in columns:
                    database.execute("ALTER TABLE operations ADD COLUMN " + name + " " + definition)
        self.slots = []
        for slot in self.grant["slots"]:
            manifest = {"keys": [slot["parent_attempt_id"]], "ordinal": slot["ordinal"],
                        "payload_sha256": slot["payload_sha256"], "grant_sha256": self.grant_sha256}
            fingerprint = _digest(_json(manifest).encode("utf-8"))
            self.journal.install(fingerprint, self.grant_sha256, manifest)
            self.slots.append((slot, fingerprint))

    def _verify_current_grant(self):
        try:
            if _digest(self.grant_path.read_bytes()) != self.grant_sha256:
                raise GrantedEmbeddingBlocked("grant_file_changed")
        except OSError:
            raise GrantedEmbeddingBlocked("grant_file_unavailable") from None
        _validate_grant(self.grant)

    def _next_slot(self):
        for slot, fingerprint in self.slots:
            row = self.journal.row(fingerprint)
            if row is None:
                raise GrantedEmbeddingBlocked("journal_slot_missing")
            if row["state"] == "complete":
                if (row["attempt_id"] != slot["parent_attempt_id"] or row["status"] != 200
                        or row["body"] is None or _digest(row["body"]) != row["body_sha"]
                        or row["redacted"] or row["grant_category"] != "success"
                        or row["grant_returned_model"] != MODEL or not row["grant_dispatch_started"]
                        or not row["transport_completed"] or row["run_ordinal"] != slot["ordinal"]
                        or row["run_path"] != str(self.journal.path.resolve())):
                    raise GrantedEmbeddingBlocked("completed_receipt_changed")
                continue
            if row["state"] != "ready":
                raise GrantedEmbeddingBlocked("unfinished_attempt_blocks_grant")
            return slot, fingerprint
        raise GrantedEmbeddingBlocked("grant_exhausted")

    def _parse_reply(self, reply, slot, fingerprint):
        if not isinstance(reply, HTTPReply) or type(reply.status) is not int or not 100 <= reply.status <= 599:
            raise GrantedEmbeddingBlocked("invalid_http_response")
        if reply.status != 200:
            category = ("authentication" if reply.status == 401 else "permission_or_region" if reply.status == 403
                        else "rate_limited" if reply.status == 429 else "server_error" if reply.status >= 500
                        else "redirect" if 300 <= reply.status < 400 else "client_error")
            raise GrantedEmbeddingBlocked(category)
        row = self.journal.row(fingerprint)
        if row["redacted"]:
            raise GrantedEmbeddingBlocked("redacted_response")
        if not isinstance(reply.body, bytes) or len(reply.body) > BODY_LIMIT:
            raise GrantedEmbeddingBlocked("response_limit")
        try:
            data = _strict_json(reply.body)
        except (ValueError, UnicodeError):
            raise GrantedEmbeddingBlocked("invalid_json") from None
        if not isinstance(data, dict) or data.get("model") != MODEL:
            raise GrantedEmbeddingBlocked("model_mismatch")
        with self.journal.connect() as database:
            database.execute("UPDATE operations SET grant_returned_model=? WHERE fingerprint=?", (MODEL, fingerprint))
        usage = data.get("usage")
        if not isinstance(usage, dict) or type(usage.get("total_tokens")) is not int:
            raise GrantedEmbeddingBlocked("invalid_usage")
        tokens = usage["total_tokens"]
        if (not 0 <= tokens <= slot["max_input_tokens"]
                or ("prompt_tokens" in usage and (type(usage["prompt_tokens"]) is not int or usage["prompt_tokens"] != tokens))):
            raise GrantedEmbeddingBlocked("invalid_usage")
        # Usage is durably available even if the vector contract subsequently fails.
        with self.journal.connect() as database:
            database.execute("UPDATE operations SET grant_input_tokens=? WHERE fingerprint=?", (tokens, fingerprint))
        rows = data.get("data")
        count = len(slot["payload"]["input"])
        if not isinstance(rows, list) or len(rows) != count:
            raise GrantedEmbeddingBlocked("invalid_contract")
        indices = []
        for row in rows:
            if not isinstance(row, dict) or type(row.get("index")) is not int:
                raise GrantedEmbeddingBlocked("invalid_contract")
            indices.append(row["index"])
            vector = row.get("embedding")
            if (not isinstance(vector, list) or len(vector) != 256
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                    or not any(vector)):
                raise GrantedEmbeddingBlocked("invalid_contract")
        if set(indices) != set(range(count)):
            raise GrantedEmbeddingBlocked("invalid_contract")
        return data

    def __call__(self, endpoint, payload):
        self._verify_current_grant()
        if endpoint != BAILIAN_BEIJING_ENDPOINT:
            raise GrantedEmbeddingBlocked("endpoint_mismatch")
        _payload_bounds(payload)
        raw = _json(payload).encode("utf-8")
        slot, fingerprint = self._next_slot()
        if _digest(raw) != slot["payload_sha256"]:
            raise GrantedEmbeddingBlocked("payload_does_not_match_next_slot")
        if self.offline_test:
            key = "SYNTHETIC_GRANT_KEY_NEVER_LOG"
        else:
            try:
                key = self.credential_reader()
            except Exception:
                raise GrantedEmbeddingBlocked("credential_unavailable") from None
        if (not isinstance(key, str) or not 1 <= len(key) <= 16384
                or any(ord(char) < 33 or ord(char) > 126 for char in key)):
            raise GrantedEmbeddingBlocked("credential_unavailable")
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"}
        # start() atomically admits only one in-flight operation for this grant.
        try:
            self.journal.start(fingerprint)
        except Exception:
            raise GrantedEmbeddingBlocked("concurrent_or_unresolved_attempt") from None
        try:
            self.journal.attach(fingerprint, slot["parent_attempt_id"], str(self.journal.path.resolve()), slot["ordinal"])
            wrapped = _ReceiptHTTP(self.http, self.journal, fingerprint, payload, None)
            wrapped.endpoint = endpoint
            with self.journal.connect() as database:
                database.execute("UPDATE operations SET grant_dispatch_started=1 WHERE fingerprint=?", (fingerprint,))
            reply = wrapped.exchange(endpoint, raw, headers, 60, BODY_LIMIT)
            result = self._parse_reply(reply, slot, fingerprint)
            with self.journal.connect() as database:
                changed = database.execute("UPDATE operations SET state='complete',grant_category='success',transport_completed=1 WHERE fingerprint=? AND state='received'", (fingerprint,))
                if changed.rowcount != 1:
                    raise GrantedEmbeddingBlocked("receipt_state_changed")
            return result
        except BaseException as error:
            if isinstance(error, GrantedEmbeddingBlocked):
                category = error.category
            elif isinstance(error, ProviderFailure):
                category = error.category if error.category in {"timeout", "network_error", "tls_error", "response_limit"} else "network_error"
            elif isinstance(error, TimeoutError):
                category = "timeout"
            else:
                category = "interrupted" if not isinstance(error, Exception) else "local_or_network_error"
            with self.journal.connect() as database:
                database.execute("UPDATE operations SET state=CASE WHEN state='received' THEN 'failed' ELSE 'unknown' END,grant_category=? WHERE fingerprint=? AND state IN ('inflight','received')", (category, fingerprint))
            if not isinstance(error, Exception):
                raise
            raise GrantedEmbeddingBlocked(category) from None

    def summary(self):
        slots = []
        for slot, fingerprint in self.slots:
            row = self.journal.row(fingerprint)
            slots.append({"ordinal": slot["ordinal"], "parent_attempt_id": slot["parent_attempt_id"],
                          "state": row["state"], "dispatch_started": bool(row["grant_dispatch_started"]),
                          "http_status": row["status"], "receipt_sha256": row["body_sha"],
                          "receipt_redacted": bool(row["redacted"]), "category": row["grant_category"],
                          "returned_model": row["grant_returned_model"],
                          "input_tokens": row["grant_input_tokens"],
                          "output_tokens": 0 if row["grant_input_tokens"] is not None else None,
                          "reserved_micro_cny": SLOT_HOLD})
        return {"grant_id": self.grant["grant_id"], "grant_sha256": self.grant_sha256,
                "model": MODEL, "mode": self.mode, "slots": slots,
                "dispatches_started": sum(row["dispatch_started"] for row in slots),
                "received_responses": sum(row["http_status"] is not None for row in slots),
                "complete": all(row["state"] == "complete" for row in slots),
                "maximum_reserved_micro_cny": 3 * SLOT_HOLD, "remote_budget_created": False,
                "parent_reconciliation_required": True}
