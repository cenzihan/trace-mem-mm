"""Validated subset of AML's published 2026-10-24 multimodal contract."""
import base64
import binascii
import hashlib
import json
import re
from datetime import datetime, timezone

IMAGE_LIMIT = 10 * 1024 * 1024
MEDIA_LIMIT = 30 * 1024 * 1024


class ContractError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_digest(value):
    """Hash identical canonical UTF-8 bytes without joining the whole request.

    A single non-BMP character otherwise makes the entire base64-heavy JSON
    string four bytes per character before allocating a second UTF-8 copy.
    Encoder chunks preserve the existing request IDs and replay semantics.
    """
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ContractError("missing required fields or invalid object")
    if value.keys() - set(required) - set(optional):
        raise ContractError("unsupported fields; no silent ignoring")


def image_bytes(uri):
    if not isinstance(uri, str):
        raise ContractError("image URL must be an inline Data URI")
    match = re.fullmatch(r"data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=]+)", uri)
    if not match or len(match[2]) > ((IMAGE_LIMIT + 2) // 3) * 4:
        raise ContractError("invalid inline media or image exceeds 10 MiB")
    try:
        raw = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ContractError("invalid Base64") from exc
    if not raw or len(raw) > IMAGE_LIMIT:
        raise ContractError("image must be non-empty and <=10 MiB decoded")
    mime = match[1]
    valid = ((mime == "png" and raw.startswith(b"\x89PNG\r\n\x1a\n")) or
             (mime == "jpeg" and raw.startswith(b"\xff\xd8\xff")) or
             (mime == "webp" and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"))
    if not valid:
        raise ContractError("media signature does not match MIME")
    return raw, mime


def content_parts(content):
    """Normalize plain strings to text parts; retain all array parts in order."""
    if isinstance(content, str):
        return [{"type": "text", "text": nonempty(content, "content")}]
    if not isinstance(content, list) or not content:
        raise ContractError("content/query must be non-empty text or ordered parts")
    for part in content:
        if not isinstance(part, dict):
            raise ContractError("content part must be an object")
        if part.get("type") == "text":
            fields(part, ("type", "text"))
            nonempty(part["text"], "text")
        elif part.get("type") == "image_url":
            fields(part, ("type", "image_url"))
            fields(part["image_url"], ("url",))
            image_bytes(part["image_url"]["url"])
        else:
            raise ContractError("only text and image_url parts are supported")
    return content


def media_size(parts):
    return sum(len(image_bytes(p["image_url"]["url"])[0]) for p in parts if p["type"] == "image_url")


def validate_add(payload):
    fields(payload, ("request_id", "user_id", "session_id", "messages"))
    for key in ("request_id", "user_id", "session_id"):
        nonempty(payload[key], key)
    if not isinstance(payload["messages"], list) or not payload["messages"]:
        raise ContractError("messages must be a non-empty ordered array")
    total = 0
    for msg in payload["messages"]:
        fields(msg, ("role", "content"), ("timestamp",))
        if msg["role"] not in ("user", "assistant"):
            raise ContractError("role must be user or assistant")
        if "timestamp" in msg:
            ts = msg["timestamp"]
            if type(ts) is not int:
                raise ContractError("timestamp must be integer Unix milliseconds")
            try:
                datetime.fromtimestamp(ts / 1000, timezone.utc)
            except (OverflowError, OSError, ValueError) as exc:
                raise ContractError("timestamp outside supported datetime range") from exc
        total += media_size(content_parts(msg["content"]))
    if total > MEDIA_LIMIT:
        raise ContractError("Add media exceeds 30 MiB")
    return payload


def validate_search(payload):
    fields(payload, ("query", "user_id", "top_k"), ("options",))
    nonempty(payload["user_id"], "user_id")
    if type(payload["top_k"]) is not int or payload["top_k"] < 1:
        raise ContractError("top_k must be a positive integer")
    if "options" in payload:
        if not isinstance(payload["options"], list):
            raise ContractError("options must be a top-level string array")
        for option in payload["options"]:
            nonempty(option, "option")
    parts = content_parts(payload["query"])
    # Local capacity restriction, not an invented global AML query limit.
    if media_size(parts) > MEDIA_LIMIT:
        raise ContractError("local query media capacity is 30 MiB")
    return parts


def image_hashes(parts):
    return {hashlib.sha256(image_bytes(p["image_url"]["url"])[0]).hexdigest()
            for p in parts if p["type"] == "image_url"}


def text_of(parts):
    return " ".join(p["text"] for p in parts if p["type"] == "text")
