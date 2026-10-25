"""Dedicated current-user Windows Credential Manager storage; no environment fallback."""
import ctypes
import hmac
import os
from ctypes import wintypes

TARGET = "agent-memory-multimodal:OpenAI:api.openai.com"
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168


class CredentialError(RuntimeError):
    pass


class CREDENTIALW(ctypes.Structure):
    _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD), ("Attributes", wintypes.LPVOID),
                ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]


def api():
    if os.name != "nt":
        raise CredentialError("Windows Credential Manager is required")
    library = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    library.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    library.CredReadW.restype = wintypes.BOOL
    library.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
    library.CredWriteW.restype = wintypes.BOOL
    library.CredFree.argtypes = [wintypes.LPVOID]
    library.CredFree.restype = None
    return library


def validate_key(key):
    if not isinstance(key, str) or not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise CredentialError("credential has invalid format")
    if len(key.encode("utf-16-le")) > 2560:
        raise CredentialError("credential exceeds Windows blob capacity")


def read_key():
    library = api()
    pointer = ctypes.POINTER(CREDENTIALW)()
    if not library.CredReadW(TARGET, CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        code = ctypes.get_last_error()
        if code == ERROR_NOT_FOUND:
            return None
        raise CredentialError(f"credential read failed (Windows code {code})")
    try:
        item = pointer.contents
        if item.CredentialBlobSize > 2560:
            raise CredentialError("credential blob exceeds supported capacity")
        raw = ctypes.string_at(item.CredentialBlob, item.CredentialBlobSize)
        try:
            # This application and Windows Generic Credential UI use UTF-16LE.
            # Also accept strictly ASCII UTF-8 blobs created by compatible clients.
            key = raw.decode("utf-16-le") if b"\x00" in raw else raw.decode("utf-8")
        except UnicodeError as exc:
            raise CredentialError("credential encoding unsupported") from exc
        validate_key(key)
        return key
    finally:
        library.CredFree(pointer)


def write_key(key):
    """Caller must obtain key from a private local input, never a CLI argument."""
    validate_key(key)
    library = api()
    raw = bytearray(key.encode("utf-16-le"))
    buffer = (ctypes.c_ubyte * len(raw)).from_buffer(raw)
    item = CREDENTIALW()
    item.Type = CRED_TYPE_GENERIC
    item.TargetName = TARGET
    item.Comment = "AML project OpenAI key; current user, local computer only"
    item.CredentialBlobSize = len(raw)
    item.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    item.Persist = CRED_PERSIST_LOCAL_MACHINE
    item.UserName = "OpenAI"
    try:
        if not library.CredWriteW(ctypes.byref(item), 0):
            raise CredentialError(f"credential write failed (Windows code {ctypes.get_last_error()})")
    finally:
        ctypes.memset(ctypes.addressof(buffer), 0, len(raw))
    retrieved = read_key()
    if retrieved is None or not hmac.compare_digest(retrieved, key):
        raise CredentialError("credential read-back verification failed")
    return {"stored": True, "readback_verified": True, "target": TARGET,
            "storage": "Windows Credential Manager; current user/local machine"}


def probe():
    key = read_key()
    return {"target": TARGET, "credential_exists": key is not None,
            "readback_verified": key is not None, "value_disclosed": False}
