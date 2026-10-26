"""Read-only export binding for the existing Bailian Windows credential target.

Only explicit read_bailian() calls access Credential Manager. This module has no
credential writes, connectivity checks, desktop input, network or CLI behavior.
Original setup scripts remain private; see docs/export-provenance.md.
"""
import ctypes
import unicodedata
from agent_memory.windows_credentials import api, CREDENTIALW, CRED_TYPE_GENERIC

TARGET = 'agent-memory-multimodal:Bailian:dashscope.aliyuncs.com:cn-beijing'


def validate_bailian(key):
    if not isinstance(key, str) or not key:
        raise ValueError('empty_credential_entry')
    if any((char.isspace() or unicodedata.category(char).startswith('C') for char in key)):
        raise ValueError('unsafe_credential_control_or_whitespace')
    if len(key.encode('utf-16-le')) > 2560:
        raise ValueError('local_credential_store_capacity_exceeded')
    try:
        key.encode('latin-1')
    except UnicodeError:
        raise ValueError('local_http_header_encoding_unsupported') from None


def read_bailian():
    library = api()
    pointer = ctypes.POINTER(CREDENTIALW)()
    if not library.CredReadW(TARGET, CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        raise ValueError('bailian_credential_unavailable')
    try:
        item = pointer.contents
        if not 0 < item.CredentialBlobSize <= 2560:
            raise ValueError('credential_blob_invalid')
        raw = ctypes.string_at(item.CredentialBlob, item.CredentialBlobSize)
        key = raw.decode('utf-16-le')
        validate_bailian(key)
        return key
    finally:
        library.CredFree(pointer)
