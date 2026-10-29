"""One-pass selection over longer bounded prefixes of unchanged originals.

This candidate changes only the text visibility budget. It does not use an
identifier heuristic, a second selection pass, generated facts or answers.
"""
import json

from .joint_selection import (selection_payload, selected_indices, strict_json,
    MAX_REQUEST_BYTES, TEXT_BYTES)
from .protocol import ContractError, text_of
from .provider_http import OPENAI_ENDPOINT
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport

VERSION = 'tracemem-mm-extended-views-v4.7.0-rc1'
MAX_SOURCE_TEXT_BYTES = 6000
MAX_TOTAL_TEXT_BYTES = 94000  # Below the unchanged transport's 96000-byte cap.


def text_bytes(payload):
    return (len(payload['messages'][0]['content'].encode()) +
            sum(len(p['text'].encode()) for p in payload['messages'][1]['content']
                if p['type'] == 'text'))


def extended_payload(parts, sources, options=None):
    payload = selection_payload(parts, sources, options)
    content = payload['messages'][1]['content']
    header = strict_json(content[0]['text'])
    originals = [text_of(s['content']) for s in sources]

    def apply(limit):
        for row, text in zip(header['candidates'], originals):
            view = text.encode()[:limit].decode('utf-8', errors='ignore')
            row['text_prefix'] = view
            row['text_truncated'] = view != text
        content[0]['text'] = json.dumps(header, ensure_ascii=False)

    def fits():
        return (text_bytes(payload) <= MAX_TOTAL_TEXT_BYTES and
                len(json.dumps(payload, ensure_ascii=False).encode()) <= MAX_REQUEST_BYTES)

    # Allocate a common prefix limit, independent of candidate identities,
    # relevance scores, query/option words and source position. Short originals
    # stay complete; long originals all retain at least the baseline prefix.
    apply(TEXT_BYTES)
    if not fits():raise ContractError('extended source view capacity', 413)
    low, high = TEXT_BYTES, MAX_SOURCE_TEXT_BYTES
    while low < high:
        middle = (low + high + 1) // 2
        apply(middle)
        if fits():low = middle
        else:high = middle - 1
    apply(low)
    if not fits():raise ContractError('extended source view capacity', 413)
    return payload


class ExtendedSelector:
    def __init__(self, transport):self.transport = transport

    def select(self, parts, sources, options=None):
        payload = extended_payload(parts, sources, options)
        value = self.transport(OPENAI_ENDPOINT, payload)
        return selected_indices(strict_json(value['choices'][0]['message']['content']), len(sources))


class ExtendedSystem(RecoverableOptionsSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ExtendedSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
            selection='one-pass-original-prefixes-with-shared-text-budget',
            selection_text_prefix_bytes=MAX_SOURCE_TEXT_BYTES,
            selection_text_prefix_minimum_bytes=TEXT_BYTES,
            selection_total_text_bytes=MAX_TOTAL_TEXT_BYTES,
            selection_max_calls_per_search=1,
            limitations='candidate-pool-and-eight-image-coverage-unchanged; long-text-still-bounded')


def create_extended_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_EXTENDED_VIEWS_ENABLED') != 'approved-bounded-original-views':
        raise ValueError('explicit extended original view activation required')
    return create_app(env, _system_type=ExtendedSystem,
        _transport_type=RetrySelectionTransport, _stop_at_chat_done=True)
