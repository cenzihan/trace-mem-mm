"""Bounded image-detail ablation; original evidence and selection rules unchanged."""
from .joint_selection import (selection_payload, selected_indices, strict_json,
                              MAX_REQUEST_BYTES)
from .protocol import ContractError
from .provider_http import OPENAI_ENDPOINT
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport
import json

VERSION = 'tracemem-mm-high-detail-v4.9.0-rc1'


def high_detail_payload(parts, sources, options=None):
    payload = selection_payload(parts, sources, options)
    for part in payload['messages'][1]['content']:
        if part['type'] == 'image_url':
            part['image_url']['detail'] = 'high'
    if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_REQUEST_BYTES:
        raise ContractError('joint selection request capacity', 413)
    return payload


class HighDetailSelector:
    def __init__(self, transport):
        self.transport = transport

    def select(self, parts, sources, options=None):
        value = self.transport(OPENAI_ENDPOINT, high_detail_payload(parts, sources, options))
        return selected_indices(strict_json(value['choices'][0]['message']['content']), len(sources))


class HighDetailSystem(RecoverableOptionsSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = HighDetailSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    selection_image_detail='high',
                    selection_preview='JPEG512-q85-max256KiB',
                    calibrated_abstention=False)


def create_high_detail_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_HIGH_DETAIL_ENABLED') != 'approved-bounded-image-detail':
        raise ValueError('explicit high-detail activation required')
    return create_app(env, _system_type=HighDetailSystem,
                      _transport_type=RetrySelectionTransport, _stop_at_chat_done=True)
