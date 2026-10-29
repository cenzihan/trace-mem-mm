"""Conservative support expansion through explicit shared source identifiers.

Matching an identifier only admits a source for model review; it is not a claim
that the source proves the query. Pure names, dates and image-only relationships
are deliberately left to the unchanged first retrieval step.
"""
import json
import re

from .anchored_support import (AnchoredSystem, AnchoredServiceTransport,
    anchored_payload)
from .joint_selection import JointSelector, selected_indices, strict_json
from .provider_http import OPENAI_ENDPOINT

VERSION = 'tracemem-mm-identifier-support-v4.6.3-rc1'
_TOKEN = re.compile(r'(?<![\w-])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?![\w-])')


def identifiers(text):
    # Exact, case-sensitive, whole tokens; no fuzzy identity or model facts.
    return {v for v in _TOKEN.findall(text) if 3 <= len(v) <= 64
            and any(c.isalpha() for c in v) and any(c.isdigit() for c in v)}


def linked_payload(parts, sources, options, primary):
    payload, mapping = anchored_payload(parts, sources, options, primary)
    content = payload['messages'][1]['content']
    header = strict_json(content[0]['text'])
    anchor_ids = set().union(*(identifiers(v['text_prefix'])
                              for v in header['reference_anchors']))
    kept = [i for i, view in enumerate(header['candidates'])
            if anchor_ids & identifiers(view['text_prefix'])]
    if not kept:return None, []
    header['candidates'] = [dict(header['candidates'][i], index=n)
                            for n, i in enumerate(kept)]
    filtered = [dict(type='text', text=json.dumps(header, ensure_ascii=False))]
    # All images remain paired with their role labels. No new image is inferred
    # or attached, and removing a source cannot expand the original image budget.
    for offset in range(1, len(content), 2):
        label = strict_json(content[offset]['text'])
        if 'anchor_image_for' in label or label.get('image_for') == 'query':
            filtered.extend(content[offset:offset+2])
        elif label['image_for'] in kept:
            filtered.append(dict(type='text', text=json.dumps(
                {'image_for':kept.index(label['image_for'])})))
            filtered.append(content[offset+1])
    payload['messages'][1]['content'] = filtered
    return payload, [mapping[i] for i in kept]


class IdentifierSelector:
    def __init__(self, transport):
        self.transport = transport
        self.first = JointSelector(transport)

    def select(self, parts, sources, options=None):
        primary = self.first.select(parts, sources, options)
        if not primary or len(primary) == 8 or len(primary) == len(sources):return primary
        payload, mapping = linked_payload(parts, sources, options, primary)
        if payload is None:return primary
        result = self.transport(OPENAI_ENDPOINT, payload)
        support = selected_indices(strict_json(result['choices'][0]['message']['content']), len(mapping))
        return (primary + [mapping[i] for i in support])[:8]


class IdentifierSystem(AnchoredSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = IdentifierSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
            selection='primary-preserving-exact-identifier-linked-support',
            support_eligibility='shared-case-sensitive-alphanumeric-identifier-in-visible-source-text',
            limitations='no-added-support-for-pure-name-or-image-only-links')


def create_identifier_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_IDENTIFIER_SUPPORT_ENABLED') != 'approved-explicit-source-links':
        raise ValueError('explicit identifier support activation required')
    return create_app(env, _system_type=IdentifierSystem,
        _transport_type=AnchoredServiceTransport, _stop_at_chat_done=True)
