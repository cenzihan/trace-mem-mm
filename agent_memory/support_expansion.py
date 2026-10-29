"""Preserve primary evidence; a bounded second pass may append support only.

This is an opt-in candidate. Neither the successful baseline nor the earlier
replacement-review candidate changes. Model-selected support is not a proof.
"""
import copy
import json

from .completeness_selection import review_payload, checked_selection_payload, REVIEW_SYSTEM
from .joint_selection import JointSelector, SelectionBudgetedTransport, OPTIONS_SYSTEM, selected_indices, strict_json, MAX_REQUEST_BYTES
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport
from .provider_http import OPENAI_ENDPOINT

VERSION = 'tracemem-mm-support-v4.6.1-rc1'
SUPPORT_SYSTEM = OPTIONS_SYSTEM + (
    ' This pass may only propose additional supporting sources for draft_selected. '
    'The draft is a fallible primary retrieval, not truth or an answer key. '
    'Do not remove, replace, reorder, or repeat its indices. Read the original '
    'query and candidate evidence, and select only omitted sources that explicitly '
    'establish a necessary alias, ownership, event, earlier/later state, or comparison '
    'link to the primary evidence. An association needs an explicit shared entity '
    'or event established by source content; a shared word, option match, archive '
    'timestamp, or session label alone is insufficient. Do not add background '
    'sources merely to increase coverage. Return empty selected when no additional '
    'source supplies a necessary supported link. At most eight unique indices; '
    'return only {"selected":[indices]}, no answers, rationale, or generated facts.'
)


def support_payload(parts, sources, options, primary):
    payload = review_payload(parts, sources, options, primary)
    payload['messages'][0]['content'] = SUPPORT_SYSTEM
    if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_REQUEST_BYTES:
        raise ValueError('serialized support request exceeded')
    return payload


def checked_support_payload(payload):
    if payload['messages'][0]['content'] != SUPPORT_SYSTEM:
        # Do not accept the prior replacement-review prompt in this candidate.
        from .joint_selection import candidate_count
        candidate_count(payload)
        return payload
    checked = copy.deepcopy(payload)
    checked['messages'][0]['content'] = REVIEW_SYSTEM
    return checked_selection_payload(checked)


class SupportBudgetedTransport(SelectionBudgetedTransport):
    def _bounds(self, payload):
        if self.model == 'gpt-4o-mini':
            if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_REQUEST_BYTES:
                raise ValueError('serialized support request exceeded')
            payload = checked_support_payload(payload)
        return super()._bounds(payload)

    def _contract(self, data, payload):
        if self.model == 'gpt-4o-mini':payload = checked_support_payload(payload)
        return super()._contract(data, payload)


class SupportServiceTransport(RetrySelectionTransport):
    transport_type = SupportBudgetedTransport


class SupportSelector:
    def __init__(self, transport):
        self.transport = transport
        self.first = JointSelector(transport)

    def select(self, parts, sources, options=None):
        primary = self.first.select(parts, sources, options)
        # Preserve baseline abstention and avoid a useless call at capacity.
        if not primary or len(primary) == 8:return primary
        result = self.transport(OPENAI_ENDPOINT, support_payload(parts, sources, options, primary))
        support = selected_indices(strict_json(result['choices'][0]['message']['content']), len(sources))
        # A valid empty support result is not a fallback. Transport/contract failures
        # still propagate. The primary prefix is immutable after successful review.
        return (primary + [i for i in support if i not in primary])[:8]


class SupportSystem(RecoverableOptionsSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = SupportSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    selection='primary-preserving-linked-support-expansion',
                    selection_max_calls_per_search=2,
                    selection_max_provider_attempts_per_search=12,
                    primary_order_preserved=True, empty_primary_preserved=True,
                    support_verified='model-selection-not-semantic-proof')


def create_support_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_SUPPORT_ENABLED') != 'approved-primary-support':
        raise ValueError('explicit support candidate activation required')
    return create_app(env, _system_type=SupportSystem,
                      _transport_type=SupportServiceTransport, _stop_at_chat_done=True)
