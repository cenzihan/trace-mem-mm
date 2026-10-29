"""Review a provisional selection for missing links using the same original views.

Two bounded calls, flat source indices, no generated memory facts or answer.
The second selection is authoritative, including an empty selection.
"""
import copy
import json

from .joint_selection import (JointSelector, SelectionBudgetedTransport,
    selection_payload, selected_indices, strict_json, candidate_count,
    SYSTEM, OPTIONS_SYSTEM, MAX_REQUEST_BYTES)
from .protocol import ContractError
from .provider_http import OPENAI_ENDPOINT
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport

VERSION = 'tracemem-mm-completeness-v4.6.0-rc1'
REVIEW_SYSTEM = OPTIONS_SYSTEM + (
    ' This is a second evidence-completeness review. draft_selected is a fallible '
    'provisional selection, not evidence or an answer key. Re-examine ALL supplied '
    'candidate views against the original query. When a source identifies an entity '
    'by an alias, a later event, ownership, or an earlier state, retrieve the separate '
    'source that actually establishes that link as well as the requested property. '
    'A comparison needs the sources for both compared states. Sources can jointly '
    'establish a fact even when neither alone states the entire relationship. '
    'Do not invent missing links, infer identity from shared words alone, or infer '
    'events from archive timestamps or session labels. Remove draft sources that '
    'only match an option or a different entity. If the original subject/event lacks '
    'support, return an empty selection. Return the FINAL ordered set of at most '
    'eight unique source indices, with direct evidence and its necessary linking '
    'sources together. Return only {"selected":[indices]}; no graph, quotes, '
    'rationale, answer, or option choice.'
)


def review_payload(parts, sources, options, draft):
    payload = selection_payload(parts, sources, options)
    selected_indices({'selected': draft}, len(sources))
    header = strict_json(payload['messages'][1]['content'][0]['text'])
    header['draft_selected'] = list(draft)
    payload['messages'][1]['content'][0]['text'] = json.dumps(header, ensure_ascii=False)
    payload['messages'][0]['content'] = REVIEW_SYSTEM
    if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_REQUEST_BYTES:
        raise ContractError('completeness review request capacity', 413)
    return payload


def checked_selection_payload(payload):
    """Validate review metadata, then use the existing strict index contract."""
    if payload['messages'][0]['content'] != REVIEW_SYSTEM:
        candidate_count(payload)  # Only the original fixed first-pass prompt.
        return payload
    checked = copy.deepcopy(payload)
    header = strict_json(checked['messages'][1]['content'][0]['text'])
    checked['messages'][0]['content'] = OPTIONS_SYSTEM if header.get('options') else SYSTEM
    count = candidate_count(checked)
    selected_indices({'selected': header['draft_selected']}, count)
    return checked


class CompletenessBudgetedTransport(SelectionBudgetedTransport):
    def _bounds(self, payload):
        if self.model == 'gpt-4o-mini':
            if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_REQUEST_BYTES:
                raise ValueError('serialized completeness request exceeded')
            return super()._bounds(checked_selection_payload(payload))
        return super()._bounds(payload)

    def _contract(self, data, payload):
        checked = checked_selection_payload(payload) if self.model == 'gpt-4o-mini' else payload
        return super()._contract(data, checked)


class CompletenessServiceTransport(RetrySelectionTransport):
    transport_type = CompletenessBudgetedTransport


class CompletenessSelector:
    def __init__(self, transport):
        self.transport = transport
        self.first = JointSelector(transport)

    def select(self, parts, sources, options=None):
        draft = self.first.select(parts, sources, options)
        payload = review_payload(parts, sources, options, draft)
        result = self.transport(OPENAI_ENDPOINT, payload)
        # Never union the draft, append unselected sources, or hide review failure.
        return selected_indices(strict_json(result['choices'][0]['message']['content']), len(sources))


class CompletenessSystem(RecoverableOptionsSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = CompletenessSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    selection='provisional-then-original-evidence-completeness-review',
                    selection_max_calls_per_search=2,
                    selection_max_provider_attempts_per_search=12,
                    completeness_verified='model-selection-not-semantic-proof')


def create_completeness_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_COMPLETENESS_ENABLED') != 'approved-evidence-completeness':
        raise ValueError('explicit completeness candidate activation required')
    return create_app(env, _system_type=CompletenessSystem,
                      _transport_type=CompletenessServiceTransport, _stop_at_chat_done=True)
