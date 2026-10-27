"""Cover every candidate first image without widening any provider request guard."""
from .joint_selection import (MAX_IMAGES, MAX_SOURCES, MAX_REQUEST_BYTES,
    SelectionBudgetedTransport, SelectionServiceTransport, selection_payload,
    selected_indices, strict_json)
from .protocol import ContractError
from .provider_http import OPENAI_ENDPOINT

MAX_CALLS = 5


def batches(parts, sources):
    if not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError('bounded source pool required')
    query_image = any(p['type'] == 'image_url' for p in parts)
    images = [i for i, source in enumerate(sources)
              if any(p['type'] == 'image_url' for p in source['content'])]
    capacity = MAX_IMAGES - int(query_image)
    if len(images) <= capacity:
        return [list(range(len(sources)))]
    texts = set(range(len(sources))) - set(images)
    result = []
    for offset in range(0, len(images), capacity):
        admitted = texts | set(images[offset:offset+capacity])
        result.append([i for i in range(len(sources)) if i in admitted])
    if len(result) > MAX_CALLS:raise ValueError('selection call plan exceeds bound')
    return result


class CoverageServiceTransport(SelectionServiceTransport):
    def preflight_selection(self, payloads):
        if not 1 <= len(payloads) <= MAX_CALLS:
            raise ValueError('bounded selection plan required')
        validator = object.__new__(SelectionBudgetedTransport)
        validator.model = 'gpt-4o-mini'
        validator.output_token_cap = 512
        # Validate every batch before the first GPT reservation or dispatch.
        for payload in payloads:validator._bounds(payload)
        self.ready()
        if self.ledger.totals()['remaining_micro_cny'] < 5000000 + 250000*len(payloads):
            raise ContractError('insufficient original budget for complete selection plan',503)
        # This is read-only admission, not a new grant. Each call still reserves
        # in the original money ledger and observes the unchanged global journal.


class CoverageSelector:
    def __init__(self, transport):self.transport = transport

    def select(self, parts, sources, options=None):
        groups = batches(parts, sources)
        payloads = [selection_payload(parts, [sources[i] for i in group], options)
                    for group in groups]
        self.transport.preflight_selection(payloads)
        ranks = {}
        for group, payload in zip(groups, payloads):
            value = self.transport(OPENAI_ENDPOINT, payload)
            selected = selected_indices(
                strict_json(value['choices'][0]['message']['content']), len(group))
            for rank, index in enumerate(selected):
                original = group[index]
                ranks[original] = min(rank, ranks.get(original,rank))
        # No partial return on an exception, no votes from repeated text views,
        # and no unselected fallback tail. Ties use the original candidate order.
        return sorted(ranks,key=lambda index:(ranks[index],index))
