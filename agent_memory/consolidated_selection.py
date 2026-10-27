"""Globally compare bounded survivors after candidate-image coverage."""
from .coverage_selection import (CoverageServiceTransport, CoverageSelector, batches,
                                 MAX_CALLS)
from .joint_selection import (MAX_IMAGES, selection_payload, selected_indices,
                              strict_json)
from .provider_http import OPENAI_ENDPOINT
from .protocol import ContractError

MAX_TOTAL_CALLS = MAX_CALLS + 1


def finalists(parts, sources, selections):
    """Round-robin first-image survivors, all selected text, no duplicate votes."""
    capacity = MAX_IMAGES - int(any(p['type'] == 'image_url' for p in parts))
    is_image = lambda i: any(p['type'] == 'image_url' for p in sources[i]['content'])
    texts = {i for row in selections for i in row if not is_image(i)}
    rows = [[i for i in row if is_image(i)] for row in selections]
    images = []
    for rank in range(max(map(len, rows), default=0)):
        for row in rows:
            if rank < len(row) and row[rank] not in images and len(images) < capacity:
                images.append(row[rank])
    admitted = texts | set(images)
    return [i for i in range(len(sources)) if i in admitted]


class ConsolidatingServiceTransport(CoverageServiceTransport):
    def preflight_consolidation(self, payloads):
        self.preflight_selection(payloads)
        if self.ledger.totals()['remaining_micro_cny'] < 5000000 + 250000*(len(payloads)+1):
            raise ContractError('insufficient original budget for final selection',503)


class ConsolidatingSelector(CoverageSelector):
    def select(self, parts, sources, options=None):
        groups = batches(parts, sources)
        if len(groups) == 1:
            return super().select(parts, sources, options)
        payloads = [selection_payload(parts,[sources[i] for i in group],options)
                    for group in groups]
        self.transport.preflight_consolidation(payloads)
        selections = []
        for group,payload in zip(groups,payloads):
            value = self.transport(OPENAI_ENDPOINT,payload)
            local = selected_indices(strict_json(value['choices'][0]['message']['content']),len(group))
            selections.append([group[i] for i in local])
        group = finalists(parts,sources,selections)
        if not group:return []
        payload = selection_payload(parts,[sources[i] for i in group],options)
        self.transport.preflight_selection([payload])
        value = self.transport(OPENAI_ENDPOINT,payload)
        local = selected_indices(strict_json(value['choices'][0]['message']['content']),len(group))
        return [group[i] for i in local]

