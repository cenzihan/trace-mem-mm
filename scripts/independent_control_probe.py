"""Prospective scalar diagnostics for isolated, preplanned independent controls.

Not imported by any service factory. Never attach this helper to official traffic
or a store containing official data. Provenance is a caller declaration, not a
license/overlap audit. Targets must be fixed before a new independent test; this
helper neither authorizes calls nor retries a failed/closed test.
"""
import copy
import json
from pathlib import Path

from agent_memory.consolidated_selection import ConsolidatingSelector, finalists
from agent_memory.coverage_selection import CoverageSelector, batches
from agent_memory.joint_selection import (
    JointSelector, selected_indices, selection_payload, strict_json)


def _rank(indices, targets):
    return next((rank for rank, index in enumerate(indices, 1) if index in targets), None)


class IndependentControlProbe:
    """One selection and one final-return observation; no IDs/content in output."""
    def __init__(self, *, provenance, target_content):
        if provenance not in ('synthetic', 'independent-public'):
            raise ValueError('independent control provenance required')
        if not isinstance(target_content, list) or not target_content:
            raise ValueError('complete predeclared target content required')
        self._target = copy.deepcopy(target_content)
        self._used = False
        self._return_observed = False
        self._report = dict(provenance=provenance, diagnostic_valid=True,
            outcome='not_run', pool_count=None, target_pool_count=None,
            target_pool_rank=None, calls=[], selected_count=None,
            target_selected_rank=None, returned_count=None,
            target_returned_rank=None)

    def wrap(self, selector):
        if type(selector) not in (JointSelector, CoverageSelector, ConsolidatingSelector):
            raise ValueError('unsupported selector; diagnostic mapping requires review')
        probe = self

        class Wrapped:
            def select(self, parts, sources, options=None):
                if probe._used:
                    raise ValueError('control probe is single-use, never a replay mechanism')
                probe._used = True
                if not 1 <= len(sources) <= 32:
                    raise ValueError('bounded independent pool required')
                targets = {i for i, s in enumerate(sources) if s['content'] == probe._target}
                probe._report.update(pool_count=len(sources), target_pool_count=len(targets),
                    target_pool_rank=_rank(range(len(sources)), targets), outcome='running')
                observer = _ObservedTransport(selector.transport, type(selector), probe,
                                              parts, sources, options, targets)
                try:
                    result = type(selector)(observer).select(parts, sources, options)
                except Exception:
                    probe._report['outcome'] = 'selection_failed'
                    raise
                probe._report.update(outcome='selection_complete', selected_count=len(result),
                                     target_selected_rank=_rank(result, targets))
                return result
        return Wrapped()

    def observe_returned(self, rows):
        """Call on actual packed original results, not the selector's index list."""
        if self._report['outcome'] != 'selection_complete' or self._return_observed:
            raise ValueError('one completed selection and one actual return required')
        if not isinstance(rows, list) or len(rows) > 100:
            raise ValueError('bounded original result list required')
        self._return_observed = True
        rank = next((i for i, row in enumerate(rows, 1)
                     if row['content'] == self._target), None)
        self._report.update(returned_count=len(rows), target_returned_rank=rank)
        self._target = None

    def snapshot(self):
        # Only known fields built from numeric/boolean results and fixed labels.
        value = copy.deepcopy(self._report)
        if not value['diagnostic_valid']:
            # Preserve evidence of unavailable diagnostics without plausible ranks.
            value['calls'] = []
        if len(json.dumps(value).encode()) > 8192:
            raise ValueError('scalar diagnostic capacity exceeded')
        return value

    def write_once(self, path):
        """Private output only; failed/existing destination never changes a result."""
        try:
            with Path(path).open('x', encoding='utf-8') as out:
                out.write(json.dumps(self.snapshot(), indent=2) + '\n')
            return True
        except OSError:
            return False


class _ObservedTransport:
    def __init__(self, transport, kind, probe, parts, sources, options, targets):
        self.inner, self.kind, self.probe = transport, kind, probe
        self.parts, self.sources, self.options, self.targets = parts, sources, options, targets
        self.groups = ([list(range(len(sources)))] if kind is JointSelector
                       else batches(parts, sources))
        self.selections = []
        self.ordinal = 0

    def __getattr__(self, name):
        # Original admission, money ledger and journal remain the only authority.
        return getattr(self.inner, name)

    def __call__(self, endpoint, payload):
        ordinal = self.ordinal
        self.ordinal += 1
        event = None
        try:
            if ordinal >= 6:
                raise ValueError('diagnostic call bound exceeded')
            final = ordinal >= len(self.groups)
            if final:
                if self.kind is not ConsolidatingSelector or ordinal != len(self.groups):
                    raise ValueError('unexpected final stage')
                group = finalists(self.parts, self.sources, self.selections)
            else:
                group = self.groups[ordinal]
            expected = selection_payload(self.parts, [self.sources[i] for i in group], self.options)
            if payload != expected:
                raise ValueError('selector payload map changed; diagnostics unavailable')
            event = dict(stage='final' if final else 'initial', ordinal=ordinal + 1,
                candidate_count=len(group), target_input_rank=_rank(group, self.targets),
                image_count=sum(p['type'] == 'image_url' for p in payload['messages'][1]['content']),
                selected_count=None, target_selected_rank=None, outcome='dispatched')
            self.probe._report['calls'].append(event)
        except Exception:
            self.probe._report['diagnostic_valid'] = False
        try:
            value = self.inner(endpoint, payload)  # Exactly one original dispatch.
        except Exception:
            if event is not None:event['outcome'] = 'provider_failed'
            raise
        if event is not None:
            try:
                local = selected_indices(strict_json(value['choices'][0]['message']['content']), len(group))
                selected = [group[i] for i in local]
                if not final:self.selections.append(selected)
                event.update(outcome='complete', selected_count=len(selected),
                             target_selected_rank=_rank(selected, self.targets))
            except Exception:
                # The selector still validates and raises its original error.
                event['outcome'] = 'invalid_response'
        return value
