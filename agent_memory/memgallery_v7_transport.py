"""Narrow additive transport compatibility for the already frozen V7 payload.

The parent transport predates the V7 temperature field. This subclass admits
exact integer zero only, validates every other field through the original
implementation, and sends the original unchanged payload. No policy/budget,
receipt, credential, endpoint, model or retry behavior is replaced.
"""
from .provider_http import BudgetedJSONTransport


class MemGalleryV7TemperatureZeroTransport(BudgetedJSONTransport):
    def _bounds(self, payload):
        if (self.model != 'gpt-4o-mini' or not isinstance(payload, dict)
                or type(payload.get('temperature')) is not int or payload['temperature'] != 0):
            raise ValueError('V7 transport requires exact integer temperature zero')
        projected = dict(payload)
        projected.pop('temperature')
        return super()._bounds(projected)
