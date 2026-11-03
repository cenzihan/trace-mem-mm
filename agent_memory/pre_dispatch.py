"""Explicit rejection of a request before any provider dispatch."""


class PreDispatchRejection(ValueError):
    """The current provider request was rejected before its journal claim."""
