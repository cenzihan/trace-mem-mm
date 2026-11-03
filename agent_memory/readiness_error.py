"""Fixed, non-content diagnostic reasons for service readiness."""


class ReadinessFailure(ValueError):
    ALLOWED = frozenset({'embedding_latch', 'llm_latch'})

    def __init__(self, code):
        if code not in self.ALLOWED:
            raise ValueError('invalid readiness reason')
        self.code = code
        super().__init__('SDK provider failure requires operator reconciliation')
