"""Keep exact-image matches in retrieval, but require selection before returning them."""
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport

VERSION = 'tracemem-mm-model-image-selection-v4.10.0-rc1'


class ModelImageSelectionSystem(RecoverableOptionsSystem):
    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    exact_image_policy='candidate-priority-without-forced-return',
                    empty_selection='empty-even-with-exact-image-match',
                    generated_answers=False)

    def _order_sources(self, payload, parts, rows, order, identities):
        # The upstream retrieval order already contains exact-image matches.
        # Preserve that pool and the unchanged selector; remove only the final
        # unconditional identity union that can override an empty selection.
        selected = super()._order_sources(payload, parts, rows, order, set())
        # Retain the baseline priority of identities that the model did select.
        return list(dict.fromkeys([*sorted(identities.intersection(selected)), *selected]))


def create_model_image_selection_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_MODEL_IMAGE_SELECTION') != 'approved-model-selected-images':
        raise ValueError('explicit candidate activation required')
    return create_app(env, _system_type=ModelImageSelectionSystem,
                      _transport_type=RetrySelectionTransport, _stop_at_chat_done=True)
