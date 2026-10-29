"""Explicit ablation: no automatic same-Add neighbors after evidence selection."""
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport

VERSION = 'tracemem-mm-selected-sources-v4.8.1-rc1'


class SelectedSourcesSystem(RecoverableOptionsSystem):
    context_anchor_limit = 0

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    retrieval='native-raw+BM25+exact-image+joint-selection',
                    same_add_context='disabled-explicit-ablation',
                    exact_image_identity='unchanged-independent-inclusion',
                    calibrated_abstention=False)


def create_selected_sources_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_SELECTED_SOURCES_ENABLED') != 'approved-selected-originals':
        raise ValueError('explicit selected-sources activation required')
    return create_app(env, _system_type=SelectedSourcesSystem,
                      _transport_type=RetrySelectionTransport, _stop_at_chat_done=True)
