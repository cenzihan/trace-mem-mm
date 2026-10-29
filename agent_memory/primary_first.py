"""Keep selected original sources ahead of optional same-Add neighbors."""
from .recoverable_options import RecoverableOptionsSystem, RetrySelectionTransport

VERSION = 'tracemem-mm-primary-first-v4.8.0-rc1'


class PrimaryFirstSystem(RecoverableOptionsSystem):
    context_primary_first = True

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    context_order='all-primary-sources-before-same-add-neighbors',
                    media_overflow='skip-whole-record-unchanged')


def create_primary_first_app(environ=None):
    import os
    from .mem0_app import create_app
    env = os.environ if environ is None else environ
    if env.get('TRACEMEM_PRIMARY_FIRST_ENABLED') != 'approved-primary-first-originals':
        raise ValueError('explicit primary-first activation required')
    return create_app(env, _system_type=PrimaryFirstSystem,
                      _transport_type=RetrySelectionTransport, _stop_at_chat_done=True)
