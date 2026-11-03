"""Bounded immutable media results, scoped to one explicitly opted-in Add."""
from contextlib import contextmanager
from contextvars import ContextVar

_current = ContextVar('request_media_results', default=None)
MAX_BYTES = 32 * 1024 * 1024
MAX_ENTRIES = 128


@contextmanager
def media_request_cache():
    state = {'values': {}, 'bytes': 0}
    token = _current.set(state)
    try:
        yield state
    finally:
        state['values'].clear()
        state['bytes'] = 0
        _current.reset(token)


def media_result(kind, uri, compute):
    state = _current.get()
    if state is None or not isinstance(uri, str):
        return compute()
    key = (kind, uri)
    if key in state['values']:
        return state['values'][key]
    value = compute()  # Never retain failed validation or a mutable result.
    if kind == 'decoded':
        assert isinstance(value, tuple) and isinstance(value[0], bytes)
        size = len(value[0])
    elif kind == 'preview':
        assert isinstance(value, str)
        size = len(value.encode('ascii'))
    else:
        raise ValueError('unsupported media result')
    if len(state['values']) < MAX_ENTRIES and state['bytes'] + size <= MAX_BYTES:
        state['values'][key] = value
        state['bytes'] += size
    return value
