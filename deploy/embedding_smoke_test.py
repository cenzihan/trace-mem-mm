"""Three-request, synthetic embedding smoke for an explicitly granted live profile.

The server owns provider dispatch, receipts and accounting. This client only
checks loopback HTTP behavior; its success is not proof of provider billing or
model execution. Run seed once, recreate the container, then run verify once.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


SOURCE_TEXTS = (
    'The spare door keys are stored in the amber cabinet near the workshop entrance.',
    'The weekly planning meeting takes place on Friday in the east room.',
)
QUERY_TEXTS = {
    'seed': 'Where are the spare door keys kept?',
    'verify': 'Which cabinet near the workshop entrance stores the reserve door keys?',
}
USER_ID = 'embedding-smoke-user-a'
OTHER_USER_ID = 'embedding-smoke-user-b'
PROFILE = 'live_prepaid_embedding_smoke'


class SmokeFailure(RuntimeError):
    """Only fixed, operator-safe messages may be used here."""


def add_payload():
    """Return a fresh fixture, never a mutable shared request object."""
    return {
        'request_id': 'embedding-smoke-add-v1',
        'user_id': USER_ID,
        'session_id': 'embedding-smoke-session-v1',
        'messages': [{'role': 'user', 'content': text} for text in SOURCE_TEXTS],
    }


def query_payload(phase):
    if phase not in QUERY_TEXTS:
        raise ValueError('unknown embedding smoke phase')
    return {'user_id': USER_ID, 'query': QUERY_TEXTS[phase], 'top_k': 2}


def expected_embedding_payloads():
    """Freeze the grant's three exact requests, including native index spacing.

    Default deterministic cards append one space to no-facts source text.
    Tests compare these frozen requests with the actual native Add/Search path.
    Query strings do not receive that index-only suffix.
    """
    inputs = ([text + ' ' for text in SOURCE_TEXTS],
              [QUERY_TEXTS['seed']], [QUERY_TEXTS['verify']])
    return [dict(model='text-embedding-v4', input=list(texts), dimensions=256,
                 encoding_format='float') for texts in inputs]


def local_target(base_url):
    try:
        target = urlsplit(base_url)
        port = target.port if target.port is not None else 8000
    except ValueError:
        raise SmokeFailure('invalid_loopback_target') from None
    if (target.scheme != 'http' or target.hostname not in ('127.0.0.1', 'localhost')
            or target.username is not None or target.password is not None
            or target.path not in ('', '/') or target.query or target.fragment
            or not 1 <= port <= 65535):
        raise SmokeFailure('loopback_http_required')
    return target.hostname, port


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def exercise(phase, base_url):
    hostname, port = local_target(base_url)
    auth_mode = os.environ.get('TRACEMEM_AUTH_MODE', 'Bearer')
    require(auth_mode in ('Bearer', 'Token', 'X-Api-Key'), 'authenticated_mode_required')
    filename = os.environ.get('TRACEMEM_AUTH_TOKEN_FILE')
    require(bool(filename), 'service_token_file_required')
    with Path(filename).open('rb') as source:
        raw = source.read(4097)
    token = raw.decode('ascii').rstrip('\r\n')
    require(len(raw) <= 4096 and 32 <= len(token) <= 4096
            and all(33 <= ord(char) <= 126 for char in token), 'invalid_service_token')

    def request(method, path, payload=None, authenticated=True):
        connection = http.client.HTTPConnection(hostname, port, timeout=150)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            if auth_mode == 'X-Api-Key':
                headers['X-Api-Key'] = token
            else:
                headers['Authorization'] = auth_mode + ' ' + token
        try:
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            raw_response = response.read(2 * 1024 * 1024 + 1)
            require(len(raw_response) <= 2 * 1024 * 1024, 'response_limit_exceeded')
            return response.status, json.loads(raw_response)
        finally:
            connection.close()

    status, health = request('GET', '/health', authenticated=False)
    require(status == 200 and isinstance(health, dict)
            and health.get('status') == 'ok' and health.get('mode') == PROFILE
            and health.get('dense') == 'text-embedding-v4'
            and health.get('reranker') == 'disabled'
            and health.get('cards') == 'explicit-source-cards-no-LLM',
            'expected_embedding_only_profile_required')
    add = add_payload()
    query = query_payload(phase)
    require(request('POST', '/add', add, False)[0] == 401, 'add_auth_rejection_failed')
    require(request('POST', '/search', query, False)[0] == 401, 'search_auth_rejection_failed')
    echo = dict(success=True, **{key: add[key] for key in ('request_id', 'user_id', 'session_id')})
    require(request('POST', '/add', add) == (200, echo), 'add_or_restart_idempotency_failed')
    if phase == 'seed':
        require(request('POST', '/add', add_payload()) == (200, echo), 'add_idempotency_failed')
        conflicting = add_payload()
        conflicting['messages'][0]['content'] = 'A changed synthetic request body must be rejected.'
        require(request('POST', '/add', conflicting)[0] == 409, 'conflicting_add_rejection_failed')
    other_query = dict(query, user_id=OTHER_USER_ID)
    require(request('POST', '/search', other_query) == (200, {'data': []}), 'user_isolation_failed')
    status, result = request('POST', '/search', query)
    require(status == 200 and isinstance(result, dict), 'search_failed')
    rows = result.get('data')
    require(isinstance(rows, list) and len(rows) == 2, 'original_record_count_mismatch')
    expected = [[dict(type='text', text=text)] for text in SOURCE_TEXTS]
    require(all(isinstance(row, dict) for row in rows)
            and [row.get('content') for row in rows] == expected, 'target_rank_or_original_text_mismatch')
    require(all(isinstance(row.get('id'), str) for row in rows)
            and len({row['id'] for row in rows}) == 2, 'distinct_record_identity_required')
    return dict(state='passed', phase=phase, synthetic=True, profile=PROFILE,
                authenticated_add_search=True, user_isolation=True,
                idempotent_add=True, original_text_preserved=True,
                records_returned=2, target_rank=1,
                embedding_requests_expected=2 if phase == 'seed' else 1,
                provider_receipts_verified=False, llm_required=False, rerank_required=False,
                official_evaluation=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('seed', 'verify'))
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args(argv)
    try:
        result = exercise(args.phase, args.base_url)
    except (SmokeFailure, OSError, ValueError, http.client.HTTPException) as exc:
        message = str(exc) if isinstance(exc, SmokeFailure) else 'local_smoke_operation_failed'
        print(json.dumps(dict(state='failed', phase=args.phase, error=message,
                              synthetic=True, official_evaluation=False)))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
