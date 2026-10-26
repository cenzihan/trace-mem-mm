"""Read-only proof of the original local admission failure, never a retry.

Only the aggregate report is new. A pure admission reproduction has throwing
sentinels for credentials, ledger and HTTP and must stop before all of them.
The original claimed guard and unknown journal entry remain untouched.
"""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_memgallery_v7_encoding import canonical, digest, sha, load, lines, ro, history, require
from agent_memory.memgallery_v7_reader import reader_payload
from agent_memory.provider_http import BudgetedJSONTransport, OPENAI_ENDPOINT

CHAT = Path('runtime/memgallery-v7/chat')
REPORT = Path('reports/memgallery-v7-chat-preflight-failure-audit.json')


def audit(root=ROOT):
    root = Path(root).resolve(); chat = root / CHAT
    progress_path = root / 'reports/memgallery-v7-chat-progress.json'
    frozen, live, progress = (load(chat / 'FROZEN_READER.json'), load(chat / 'FROZEN_LIVE.json'), load(progress_path))
    require(live['reader_sha256'] == sha(chat / 'FROZEN_READER.json') and live['baseline']['attempts'] == 1483,
            'original preflight freeze/baseline mismatch')
    require(progress.get('state') == 'stopped_no_automatic_retry' and progress.get('phase') == 'reader'
            and progress.get('error_type') == 'ValueError' and progress.get('completed') == 0
            and progress.get('new_this_invocation') == 0 and progress.get('budget') == live['baseline'],
            'no exact unchanged before/after ledger snapshot for local failure')
    for name, expected in {**frozen['input_hashes'], **frozen['code_hashes']}.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and 'scorer-only' not in path.parts and sha(path) == expected,
                'frozen original input/code changed')
    packs_path = root / frozen['packs_file']
    require(packs_path.resolve() == chat / 'reader-packs.jsonl' and sha(packs_path) == frozen['packs_sha256'],
            'original reader packs changed')
    packs = lines(packs_path)
    require(len(packs) == 64, 'all64 original reader packs required')
    first = packs[0]; payload = reader_payload(first['pack'])
    require(type(payload.get('temperature')) is int and payload['temperature'] == 0, 'exact original temperature0 required')
    paths = [chat / 'FROZEN_READER.json', chat / 'FROZEN_LIVE.json', packs_path,
             chat / 'run.sqlite', chat / 'receipts/private-embedding.sqlite', progress_path]
    before = {p.relative_to(root).as_posix(): sha(p) for p in paths}
    with closing(ro(chat / 'run.sqlite')) as db:
        runs = [tuple(r) for r in db.execute('SELECT max_requests,cap,state FROM run')]
        calls = [tuple(r) for r in db.execute('SELECT ordinal,model,held,state,returned_model FROM calls')]
    require(runs == [(72, 18000000, 'active')]
            and calls == [(1, 'gpt-4o-mini', 250000, 'claimed', None)], 'original single claimed guard changed')
    with closing(ro(chat / 'receipts/private-embedding.sqlite')) as db:
        operations = [dict(r) for r in db.execute('SELECT * FROM operations')]
        claims = [tuple(r) for r in db.execute('SELECT scope,chunk_key,fingerprint FROM claims')]
    require(len(operations) == 1 and len(claims) == 1, 'original journal has extra/missing operations')
    row = operations[0]
    request_id = digest((first['query_id'] + '\0' + first['variant']).encode('utf-8'))
    manifest = dict(keys=[request_id], payload_sha256=digest(canonical(payload).encode('utf-8')),
                    run_path=str((chat / 'run.sqlite').resolve()))
    namespace = 'memgallery-v7:' + sha(chat / 'FROZEN_LIVE.json')
    fingerprint = digest(canonical(dict(namespace=namespace, manifest=manifest)).encode('utf-8'))
    require(row['fingerprint'] == fingerprint and row['scope'] == namespace
            and row['manifest'] == canonical(manifest) and row['state'] == 'unknown'
            and row['transport_completed'] == 0
            and all(row[key] is None for key in ('attempt_id', 'status', 'body', 'body_sha', 'redacted', 'run_path', 'run_ordinal'))
            and claims == [(namespace, request_id, fingerprint)], 'journal is not the exact unbound local preflight failure')
    require(not (chat / 'reader-answers.jsonl').exists() and not (chat / 'READER_COMPLETED.json').exists(),
            'original failed run unexpectedly contains reader outcomes')
    # No constructor, credentials, real HTTP or ledger instance is created.
    touched = []
    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError('admission reproduction reached a forbidden external/accounting action')
    class Sentinel:
        reserve = exchange = forbidden
    transport = BudgetedJSONTransport.__new__(BudgetedJSONTransport)
    transport.model = 'gpt-4o-mini'; transport.endpoint = OPENAI_ENDPOINT
    transport.output_token_cap = 1536; transport.credential_reader = forbidden
    transport.ledger = Sentinel(); transport.http = Sentinel()
    try:
        transport(OPENAI_ENDPOINT, payload)
    except ValueError as error:
        require(str(error) == 'unsupported chat parameters or persistence', 'unexpected admission failure')
    else:
        raise ValueError('frozen old transport no longer rejects the exact payload')
    require(not touched, 'admission reached credential/ledger/HTTP')
    with closing(ro(root / 'runtime/development-budget.sqlite')) as db:
        require(history(db, 1483) == live['historical_sha256'], 'historical ledger changed after local rejection')
    require(before == {p.relative_to(root).as_posix(): sha(p) for p in paths}, 'old failed-run artifacts changed during audit')
    return dict(schema='memgallery-v7-chat-local-preflight-failure-audit-v1', state='confirmed_no_provider_dispatch',
        audited_at_utc=datetime.now(timezone.utc).isoformat(), original_provider_attempts_before=1483,
        original_provider_attempts_after=1483, original_provider_attempts_added=0,
        original_budget_accounted_before=live['baseline']['accounted_micro_cny'],
        original_budget_accounted_after=progress['budget']['accounted_micro_cny'],
        original_provider_cost_added_micro_cny=0, credential_reads_by_failed_attempt=0,
        original_http_dispatches=0, pure_admission_reproduction=True, audit_http_dispatches=0,
        failure='legacy local chat-parameter admission rejects temperature0 before credentials, ledger reserve or HTTP',
        journal_state='unknown_no_attempt_no_status_no_body', original_guard_state='active_one_claimed',
        original_guard_claims_preserved=1, nominal_guard_preflight_preserved_micro_cny=250000,
        no_retry_or_recovery_performed=True, original64packs_sha256=sha(packs_path),
        original_artifact_hashes=before, frozen_code_hashes=frozen['code_hashes'],
        historical1483_rows_unchanged=True, existing_artifacts_modified=False,
        source_question_or_answer_content_in_report=False, audit_script_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    import json
    result = audit()
    destination = ROOT / REPORT
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state=result['state'], provider_attempts_added=0, report=REPORT.as_posix())))
