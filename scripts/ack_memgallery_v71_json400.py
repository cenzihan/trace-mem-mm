"""Acknowledge exactly the known first V7.1 judge HTTP400; never resend.

Only --ack writes: one explicit rejection row and the original guard closure,
then a new exclusive report. The provider journal, ledger, all frozen inputs and
answers stay byte-identical. Missing/unknown/other errors cannot pass this gate.
"""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_memgallery_v7_encoding import canonical, digest, sha, load, lines, ro, history, require
from scripts import audit_memgallery_v7_chat as prior
from agent_memory.memgallery_v7_reader import proxy_judge_payload
from agent_memory.visual_recovery import _strict_json

CHAT = Path('runtime/memgallery-v7/chat-citation-v71')
REPORT = Path('reports/memgallery-v71-judge-json-rejection.json')
ERROR_MESSAGE = "'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'."
ERROR_BODY = ('{\n  "error": {\n    "message": "' + ERROR_MESSAGE + '",\n'
              '    "type": "invalid_request_error",\n    "param": "messages",\n    "code": null\n  }\n}').encode()
ERROR_SHA256 = 'c259bb7f479b9faee0845ca7c0c7d8fca618e090ab8f4270edd4de252b2488f3'
REASON = 'Known HTTP400: JSON-object response format requires json in messages; zero model grades; no retry; full hold retained.'


def inspect(root=ROOT):
    root = Path(root).resolve(); chat = root / CHAT
    live, reader, judge = (load(chat / name) for name in ('FROZEN_LIVE.json','FROZEN_READER.json','FROZEN_JUDGE.json'))
    completed = load(chat / 'READER_COMPLETED.json')
    require(live['baseline']['attempts'] == 1547 and live['reader_sha256'] == sha(chat / 'FROZEN_READER.json'), 'unexpected live baseline/freeze')
    require(reader['planned_reader_answers'] == 64 and reader['max_requests'] == 72 and reader['max_preflight_cny'] == '18'
            and reader['fixed_hold_cny'] == '.25', 'unexpected reader scope')
    require(completed['state'] == 'complete' and completed['phase'] == 'reader' and completed['count'] == 64
            and completed['frozen_live_sha256'] == sha(chat / 'FROZEN_LIVE.json')
            and completed['output_sha256'] == sha(chat / 'reader-answers.jsonl'), '64 reader completion not bound')
    interface = load(chat / 'FROZEN_INTERFACE.json')
    prior.verify_hashes(root, interface['hashes'])
    prior.verify_hashes(root, reader['input_hashes']); prior.verify_hashes(root, reader['code_hashes'])
    skipped = prior.verify_hashes(root, judge['input_hashes'], skip_scorer_only=True)
    prior.verify_hashes(root, judge['code_hashes'])
    requests_path = prior.safe_path(root, judge['requests_file'])
    require(requests_path == chat / 'judge-requests.jsonl' and sha(requests_path) == judge['requests_sha256']
            and judge['judge_requests'] == 8 and judge['reader_answers'] == 64, 'eight frozen judge requests required')
    requests = lines(requests_path)
    mapping_path = (root / judge['mapping_file']).resolve()
    require(mapping_path.is_relative_to(chat / 'scorer-only') and sha(mapping_path) == judge['mapping_sha256'], 'private mapping hash differs')
    require(len(requests) == 8 and not (chat / 'JUDGE_COMPLETED.json').exists()
            and (not (chat / 'judge-results.jsonl').exists() or (chat / 'judge-results.jsonl').stat().st_size == 0), 'no judge grades may exist')
    payloads = [proxy_judge_payload(r['pack']) for r in requests]
    require(all(p['response_format'] == {'type':'json_object'} and p['max_completion_tokens'] == 1536
                and 'json' not in canonical(p['messages']).lower() for p in payloads), 'not the exact missing-json failure class')
    namespace = 'memgallery-v7:' + sha(chat / 'FROZEN_LIVE.json')
    manifest = dict(keys=[requests[0]['request_id']], payload_sha256=digest(canonical(payloads[0]).encode()),
                    run_path=str((chat / 'run.sqlite').resolve()))
    fingerprint = digest(canonical(dict(namespace=namespace, manifest=manifest)).encode())
    with closing(ro(chat / 'receipts/private-embedding.sqlite')) as db:
        ops = [dict(r) for r in db.execute('SELECT * FROM operations ORDER BY run_ordinal')]
        claims = [tuple(r) for r in db.execute('SELECT scope,chunk_key,fingerprint FROM claims')]
    require(len(ops) == 65 and [r['run_ordinal'] for r in ops] == list(range(1,66)), 'exactly64 successful receipts and one rejection required')
    expected_claims = []
    answers = lines(chat / 'reader-answers.jsonl')
    require(len(answers) == 64, 'exactly64 answers required')
    for ordinal, (op, answer) in enumerate(zip(ops[:64], answers), 1):
        require(op['scope'] == namespace and op['run_path'] == manifest['run_path'] and op['state'] == 'complete'
                and op['transport_completed'] == 1 and op['status'] == 200 and op['redacted'] == 0
                and digest(op['body']) == op['body_sha'] == answer['receipt_sha256']
                and op['attempt_id'] == answer['attempt_id'] and answer['run_ordinal'] == ordinal, 'original64 receipt binding differs')
        own = _strict_json(op['manifest'])
        require(own['keys'] == [answer['request_id']] and own['run_path'] == manifest['run_path']
                and op['fingerprint'] == digest(canonical(dict(namespace=namespace,manifest=own)).encode()), 'reader receipt fingerprint differs')
        expected_claims.append((namespace,answer['request_id'],op['fingerprint']))
    op = ops[-1]
    require(op['fingerprint'] == fingerprint and op['scope'] == namespace and op['manifest'] == canonical(manifest)
            and op['run_path'] == manifest['run_path'] and op['state'] == 'received' and op['status'] == 400
            and op['transport_completed'] == 0 and op['redacted'] == 0 and op['attempt_id'], 'unique65 is not a known received HTTP400')
    require(op['body'] == ERROR_BODY and len(op['body']) == 219 and digest(op['body']) == op['body_sha'] == ERROR_SHA256,
            'raw219-byte error must match exactly')
    require(_strict_json(op['body']) == {'error':{'message':ERROR_MESSAGE,'type':'invalid_request_error','param':'messages','code':None}}, 'error structure differs')
    expected_claims.append((namespace,requests[0]['request_id'],fingerprint))
    require(sorted(claims) == sorted(expected_claims) and len({r['attempt_id'] for r in ops}) == 65, 'extra claims or duplicate attempts')
    with closing(ro(root / 'runtime/development-budget.sqlite')) as db:
        require(history(db,1547) == live['historical_sha256'], 'old1547 ledger history changed')
        ledger_history = history(db,1612)
        attempts = [dict(db.execute('SELECT rowid AS original_rowid,* FROM attempts WHERE id=?',(r['attempt_id'],)).fetchone()) for r in ops]
        observations = [dict(db.execute('SELECT * FROM provider_observations WHERE attempt_id=?',(r['attempt_id'],)).fetchone()) for r in ops]
        count = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    require([r['original_rowid'] for r in attempts] == list(range(1548,1613)), 'new65 must have exactly contiguous unique ledger rows')
    for n, (attempt, observed) in enumerate(zip(attempts,observations),1):
        require(attempt['model'] == 'gpt-4o-mini' and attempt['status'] == 'uncertain'
                and attempt['amount_kind'] == 'retained_upper_bound' and attempt['conversion_basis'] == 'budget_assumption_not_market_fx'
                and attempt['reserved'] == attempt['accounted'] == 250000 and observed['mode'] == 'live', 'ledger fixed hold changed')
        if n <= 64:
            require(observed['category'] == 'success' and observed['http_status'] == 200, 'old64 were not successful HTTP calls')
    attempt, observed = attempts[-1], observations[-1]
    require(observed['category'] == 'client_error' and observed['http_status'] == 400
            and all(observed[k] is None for k in ('returned_model','input_tokens','output_tokens','estimated_usd'))
            and attempt['input_tokens'] is None and attempt['output_tokens'] is None
            and attempt['max_input'] == 128000 and attempt['max_output'] == 1536, 'not an observed no-usage client rejection')
    with closing(ro(chat / 'run.sqlite')) as db:
        runs = [dict(r) for r in db.execute('SELECT * FROM run')]
        calls = [dict(r) for r in db.execute('SELECT * FROM calls ORDER BY ordinal')]
        has_ack = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='known_json400_rejections'").fetchone()
        acks = [dict(r) for r in db.execute('SELECT * FROM known_json400_rejections')] if has_ack else []
    require(len(runs) == 1 and runs[0]['max_requests'] == 72 and runs[0]['cap'] == 18000000 and len(calls) == 65
            and [r['ordinal'] for r in calls] == list(range(1,66)) and all(r['held'] == 250000 and r['model'] == 'gpt-4o-mini' for r in calls)
            and all(r['state'] == 'complete' for r in calls[:64]), 'guard counts/caps/holds or completed64 changed')
    acknowledged = runs[0]['state'] == 'complete' and calls[-1]['state'] == 'rejected_response'
    if acknowledged:
        require(len(acks) == 1 and acks[0]['ordinal'] == 65 and acks[0]['attempt_id'] == op['attempt_id']
                and acks[0]['receipt_sha256'] == ERROR_SHA256 and acks[0]['category'] == 'client_error'
                and acks[0]['http_status'] == 400 and acks[0]['reason'] == REASON
                and acks[0]['ledger_history_sha256'] == ledger_history and acks[0]['acknowledged_at'] == calls[-1]['finished_at']
                and calls[-1]['returned_model'] is None, 'closed guard lacks exact durable acknowledgement')
    else:
        require(runs[0]['state'] == 'active' and calls[-1]['state'] == 'claimed' and not acks and count == 1612
                and calls[-1]['finished_at'] is None and calls[-1]['returned_model'] is None, 'unknown/other guard state cannot be acknowledged')
    files = [chat / n for n in ('FROZEN_LIVE.json','FROZEN_READER.json','FROZEN_INTERFACE.json','FROZEN_JUDGE.json',
        'READER_COMPLETED.json','reader-packs.jsonl','reader-answers.jsonl','judge-requests.jsonl','receipts/private-embedding.sqlite')]
    files.append(mapping_path)  # Hash bytes only; never parse private mapping/gold.
    hashes = {p.relative_to(root).as_posix():sha(p) for p in files}
    return dict(root=root,chat=chat,acknowledged=acknowledged,ack=acks[0] if acks else None,operation=op,
                artifact_sha256=hashes,guard_sha256=sha(chat/'run.sqlite'),ledger_history_sha256=ledger_history,
                historical1547_sha256=live['historical_sha256'],skipped_scorer_only_inputs=skipped)


def acknowledge(root=ROOT):
    evidence = inspect(root); root,chat = evidence['root'],evidence['chat']
    if not evidence['acknowledged']:
        with closing(sqlite3.connect(chat / 'run.sqlite')) as db, db:
            db.execute('BEGIN IMMEDIATE')
            require(sha(chat/'run.sqlite') == evidence['guard_sha256'], 'guard changed before acknowledgement')
            require(evidence['artifact_sha256'] == {p:sha(root/p) for p in evidence['artifact_sha256']}, 'receipt/frozen inputs changed before acknowledgement')
            with closing(ro(root/'runtime/development-budget.sqlite')) as ledger:
                require(history(ledger,1612) == evidence['ledger_history_sha256'], 'ledger changed before acknowledgement')
            now = datetime.now(timezone.utc).isoformat()
            db.execute('''CREATE TABLE known_json400_rejections(ordinal INTEGER PRIMARY KEY,attempt_id TEXT UNIQUE,
                receipt_sha256 TEXT,category TEXT,http_status INTEGER,reason TEXT,guard_before_sha256 TEXT,
                ledger_history_sha256 TEXT,acknowledged_at TEXT)''')
            db.execute('INSERT INTO known_json400_rejections VALUES(?,?,?,?,?,?,?,?,?)',
                (65,evidence['operation']['attempt_id'],ERROR_SHA256,'client_error',400,REASON,evidence['guard_sha256'],evidence['ledger_history_sha256'],now))
            result = db.execute("UPDATE calls SET state='rejected_response',finished_at=? WHERE ordinal=65 AND state='claimed'",(now,))
            require(result.rowcount == 1, 'unique claimed65 changed')
            db.execute("UPDATE run SET state='complete'")
    verified = inspect(root)
    require(verified['artifact_sha256'] == evidence['artifact_sha256'] and verified['ledger_history_sha256'] == evidence['ledger_history_sha256'], 'immutable evidence changed')
    result = dict(schema='memgallery-v71-known-json400-ack-v1',state='passed',acknowledgement='known_json400_acknowledged_guard_closed',
        acknowledged_at_utc=verified['ack']['acknowledged_at'],ordinal=65,attempt_id=verified['operation']['attempt_id'],
        category='client_error',http_status=400,error_type='invalid_request_error',error_param='messages',
        error_message=ERROR_MESSAGE,receipt_sha256=ERROR_SHA256,receipt_bytes=219,real_http_requests=1,model_grades_generated=0,
        usage_available=False,actual_usage_cost=None,ledger_hold_retained_micro_cny=250000,automatic_retries=0,
        reader_requests=64,judge_requests=1,reader_calls_completed=64,judge_calls_rejected=1,guard_state='complete',guard_claims_preserved=65,
        guard_max_requests_preserved=72,guard_cap_micro_cny_preserved=18000000,guard_held_micro_cny=16250000,
        guard_before_sha256=verified['ack']['guard_before_sha256'],guard_after_sha256=verified['guard_sha256'],
        historical1547_rows_unchanged=True,historical1547_sha256=verified['historical1547_sha256'],
        ledger_prefix_rows=1612,ledger_prefix_sha256=verified['ledger_history_sha256'],
        journal_unchanged=True,artifact_sha256=dict(verified['artifact_sha256'], **{(CHAT/'run.sqlite').as_posix():verified['guard_sha256']}),
        skipped_scorer_only_inputs=verified['skipped_scorer_only_inputs'],
        future_separate_judge_max_requests=8,future_separate_judge_hold_micro_cny=2000000,
        maximum_cumulative_real_provider_requests=164,maximum_cumulative_provider_hold_micro_cny=35600000,
        maximum_nominal_claims_including_original_local_preflight=165,maximum_nominal_hold_micro_cny=35850000,
        reason=REASON,script_sha256=sha(Path(__file__)))
    path = root / REPORT
    if path.exists():
        require(load(path) == result, 'existing acknowledgement report differs; never overwrite')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ack',action='store_true');args=parser.parse_args()
    if args.ack:
        result=acknowledge();print(canonical(dict(state=result['state'],report=REPORT.as_posix(),real_http_requests=1,model_grades_generated=0)))
    else:
        result=inspect();print(canonical(dict(state='already_acknowledged' if result['acknowledged'] else 'known_json400_verified_no_changes')))


if __name__=='__main__':main()
