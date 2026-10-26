"""Synthetic cumulative164 accounting, no API, native QA or live ledger."""
from contextlib import closing
from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
import unittest
from scripts import audit_memgallery_v71_judge_json as audit
from tests import test_memgallery_v71_json400 as known


def fixture(root):
    known.fixture(root);audit.rejection.acknowledge(root)
    put,put_lines,guard=known.previous.legacy.put,known.previous.legacy.put_lines,known.previous.legacy.guard
    work=root/audit.WORK;work.mkdir(parents=True)
    old=root/audit.rejection.CHAT;old_requests=audit.lines(old/'judge-requests.jsonl')
    requests=[]
    for previous in old_requests:
        payload=audit.proxy_judge_payload(previous['pack']);payload['messages'][0]['content']+='\nReturn JSON.'
        rid=audit.digest(audit.canonical(['judge-json-v71',previous['request_id'],payload]).encode())
        requests.append(dict(request_id=rid,payload=payload))
    put_lines(work/'judge-requests.jsonl',requests)
    mapping=work/'scorer-only/judge-mapping.json';mapping.parent.mkdir();mapping.write_text('{}',encoding='utf-8')
    frozen=dict(judge_requests=8,reader_answers=64,grades=[0,.25,.5,.75,1],max_requests=8,max_preflight_cny='2',fixed_hold_cny='.25',
        requests_file=(audit.WORK/'judge-requests.jsonl').as_posix(),requests_sha256=audit.sha(work/'judge-requests.jsonl'),
        mapping_file=mapping.relative_to(root).as_posix(),mapping_sha256=audit.sha(mapping),
        input_hashes={audit.rejection.REPORT.as_posix():audit.sha(root/audit.rejection.REPORT)},
        code_hashes={'synthetic-code.py':audit.sha(root/'synthetic-code.py')})
    put(root,audit.WORK/'FROZEN_JUDGE.json',frozen)
    with closing(audit.ro(root/'runtime/development-budget.sqlite')) as db:history=audit.history(db,1612)
    live=dict(baseline={'attempts':1612},judge_sha256=audit.sha(work/'FROZEN_JUDGE.json'),max_requests=8,max_preflight_cny='2',
        automatic_retries=False,old_guard_reopened=False,historical_sha256=history,code_hashes=frozen['code_hashes'])
    put(root,audit.WORK/'FROZEN_LIVE.json',live)
    guard(work/'run.sqlite',8,2000000,8,'gpt-4o-mini')
    namespace='memgallery-v71-judge-json:'+audit.sha(work/'FROZEN_LIVE.json')
    journal=work/'receipts/private-embedding.sqlite';journal.parent.mkdir()
    with closing(sqlite3.connect(journal)) as db,db:
        db.executescript('CREATE TABLE operations(fingerprint TEXT,scope TEXT,manifest TEXT,state TEXT,attempt_id TEXT,status INTEGER,body BLOB,body_sha TEXT,redacted INTEGER,transport_completed INTEGER,run_path TEXT,run_ordinal INTEGER);CREATE TABLE claims(scope TEXT,chunk_key TEXT,fingerprint TEXT);')
    results=[]
    for ordinal,request in enumerate(requests,1):
        inner=dict(summary='',evidence_quotes=[],judgments=[dict(item_id=i,score=.75,reason='Synthetic.') for i in range(8)])
        response=dict(model='gpt-4o-mini',choices=[dict(finish_reason='stop',message={'content':audit.canonical(inner)})],
                      usage=dict(prompt_tokens=200,completion_tokens=30,total_tokens=230))
        manifest=dict(keys=[request['request_id']],payload_sha256=audit.digest(audit.canonical(request['payload']).encode()),run_path=str((work/'run.sqlite').resolve()))
        fingerprint=audit.digest(audit.canonical(dict(namespace=namespace,manifest=manifest)).encode())
        body=audit.canonical(response).encode();attempt='synthetic-json-judge-'+str(ordinal)
        with closing(sqlite3.connect(journal)) as db,db:
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(fingerprint,namespace,audit.canonical(manifest),'complete',attempt,200,body,audit.digest(body),0,1,manifest['run_path'],ordinal))
            db.execute('INSERT INTO claims VALUES(?,?,?)',(namespace,request['request_id'],fingerprint))
        with closing(sqlite3.connect(root/'runtime/development-budget.sqlite')) as db,db:
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(attempt,'gpt-4o-mini','uncertain','retained_upper_bound','budget_assumption_not_market_fx',250000,250000,200,30,128000,1536,'1.5','6'))
            db.execute('INSERT INTO provider_observations VALUES(?,?,?,?,?,?,?,?)',(attempt,'live','success',200,'gpt-4o-mini',200,30,'0.000048'))
        results.append(dict(request_id=request['request_id'],response=response,attempt_id=attempt,receipt_sha256=audit.digest(body),run_ordinal=ordinal))
    put_lines(work/'judge-results.jsonl',results)
    put(root,audit.WORK/'JUDGE_COMPLETED.json',dict(state='complete',phase='judge',count=8,output_sha256=audit.sha(work/'judge-results.jsonl'),
        frozen_live_sha256=audit.sha(work/'FROZEN_LIVE.json'),judge_frozen_sha256=audit.sha(work/'FROZEN_JUDGE.json')))
    put(root,'reports/memgallery-v71-judge-json-progress.json',dict(state='judge_complete',phase='judge'))


class MemGalleryV71JudgeJsonAuditTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='v71-final-audit-synthetic-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();fixture(self.root)

    def test_total164_keeps_failed64_and_rejected400_without_modifications(self):
        before={p:audit.sha(p) for p in self.root.rglob('*') if p.is_file()};result=audit.audit(self.root)
        self.assertEqual(result['total_real_provider_requests'],164)
        self.assertEqual(result['cumulative_provider_hold_micro_cny'],35600000)
        self.assertEqual(result['nominal_hold_including_local_preflight_micro_cny'],35850000)
        self.assertEqual(result['rejected_judge_http_requests'],1);self.assertIsNone(result['rejected_request_actual_usage_cost'])
        self.assertEqual(Decimal(result['known_successful_chat_usage_estimate_usd']),Decimal('.005184'))
        self.assertEqual(before,{p:audit.sha(p) for p in before})

    def test_unknown_new_receipt_or_usage_tamper_or_old_hold_change_blocks(self):
        cases=[(self.root/audit.WORK/'receipts/private-embedding.sqlite',"UPDATE operations SET state='unknown' WHERE run_ordinal=1"),
               (self.root/'runtime/development-budget.sqlite',"UPDATE provider_observations SET estimated_usd='99' WHERE attempt_id='synthetic-json-judge-1'"),
               (self.root/'runtime/development-budget.sqlite',"UPDATE attempts SET accounted=0 WHERE id='synthetic-v71-65'")]
        for path,sql in cases:
            data=path.read_bytes()
            with closing(sqlite3.connect(path)) as db,db:db.execute(sql)
            with self.subTest(sql=sql),self.assertRaises(ValueError):audit.audit(self.root)
            path.write_bytes(data)

    def test_incomplete_judge_and_saved_response_tamper_block(self):
        path=self.root/'reports/memgallery-v71-judge-json-progress.json';data=path.read_bytes()
        known.previous.legacy.put(self.root,path.relative_to(self.root),dict(state='running',phase='judge'))
        with self.assertRaises(ValueError):audit.audit(self.root)
        path.write_bytes(data)
        path=self.root/audit.WORK/'judge-results.jsonl';rows=audit.lines(path);rows[0]['attempt_id']='wrong'
        known.previous.legacy.put_lines(path,rows)
        with self.assertRaises(ValueError):audit.audit(self.root)


if __name__=='__main__':unittest.main()
