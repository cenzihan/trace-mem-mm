"""Full synthetic old64 + new72 accounting; no native dataset, API or live ledger."""
from contextlib import closing
from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import audit_memgallery_v71_chat as audit
from tests import test_memgallery_v7_chat_audit as legacy


def sources():
    rows = {'src': dict(text='Synthetic source.', source_session='S', source_round='R', source_date='', image_ids=[])}
    for i in range(19):
        rows['synthetic-source-' + str(i)] = dict(text='Synthetic extra source ' + str(i),
            source_session='S' + str(i), source_round='R' + str(i), source_date='', image_ids=[])
    return rows


def fixture(root):
    old_packer = legacy.pack_memgallery_evidence
    def twenty(query, ranked, source, count_tokens):
        return old_packer(query, list(sources()), sources(), count_tokens)
    with patch.object(legacy, 'pack_memgallery_evidence', side_effect=twenty):
        legacy.fixture(root)
    legacy.MemGalleryV7ChatAuditTests.make_reader_only_failure(SimpleNamespace(root=root))
    previous = audit.prior.audit_reader_only(root)
    legacy.put(root, audit.OLD_AUDIT, previous)
    chat = root / audit.CHAT; chat.mkdir(parents=True)
    previous_packs = audit.lines(root / audit.OLD / 'reader-packs.jsonl')
    packs = []
    for old in previous_packs:
        query = dict(index=old['index'], point='FR', question='Synthetic question?')
        pack = audit.native.pack_memgallery_evidence(query, list(sources()), sources(), lambda s: len(s)//4+1)
        packs.append(dict(query_id=old['query_id'], index=old['index'], point=old['point'], variant=old['variant'], pack=pack))
    legacy.put_lines(chat / 'reader-packs.jsonl', packs)
    code = 'synthetic-code.py'
    frozen = dict(planned_reader_answers=64, max_requests=72, max_preflight_cny='18', fixed_hold_cny='.25',
        packs_file=(audit.CHAT / 'reader-packs.jsonl').as_posix(), packs_sha256=audit.sha(chat / 'reader-packs.jsonl'),
        input_hashes={(audit.CHAT / 'reader-packs.jsonl').as_posix(): audit.sha(chat / 'reader-packs.jsonl')},
        code_hashes={code:audit.sha(root/code)})
    legacy.put(root, audit.CHAT / 'FROZEN_READER.json', frozen)
    interface_files = [root / audit.OLD_AUDIT, chat / 'FROZEN_READER.json', chat / 'reader-packs.jsonl']
    interface_files += [root / audit.OLD / name for name in ('FAILURE_COMPLETED.json','FROZEN_READER.json','FROZEN_LIVE.json',
                       'reader-packs.jsonl','reader-answers.jsonl','run.sqlite','receipts/private-embedding.sqlite')]
    legacy.put(root, audit.CHAT / 'FROZEN_INTERFACE.json', dict(state='new_alias_interface_frozen_no_calls',
        same_queries_rankings_complete20sources=True,new_provider_calls_max=72,new_preflight_cny='18',
        aggregate_real_calls_max=163,aggregate_provider_preflight_cny='35.35',
        hashes={p.relative_to(root).as_posix():audit.sha(p) for p in interface_files}))
    ledger = root / 'runtime/development-budget.sqlite'
    with closing(audit.ro(ledger)) as db:
        historical = audit.history(db,1547)
    legacy.put(root, audit.CHAT / 'FROZEN_LIVE.json', dict(baseline={'attempts':1547},historical_sha256=historical,
                                                       reader_sha256=audit.sha(chat / 'FROZEN_READER.json')))
    live_hash = audit.sha(chat / 'FROZEN_LIVE.json'); namespace = 'memgallery-v7:' + live_hash
    legacy.guard(chat / 'run.sqlite',72,18000000,72,'gpt-4o-mini')
    journal = chat / 'receipts/private-embedding.sqlite'; journal.parent.mkdir()
    with closing(sqlite3.connect(journal)) as db,db:
        db.executescript('CREATE TABLE operations(fingerprint TEXT,scope TEXT,manifest TEXT,state TEXT,attempt_id TEXT,'
            'status INTEGER,body BLOB,body_sha TEXT,redacted INTEGER,transport_completed INTEGER,run_path TEXT,run_ordinal INTEGER);'
            'CREATE TABLE claims(scope TEXT,chunk_key TEXT,fingerprint TEXT);')
    def receipt(rid,payload,response,ordinal):
        response['usage'] = dict(prompt_tokens=200,completion_tokens=30,total_tokens=230)
        manifest = dict(keys=[rid],payload_sha256=audit.digest(audit.canonical(payload).encode()),run_path=str((chat/'run.sqlite').resolve()))
        fingerprint = audit.digest(audit.canonical(dict(namespace=namespace,manifest=manifest)).encode())
        body = audit.canonical(response).encode(); body_hash = audit.digest(body); attempt = 'synthetic-v71-' + str(ordinal)
        with closing(sqlite3.connect(journal)) as db,db:
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (fingerprint,namespace,audit.canonical(manifest),'complete',attempt,200,body,body_hash,0,1,manifest['run_path'],ordinal))
            db.execute('INSERT INTO claims VALUES(?,?,?)',(namespace,rid,fingerprint))
        with closing(sqlite3.connect(ledger)) as db,db:
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (attempt,'gpt-4o-mini','uncertain','retained_upper_bound','budget_assumption_not_market_fx',
                 250000,250000,200,30,128000,payload['max_completion_tokens'],'1.5','6'))
            db.execute('INSERT INTO provider_observations VALUES(?,?,?,?,?,?,?,?)',
                (attempt,'live','success',200,'gpt-4o-mini',200,30,'0.000048'))
        return dict(request_id=rid,attempt_id=attempt,receipt_sha256=body_hash,run_ordinal=ordinal)
    answers = []
    for ordinal,meta in enumerate(packs,1):
        alias = meta['pack']['alias_by_native_id']['src']
        value = dict(summary='',evidence_quotes=[],answer='Synthetic new answer.',cited_ids=[alias],confidence='supported')
        response = dict(model='gpt-4o-mini',choices=[dict(finish_reason='stop',message={'content':audit.canonical(value)})])
        rid = audit.digest((meta['query_id']+'\0'+meta['variant']).encode())
        saved = receipt(rid,audit.native.reader_payload(meta['pack']),response,ordinal)
        saved.update({k:meta[k] for k in ('query_id','index','point','variant')})
        saved['answer'] = audit.projection(response,meta['pack']); answers.append(saved)
    legacy.put_lines(chat / 'reader-answers.jsonl',answers)
    legacy.put(root,audit.CHAT/'READER_COMPLETED.json',dict(state='complete',phase='reader',count=64,
        output_sha256=audit.sha(chat/'reader-answers.jsonl'),frozen_live_sha256=live_hash))
    requests,results = [],[]
    for i in range(8):
        pack = legacy.build_proxy_judge([dict(item_id=j,point='FR',question='Synthetic?',reference='Synthetic reference.',
                                            prediction='Synthetic new answer.') for j in range(8)],lambda s:len(s)//4+1)
        rid = 'synthetic-v71-judge-'+str(i); requests.append(dict(request_id=rid,pack=pack))
        value = dict(summary='',evidence_quotes=[],judgments=[dict(item_id=j,score=.75,reason='Synthetic.') for j in range(8)])
        response = dict(model='gpt-4o-mini',choices=[dict(finish_reason='stop',message={'content':audit.canonical(value)})])
        saved = receipt(rid,audit.proxy_judge_payload(pack),response,65+i); saved['response']=response;results.append(saved)
    legacy.put_lines(chat/'judge-requests.jsonl',requests);legacy.put_lines(chat/'judge-results.jsonl',results)
    legacy.put(root,audit.CHAT/'FROZEN_JUDGE.json',dict(judge_requests=8,reader_answers=64,grades=[0,.25,.5,.75,1],
        requests_file=(audit.CHAT/'judge-requests.jsonl').as_posix(),requests_sha256=audit.sha(chat/'judge-requests.jsonl'),
        input_hashes={code:audit.sha(root/code),'runtime/memgallery-v7/scorer-only/gold.json':'f'*64},
        code_hashes={code:audit.sha(root/code)}))
    legacy.put(root,audit.CHAT/'JUDGE_COMPLETED.json',dict(state='complete',phase='judge',count=8,
        output_sha256=audit.sha(chat/'judge-results.jsonl'),frozen_live_sha256=live_hash))
    legacy.put(root,'reports/memgallery-v71-chat-progress.json',dict(state='judge_complete',phase='judge'))


class MemGalleryV71ChatAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v71-audit-synthetic-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve();fixture(self.root)

    def test_all163_calls_old_usage_included_and_all_files_unchanged(self):
        before = {p.relative_to(self.root).as_posix():audit.sha(p) for p in self.root.rglob('*') if p.is_file()}
        result = audit.audit(self.root)
        self.assertEqual(result['total_real_provider_requests'],163)
        self.assertEqual(result['original_reader_requests_preserved'],64)
        self.assertEqual(result['original_reader_invalid_outputs_preserved'],1)
        self.assertEqual(result['new_reader_invalid_outputs_retained'],0)
        self.assertEqual(result['cumulative_provider_preflight_micro_cny'],35350000)
        self.assertEqual(result['nominal_guard_preflight_including_preflight_micro_cny'],35600000)
        self.assertEqual(Decimal(result['old_reader_usage_estimate_usd']),Decimal('.001728'))
        self.assertEqual(Decimal(result['new_chat_usage_estimate_usd']),Decimal('.003456'))
        self.assertEqual(Decimal(result['combined_usage_estimate_cny_budget_assumption']),Decimal('.064387'))
        self.assertEqual(result['original_calls_replayed'],0)
        self.assertEqual(before,{p.relative_to(self.root).as_posix():audit.sha(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertNotIn('Synthetic new answer.',audit.canonical(result))

    def test_new_incomplete_stage_stops_before_legacy_or_ledger(self):
        legacy.put(self.root,'reports/memgallery-v71-chat-progress.json',dict(state='running',phase='reader'))
        with patch.object(audit.prior,'audit_reader_only',side_effect=AssertionError('old auditor must not run')):
            with self.assertRaises(ValueError):audit.audit(self.root)

    def test_old_cost_and_new_usd_and_new_unknown_each_block(self):
        path = self.root/'runtime/development-budget.sqlite'
        for statement in (
            "UPDATE attempts SET accounted=0 WHERE id='synthetic-chat-1'",
            "UPDATE provider_observations SET estimated_usd='4' WHERE attempt_id='synthetic-v71-1'",
        ):
            before = path.read_bytes()
            with closing(sqlite3.connect(path)) as db,db:db.execute(statement)
            with self.subTest(statement=statement),self.assertRaises(ValueError):audit.audit(self.root)
            path.write_bytes(before)
        with closing(sqlite3.connect(self.root/audit.CHAT/'receipts/private-embedding.sqlite')) as db,db:
            db.execute("UPDATE operations SET state='unknown' WHERE run_ordinal=1")
        with self.assertRaises(ValueError):audit.audit(self.root)

    def test_same_question_source_order_and_alias_payload_are_checked(self):
        old = audit.lines(self.root/audit.OLD/'reader-packs.jsonl')
        new = audit.lines(self.root/audit.CHAT/'reader-packs.jsonl')
        self.assertTrue(audit.same_inputs(old,new))
        old[0]['pack']['source_memory_ids'].reverse()
        with self.assertRaises(ValueError):audit.same_inputs(old,new)
        old = audit.lines(self.root/audit.OLD/'reader-packs.jsonl')
        old[0]['pack']['question_sha256']='f'*64
        with self.assertRaises(ValueError):audit.same_inputs(old,new)
        new[0]['pack']['payload']['messages'][1]['content'][0]['text']+='changed'
        with self.assertRaises(ValueError):audit.same_inputs(old,new)


if __name__=='__main__':unittest.main()
