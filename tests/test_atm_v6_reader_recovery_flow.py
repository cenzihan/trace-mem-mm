"""Synthetic receipt provenance/recovery/grading; no providers or native data."""
import copy
import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch

from agent_memory.atm_reader import reader_payload
from scripts import recover_atm_v6_reader as recovery
from scripts import score_atm_v6_reader as old
from tests import test_atm_v6_reader_scoring as fixtures


def response(value):
    return dict(model='gpt-4o-mini-2024-07-18', choices=[dict(finish_reason='stop',
        message=dict(content=json.dumps(value, ensure_ascii=False)))])


class RecoveryFlowTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.ATMV6ReaderScoringTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.root = self.fx.root
        for name in recovery.CODE:
            if not (self.root/name).exists():
                self.fx.write(name, b'# Synthetic immutable recovery fixture\n')
        packs = {(row['query_id'], row['variant']): row['pack']
                 for row in old.lines((self.root/old.PACKS).read_bytes())}
        outcomes = old.lines((self.root/old.ANSWERS).read_bytes())
        records = []
        for index, row in enumerate(outcomes):
            answer = row['answer']
            value = dict(summary='Synthetic useful summary', evidence_quotes=['Synthetic quoted text'],
                **{name: answer[name] for name in ('answer', 'answer_ids', 'cited_ids', 'confidence')})
            if index == 0:
                value['confidence'] = 'insufficient'  # A substantive error must remain invalid.
            raw_response = response(value)
            row['receipt_sha256'] = recovery.digest(recovery.canonical(raw_response).encode())
            row['answer'] = dict(answer='', answer_ids=[], cited_ids=[], schema_status='invalid',
                                citations_valid=False, answer_ids_valid=False, model=raw_response['model'])
            records.append((row, reader_payload(packs[(row['query_id'], row['variant'])]), raw_response))
        self.fx.lines(old.ANSWERS, outcomes)
        self.fx.rebind_answers()
        self.make_journal(recovery.OLD_JOURNAL, recovery.OLD_GUARD,
            'atm-v6:'+self.fx.hash(old.LIVE_FROZEN), records, 80, 20000000)

    def make_journal(self, journal, guard, namespace, records, maximum, cap):
        path = self.root/journal
        path.parent.mkdir(parents=True, exist_ok=True)
        guard_path = self.root/guard
        guard_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript('''CREATE TABLE operations(fingerprint TEXT PRIMARY KEY,scope TEXT,manifest TEXT,
                state TEXT,attempt_id TEXT,status INTEGER,body BLOB,body_sha TEXT,redacted INTEGER,
                transport_completed INTEGER,run_path TEXT,run_ordinal INTEGER);
                CREATE TABLE claims(scope TEXT,chunk_key TEXT,fingerprint TEXT);''')
            for row, payload, body in records:
                manifest = dict(keys=[row['request_id']],payload_sha256=recovery.digest(recovery.canonical(payload).encode()),
                                run_path=str(guard_path.resolve()))
                fingerprint = recovery.digest(recovery.canonical(dict(namespace=namespace,manifest=manifest)).encode())
                raw = recovery.canonical(body).encode()
                db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (fingerprint,namespace,recovery.canonical(manifest),'complete',row['attempt_id'],200,raw,
                     recovery.digest(raw),0,1,str(guard_path.resolve()),row['run_ordinal']))
                db.execute('INSERT INTO claims VALUES(?,?,?)',(namespace,row['request_id'],fingerprint))
        with closing(sqlite3.connect(guard_path)) as db, db:
            db.executescript('''CREATE TABLE run(max_requests INTEGER,cap INTEGER,state TEXT);
                CREATE TABLE calls(ordinal INTEGER,model TEXT,state TEXT,returned_model TEXT);''')
            db.execute('INSERT INTO run VALUES(?,?,?)',(maximum,cap,'complete'))
            db.executemany('INSERT INTO calls VALUES(?,?,?,?)',[(r['run_ordinal'],'gpt-4o-mini','complete',b['model'])
                                                               for r,_,b in records])

    def make_judge_results(self):
        requests = old.lines((self.root/recovery.JUDGE_REQUESTS).read_bytes())
        self.fx.dump(recovery.LIVE_FROZEN,dict(code_hashes={name:self.fx.hash(name) for name in recovery.CODE}))
        namespace = 'atm-v6-recovery:'+self.fx.hash(recovery.LIVE_FROZEN)
        records, rows = [], []
        for ordinal, request in enumerate(requests,1):
            body = response(dict(summary='Graded synthetic answers', evidence_quotes=['Auxiliary synthetic quote'],
                judgments=[dict(item_id=i,label='correct',score=1,reason='Synthetic grade') for i in range(8)]))
            row = dict(request_id=request['request_id'],attempt_id='new-recovery-'+str(ordinal),run_ordinal=ordinal,
                       receipt_sha256=recovery.digest(recovery.canonical(body).encode()),response=body)
            rows.append(row);records.append((row,request['payload'],body))
        self.make_journal(recovery.JUDGE_JOURNAL,recovery.JUDGE_GUARD,namespace,records,5,1250000)
        self.fx.lines(recovery.JUDGE_RESULTS,rows)
        self.fx.dump(recovery.JUDGE_RUN,dict(state='judge_complete',completed_requests=len(requests),
            results_file=recovery.JUDGE_RESULTS,results_sha256=self.fx.hash(recovery.JUDGE_RESULTS),
            judge_frozen_sha256=self.fx.hash(recovery.JUDGE_FROZEN),
            live_frozen_sha256=self.fx.hash(recovery.LIVE_FROZEN),receipt_namespace=namespace))

    def test_receipt_recovery_no_gold_originals_preserved_and_semantic_failure_retained(self):
        names=(old.ANSWERS,recovery.OLD_JOURNAL,recovery.OLD_GUARD,old.READER_RUN)
        before={name:self.fx.hash(name) for name in names}
        with self.fx.no_gold():
            report=recovery.recover(self.root)
            verified=recovery.verify_recovery(self.root)
        self.assertEqual((report['valid_schema'],report['invalid_schema']),(63,1))
        self.assertEqual(len(verified['answers']),64)
        self.assertEqual(before,{name:self.fx.hash(name) for name in names})
        with self.assertRaises(FileExistsError):recovery.recover(self.root)

    def test_changed_receipt_scope_payload_binding_or_guard_fails_before_gold(self):
        with closing(sqlite3.connect(self.root/recovery.OLD_JOURNAL)) as db,db:
            db.execute("UPDATE operations SET scope='wrong' WHERE run_ordinal=1")
        with self.fx.no_gold(),self.assertRaises(ValueError):recovery.recover(self.root)
        self.assertFalse((self.root/recovery.ANSWERS).exists())

    def test_receipt_body_tamper_rejected_without_output(self):
        with closing(sqlite3.connect(self.root/recovery.OLD_JOURNAL)) as db,db:
            db.execute("UPDATE operations SET body=? WHERE run_ordinal=1",(b'{}',))
        with self.fx.no_gold(),self.assertRaises(ValueError):recovery.recover(self.root)
        self.assertFalse((self.root/recovery.FROZEN).exists())

    def test_recovery_prepare_blinds_two_questions_and_keeps_invalid_zero_then_finalize(self):
        recovery.recover(self.root)
        prepared=recovery.prepare(self.root,expected_open_questions=4)
        self.assertEqual(prepared['judge_requests'],2)
        requests=old.lines((self.root/recovery.JUDGE_REQUESTS).read_bytes())
        for row in requests:
            payload=row['payload'];body=json.loads(payload['messages'][1]['content'])
            self.assertEqual(len(body['items']),8)
            self.assertEqual({item['item_id'] for item in body['items']},set(range(8)))
            self.assertEqual(len({item['question'] for item in body['items']}),2)
            self.assertTrue(all(set(item)=={'item_id','question','reference','prediction'} for item in body['items']))
            self.assertLessEqual(sum(len(m['content'].encode()) for m in payload['messages']),32768)
        self.make_judge_results()
        final=recovery.finalize(self.root,bootstrap_samples=20)
        self.assertEqual(final['state'],'complete')
        self.assertFalse(final['old_judge_results_reused'])
        self.assertEqual(sum(v['invalid_answers'] for v in final['groups']['all']['variants'].values()),1)
        self.assertFalse((self.root/old.SCORES).exists())

    def test_incomplete_original_reader_never_opens_gold_for_prepare(self):
        recovery.recover(self.root)
        rows=old.lines((self.root/old.ANSWERS).read_bytes())[:-1]
        self.fx.lines(old.ANSWERS,rows);self.fx.rebind_answers()
        with self.fx.no_gold(),self.assertRaises(ValueError):recovery.prepare(self.root,expected_open_questions=4)
        self.assertFalse((self.root/recovery.JUDGE_REQUESTS).exists())

    def test_decode_eight_item_labels_scores_unique_ids_and_auxiliary_types(self):
        value=dict(summary='arbitrary auxiliary summary',evidence_quotes=['allowed auxiliary text'],
            judgments=[dict(item_id=i,label='partial',score=.5,reason='Synthetic reason') for i in range(8)])
        self.assertEqual(recovery.decode_judge(response(value),range(8)),{i:.5 for i in range(8)})
        for change in ('duplicate','mismatched_score','wrong_auxiliary_type','missing'):
            altered=copy.deepcopy(value)
            if change=='duplicate':altered['judgments'][7]['item_id']=0
            elif change=='mismatched_score':altered['judgments'][0]['score']=1
            elif change=='wrong_auxiliary_type':altered['summary']=[]
            else:altered['judgments'].pop()
            with self.assertRaises(ValueError):recovery.decode_judge(response(altered),range(8))


if __name__=='__main__':unittest.main()
