"""Synthetic JSON-only correction and distinct-guard receipt contracts."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts import score_memgallery_v71_judge_json as score
from tests.test_memgallery_v7_scoring import questions, gold

base = score.base


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base.canonical(value) + '\n', encoding='utf-8')


def put_lines(root, name, rows):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(base.canonical(r) + '\n' for r in rows), encoding='utf-8')


def fake_reader():
    answers = [dict(index=i, variant=v, point='AR' if i % 2 else 'FR',
        answer=dict(schema_status='valid', answer='Synthetic original prediction'))
        for i in range(16) for v in base.VARIANTS]
    answers[0]['answer'] = dict(schema_status='invalid', answer='')
    return dict(answers=answers, selected=list(range(16)), input_hashes={},
                reader_attempts={'old-reader-' + str(i) for i in range(64)})


def fixture(root):
    reader = fake_reader()
    requests, mapping = base.judge_batches(questions(), list(range(16)), reader['answers'], gold(), lambda _: 10)
    old = score.prior.PATHS
    for name in score.CODE:
        put(root, name, {'synthetic_code_fixture': True})
    for key in ('ANSWERS', 'READER_COMPLETE', 'JOURNAL'):
        put(root, old[key], {'synthetic': key})
    reader['input_hashes'] = {old['ANSWERS']: base.sha(root, old['ANSWERS'])}
    put_lines(root, old['JUDGE_REQUESTS'], requests)
    put(root, old['JUDGE_MAPPING'], mapping)
    frozen = dict(judge_requests=8, reader_answers=64, requests_file=old['JUDGE_REQUESTS'],
        requests_sha256=base.sha(root, old['JUDGE_REQUESTS']), mapping_file=old['JUDGE_MAPPING'],
        mapping_sha256=base.sha(root, old['JUDGE_MAPPING']), grades=[0,.25,.5,.75,1],
        code_hashes={name: base.sha(root, name) for name in score.CODE},
        input_hashes={**reader['input_hashes'], base.GOLD: 'f'*64})
    put(root, old['JUDGE_FROZEN'], frozen)
    with closing(sqlite3.connect(root / old['GUARD'])) as db, db:
        db.executescript("CREATE TABLE run(max_requests INTEGER,cap INTEGER,state TEXT); INSERT INTO run VALUES(72,18000000,'complete'); CREATE TABLE calls(ordinal INTEGER,model TEXT,held INTEGER,state TEXT,returned_model TEXT); CREATE TABLE known_json400_rejections(ordinal INTEGER,http_status INTEGER,category TEXT,receipt_sha256 TEXT);")
        db.executemany('INSERT INTO calls VALUES(?,?,?,?,?)',
            [(n,'gpt-4o-mini',250000,'complete','gpt-4o-mini') for n in range(1,65)] +
            [(65,'gpt-4o-mini',250000,'rejected_response',None)])
        db.execute('INSERT INTO known_json400_rejections VALUES(65,400,?,?)',
            ('client_error','c259bb7f479b9faee0845ca7c0c7d8fca618e090ab8f4270edd4de252b2488f3'))
    put(root, score.REJECTION, dict(state='passed', artifact_sha256={old[name]:base.sha(root,old[name])
        for name in ('GUARD','JOURNAL','ANSWERS','JUDGE_FROZEN','JUDGE_REQUESTS')}))
    return reader, requests, mapping


def completed_judge(root):
    put(root, score.LIVE, dict(judge_sha256=base.sha(root,score.FROZEN),
        code_hashes={name:base.sha(root,name) for name in score.CODE}))
    namespace = 'memgallery-v71-judge-json:' + base.sha(root,score.LIVE)
    journal = root / score.JOURNAL; journal.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(journal)) as db, db:
        db.executescript('CREATE TABLE operations(fingerprint TEXT,scope TEXT,manifest TEXT,state TEXT,attempt_id TEXT,status INTEGER,body BLOB,body_sha TEXT,redacted INTEGER,transport_completed INTEGER,run_path TEXT,run_ordinal INTEGER); CREATE TABLE claims(scope TEXT,chunk_key TEXT,fingerprint TEXT);')
    with closing(sqlite3.connect(root/score.GUARD)) as db, db:
        db.executescript("CREATE TABLE run(max_requests INTEGER,cap INTEGER,state TEXT); INSERT INTO run VALUES(8,2000000,'complete'); CREATE TABLE calls(ordinal INTEGER,model TEXT,held INTEGER,state TEXT,returned_model TEXT);")
        db.executemany('INSERT INTO calls VALUES(?,?,?,?,?)',[(i,'gpt-4o-mini',250000,'complete','gpt-4o-mini') for i in range(1,9)])
    rows=[]
    for ordinal, request in enumerate(base.lines((root/score.REQUESTS).read_bytes()),1):
        value=dict(summary='',evidence_quotes=[],judgments=[dict(item_id=i,score=.75,reason='Synthetic') for i in range(8)])
        response=dict(model='gpt-4o-mini',choices=[dict(finish_reason='stop',message=dict(content=base.canonical(value)))])
        body=base.canonical(response).encode(); attempt='new-judge-'+str(ordinal)
        manifest=dict(keys=[request['request_id']],payload_sha256=base.digest(base.canonical(request['payload']).encode()),
            run_path=str(base.path_at(root,score.GUARD)))
        fp=base.digest(base.canonical(dict(namespace=namespace,manifest=manifest)).encode())
        with closing(sqlite3.connect(journal)) as db, db:
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (fp,namespace,base.canonical(manifest),'complete',attempt,200,body,base.digest(body),0,1,manifest['run_path'],ordinal))
            db.execute('INSERT INTO claims VALUES(?,?,?)',(namespace,request['request_id'],fp))
        rows.append(dict(request_id=request['request_id'],response=response,attempt_id=attempt,
            receipt_sha256=base.digest(body),run_ordinal=ordinal))
    put_lines(root,score.RESULTS,rows)
    put(root,score.COMPLETE,dict(state='complete',phase='judge',count=8,
        frozen_live_sha256=base.sha(root,score.LIVE),output_sha256=base.sha(root,score.RESULTS)))
    put(root,score.PROGRESS,dict(state='judge_complete',phase='judge'))


class JudgeJSONCorrectionTests(unittest.TestCase):
    def test_payload_diff_is_exact_suffix_and_user_identical(self):
        reader=fake_reader(); requests,mapping=base.judge_batches(questions(),list(range(16)),reader['answers'],gold(),lambda _:10)
        new,remap,stats=score.corrected_batches(requests,mapping,reader,lambda _:10)
        again=score.corrected_batches(requests,mapping,reader,lambda _:10)
        self.assertEqual((new,remap,stats),again)
        for old,row,mapped,original_map in zip(requests,new,remap,mapping):
            payload=base.proxy_judge_payload(old['pack'])
            self.assertEqual(payload['messages'][1],row['payload']['messages'][1])
            self.assertEqual(mapped['items'],original_map['items'])
            self.assertNotEqual(row['request_id'],old['request_id'])
            score.verify_correction(payload,row['payload'])
            modified=copy.deepcopy(row['payload']);modified['temperature']=1
            with self.assertRaises(ValueError):score.verify_correction(payload,modified)

    def test_budget_overflow_blocks_without_truncating(self):
        reader=fake_reader();requests,_=base.judge_batches(questions(),list(range(16)),reader['answers'],gold(),lambda _:10)
        with self.assertRaises(ValueError):score.corrected_payload(requests[0]['pack'],lambda _:8192)

    def test_blind_mapping_or_prediction_change_blocks(self):
        reader=fake_reader();requests,mapping=base.judge_batches(questions(),list(range(16)),reader['answers'],gold(),lambda _:10)
        reader['answers'][1]['answer']['answer']='Changed after freeze'
        with self.assertRaises(ValueError):score.corrected_batches(requests,mapping,reader,lambda _:10)
        reader=fake_reader();mapping[1]['items'][0]=copy.deepcopy(mapping[0]['items'][0])
        with self.assertRaises(ValueError):score.corrected_batches(requests,mapping,reader,lambda _:10)

    def test_old65_unknown_or_open_guard_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            self.assertEqual(set(score.old_guard_calls(root)),set(range(1,65)))
            with closing(sqlite3.connect(root/score.prior.PATHS['GUARD'])) as db,db:
                db.execute("UPDATE calls SET state='claimed' WHERE ordinal=65")
            with self.assertRaises(ValueError):score.old_guard_calls(root)

    def test_prepare_never_reads_native_gold_and_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reader,_,_=fixture(root)
            old={p:base.sha(root,p) for p in score.prior.PATHS.values() if (root/p).is_file()}
            with patch.object(score,'verify_reader',return_value=reader),patch.object(base,'read_gold_after_completion',side_effect=AssertionError('gold forbidden')):
                report=score.prepare(root,count_tokens=lambda _:10)
                self.assertEqual(report['invalid_reader_answers'],1)
                with self.assertRaises(FileExistsError):score.prepare(root,count_tokens=lambda _:10)
            self.assertFalse((root/base.GOLD).exists())
            frozen=base.load(root,score.FROZEN)
            self.assertNotIn(base.GOLD,frozen['input_hashes'])
            self.assertEqual(old,{p:base.sha(root,p) for p in old})

    def test_complete_new_receipts_score_invalid_zero_with_same_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reader,_,_=fixture(root)
            with patch.object(score,'verify_reader',return_value=reader),patch.object(base,'local_tokenizer',return_value=lambda _:10):
                score.prepare(root,count_tokens=lambda _:10);completed_judge(root)
                result=score.finalize(root,bootstrap_samples=20)
            self.assertEqual(result['invalid_reader_answers'],1)
            self.assertEqual(result['groups']['all']['variants']['dense']['proxy_mean'],.75*15/16)
            self.assertEqual(result['groups']['all']['variants']['hybrid']['proxy_mean'],.75)
            self.assertFalse(result['original_native_gold_reopened'])

    def test_wrong_new_namespace_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reader,_,_=fixture(root)
            with patch.object(score,'verify_reader',return_value=reader),patch.object(base,'local_tokenizer',return_value=lambda _:10):
                score.prepare(root,count_tokens=lambda _:10);completed_judge(root)
                with closing(sqlite3.connect(root/score.JOURNAL)) as db,db:
                    db.execute("UPDATE operations SET scope='old-namespace' WHERE run_ordinal=1")
                with self.assertRaises(ValueError):score.finalize(root,bootstrap_samples=20)
            self.assertFalse((root/score.SCORES).exists())

    def test_bindings_restore_original_globals_after_failure(self):
        before=base.GUARD
        with self.assertRaises(RuntimeError):
            with score.bindings(GUARD='synthetic-new'):
                self.assertEqual(base.GUARD,'synthetic-new');raise RuntimeError()
        self.assertEqual(base.GUARD,before)


if __name__=='__main__':unittest.main()
