"""Synthetic known HTTP400 acknowledgement; never native data or live APIs."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from scripts import ack_memgallery_v71_json400 as ack
from tests import test_memgallery_v71_chat_audit as previous


def fixture(root):
    previous.fixture(root); chat=root/ack.CHAT
    mapping=chat/'scorer-only/judge-mapping.json';mapping.parent.mkdir();mapping.write_text('{}',encoding='utf-8')
    judge=ack.load(chat/'FROZEN_JUDGE.json')
    judge.update(mapping_file=mapping.relative_to(root).as_posix(),mapping_sha256=ack.sha(mapping))
    previous.legacy.put(root,ack.CHAT/'FROZEN_JUDGE.json',judge)
    for name in ('JUDGE_COMPLETED.json','judge-results.jsonl'):(chat/name).unlink()
    with closing(sqlite3.connect(chat/'run.sqlite')) as db,db:
        db.execute('ALTER TABLE calls ADD COLUMN started_at TEXT')
        db.execute('ALTER TABLE calls ADD COLUMN finished_at TEXT')
        db.execute('DELETE FROM calls WHERE ordinal>65')
        db.execute("UPDATE run SET state='active'")
        db.execute("UPDATE calls SET state='claimed',returned_model=NULL WHERE ordinal=65")
    with closing(sqlite3.connect(chat/'receipts/private-embedding.sqlite')) as db,db:
        db.execute('DELETE FROM claims WHERE fingerprint IN (SELECT fingerprint FROM operations WHERE run_ordinal>65)')
        db.execute('DELETE FROM operations WHERE run_ordinal>65')
        db.execute("UPDATE operations SET state='received',status=400,body=?,body_sha=?,transport_completed=0 WHERE run_ordinal=65",
                   (ack.ERROR_BODY,ack.ERROR_SHA256))
    with closing(sqlite3.connect(root/'runtime/development-budget.sqlite')) as db,db:
        db.execute('DELETE FROM provider_observations WHERE attempt_id IN (SELECT id FROM attempts WHERE rowid>1612)')
        db.execute('DELETE FROM attempts WHERE rowid>1612')
        db.execute("UPDATE attempts SET input_tokens=NULL,output_tokens=NULL WHERE id='synthetic-v71-65'")
        db.execute("UPDATE provider_observations SET category='client_error',http_status=400,returned_model=NULL,input_tokens=NULL,output_tokens=NULL,estimated_usd=NULL WHERE attempt_id='synthetic-v71-65'")


class MemGalleryV71Json400Tests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='v71-known400-synthetic-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();fixture(self.root)

    def test_readonly_verify_then_atomic_ack_preserves_everything_else(self):
        files=[p for p in self.root.rglob('*') if p.is_file()]
        before={p:ack.sha(p) for p in files}
        result=ack.inspect(self.root);self.assertFalse(result['acknowledged'])
        self.assertEqual(before,{p:ack.sha(p) for p in files})
        report=ack.acknowledge(self.root)
        self.assertEqual(report['real_http_requests'],1);self.assertEqual(report['model_grades_generated'],0)
        self.assertEqual(report['ledger_hold_retained_micro_cny'],250000)
        self.assertEqual(report['maximum_cumulative_real_provider_requests'],164)
        guard=self.root/ack.CHAT/'run.sqlite'
        self.assertEqual({p:h for p,h in before.items() if p!=guard},{p:ack.sha(p) for p in files if p!=guard})
        with closing(ack.ro(guard)) as db:
            self.assertEqual(tuple(db.execute('SELECT max_requests,cap,state FROM run').fetchone()),(72,18000000,'complete'))
            self.assertEqual(tuple(db.execute('SELECT count(*),sum(held) FROM calls').fetchone()),(65,16250000))
            self.assertEqual(db.execute("SELECT count(*) FROM calls WHERE state='complete'").fetchone()[0],64)
            self.assertEqual(db.execute("SELECT state FROM calls WHERE ordinal=65").fetchone()[0],'rejected_response')
        after=ack.sha(guard)
        self.assertEqual(ack.acknowledge(self.root),report);self.assertEqual(ack.sha(guard),after)

    def test_unknown_or_other400_or_usage_or_released_hold_cannot_mutate_guard(self):
        chat=self.root/ack.CHAT; guard=chat/'run.sqlite';guard_hash=ack.sha(guard)
        cases=[(chat/'receipts/private-embedding.sqlite',"UPDATE operations SET state='unknown' WHERE run_ordinal=65"),
               (chat/'receipts/private-embedding.sqlite',"UPDATE operations SET body=x'7b7d' WHERE run_ordinal=65"),
               (self.root/'runtime/development-budget.sqlite',"UPDATE provider_observations SET input_tokens=1 WHERE attempt_id='synthetic-v71-65'"),
               (self.root/'runtime/development-budget.sqlite',"UPDATE attempts SET accounted=0 WHERE id='synthetic-v71-65'"),
               (self.root/'runtime/development-budget.sqlite',"UPDATE attempts SET accounted=1 WHERE rowid=100")]
        for path,sql in cases:
            data=path.read_bytes()
            with closing(sqlite3.connect(path)) as db,db:db.execute(sql)
            with self.subTest(sql=sql),self.assertRaises(ValueError):ack.acknowledge(self.root)
            self.assertEqual(ack.sha(guard),guard_hash);self.assertFalse((self.root/ack.REPORT).exists())
            path.write_bytes(data)

    def test_crash_after_guard_commit_recovers_report_without_second_write(self):
        original=Path.open
        def crash(path,*args,**kwargs):
            if path==self.root/ack.REPORT and args and args[0]=='x':raise OSError('synthetic report write crash')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',crash),self.assertRaises(OSError):ack.acknowledge(self.root)
        guard_hash=ack.sha(self.root/ack.CHAT/'run.sqlite')
        self.assertTrue(ack.inspect(self.root)['acknowledged'])
        report=ack.acknowledge(self.root)
        self.assertEqual(report['guard_after_sha256'],guard_hash)
        self.assertEqual(ack.sha(self.root/ack.CHAT/'run.sqlite'),guard_hash)

    def test_closed_without_ack_or_tampered_report_rejected(self):
        guard=self.root/ack.CHAT/'run.sqlite';data=guard.read_bytes()
        with closing(sqlite3.connect(guard)) as db,db:
            db.execute("UPDATE run SET state='complete'")
            db.execute("UPDATE calls SET state='rejected_response' WHERE ordinal=65")
        with self.assertRaises(ValueError):ack.acknowledge(self.root)
        guard.write_bytes(data);ack.acknowledge(self.root)
        report=ack.load(self.root/ack.REPORT);report['real_http_requests']=0
        previous.legacy.put(self.root,ack.REPORT,report)
        with self.assertRaises(ValueError):ack.acknowledge(self.root)


if __name__=='__main__':unittest.main()
