"""Durability, true child exit and credential-echo exclusion, synthetic data only."""
import base64
import contextlib
import copy
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.provider_http import BudgetedJSONTransport,HTTPReply,ProviderFailure,OPENAI_ENDPOINT,RetryPolicy
from agent_memory.visual_batch8 import GPT4oMiniVisualBatch8 as Batch,VisualInput,VisualBatchFailure
from agent_memory.visual_recovery import PrivateVisualJournal,DurableVisualExecutor,VisualRecoveryBlocked

ROOT=Path(__file__).resolve().parents[1]
SYNTHETIC='SYNTHETIC_KEY_NEVER_LOG_THIS'


def prepared(namespace='unit-user',large_original=False):
    raw=(ROOT/'fixtures/red.png').read_bytes()
    uri='data:image/png;base64,'+base64.b64encode(raw).decode()
    original=raw+b'\0'*(10*1024*1024) if large_original else raw
    return Batch.prepare([VisualInput('unit-native','unit-occurrence',None,uri,'unit-small-request',original)],namespace)


def good_response():
    value=dict(summary='unit',evidence_quotes=[],images=[dict(index=0,description='unit red square',uncertainties=[])])
    return dict(model='gpt-4o-mini-2024-07-18',usage=dict(prompt_tokens=2840,completion_tokens=30,total_tokens=2870),
                choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])


class FakeHTTP:
    is_fake_http=True
    def __init__(self,result=None):self.result=result or HTTPReply(200,json.dumps(good_response()).encode());self.calls=0
    def exchange(self,*args):
        self.calls+=1
        if isinstance(self.result,BaseException):raise self.result
        return self.result


def factory(ledger):
    def build(http,observer):
        rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat(),conversion_basis='budget_assumption_not_market_fx')
        return BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,rate=rate,
            mode='offline_fake',http=http,fixed_reserve_cny='.25',output_token_cap=2048,attempt_observer=observer,
            credential_reader=lambda:(_ for _ in ()).throw(AssertionError('fake must not read credentials')))
    return build


class VisualRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.ledger=BudgetLedger(self.root/'budget.sqlite');self.journal=PrivateVisualJournal(self.root/'private',offline_test=True)
        self.batch=prepared()
    def tearDown(self):self.tmp.cleanup()
    def run_once(self,http,hook=None):return DurableVisualExecutor(self.journal,http,factory(self.ledger),after_received=hook).execute(self.batch)
    def held(self):
        with self.ledger.connect() as db:return db.execute('SELECT COUNT(*),SUM(accounted) FROM attempts').fetchone()

    def test_success_cache_reopen_never_constructs_transport(self):
        http=FakeHTTP();result=self.run_once(http);self.assertEqual(http.calls,1)
        reopened=PrivateVisualJournal(self.root/'private',offline_test=True)
        def forbidden(*args):self.fail('cache must not reserve or construct transport')
        self.assertEqual(DurableVisualExecutor(reopened,http,forbidden).execute(self.batch),result)
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_http200_local_shape_rejection_saved_and_reparsed_without_request(self):
        value=good_response();value['choices'][0]['message']['content']=json.dumps(dict(summary='unit',evidence_quotes=[],images=[]))
        http=FakeHTTP(HTTPReply(200,json.dumps(value).encode()))
        self.assertRaises(VisualBatchFailure,self.run_once,http)
        row=self.journal.row(self.batch.request_fingerprint)
        self.assertEqual(row['state'],'invalid_response');self.assertEqual(row['category'],'item_count');self.assertEqual(row['status'],200)
        self.assertIsNotNone(row['body']);self.assertIsNotNone(row['attempt_id'])
        self.assertRaises(VisualBatchFailure,self.journal.recover,self.batch.request_fingerprint)
        self.assertRaises(VisualBatchFailure,self.run_once,http)
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_provider_generic_parser_rejection_also_saved_before_parsing(self):
        http=FakeHTTP(HTTPReply(200,b'{"model":"gpt-4o-mini","malformed":"unit"}'))
        self.assertRaises(ProviderFailure,self.run_once,http)
        row=self.journal.row(self.batch.request_fingerprint);self.assertEqual(row['state'],'received')
        self.assertIsNotNone(row['body']);self.assertRaises(VisualRecoveryBlocked,self.journal.recover,self.batch.request_fingerprint)
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_local_parser_bug_can_be_fixed_and_recovered_without_paid_replay(self):
        http=FakeHTTP()
        with patch.object(Batch,'validate',side_effect=VisualBatchFailure('synthetic_parser_bug')):
            self.assertRaises(VisualBatchFailure,self.run_once,http)
        self.assertEqual(self.journal.row(self.batch.request_fingerprint)['state'],'invalid_response')
        result=self.journal.recover(self.batch.request_fingerprint)
        self.assertEqual(result[0]['description'],'unit red square')
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_timeout_retains_hold_and_does_not_replay_on_restart(self):
        http=FakeHTTP(TimeoutError());self.assertRaises(ProviderFailure,self.run_once,http)
        self.assertEqual(self.journal.row(self.batch.request_fingerprint)['state'],'unknown')
        reopened=PrivateVisualJournal(self.root/'private',offline_test=True)
        self.assertRaises(VisualRecoveryBlocked,DurableVisualExecutor(reopened,http,factory(self.ledger)).execute,self.batch)
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_child_process_death_after_saved_receipt_recovers_offline(self):
        # os._exit bypasses exception handlers/finally; SQLite FULL receipt already committed.
        child_root=self.root/'child';child_root.mkdir()
        code='''import os,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_visual_recovery import *
r=Path(sys.argv[1]);l=BudgetLedger(r/'budget.sqlite');j=PrivateVisualJournal(r/'private',offline_test=True)
DurableVisualExecutor(j,FakeHTTP(),factory(l),after_received=lambda:os._exit(44)).execute(prepared())
'''
        result=subprocess.run([sys.executable,'-c',code,str(child_root)],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
        self.assertEqual(result.returncode,44)
        j=PrivateVisualJournal(child_root/'private',offline_test=True);self.assertEqual(j.row(self.batch.request_fingerprint)['state'],'received')
        result=j.recover(self.batch.request_fingerprint);self.assertEqual(result[0]['description'],'unit red square')
        ledger=BudgetLedger(child_root/'budget.sqlite')
        with ledger.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*),SUM(accounted) FROM attempts').fetchone(),(1,250000))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM provider_observations').fetchone()[0],0)

    def test_receipt_write_failure_blocks_after_response_without_duplicate(self):
        http=FakeHTTP();original=self.journal.received
        def fail(*args):raise OSError('synthetic receipt failure')
        self.journal.received=fail
        self.assertRaises(ProviderFailure,self.run_once,http)
        self.journal.received=original
        self.assertEqual(self.journal.row(self.batch.request_fingerprint)['state'],'unknown')
        self.assertRaises(VisualRecoveryBlocked,self.run_once,http)
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_large_original_raw_bytes_separate_from_accepted_request(self):
        batch=prepared(large_original=True);self.journal.install(batch)
        manifest=batch.manifest[0]
        original=self.journal.get_blob(manifest['original_sha256']);request=self.journal.get_blob(manifest['request_sha256'])
        self.assertGreater(len(original),10*1024*1024);self.assertLess(len(request),1024)
        self.assertNotEqual(manifest['original_sha256'],manifest['request_sha256'])
        restored=self.journal.prepared(batch.request_fingerprint)
        self.assertEqual(restored.original_sources[0].original_bytes,original)
        self.assertEqual(restored.payload,batch.payload)

    def test_header_and_key_echo_not_written_or_logged(self):
        value=good_response();value['headers']={'Authorization':'Bearer '+SYNTHETIC}
        value['echo']='Bearer '+SYNTHETIC;value['escaped']=''.join('\\u%04x'%ord(c) for c in SYNTHETIC)
        # Escaped JSON value also decoded before sanitization.
        raw=json.dumps(value).encode()
        output=io.StringIO()
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):self.run_once(FakeHTTP(HTTPReply(200,raw)))
        row=self.journal.row(self.batch.request_fingerprint)
        self.assertTrue(row['redacted']);self.assertNotIn(SYNTHETIC.encode(),row['body'])
        self.assertNotIn(value['escaped'],json.loads(row['body'])['escaped'])
        for path in (self.root/'private').rglob('*'):
            if path.is_file():self.assertNotIn(SYNTHETIC.encode(),path.read_bytes())
        self.assertNotIn(SYNTHETIC,output.getvalue())

    def test_credential_echo_in_json_keys_is_redacted_before_storage(self):
        value=good_response();value[SYNTHETIC]='synthetic metadata'
        value['extra']={'Bearer '+SYNTHETIC:'synthetic nested metadata'}
        self.run_once(FakeHTTP(HTTPReply(200,json.dumps(value).encode())))
        row=self.journal.row(self.batch.request_fingerprint)
        self.assertTrue(row['redacted'])
        for path in (self.root/'private').rglob('*'):
            if path.is_file():self.assertNotIn(SYNTHETIC.encode(),path.read_bytes())

    def test_malformed_json_unicode_credential_echo_is_redacted(self):
        encoded=''.join('\\u%04x'%ord(c) for c in SYNTHETIC)
        raw=('{"echo":"'+encoded+'",').encode()
        http=FakeHTTP(HTTPReply(200,raw))
        self.assertRaises(ProviderFailure,self.run_once,http)
        row=self.journal.row(self.batch.request_fingerprint)
        self.assertTrue(row['redacted'])
        self.assertRaises(VisualRecoveryBlocked,self.journal.recover,self.batch.request_fingerprint)
        for path in (self.root/'private').rglob('*'):
            if path.is_file():
                saved=path.read_bytes()
                self.assertNotIn(SYNTHETIC.encode(),saved);self.assertNotIn(encoded.encode(),saved)
        self.assertRaises(VisualRecoveryBlocked,self.run_once,http)
        self.assertEqual(http.calls,1)

    def test_ready_manifest_or_namespace_corruption_cannot_construct_transport(self):
        for field in ('namespace','manifest'):
            with self.subTest(field=field):
                batch=prepared('unit-corruption-'+field);self.journal.install(batch)
                changed='another-namespace' if field=='namespace' else '[]'
                with self.journal.connect() as db:
                    db.execute('UPDATE operations SET '+field+'=? WHERE fingerprint=?',
                               (changed,batch.request_fingerprint))
                calls=[];http=FakeHTTP()
                def forbidden_factory(*args):
                    calls.append(True);raise AssertionError('corrupted ready operation reached factory')
                runner=DurableVisualExecutor(self.journal,http,forbidden_factory)
                self.assertRaises(VisualRecoveryBlocked,runner.execute,batch)
                self.assertEqual(calls,[]);self.assertEqual(http.calls,0)
                self.assertEqual(self.held(),(0,None))

    def test_nonfinite_and_duplicate_json_never_recover_as_success(self):
        valid=json.dumps(good_response())
        cases={'nan':'{"extra":NaN,'+valid[1:],
               'infinity':'{"extra":Infinity,'+valid[1:],
               'duplicate_outer':'{"model":"gpt-4o-mini",'+valid[1:]}
        value=good_response()
        value['choices'][0]['message']['content']='{"summary":"first",'+value['choices'][0]['message']['content'][1:]
        cases['duplicate_inner']=json.dumps(value)
        for name,raw in cases.items():
            with self.subTest(name=name):
                batch=prepared('unit-json-'+name);fingerprint=self.journal.install(batch)
                self.journal.start(fingerprint);self.journal.attach_attempt(fingerprint,'synthetic-'+name)
                self.journal.received(fingerprint,HTTPReply(200,raw.encode()),{})
                self.assertRaises(VisualRecoveryBlocked,self.journal.recover,fingerprint)
                self.assertEqual(self.journal.row(fingerprint)['state'],'invalid_response')
                self.assertIsNone(self.journal.row(fingerprint)['result'])

    def test_namespace_order_and_corruption_fail_closed(self):
        self.run_once(FakeHTTP());self.assertNotEqual(prepared('other-user').request_fingerprint,self.batch.request_fingerprint)
        with self.journal.connect() as db:db.execute('UPDATE operations SET body=?',(b'corrupt',))
        self.assertRaises(VisualRecoveryBlocked,self.journal.recover,self.batch.request_fingerprint)
        manifest=self.batch.manifest[0];self.journal._blob_path(manifest['request_sha256']).write_bytes(b'corrupt')
        self.assertRaises(VisualRecoveryBlocked,self.journal.prepared,self.batch.request_fingerprint)

    def test_oversized_usage_saved_but_never_promoted_to_success_cache(self):
        value=good_response();value['usage']=dict(prompt_tokens=128001,completion_tokens=0,total_tokens=128001)
        http=FakeHTTP(HTTPReply(200,json.dumps(value).encode()))
        self.assertRaises(ProviderFailure,self.run_once,http)
        self.assertRaises(VisualRecoveryBlocked,self.journal.recover,self.batch.request_fingerprint)
        self.assertNotEqual(self.journal.row(self.batch.request_fingerprint)['state'],'complete')
        self.assertEqual(http.calls,1);self.assertEqual(self.held(),(1,250000))

    def test_atomic_claim_allows_only_one_dispatch(self):
        self.journal.install(self.batch)
        def claim(_):
            try:self.journal.start(self.batch.request_fingerprint);return True
            except VisualRecoveryBlocked:return False
        with ThreadPoolExecutor(max_workers=4) as pool:self.assertEqual(sum(pool.map(claim,range(4))),1)
        self.assertEqual(self.journal.row(self.batch.request_fingerprint)['state'],'inflight')

    def test_factory_without_budget_binding_cannot_send(self):
        http=FakeHTTP()
        def bad_factory(wrapped,observer):return lambda endpoint,payload:wrapped.exchange(endpoint,b'{}',{},1,1024)
        runner=DurableVisualExecutor(self.journal,http,bad_factory)
        self.assertRaises(VisualRecoveryBlocked,runner.execute,self.batch)
        self.assertEqual(http.calls,0);self.assertEqual(self.held(),(0,None))

    def test_factory_cannot_enable_automatic_retry(self):
        http=FakeHTTP()
        def bad_factory(wrapped,observer):
            transport=factory(self.ledger)(wrapped,observer)
            transport.retry=RetryPolicy(max_attempts=2,retry_429=True)
            return transport
        self.assertRaises(VisualRecoveryBlocked,DurableVisualExecutor(self.journal,http,bad_factory).execute,self.batch)
        self.assertEqual(http.calls,0);self.assertEqual(self.held(),(0,None))

    def test_private_files_excluded_from_source_export(self):
        self.journal.install(self.batch)
        self.assertRaises(ValueError,PrivateVisualJournal,ROOT/'docs'/'private-journal')
        self.assertIn('runtime/',(ROOT/'.gitignore').read_text())
        # Curated checkout excludes private artifacts; never import a historical publisher.
        ignored=(ROOT/'.gitignore').read_text(encoding='utf-8').splitlines()
        self.assertIn('data/',ignored)
        self.assertIn('*.sqlite*',ignored)
        if (ROOT/'.git').exists():
            import subprocess
            tracked=subprocess.check_output(['git','ls-files','--','runtime','data'],cwd=ROOT,text=True)
            self.assertEqual(tracked.strip(),'')


if __name__=='__main__':unittest.main()
