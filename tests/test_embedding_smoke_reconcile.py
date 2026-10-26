"""Synthetic original holds and raw receipts only; never live ledger or HTTP."""
import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_memory.budget import BudgetLedger, VerifiedRate
from scripts import prepare_embedding_smoke_grant as prep
from scripts import reconcile_embedding_smoke_grant as recon


class EmbeddingSmokeReconcileTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='embedding-reconcile-synthetic-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name).resolve()
        self.ledger=BudgetLedger(self.root/'budget.sqlite')
        now=datetime(2020,1,2,12,tzinfo=timezone.utc)  # Deliberately expired dispatch grant.
        old=self.ledger.reserve(VerifiedRate('gpt-4o-mini','1','1','synthetic','synthetic'),10,fixed_reserve_cny='.01')
        self.ledger.mark_dispatched(old)
        self.ledger.observe_provider(old,mode='offline_fake',category='success',http_status=200,
            returned_model='gpt-4o-mini',input_tokens=3,output_tokens=0)
        self.old_id=old
        payloads=[dict(model=prep.MODEL,input=values,dimensions=256,encoding_format='float')
            for values in (['Synthetic source1','Synthetic source2'],['Synthetic query1'],['Synthetic query2'])]
        self.directory=self.root/'grant'
        prep.prepare(Path(self.ledger.path),self.directory,(now-timedelta(minutes=5)).isoformat(),
            reserve_live=True,now=now,payloads=payloads)
        self.grant_path=self.directory/'grant.json';self.grant=json.loads(self.grant_path.read_text())
        self.receipts=self.root/'private-audit.json';self.output=self.root/'reconciled'
        rows=[]
        for slot in self.grant['slots']:
            tokens=9+slot['ordinal']
            body=prep._json(dict(model=prep.MODEL,usage=dict(total_tokens=tokens,prompt_tokens=tokens),
                data=[dict(index=i,embedding=[float(i+1)]*256) for i in range(len(slot['payload']['input']))])).encode()
            rows.append(dict(ordinal=slot['ordinal'],parent_attempt_id=slot['parent_attempt_id'],status=200,
                body_base64=base64.b64encode(body).decode(),body_sha256=prep.sha(body),redacted=False,
                dispatch_started=True,category='success',returned_model=prep.MODEL,input_tokens=tokens,output_tokens=0))
        self.value=dict(version=1,grant_id=self.grant['grant_id'],grant_sha256=prep.sha(self.grant_path.read_bytes()),
            records=rows,memory=dict(records=2,vector_count=2,dimensions=256,source_vectors_match_receipt=True))
        self.write_receipts()
        self.before=prep.read_ledger(Path(self.ledger.path))
        self.reserve=self.enterContext(patch.object(BudgetLedger,'reserve',side_effect=AssertionError('no new reservations')))
        self.http=self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',side_effect=AssertionError('no HTTP')))

    def write_receipts(self):
        self.receipts.write_text(prep._json(self.value),encoding='utf-8')

    def run_reconcile(self,apply=True):
        return recon.reconcile(Path(self.ledger.path),self.grant_path,self.receipts,self.output,apply=apply)

    def snapshot(self):
        return prep.read_ledger(Path(self.ledger.path))

    def test_all_three_original_rows_settle_exact_usage_idempotently_after_grant_expiry(self):
        result=self.run_reconcile();after=self.snapshot()
        self.assertEqual(result['usage_estimate_micro_cny'],17)
        self.assertEqual(result['reconciled_attempts'],3)
        self.assertEqual(len(after['attempts']),len(self.before['attempts']))
        self.assertEqual(after['attempts'][0],self.before['attempts'][0])
        self.assertEqual(after['observations'][0],self.before['observations'][0])
        self.assertEqual(after['accounted'],self.before['accounted']-150000+17)
        report=(self.output/'reconciliation.json').read_bytes()
        self.assertEqual(self.run_reconcile(),result)
        self.assertEqual(self.snapshot(),after)
        self.assertEqual((self.output/'reconciliation.json').read_bytes(),report)
        self.reserve.assert_not_called();self.http.assert_not_called()

    def test_dry_run_validates_without_output_or_mutation(self):
        result=self.run_reconcile(apply=False)
        self.assertEqual(result['state'],'validated_no_changes')
        self.assertEqual(self.snapshot(),self.before);self.assertFalse(self.output.exists())

    def test_third_unknown_failure_or_redaction_blocks_every_update(self):
        original=json.loads(prep._json(self.value))
        for changes in (dict(status=None,category='timeout',dispatch_started=True),
                        dict(status=403,category='permission_or_region'),dict(redacted=True)):
            self.value=json.loads(prep._json(original));self.value['records'][2].update(changes);self.write_receipts()
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.run_reconcile()
            self.assertEqual(self.snapshot(),self.before)
        self.assertFalse(self.output.exists())

    def test_raw_sha_model_dimensions_usage_and_duplicate_json_are_independently_validated(self):
        record=self.value['records'][2]
        body=base64.b64decode(record['body_base64']);original=json.loads(body)
        mutations=[]
        wrong=json.loads(body);wrong['model']='other';mutations.append(prep._json(wrong).encode())
        wrong=json.loads(body);wrong['data'][0]['embedding']=[1.0]*255;mutations.append(prep._json(wrong).encode())
        wrong=json.loads(body);wrong['usage']['total_tokens']=True;mutations.append(prep._json(wrong).encode())
        wrong=json.loads(body);wrong['data'][0]['embedding'][0]=float('nan');mutations.append(json.dumps(wrong).encode())
        mutations.append(body[:-1]+b',"model":"text-embedding-v4"}')
        for changed in mutations:
            record['body_base64']=base64.b64encode(changed).decode();record['body_sha256']=prep.sha(changed)
            self.write_receipts()
            with self.assertRaises(ValueError):self.run_reconcile()
            self.assertEqual(self.snapshot(),self.before)
        record['body_base64']=base64.b64encode(body).decode();record['body_sha256']='0'*64;self.write_receipts()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.assertEqual(self.snapshot(),self.before)

    def test_dispatch_bookkeeping_interruption_resumes_same_id_without_reserving(self):
        with patch.object(BudgetLedger,'observe_provider',side_effect=OSError('synthetic crash')):
            with self.assertRaises(OSError):self.run_reconcile()
        context=recon.inputs(Path(self.ledger.path),self.grant_path,self.receipts)
        states=recon.ledger_state(context)
        self.assertEqual(list(states.values()),['dispatched','reserved','reserved'])
        self.assertEqual(self.snapshot()['accounted'],self.before['accounted'])
        self.run_reconcile();self.assertEqual(len(self.snapshot()['attempts']),len(self.before['attempts']))
        self.reserve.assert_not_called()

    def test_committed_observation_before_checkpoint_resumes_without_duplicate_observation(self):
        original=recon.frozen_write
        def fail(path,value):
            if path.name=='slot-1.json':raise OSError('synthetic disk failure')
            return original(path,value)
        with patch.object(recon,'frozen_write',side_effect=fail),self.assertRaises(OSError):self.run_reconcile()
        partial=self.snapshot();self.assertEqual(len(partial['observations']),len(self.before['observations'])+1)
        self.run_reconcile();after=self.snapshot()
        self.assertEqual(len(after['observations']),len(self.before['observations'])+3)
        self.assertEqual(partial['observations'][1],after['observations'][1])

    def test_original_or_target_row_tamper_blocks_before_any_other_update(self):
        with self.ledger.connect() as db:
            db.execute('UPDATE attempts SET max_input=999 WHERE id=?',(self.grant['slots'][2]['parent_attempt_id'],))
        before=self.snapshot()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.assertEqual(self.snapshot(),before)
        with self.ledger.connect() as db:
            db.execute('UPDATE attempts SET max_input=8192 WHERE id=?',(self.grant['slots'][2]['parent_attempt_id'],))
            db.execute("UPDATE attempts SET price_source='tampered-old' WHERE id=?",(self.old_id,))
        before=self.snapshot()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.assertEqual(self.snapshot(),before)

    def test_changed_receipts_or_existing_foreign_output_cannot_resume(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.assertEqual(self.snapshot(),self.before)

    def test_changed_valid_receipt_after_interruption_conflicts_with_frozen_plan(self):
        with patch.object(BudgetLedger,'observe_provider',side_effect=OSError('synthetic crash')):
            with self.assertRaises(OSError):self.run_reconcile()
        before=self.snapshot();row=self.value['records'][2]
        body=json.loads(base64.b64decode(row['body_base64']))
        body['usage']={'total_tokens':20,'prompt_tokens':20};raw=prep._json(body).encode()
        row.update(body_base64=base64.b64encode(raw).decode(),body_sha256=prep.sha(raw),input_tokens=20)
        self.write_receipts()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.assertEqual(self.snapshot(),before)

    def test_grant_or_parent_identity_mismatch_blocks(self):
        self.value['grant_sha256']='0'*64;self.write_receipts()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.value['grant_sha256']=prep.sha(self.grant_path.read_bytes())
        self.value['records'][2]['parent_attempt_id']=self.value['records'][0]['parent_attempt_id'];self.write_receipts()
        with self.assertRaises(ValueError):self.run_reconcile()
        self.assertEqual(self.snapshot(),self.before)


class EmbeddingSmokeReconcileIntegrationTests(unittest.TestCase):
    def test_grant_native_add_search_raw_sqlite_audit_and_parent_reconciliation(self):
        from agent_memory.granted_embedding_smoke import GrantedEmbeddingTransport
        from agent_memory.providers import TextEmbeddingV4
        from agent_memory.system import AgentMemorySystem, RetrievalConfig
        from deploy import audit_embedding_smoke as auditor
        from deploy.embedding_smoke_test import add_payload, query_payload
        from tests.test_embedding_smoke_audit import FakeHTTP
        with tempfile.TemporaryDirectory(prefix='embedding-reconcile-full-synthetic-') as temporary:
            root=Path(temporary).resolve();ledger=BudgetLedger(root/'budget.sqlite')
            now=datetime.now(timezone.utc);directory=root/'grant'
            prep.prepare(Path(ledger.path),directory,now.isoformat(),reserve_live=True,now=now)
            path=directory/'grant.json';digest=prep.sha(path.read_bytes());state=root/'remote'
            credential=Mock(side_effect=AssertionError('no credentials'))
            with patch.object(BudgetLedger,'reserve',side_effect=AssertionError('no extra reservations')), \
                    patch('agent_memory.provider_http.NativeHTTPS.exchange',side_effect=AssertionError('no HTTP')):
                transport=GrantedEmbeddingTransport(path,state/'receipts',credential,http=FakeHTTP(),offline_test=True)
                embedding=TextEmbeddingV4(prep.BAILIAN_BEIJING_ENDPOINT,transport,dimensions=256)
                service=AgentMemorySystem(state/'memory.sqlite',RetrievalConfig(dense=True),embedding=embedding)
                service.add(add_payload());service.search(query_payload('seed'))
                recreated=AgentMemorySystem(state/'memory.sqlite',RetrievalConfig(dense=True),embedding=embedding)
                recreated.search(query_payload('verify'))
                public,private=auditor.audit(path,digest,state)
                private_file=root/'private-audit.json';auditor.export_private(private_file,private)
                result=recon.reconcile(Path(ledger.path),path,private_file,root/'reconciled',apply=True)
            self.assertEqual(public['provider_receipts_verified'],3)
            self.assertEqual(result['usage_estimate_micro_cny'],15)
            self.assertEqual(ledger.totals()['attempts'],3)
            self.assertEqual(ledger.totals()['accounted_micro_cny'],15)
            credential.assert_not_called()


if __name__=='__main__':unittest.main()
