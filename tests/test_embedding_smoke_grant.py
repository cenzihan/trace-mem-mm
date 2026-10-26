"""Temporary synthetic ledgers only; no credentials, HTTP or actual shared budget."""
from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_memory.budget import BudgetLedger, VerifiedRate
from scripts import prepare_embedding_smoke_grant as grant

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
CHECKED = (NOW - timedelta(minutes=5)).isoformat()


def payloads():
    return [dict(model='text-embedding-v4', input=values, dimensions=256, encoding_format='float')
        for values in (['Synthetic source one.', 'Synthetic source two.'], ['Synthetic first query.'], ['Synthetic second query.'])]


class EmbeddingSmokeGrantTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='embedding-grant-synthetic-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve(); self.path = self.root / 'budget.sqlite'
        self.ledger = BudgetLedger(self.path)
        rate = VerifiedRate('gpt-4o-mini', '1', '1', 'synthetic price', CHECKED)
        old = self.ledger.reserve(rate, 10, fixed_reserve_cny='.01')
        self.ledger.mark_dispatched(old)
        self.ledger.observe_provider(old, mode='offline_fake', category='success', http_status=200,
            returned_model='gpt-4o-mini', input_tokens=4, output_tokens=0)
        self.output = self.root / 'new-grant'
        self.no_network = self.enterContext(patch('agent_memory.provider_http.NativeHTTPS.exchange',
            side_effect=AssertionError('real HTTP forbidden')))

    def prepare(self, **changes):
        args = dict(reserve_live=True, now=NOW, payloads=payloads()); args.update(changes)
        return grant.prepare(self.path, self.output, CHECKED, **args)

    def test_three_holds_exact_payload_hashes_old_rows_unchanged_not_dispatched(self):
        before = grant.read_ledger(self.path)
        result = self.prepare(); after = grant.read_ledger(self.path)
        self.assertEqual(after['attempts'][:len(before['attempts'])], before['attempts'])
        self.assertEqual(after['observations'], before['observations'])
        self.assertEqual(after['accounted'] - before['accounted'], 150000)
        self.assertEqual(len(after['attempts']) - len(before['attempts']), 3)
        value = json.loads((self.output / 'grant.json').read_text())
        self.assertEqual(value['expires_at'], (NOW + timedelta(hours=2)).isoformat())
        self.assertEqual(value['endpoint'], grant.BAILIAN_BEIJING_ENDPOINT)
        self.assertEqual([s['max_input_tokens'] for s in value['slots']], [16384,8192,8192])
        for ordinal, slot in enumerate(value['slots'], 1):
            self.assertEqual(slot['ordinal'], ordinal)
            self.assertEqual(slot['payload'], payloads()[ordinal-1])
            self.assertEqual(slot['payload_sha256'], grant.sha(grant._json(slot['payload']).encode()))
            self.assertEqual(slot['reserved_micro_cny'], 50000)
            checkpoint = json.loads((self.output / ('reservation-'+str(ordinal)+'.json')).read_text())
            self.assertEqual(checkpoint['completed_reservations'], ordinal)
            self.assertEqual(checkpoint['actual_provider_dispatches'], 0)
        self.assertEqual(result['actual_provider_dispatches'], 0)
        self.assertEqual(json.loads((self.output/'preparation.json').read_text())['state'], 'reserved_not_dispatched')
        self.no_network.assert_not_called()

    def test_existing_output_refused_without_extra_attempt(self):
        self.prepare(); before = grant.read_ledger(self.path)
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertEqual(grant.read_ledger(self.path), before)

    def test_missing_explicit_flag_refuses_before_payload_or_ledger(self):
        with patch.object(grant, 'read_ledger') as read, patch.object(grant, 'expected_payloads') as expected:
            with self.assertRaises(ValueError): self.prepare(reserve_live=False)
            read.assert_not_called(); expected.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_expired_future_or_naive_price_cannot_create_output_or_reserve(self):
        before = grant.read_ledger(self.path)
        for checked in ((NOW-timedelta(days=1)).isoformat(), (NOW+timedelta(seconds=1)).isoformat(), '2026-10-26T10:00:00'):
            with self.subTest(checked=checked), self.assertRaises(ValueError):
                grant.prepare(self.path, self.output, checked, reserve_live=True, now=NOW, payloads=payloads())
        self.assertFalse(self.output.exists()); self.assertEqual(grant.read_ledger(self.path), before)

    def test_wrong_missing_relative_and_lower_cap_ledgers_are_not_created_or_migrated(self):
        missing = self.root/'missing.sqlite'
        with self.assertRaises(FileNotFoundError):
            grant.prepare(missing,self.output,CHECKED,reserve_live=True,now=NOW,payloads=payloads())
        self.assertFalse(missing.exists())
        with self.assertRaises(ValueError): grant.read_ledger(Path('relative.sqlite'))
        wrong = self.root/'wrong.sqlite'
        with closing(sqlite3.connect(wrong)) as db, db: db.execute('CREATE TABLE memories(id TEXT)')
        before = wrong.read_bytes()
        with self.assertRaises(ValueError):
            grant.prepare(wrong,self.output,CHECKED,reserve_live=True,now=NOW,payloads=payloads())
        self.assertEqual(wrong.read_bytes(),before)
        other = self.root/'lower-cap.sqlite'; BudgetLedger(other,cap_cny='499')
        before = other.read_bytes()
        with self.assertRaises(ValueError): grant.read_ledger(other)
        self.assertEqual(other.read_bytes(),before)

    def test_marker_precedes_first_reservation_and_each_hold_is_fsynced(self):
        original = BudgetLedger.reserve; observations=[]
        def reserve(ledger,*args,**kwargs):
            self.assertTrue((self.output/'PREPARATION_STARTED.json').is_file())
            observations.append(len(list(self.output.glob('reservation-*.json'))))
            return original(ledger,*args,**kwargs)
        with patch.object(BudgetLedger,'reserve',new=reserve), patch.object(grant.os,'fsync',wraps=grant.os.fsync) as fsync:
            self.prepare()
        self.assertEqual(observations,[0,1,2]); self.assertGreaterEqual(fsync.call_count,6)

    def test_partial_failure_retains_checkpoint_and_refuses_automatic_replay(self):
        original = BudgetLedger.reserve; count=0
        def reserve(ledger,*args,**kwargs):
            nonlocal count
            count+=1
            if count==2: raise RuntimeError('synthetic interruption')
            return original(ledger,*args,**kwargs)
        before = self.ledger.totals()
        with patch.object(BudgetLedger,'reserve',new=reserve), self.assertRaises(RuntimeError): self.prepare()
        self.assertTrue((self.output/'reservation-1.json').exists())
        self.assertFalse((self.output/'grant.json').exists())
        self.assertEqual(self.ledger.totals()['accounted_micro_cny']-before['accounted_micro_cny'],50000)
        with self.assertRaises(FileExistsError): self.prepare()

    def test_checkpoint_failure_after_reserved_hold_blocks_replay(self):
        save = grant.save_exclusive
        def interrupted(path,value):
            if Path(path).name == 'reservation-1.json':
                raise OSError('synthetic disk failure after ledger commit')
            return save(path,value)
        before = self.ledger.totals()
        with patch.object(grant,'save_exclusive',side_effect=interrupted), self.assertRaises(OSError):
            self.prepare()
        self.assertTrue((self.output/'PREPARATION_STARTED.json').exists())
        self.assertFalse((self.output/'grant.json').exists())
        self.assertEqual(self.ledger.totals()['accounted_micro_cny']-before['accounted_micro_cny'],50000)
        with self.assertRaises(FileExistsError):self.prepare()

    def test_default_authored_smoke_payloads_match_remote_grant_validator(self):
        from deploy.embedding_smoke_test import expected_embedding_payloads
        from agent_memory import granted_embedding_smoke as remote
        self.prepare(payloads=None)
        value=json.loads((self.output/'grant.json').read_text())
        self.assertEqual([s['payload'] for s in value['slots']],expected_embedding_payloads())
        with patch.object(remote,'datetime') as clock:
            clock.fromisoformat=datetime.fromisoformat;clock.now.return_value=NOW
            self.assertEqual(remote._validate_grant(value),value)
        self.no_network.assert_not_called()

    def test_invalid_payload_or_insufficient_budget_never_reserves_partial_plan(self):
        wrong=payloads();wrong[-1]['dimensions']=1024
        before=grant.read_ledger(self.path)
        with self.assertRaises(ValueError):self.prepare(payloads=wrong)
        self.assertEqual(grant.read_ledger(self.path),before)
        duplicate=payloads();duplicate[-1]=duplicate[-2]
        with self.assertRaises(ValueError):self.prepare(payloads=duplicate)
        self.ledger.reserve(VerifiedRate('text-embedding-v4','0','0','synthetic',CHECKED),1,fixed_reserve_cny='499.90')
        before=grant.read_ledger(self.path)
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(grant.read_ledger(self.path),before);self.assertFalse(self.output.exists())

    def test_cli_without_reserve_live_never_calls_ledger_constructor(self):
        with patch.object(grant,'BudgetLedger') as ledger, redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            grant.main(['prepare','--ledger',str(self.path),'--output',str(self.output),'--price-checked-at',CHECKED])
        ledger.assert_not_called()


if __name__ == '__main__': unittest.main()
