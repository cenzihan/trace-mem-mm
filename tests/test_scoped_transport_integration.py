"""Synthetic scoped failures: no live provider, sensitive input or network."""
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.budget import BudgetLedger
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, HTTPReply, ProviderFailure
from agent_memory.scoped_unknown import request_owner
from agent_memory.service_transport import ServiceTransport, history_digest


class FakeHTTP:
    is_fake_http = True
    def __init__(self): self.fail = False; self.calls = 0
    def exchange(self, endpoint, body, headers, timeout, limit):
        self.calls += 1
        if self.fail: raise ProviderFailure('timeout')
        return HTTPReply(200, json.dumps(dict(model='text-embedding-v4', usage=dict(total_tokens=3),
            data=[dict(index=0, embedding=[0.1]*256)])).encode())


class ScopedTransport(ServiceTransport):
    scoped_unknown_enabled = True


class ScopedTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        self.ledger=BudgetLedger(self.p/'budget.sqlite');self.http=FakeHTTP()
        self.authority=self.p/'authority.json';self.prices=self.p/'prices.json'
        self.authority.write_text(json.dumps(dict(active_owner='server',cap_micro_cny=500000000,
            history={t:dict(rows=0,sha256=history_digest([])) for t in ('attempts','provider_observations')})))
        self.prices.write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),models={
            'gpt-4o-mini':dict(input_cny_per_million='1.5',output_cny_per_million='6',price_source='https://developers.openai.com/api/docs/models/gpt-4o-mini',conversion_basis='budget_assumption_not_market_fx'),
            'text-embedding-v4':dict(input_cny_per_million='.5',output_cny_per_million='0',price_source='https://help.aliyun.com/zh/model-studio/embedding')})))
        self.payload=dict(model='text-embedding-v4',input=['fixture-one'],dimensions=256,encoding_format='float')

    def make(self):
        return ScopedTransport(self.ledger.path,self.authority,self.prices,self.p/'receipts',{},http=self.http,offline_test=True)

    def test_unknown_owner_isolated_other_user_runs_and_old_payload_never_replayed(self):
        t=self.make();self.http.fail=True
        with request_owner('user-a'):
            with self.assertRaises(ProviderFailure): t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        self.assertEqual(self.http.calls,1);self.http.fail=False
        t=self.make();t.ready()
        with request_owner('user-a'):
            with self.assertRaises(ValueError):t(BAILIAN_BEIJING_ENDPOINT,dict(self.payload,input=['new text']))
        with request_owner('user-b'):
            with self.assertRaises(ValueError):t(BAILIAN_BEIJING_ENDPOINT,self.payload)
            t(BAILIAN_BEIJING_ENDPOINT,dict(self.payload,input=['different']))
        self.assertEqual(self.http.calls,2)
        with self.ledger.connect() as db:
            rows=db.execute('select status,reserved,accounted from attempts order by rowid').fetchall()
        self.assertEqual(rows[0][0],'uncertain');self.assertEqual(rows[0][1],rows[0][2]);self.assertEqual(rows[1][0],'settled')
        with t.journal.connect() as db:
            rows=db.execute('select state,manifest from operations order by rowid').fetchall()
        self.assertEqual(rows[0][0],'unknown');self.assertNotIn('user-a',rows[0][1]);self.assertNotIn('fixture-one',rows[0][1])

    def test_missing_owner_fails_closed_before_claim(self):
        t=self.make()
        with self.assertRaises(ValueError):t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        self.assertEqual(self.http.calls,0)
        with t.journal.connect() as db:self.assertEqual(db.execute('select count(*) from operations').fetchone()[0],0)

    def test_old_unknown_without_owner_remains_global_block(self):
        t=self.make();self.http.fail=True
        with request_owner('user-a'):
            with self.assertRaises(ProviderFailure):t(BAILIAN_BEIJING_ENDPOINT,self.payload)
        with t.journal.connect() as db:
            row=db.execute("select fingerprint,manifest from operations where state='unknown'").fetchone()
            m=json.loads(row['manifest']);m.pop('owner_sha256')
            db.execute('update operations set manifest=? where fingerprint=?',(json.dumps(m),row['fingerprint']))
        self.http.fail=False
        with self.assertRaises(ValueError):self.make().ready()
        self.assertEqual(self.http.calls,1)


if __name__ == '__main__': unittest.main()
