"""Temporary synthetic ledgers only. Unknown network outcomes must keep blocking."""
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.development_run import DevelopmentRun

class KnownResponseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.run=DevelopmentRun(self.root/'run.sqlite',create=True,max_requests=2,max_reservation_cny='.5')
        self.ledger=BudgetLedger(self.root/'budget.sqlite')
    def tearDown(self):self.tmp.cleanup()
    def observe(self,category='invalid_contract',status=200,usage=True):
        self.slot=self.run.claim('gpt-4o-mini','.25')
        rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat(),conversion_basis='budget_assumption_not_market_fx')
        attempt=self.ledger.reserve(rate,128000,1536,fixed_reserve_cny='.25');self.ledger.mark_dispatched(attempt)
        self.ledger.observe_provider(attempt,mode='offline_fake',category=category,http_status=status,
            returned_model='gpt-4o-mini-2024-07-18',input_tokens=40 if usage else None,output_tokens=10 if usage else None)
        return attempt
    def test_known_rejection_preserves_slot_hold_and_allows_explicit_next_request(self):
        attempt=self.observe();before=self.ledger.totals()
        self.run.acknowledge_known_contract_rejection(self.slot,self.ledger.path,attempt)
        self.assertEqual(before,self.ledger.totals());self.assertEqual(self.run.summary()['request_claims'],1)
        self.assertEqual(self.run.summary()['cumulative_preflight_micro_cny'],250000)
        with self.run.connect() as db:self.assertEqual(db.execute('SELECT state FROM calls').fetchone()[0],'rejected_response')
        self.assertEqual(self.run.claim('gpt-4o-mini','.25'),2)
        self.assertRaises(ValueError,self.run.claim,'gpt-4o-mini','.25')
    def test_timeout_unknown_usage_or_non200_stays_blocked(self):
        for category,status,usage in (('timeout',None,False),('invalid_contract',200,False),('invalid_contract',500,True)):
            with self.subTest(category=category,status=status,usage=usage):
                # fresh fixture per subcase; no production-ledger mutation
                other=KnownResponseRecoveryTests();other.setUp()
                try:
                    attempt=other.observe(category,status,usage)
                    self.assertRaises(ValueError,other.run.acknowledge_known_contract_rejection,other.slot,other.ledger.path,attempt)
                    self.assertRaises(ValueError,other.run.claim,'gpt-4o-mini','.25')
                finally:other.tearDown()
    def test_wrong_slot_and_double_acknowledgement_reject(self):
        attempt=self.observe()
        self.assertRaises(ValueError,self.run.acknowledge_known_contract_rejection,99,self.ledger.path,attempt)
        self.run.acknowledge_known_contract_rejection(self.slot,self.ledger.path,attempt)
        self.assertRaises(ValueError,self.run.acknowledge_known_contract_rejection,self.slot,self.ledger.path,attempt)
    def test_observation_before_slot_does_not_authorize_recovery(self):
        attempt=self.observe();self.run.acknowledge_known_contract_rejection(self.slot,self.ledger.path,attempt)
        next_slot=self.run.claim('gpt-4o-mini','.25')
        self.assertRaises(ValueError,self.run.acknowledge_known_contract_rejection,next_slot,self.ledger.path,attempt)

if __name__=='__main__':unittest.main()
