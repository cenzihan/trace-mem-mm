import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_memory.budget import BudgetLedger, VerifiedRate


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "budget.sqlite"
        self.ledger = BudgetLedger(self.path)
        # Arbitrary synthetic prices for arithmetic tests, NOT supplier quotations.
        self.rate = VerifiedRate("gpt-4o-mini", "100", "200", "synthetic test rate, not a real price", "synthetic")

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_prices_block(self):
        self.assertRaises(ValueError, self.ledger.reserve, None, 1)
        for rate in (VerifiedRate("gpt-4o-mini", "NaN", "1", "synthetic", "synthetic"),
                     VerifiedRate("CLIP", "1", "1", "synthetic", "synthetic"),
                     VerifiedRate("gpt-4o-mini", "1", "1", "", "synthetic"),
                     VerifiedRate("gpt-4o-mini", "1", "1", "synthetic", "synthetic", "USD")):
            with self.subTest(rate=rate):
                self.assertRaises(ValueError, self.ledger.reserve, rate, 1)

    def test_reserve_dispatch_settle_and_reopen(self):
        attempt = self.ledger.reserve(self.rate, 1_000_000, 1_000_000)
        self.assertEqual(self.ledger.totals()["accounted_micro_cny"], 300_000_000)
        self.ledger.mark_dispatched(attempt)
        self.ledger.settle(attempt, 500_000, 100_000)
        self.assertEqual(BudgetLedger(self.path).totals()["accounted_micro_cny"], 70_000_000)

    def test_retry_and_uncertain_outcome_consume_budget(self):
        first = self.ledger.reserve(self.rate, 2_000_000)
        self.ledger.mark_dispatched(first)
        retry = self.ledger.reserve(self.rate, 2_000_000, retry_of=first)
        self.assertNotEqual(first, retry)
        self.assertEqual(self.ledger.totals()["accounted_micro_cny"], 400_000_000)
        self.assertRaises(ValueError, self.ledger.reserve, self.rate, 2_000_000, retry_of=retry)

    def test_no_reuse_or_double_settlement(self):
        attempt = self.ledger.reserve(self.rate, 1)
        self.assertRaises(ValueError, self.ledger.settle, attempt, 1)
        self.ledger.mark_dispatched(attempt)
        self.assertRaises(ValueError, self.ledger.mark_dispatched, attempt)
        self.ledger.settle(attempt, 1)
        self.assertRaises(ValueError, self.ledger.settle, attempt, 1)

    def test_usage_over_bound_recorded_and_blocks_further_attempts(self):
        attempt = self.ledger.reserve(self.rate, 10)
        self.ledger.mark_dispatched(attempt)
        self.assertRaises(ValueError, self.ledger.settle, attempt, 11)
        self.assertEqual(self.ledger.totals()["accounted_micro_cny"], 1100)
        self.assertRaises(ValueError, self.ledger.reserve, self.rate, 1)

    def test_budget_cannot_silently_increase(self):
        self.assertRaises(ValueError, BudgetLedger, self.path, "501")
        self.assertRaises(ValueError, BudgetLedger, self.path, "499")

    def test_concurrent_reservations_cannot_overrun(self):
        def reserve(_):
            try:
                return self.ledger.reserve(self.rate, 2_000_000)
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            accepted = [v for v in pool.map(reserve, range(4)) if v]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(self.ledger.totals()["remaining_micro_cny"], 100_000_000)

    def test_fixed_hold_covers_bound_and_stays_after_usage(self):
        rate = VerifiedRate("gpt-4o-mini", "1.50", "6.00", "synthetic explicit budget assumption", "synthetic",
                            conversion_basis="budget_assumption_not_market_fx")
        self.assertRaises(ValueError, self.ledger.reserve, rate, 128000, 8)
        self.assertRaises(ValueError, self.ledger.reserve, rate, 128000, 8, fixed_reserve_cny="0.1")
        attempt = self.ledger.reserve(rate, 128000, 8, fixed_reserve_cny="1")
        self.ledger.mark_dispatched(attempt)
        self.assertRaises(ValueError, self.ledger.settle, attempt, 12, 1)
        self.ledger.record_usage_keep_hold(attempt, 12, 1)
        self.assertEqual(self.ledger.totals()["accounted_micro_cny"], 1_000_000)

    def test_fixed_hold_cannot_overrun_remaining_budget(self):
        ledger = BudgetLedger(Path(self.tmp.name) / "small.sqlite", cap_cny="1")
        ledger.reserve(self.rate, 1, fixed_reserve_cny="1")
        self.assertRaises(ValueError, ledger.reserve, self.rate, 1, fixed_reserve_cny="1")
