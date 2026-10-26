"""Synthetic manifests/caches only; no real data, providers, or credentials."""
import json
import unittest
from unittest.mock import patch

from scripts import score_atm_v5_dense as original
from scripts import score_atm_v5_dense_live as live
from scripts.prepare_atm_v5_queries import QUERY_FROZEN, SOURCE_FROZEN, digest
from tests.test_atm_v5_pipeline import ATMV5PipelineTests as Fixture


class ATMV5LiveCompatibilityTests(unittest.TestCase):
    setUp = Fixture.setUp
    write = Fixture.write
    dump = Fixture.dump
    hash = Fixture.hash
    fill = Fixture.fill
    prepare = Fixture.prepare
    no_gold = Fixture.no_gold

    def compatible_fixture(self, complete=True):
        source = json.loads((self.root / SOURCE_FROZEN).read_text())
        old = digest(b'Synthetic historical cache implementation')
        current = self.hash(live.CACHE_CODE)
        source['code_hashes'][live.CACHE_CODE] = old
        source['preserved_hashes'] = {original.SOURCE_INDEX: self.hash(original.SOURCE_INDEX)}
        self.dump(SOURCE_FROZEN, source)
        self.prepare(complete=complete)
        query = json.loads((self.root / QUERY_FROZEN).read_text())
        self.write(live.ENTRY, b'# Synthetic wrapper identity\n')
        self.dump(live.VALIDATION, dict(state='source_cache_validated_no_vectors',
            pure_whitespace_chunks=0, first_preparation_preserved=True,
            cache_bytes_unchanged=True, preserved_experiments_and_ledgers_unchanged=True,
            new_provider_requests=0, manifest_sha256=source['manifest_sha256'],
            current_code_hashes={live.CACHE_CODE: current}))
        self.dump(live.LIVE_FROZEN, dict(code_hashes={live.CACHE_CODE: current},
            source_manifest_sha256=source['manifest_sha256'],
            query_manifest_sha256=query['manifest_sha256'],
            query_scope_sha256=digest(json.dumps(query['scope'], sort_keys=True,
                ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()),
            model=query['scope']['model'], dimensions=query['scope']['dimensions'],
            preserved_hashes=source['preserved_hashes']))
        self.pins = {name: self.hash(name) for name in live.EVIDENCE_HASHES}
        self.addCleanup(patch.stopall)
        patch.object(live, 'EVIDENCE_HASHES', self.pins).start()
        patch.object(live, 'ACCEPTED_TRANSITION', (old, current)).start()

    def test_exact_historical_repair_retrieves_then_scores_without_mutating_freezes(self):
        self.compatible_fixture()
        before = {name: self.hash(name) for name in (*live.EVIDENCE_HASHES, *original.CODE)}
        verifier = original._hashes
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'hash mismatch'):
            original.retrieve(self.root, offline_test=True, expected_sources=3)
        self.assertFalse((self.root / original.WORK).exists())
        with self.no_gold():
            proof = live.compatibility(self.root)
            self.assertFalse((self.root / live.PROOF).exists())
            receipt = live.retrieve(self.root, offline_test=True, expected_sources=3)
        self.assertIs(original._hashes, verifier)
        self.assertEqual(receipt['transition'], proof['transition'])
        scored = live.score(self.root, expected_sources=3)
        self.assertEqual(scored['state'], 'scoring_complete')
        result = json.loads((self.root / original.SCORE_REPORT).read_text())
        self.assertEqual(result['datasets']['regular']['methods']['dense']['1']['fractional_recall'], 1.)
        self.assertEqual(result['datasets']['regular']['methods']['flat']['1']['fractional_recall'], 0.)
        self.assertEqual(before, {name: self.hash(name) for name in before})
        self.assertNotIn('PRIVATE_ANSWER_SENTINEL', (self.root / live.PROOF).read_text())
        with self.assertRaises(FileExistsError):
            live.retrieve(self.root, offline_test=True, expected_sources=3)

    def test_changed_evidence_is_rejected_before_gold_or_output(self):
        self.compatible_fixture()
        self.write(live.VALIDATION, (self.root / live.VALIDATION).read_bytes() + b' ')
        with self.no_gold(), self.assertRaises(ValueError):
            live.retrieve(self.root, offline_test=True, expected_sources=3)
        self.assertFalse((self.root / original.WORK).exists())
        self.assertFalse((self.root / live.PROOF).exists())

    def test_different_code_change_is_not_a_general_hash_bypass(self):
        self.compatible_fixture()
        self.write('agent_memory/shared_dense.py', b'# unreviewed ranking change')
        with self.no_gold(), self.assertRaises(ValueError):
            live.compatibility(self.root)

    def test_current_cache_must_match_live_and_query_freezes(self):
        self.compatible_fixture()
        config = json.loads((self.root / live.LIVE_FROZEN).read_text())
        config['code_hashes'][live.CACHE_CODE] = 'f' * 64
        self.dump(live.LIVE_FROZEN, config)
        self.pins[live.LIVE_FROZEN] = self.hash(live.LIVE_FROZEN)
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'one-file'):
            live.compatibility(self.root)

    def test_incomplete_cache_does_not_claim_compatibility_or_retrieval(self):
        self.compatible_fixture(complete=False)
        verifier = original._hashes
        with self.no_gold(), self.assertRaises(ValueError):
            live.retrieve(self.root, offline_test=True, expected_sources=3)
        self.assertIs(original._hashes, verifier)
        self.assertFalse((self.root / live.PROOF).exists())
        self.assertFalse((self.root / original.WORK).exists())

    def test_changed_wrapper_is_rejected_before_gold_scoring(self):
        self.compatible_fixture()
        with self.no_gold():
            live.retrieve(self.root, offline_test=True, expected_sources=3)
        self.write(live.ENTRY, b'# modified after retrieval')
        with self.no_gold(), self.assertRaises(ValueError):
            live.score(self.root, expected_sources=3)
        self.assertFalse((self.root / original.SCORE_REPORT).exists())

    def test_tampered_rankings_still_fail_before_gold(self):
        self.compatible_fixture()
        with self.no_gold():
            live.retrieve(self.root, offline_test=True, expected_sources=3)
        self.write(original.RANKINGS, (self.root / original.RANKINGS).read_bytes() + b' ')
        with self.no_gold(), self.assertRaises(ValueError):
            live.score(self.root, expected_sources=3)
        self.assertFalse((self.root / original.SCORE_REPORT).exists())


if __name__ == '__main__':
    unittest.main()
