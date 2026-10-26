"""Offline V5 entry point for the single documented source-preparation repair.

The original scorer and every old freeze remain byte-for-byte unchanged. Only
the historical embedding_cache.py preparation hash has one explicitly pinned
successor. This is not a general hash bypass or an embedding/network executor.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_atm_v5_dense as original
from scripts.prepare_atm_v5_queries import (
    QUERY_FROZEN, SOURCE_FROZEN, digest, load_json, path_at, save_new, sha, verified_bytes,
)

ENTRY = 'scripts/score_atm_v5_dense_live.py'
CACHE_CODE = 'agent_memory/embedding_cache.py'
LIVE_FROZEN = 'runtime/atm-v5-live/FROZEN_RUN.json'
VALIDATION = 'reports/atm-v5-cache-validation.json'
PROOF = 'runtime/atm-v5-dense-live/FROZEN_COMPATIBILITY.json'
RETRIEVAL_REPORT = 'reports/atm-v5-dense-live-retrieval.json'
SCORING_REPORT = 'reports/atm-v5-dense-live-scoring.json'
ACCEPTED_TRANSITION = (
    'c981301af005740e83225833edd72c45c0984125b2573c6c32400350d4f204c2',
    'd7c01311efd5519413c6551772b7089f561e547096a2bef019c86a5d3f5b583d',
)
# These four existing artifacts were inspected before adding this entry point.
# A future run or repair requires a separate reviewed entry point, not new CLI
# options that weaken these pins. Synthetic tests patch them in memory only.
EVIDENCE_HASHES = {
    SOURCE_FROZEN: '4ac0fc925990acfbcf17db7554d812a9124a00d34e82af01eb3bf304b3b4d4fe',
    QUERY_FROZEN: 'aa63ba45ee3fda1f915987748f30576d096cf9727999ba1867e7a5c5e18a81f7',
    LIVE_FROZEN: 'cc3aa2414415b8d5b0ad956fb0e17518cfa59a8a8430261ad77f972386e0dd43',
    VALIDATION: 'd552b7786c2d0626bdde10cd4f215de8f01f7eca42585183336d32a0171166fe',
}


def compatibility(root=ROOT):
    """Verify metadata/code and return the exact one-file compatibility proof."""
    root = Path(root).resolve()
    evidence = {name: load_json(verified_bytes(root, name, expected))
                for name, expected in EVIDENCE_HASHES.items()}
    source, query = evidence[SOURCE_FROZEN], evidence[QUERY_FROZEN]
    live, validation = evidence[LIVE_FROZEN], evidence[VALIDATION]
    historical = source['code_hashes']
    current = validation['current_code_hashes']
    old_hash, current_hash = ACCEPTED_TRANSITION
    if (set(historical) != set(current)
            or {name for name in historical if historical[name] != current[name]} != {CACHE_CODE}
            or historical[CACHE_CODE] != old_hash or current[CACHE_CODE] != current_hash
            or query['code_hashes'].get(CACHE_CODE) != current_hash
            or live['code_hashes'].get(CACHE_CODE) != current_hash):
        raise ValueError('only the reviewed one-file source preparation transition is accepted')
    if (validation.get('state') != 'source_cache_validated_no_vectors'
            or validation.get('pure_whitespace_chunks') != 0
            or validation.get('first_preparation_preserved') is not True
            or validation.get('cache_bytes_unchanged') is not True
            or validation.get('preserved_experiments_and_ledgers_unchanged') is not True
            or validation.get('new_provider_requests') != 0
            or validation['manifest_sha256'] != source['manifest_sha256']
            or live['source_manifest_sha256'] != source['manifest_sha256']
            or live['query_manifest_sha256'] != query['manifest_sha256']
            or query['source_frozen_sha256'] != EVIDENCE_HASHES[SOURCE_FROZEN]
            or query['source_scope'] != source['scope']
            or live['model'] != query['scope']['model']
            or live['dimensions'] != query['scope']['dimensions']):
        raise ValueError('audited source/query/live manifest compatibility evidence differs')
    scope_hash = digest(json.dumps(query['scope'], ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode('utf-8'))
    if live['query_scope_sha256'] != scope_hash:
        raise ValueError('live query scope differs from the frozen query scope')
    preserved = {name: value for name, value in source['preserved_hashes'].items()
                 if name != 'runtime/development-budget.sqlite'}
    if live['preserved_hashes'] != preserved:
        raise ValueError('live preserved artifacts differ from initial source preparation')
    # The active development budget is deliberately not compared to its pre-run
    # byte hash; the live executor owns its historical-row audit and new charges.
    for values in (current, query['code_hashes'], query['input_hashes'],
                   live['code_hashes'], preserved):
        original._hashes(root, values)
    return dict(version='V5-single-source-preparation-compatibility-v1',
        entry_file=ENTRY, entry_sha256=sha(root, ENTRY), evidence_hashes=dict(EVIDENCE_HASHES),
        transition=dict(file=CACHE_CODE, historical_preparation_sha256=old_hash,
                        audited_execution_sha256=current_hash),
        historical_source_code_hashes=historical, audited_source_code_hashes=current,
        query_code_hashes=query['code_hashes'], live_code_hashes=live['code_hashes'],
        source_manifest_sha256=source['manifest_sha256'], query_manifest_sha256=query['manifest_sha256'],
        reason='Reject pure-whitespace chunks; existing source mapping was validated unchanged before encoding',
        original_scorer_unchanged=True, original_freezes_unchanged=True,
        retrieval_math_unchanged=True, gold_isolation_unchanged=True, provider_requests=0)


@contextmanager
def _retrieval_adapter(root, proof):
    """Adapt one historical map check, then bind the proof into the new freeze."""
    check, save = original._hashes, original.save_new
    calls = 0
    def checked(check_root, values):
        nonlocal calls
        if values == proof['historical_source_code_hashes']:
            if Path(check_root).resolve() != root or calls:
                raise ValueError('historical compatibility check may occur exactly once')
            calls += 1
            return check(check_root, proof['audited_source_code_hashes'])
        return check(check_root, values)
    def saved(save_root, relative, value):
        if relative == original.FROZEN:
            if calls != 1 or Path(save_root).resolve() != root:
                raise ValueError('compatibility must precede the retrieval freeze')
            # Recheck immutable evidence and code immediately before claiming
            # output. compatibility() itself must use the unadapted verifier.
            for name, expected in proof['evidence_hashes'].items():
                verified_bytes(root, name, expected)
            verified_bytes(root, ENTRY, proof['entry_sha256'])
            save(root, PROOF, proof)
            value = dict(value)
            value['input_hashes'] = {**value['input_hashes'], PROOF: sha(root, PROOF),
                                     **proof['evidence_hashes']}
            value['code_hashes'] = {**value['code_hashes'], ENTRY: proof['entry_sha256']}
            value['live_compatibility'] = dict(file=PROOF, sha256=sha(root, PROOF))
        return save(save_root, relative, value)
    original._hashes, original.save_new = checked, saved
    try:
        yield
        if calls != 1:
            raise ValueError('expected exactly one historical compatibility check')
    finally:
        original._hashes, original.save_new = check, save


def _finish(root, phase, proof, result_file, report_file):
    if compatibility(root) != proof:
        raise ValueError('compatibility evidence or execution code changed during the phase')
    proof_hash = sha(root, PROOF)
    if load_json(verified_bytes(root, PROOF, proof_hash)) != proof:
        raise ValueError('saved compatibility proof changed')
    report = dict(state=phase + '_complete', entry_file=ENTRY, entry_sha256=proof['entry_sha256'],
        compatibility_file=PROOF, compatibility_sha256=proof_hash,
        transition=proof['transition'], evidence_hashes=proof['evidence_hashes'],
        result_file=result_file, result_sha256=sha(root, result_file),
        original_scorer_unchanged=True, original_freezes_unchanged=True,
        retrieval_math_unchanged=True, gold_isolation_unchanged=True, provider_requests=0)
    save_new(root, report_file, report)
    return report


def retrieve(root=ROOT, *, offline_test=False, expected_sources=11034, timeout_seconds=3600):
    root = Path(root).resolve()
    for relative in (PROOF, RETRIEVAL_REPORT, original.WORK, original.RUN_REPORT):
        if path_at(root, relative).exists():
            raise FileExistsError('V5 compatibility retrieval already claimed; preserve outputs')
    proof = compatibility(root)
    with _retrieval_adapter(root, proof):
        original.retrieve(root, offline_test=offline_test, expected_sources=expected_sources,
                          timeout_seconds=timeout_seconds)
    return _finish(root, 'retrieval', proof, original.RUN_REPORT, RETRIEVAL_REPORT)


def score(root=ROOT, *, expected_sources=11034):
    root = Path(root).resolve()
    if path_at(root, SCORING_REPORT).exists():
        raise FileExistsError('V5 compatibility scoring already completed; preserve outputs')
    proof = compatibility(root)
    run = load_json(path_at(root, original.RUN_REPORT).read_bytes())
    frozen = load_json(verified_bytes(root, run['frozen_file'], run['frozen_sha256']))
    expected_link = dict(file=PROOF, sha256=sha(root, PROOF))
    if (frozen.get('live_compatibility') != expected_link
            or frozen['input_hashes'].get(PROOF) != expected_link['sha256']
            or frozen['code_hashes'].get(ENTRY) != proof['entry_sha256']
            or load_json(verified_bytes(root, PROOF, expected_link['sha256'])) != proof):
        raise ValueError('retrieval freeze does not bind this exact compatibility entry point')
    receipt = load_json(path_at(root, RETRIEVAL_REPORT).read_bytes())
    if (receipt['compatibility_sha256'] != expected_link['sha256']
            or receipt['entry_sha256'] != proof['entry_sha256']
            or receipt['result_file'] != original.RUN_REPORT
            or receipt['result_sha256'] != sha(root, original.RUN_REPORT)):
        raise ValueError('compatibility retrieval receipt differs from frozen retrieval')
    original.score(root, expected_sources=expected_sources)
    return _finish(root, 'scoring', proof, original.SCORE_REPORT, SCORING_REPORT)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('verify', 'retrieve', 'score'):
        raise SystemExit('Use verify for read-only checks, retrieve after encoding, then score separately.')
    result = compatibility() if sys.argv[1] == 'verify' else retrieve() if sys.argv[1] == 'retrieve' else score()
    print(json.dumps(result, indent=2))
