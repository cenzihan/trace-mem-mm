"""Post-hoc coverage lost when frozen V6 Top20 is packed for the fixed reader.

Read-only with respect to every experimental input. This writes one new aggregate
diagnostic report, never per-question IDs/content. Gold is opened only after the
64 reader outcomes and their additive recovery pass the existing verifier.
This is an exposed development-pilot diagnostic, not another test or tuning run.
"""
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.atm_reader import VARIANTS, query_id
from agent_memory.native_metrics import retrieval_metrics
from scripts import recover_atm_v6_reader as recovery
from scripts import score_atm_v6_reader as original
from scripts.prepare_atm_v5_queries import load_json, path_at, save_new, sha, verified_bytes

SCRIPT = 'scripts/audit_atm_v6_packing_coverage.py'
REPORT = 'reports/atm-v6-packing-coverage-posthoc.json'
RANKINGS = 'runtime/atm-v6-rerank/retrieval-results.jsonl'
METRICS = ('fractional_recall', 'any_hit', 'all_evidence_coverage')
SEED, SAMPLES = 20261026, 5000


def stats(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def paired_loss(values, *, samples=SAMPLES):
    if not values or any(value < -1e-12 for value in values):
        raise ValueError('a whole candidate prefix cannot improve evidence coverage')
    rng, n = random.Random(SEED), len(values)
    if all(value == values[0] for value in values):
        draws = [values[0]]
    else:
        draws = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    def quantile(p):
        position = (len(draws)-1)*p
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower]+(draws[upper]-draws[lower])*(position-lower)
    return dict(mean=statistics.mean(values), mean_percentage_points=statistics.mean(values)*100,
        questions_with_loss=sum(value > 1e-12 for value in values),
        unchanged_questions=sum(abs(value) <= 1e-12 for value in values),
        improved_questions=0, maximum_loss=max(values), ci95=[quantile(.025), quantile(.975)])


def aggregate(items):
    return dict(questions=len(items),
        original_top20={metric: statistics.mean(float(row['top20'][metric]) for row in items) for metric in METRICS},
        packed_context={metric: statistics.mean(float(row['packed'][metric]) for row in items) for metric in METRICS},
        paired_coverage_loss={metric: paired_loss([float(row['top20'][metric])-float(row['packed'][metric])
                                                for row in items]) for metric in METRICS},
        original_top20_sources=stats([row['top20_count'] for row in items]),
        retained_sources=stats([row['retained'] for row in items]),
        input_text_tokens=stats([row['tokens'] for row in items]),
        utf8_text_bytes=stats([row['bytes'] for row in items]),
        budget_shortened_prefixes=sum(row['retained'] < row['top20_count'] for row in items),
        empty_contexts=sum(row['retained'] == 0 for row in items),
        gold_source_occurrences=sum(row['gold_count'] for row in items),
        gold_source_occurrences_in_top20=sum(row['top20']['hits'] for row in items),
        gold_source_occurrences_in_context=sum(row['packed']['hits'] for row in items),
        lost_gold_source_occurrences=sum(row['top20']['hits']-row['packed']['hits'] for row in items))


def audit(root=ROOT):
    root = Path(root).resolve()
    if path_at(root, REPORT).exists():
        raise FileExistsError('post-hoc diagnostic already exists; preserve the first audit')
    verified = recovery.verify_recovery(root)
    if len(verified['answers']) != 64 or len(verified['public']) != 16:
        raise ValueError('complete frozen 16 x 4 recovered reader outcomes required')
    answer_counts = Counter(row['answer']['schema_status'] for row in verified['answers'].values())
    inputs = dict(verified['inputs'])
    # The helper verifies original receipts; all hashes are captured before gold.
    inputs[recovery.REPORT] = sha(root, recovery.REPORT)
    recovery_report = load_json(verified_bytes(root, recovery.REPORT, inputs[recovery.REPORT]))
    if (recovery_report['reader_answers'] != 64 or recovery_report['valid_schema'] != answer_counts['valid']
            or recovery_report['invalid_schema'] != answer_counts['invalid']):
        raise ValueError('recovery aggregate does not match the complete answer set')
    audit_code_hash = sha(root, SCRIPT)
    metrics_code_hash = sha(root, 'agent_memory/native_metrics.py')
    pack_raw = verified_bytes(root, original.PACKS, inputs[original.PACKS])
    packs = original.lines(pack_raw)
    rank_rows = original.lines(verified_bytes(root, RANKINGS, inputs[RANKINGS]))
    rankings = {query_id(row['dataset'], row['id']): row for row in rank_rows}
    if len(rankings) != len(rank_rows):
        raise ValueError('unique frozen retrieval query identities required')
    seen = set()
    for row in packs:
        identity = query_id(row['dataset'], row['id'])
        key = (identity, row['variant'])
        pack = row['pack']
        if (key not in verified['answers'] or key in seen or row['query_id'] != identity
                or identity not in rankings or row['qtype'] != verified['public'][identity]['qtype']):
            raise ValueError('frozen pack identity differs from reader and retrieval')
        seen.add(key)
        full = rankings[identity]['methods'][row['variant']]['ranked_ids'][:20]
        retained = pack['source_memory_ids']
        if (len(full) != 20 or len(set(full)) != 20 or pack['ranked_native_ids'] != full
                or retained != full[:len(retained)] or pack['dropped_native_ids'] != full[len(retained):]
                or pack['source_records_retained'] != len(retained)
                or pack['dropped_records'] != len(full)-len(retained)
                or pack['text_token_count'] > pack['budget']['text_tokens']
                or pack['utf8_text_bytes'] > pack['budget']['utf8_bytes']
                or pack['image_count'] != 0 or pack['context_gold_access'] is not False):
            raise ValueError('exact whole-source prefix within the frozen reader budget required')
    if seen != set(verified['answers']):
        raise ValueError('all 64 original packs must match complete reader outcomes')

    # Gold is first opened after the complete reader/recovery and pack checks.
    gold, gold_hashes = {}, {}
    for spec in verified['datasets']:
        native = load_json(verified_bytes(root, spec['gold_file'], spec['gold_sha256']))
        gold_hashes[spec['gold_file']] = spec['gold_sha256']
        if not isinstance(native, list) or len(native) != spec['count']:
            raise ValueError('native gold dataset count mismatch')
        native_seen = set()
        for row in native:
            identity = query_id(spec['name'], row['id'])
            if identity in native_seen:
                raise ValueError('duplicate native gold query identity')
            native_seen.add(identity)
            if identity not in verified['public']:
                continue
            question, labels = verified['public'][identity], row.get('evidence_ids')
            if (row.get('question') != question['question'] or row.get('qtype') != question['qtype']
                    or not isinstance(labels, list) or not labels
                    or any(not isinstance(label, str) or not label for label in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError('selected exact original question and unique gold sources required')
            gold[identity] = labels
    if set(gold) != set(verified['public']):
        raise ValueError('selected gold coverage is incomplete')
    details = []
    for row in packs:
        pack, labels = row['pack'], gold[row['query_id']]
        details.append(dict(dataset=row['dataset'], variant=row['variant'],
            top20=retrieval_metrics(pack['ranked_native_ids'], labels, 20),
            packed=retrieval_metrics(pack['source_memory_ids'], labels, 20),
            gold_count=len(labels), top20_count=len(pack['ranked_native_ids']),
            retained=len(pack['source_memory_ids']), tokens=pack['text_token_count'], bytes=pack['utf8_text_bytes']))
    groups = {dataset: {variant: aggregate([row for row in details
        if row['variant'] == variant and (dataset == 'all' or row['dataset'] == dataset)])
        for variant in VARIANTS} for dataset in ('all', 'regular', 'hard')}
    # Detect concurrent changes without rerunning models or altering old artifacts.
    recovery.verify_recovery(root)
    original.check_hashes(root, inputs)
    for name, expected in gold_hashes.items():
        verified_bytes(root, name, expected)
    verified_bytes(root, SCRIPT, audit_code_hash)
    verified_bytes(root, 'agent_memory/native_metrics.py', metrics_code_hash)
    report = dict(state='complete', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        analysis='posthoc frozen reader packing coverage; exposed development pilot only',
        questions=16, reader_packs=64, recovered_valid_answers=answer_counts['valid'],
        substantive_invalid_answers=answer_counts['invalid'], groups=groups,
        metric='macro fractional recall, any-hit and all-evidence coverage over native gold source IDs',
        paired_loss='same question and variant: original Top20 coverage minus actual retained-prefix coverage',
        diagnostics_only=True, new_test=False, used_for_tuning=False, models_called=0,
        answer_accuracy_measured=False, official_score=False, no_new_heldout_claim=True,
        all_ranked_top20_verified=True, all_contexts_whole_prefixes=True,
        all_contexts_within_fixed_budget=True, all_contexts_zero_images=True,
        gold_opened_after_64_reader_and_recovery_verified=True,
        bootstrap=dict(seed=SEED, samples=SAMPLES, unit='paired pilot question',
            caveat='small exposed pilot; descriptive unadjusted intervals; shared-evidence dependence not modeled'),
        provenance=dict(input_hashes=inputs, gold_hashes=gold_hashes,
            audit_code_hashes={SCRIPT:audit_code_hash, 'agent_memory/native_metrics.py':metrics_code_hash},
            packs_sha256=inputs[original.PACKS], selection_sha256=inputs[original.SELECTION],
            original_reader_frozen_sha256=inputs[original.READER_FROZEN],
            recovery_frozen_sha256=inputs[recovery.FROZEN], recovery_answers_sha256=inputs[recovery.ANSWERS]),
        content_or_native_ids_in_report=False, original_experiment_artifacts_unchanged=True)
    save_new(root, REPORT, report)
    return report


if __name__ == '__main__':
    result = audit()
    summary = {dataset: {variant: dict(
        top20_recall=value['original_top20']['fractional_recall'],
        packed_recall=value['packed_context']['fractional_recall'],
        loss_pp=value['paired_coverage_loss']['fractional_recall']['mean_percentage_points'],
        questions_with_loss=value['paired_coverage_loss']['fractional_recall']['questions_with_loss'])
        for variant,value in methods.items()} for dataset,methods in result['groups'].items()}
    print(json.dumps(dict(state=result['state'], questions=result['questions'], reader_packs=result['reader_packs'],
                         groups=summary, report_file=REPORT, models_called=0), ensure_ascii=False, indent=2))
