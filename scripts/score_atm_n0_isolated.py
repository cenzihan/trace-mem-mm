"""Post-retrieval native evidence scoring. Gold never enters source index or ranking."""
import hashlib
import json
import sqlite3
import sys
from collections import Counter,defaultdict
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent_memory.native_metrics import retrieval_metrics
WORK=ROOT/'runtime/atm-n0';REPORT=ROOT/'reports/atm-n0-scores.json'
KS=(1,5,10,20,50,100)

def aggregate(rows):
    valid=[r for r in rows if r['gold_count']]
    return dict(questions=len(rows),gold_questions=len(valid),no_gold_questions=len(rows)-len(valid),
        metrics={str(k):{metric:sum(r['metrics'][str(k)][metric] for r in valid)/len(valid) if valid else None
                for metric in ('fractional_recall','any_hit','all_evidence_coverage')} for k in KS})

def main():
    if REPORT.exists():raise SystemExit('N0 scoring already recorded; preserve frozen results.')
    run=json.loads((ROOT/'reports/atm-n0-run.json').read_text());assert run['state']=='retrieval_complete_gold_not_loaded' and run['questions']==1013
    index_hash=hashlib.sha256((WORK/'source-index.sqlite').read_bytes()).hexdigest();assert index_hash==run['source_index_sha256']
    raw=(ROOT/'data/atm-schema/scorer-only/atm-bench.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762'
    gold={q['id']:q for q in json.loads(raw)}
    results=[json.loads(line) for line in (WORK/'retrieval-results.jsonl').read_text().splitlines()]
    assert len(results)==1013 and {r['id'] for r in results}==set(gold)
    with closing(sqlite3.connect((WORK/'source-index.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        sources=dict(db.execute('SELECT id,modality FROM sources'))
        columns=[row[1] for row in db.execute('PRAGMA table_info(sources)')]
    assert len(sources)==11034 and not {'answer','question','gold','evidence_ids'}&set(columns)
    rows=[];qtypes=defaultdict(list);groups=defaultdict(list);by_modality=defaultdict(list);missing=set();duplicates=0;distinct_gold=set()
    for result in results:
        q=gold[result['id']];ids=result['ranked_ids'];clues=q['evidence_ids'];required=set(clues)
        assert len(ids)==len(set(ids)) and set(ids)<=set(sources) and q['qtype']==result['qtype']
        assert all(isinstance(v,str) and v for v in clues)
        duplicates+=len(clues)-len(required);missing.update(required-set(sources));distinct_gold.update(required)
        row=dict(id=result['id'],qtype=q['qtype'],gold_count=len(required),metrics={str(k):retrieval_metrics(ids,list(required),k) for k in KS})
        rows.append(row);qtypes[q['qtype']].append(row)
        modalities=sorted({sources.get(identity,'unknown') for identity in required})
        groups['+'.join(modalities) or 'no_gold'].append(row)
        for modality in ('image','video','email'):
            subset=[identity for identity in required if sources.get(identity)==modality]
            if subset:by_modality[modality].append(dict(gold_count=len(subset),metrics={str(k):retrieval_metrics(ids,subset,k) for k in KS}))
    assert hashlib.sha256((WORK/'source-index.sqlite').read_bytes()).hexdigest()==index_hash
    report=dict(state='complete',scored_at_utc=datetime.now(timezone.utc).isoformat(),condition='external_Qwen_SGM_local_BM25_not_strict_end_to_end',
        source_counts=dict(Counter(sources.values())),total_sources=len(sources),all_native_source_ids_verified=True,
        questions=len(rows),overall=aggregate(rows),by_qtype={key:aggregate(group) for key,group in sorted(qtypes.items())},
        by_gold_modality_combination={key:aggregate(group) for key,group in sorted(groups.items())},
        per_required_modality={key:aggregate(group) for key,group in sorted(by_modality.items())},
        distinct_annotated_gold_ids=len(distinct_gold),unresolved_gold_ids_count=len(missing),duplicate_gold_labels=duplicates,
        sources_not_annotated_in_regular=len(set(sources)-distinct_gold),
        metric_formula=dict(fractional='macro mean |unique(topK) intersect unique(G)| / |unique(G)| over nonempty G',
            any_hit='macro mean indicator(intersection nonempty)',all_evidence='macro mean indicator(G subset topK)',
            no_gold='excluded from recall denominator and separately counted',unit='native artifact ID, not view/card/frame',
            native_ATM_denominator_compatible_when_gold_labels_unique=duplicates==0),
        source_index_sha256_unchanged=True,ranking_hash=run['retrieval_result_sha256'],gold_only_in_isolated_scorer=True,
        provider_requests=0,reader_answer_accuracy=None,raw_media_or_audio_processed=False,official_smoke=0,official_full=0,
        native_release='78e826dc07e97466b2f54443831ef9a83ab8b27c',no_gold_normalization_or_fitted_remapping=True)
    (WORK/'isolated-per-question-scores.json').write_text(json.dumps(rows),encoding='utf-8')
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'reports/atm-n0-progress.json').write_text(json.dumps(dict(stage='complete',source_count=len(sources),questions=len(rows),provider_requests=0)),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('state','source_counts','total_sources','questions','overall','by_qtype','by_gold_modality_combination','unresolved_gold_ids_count','sources_not_annotated_in_regular')},indent=2))

if __name__=='__main__':main()
