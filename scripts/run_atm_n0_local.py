"""Full-source native ATM SGM/BM25; no gold loading, provider calls, or raw media."""
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent_memory.atm_n0 import native_sgm_corpus,NativeBM25
WORK=ROOT/'runtime/atm-n0';DATA=ROOT/'data/atm-n0-source';REPORT=ROOT/'reports/atm-n0-run.json'

def chunks_utf8(text,limit=8192):
    raw=text.encode('utf-8');start=0
    while start<len(raw):
        end=min(start+limit,len(raw))
        while end<len(raw) and raw[end]&0xC0==0x80:end-=1
        if end<=start:raise ValueError('UTF8 planning chunk failed')
        yield raw[start:end];start=end

def main():
    if WORK.exists() or REPORT.exists():raise SystemExit('N0 already claimed; inspect existing results, never duplicate process.')
    downloaded=json.loads((ROOT/'reports/atm-n0-download.json').read_text());assert downloaded['complete']
    source={}
    for row in downloaded['files']:
        raw=(DATA/row['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];source[row['file']]=json.loads(raw)
    inventory=json.loads((ROOT/'data/atm-schema/corpus-inventory.json').read_text())
    expected={m:{r['evidence_id'] for r in inventory['records'] if r['modality']==m} for m in ('image','video')}
    records,coverage=native_sgm_corpus(source['image_batch_results.json'],source['video_batch_results.json'],source['emails.json'],
        expected_images=expected['image'],expected_videos=expected['video'],expected_emails=6742)
    assert len(records)==11034
    WORK.mkdir();start=time.monotonic()
    frozen=dict(frozen_before_questions=True,namespace='atm-regular:sgm-n0:78e826dc',
        dataset_revision='78e826dc07e97466b2f54443831ef9a83ab8b27c',license='CC-BY-NC-4.0',condition='external_Qwen_SGM_research_not_strict_from_raw',
        bm25=dict(k1=1.2,b=.75,query_terms='unique sorted',zero_score_results='excluded',ties='native ID ascending'),
        source_fields=dict(media=['id','type','timestamp','location_name','short_caption','caption','ocr_text','tags'],email=['id','timestamp','short_summary','detail']),
        entities_included=False,source_text_clipped=False,all_distractors=True,source_counts=coverage['source_counts'],
        source_files=downloaded['files'],top_k=100,metric_ks=[1,5,10,20,50,100],
        tokenizer_source_sha256=hashlib.sha256((ROOT/'agent_memory/retrieval.py').read_bytes()).hexdigest(),
        formatter_and_index_source_sha256=hashlib.sha256((ROOT/'agent_memory/atm_n0.py').read_bytes()).hexdigest(),
        no_gold_in_index=True,provider_requests=0,reader=False,reranker=False,raw_media=False)
    (WORK/'FROZEN_CONFIG.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    with sqlite3.connect(WORK/'source-index.sqlite') as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE sources(namespace TEXT,id TEXT,modality TEXT,text TEXT,timestamp TEXT,description_present INTEGER,PRIMARY KEY(namespace,id))')
        db.executemany('INSERT INTO sources VALUES(?,?,?,?,?,?)',[(frozen['namespace'],r['id'],r['modality'],r['text'],r['timestamp'],r['description_present']) for r in records])
    index=NativeBM25({r['id']:r['text'] for r in records})
    tokens_by_modality={m:sum(index.lengths[r['id']] for r in records if r['modality']==m) for m in ('image','video','email')}
    chunk_hashes=set();chunks=0;stats={}
    for m in ('image','video','email'):
        selected=[r for r in records if r['modality']==m];lengths=[len(r['text'].encode()) for r in selected]
        stats[m]=dict(sources=len(selected),selected_text_bytes=sum(lengths),max_source_bytes=max(lengths),sources_over_8192_bytes=sum(n>8192 for n in lengths))
    for record in records:
        for chunk in chunks_utf8(record['text']):chunks+=1;chunk_hashes.add(hashlib.sha256(chunk).hexdigest())
    source_only=dict(coverage=coverage,source_text_stats=stats,source_token_count=sum(index.lengths.values()),source_tokens_by_modality=tokens_by_modality,
        planning_utf8_chunks=chunks,planning_unique_utf8_chunks=len(chunk_hashes),embedding_vectors_created=0,source_fields_not_raw_media=True)
    (WORK/'source-summary.json').write_text(json.dumps(source_only,indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='complete_source_index_built',source_counts=coverage['source_counts'],metadata_only=coverage['metadata_only_sources'],source_utf8_chunks=chunks)),flush=True)
    # Index is frozen and complete before reading public questions. Gold remains in a separate process.
    qpath=ROOT/'data/atm-schema/atm-bench.json.questions.json';qraw=qpath.read_bytes();questions=json.loads(qraw)
    assert len(questions)==1013 and all(set(q)=={'id','question','qtype'} for q in questions)
    answered=0;latencies=[];returned=[];zero=[]
    with (WORK/'retrieval-results.jsonl').open('x',encoding='utf-8') as target:
        for q in questions:
            if time.monotonic()-start>600:raise TimeoutError('bounded N0 CPU evaluation deadline')
            tick=time.monotonic();result=index.search(q['question'],100);elapsed=(time.monotonic()-tick)*1000
            target.write(json.dumps(dict(id=q['id'],qtype=q['qtype'],ranked_ids=[i for i,_ in result],scores=[v for _,v in result],elapsed_ms=elapsed))+'\n')
            answered+=1;latencies.append(elapsed);returned.append(len(result))
            if not result:zero.append(q['id'])
            if answered%100==0:
                target.flush();(ROOT/'reports/atm-n0-progress.json').write_text(json.dumps(dict(stage='retrieval',questions=answered,source_count=len(records),provider_requests=0)),encoding='utf-8')
                print(json.dumps(dict(stage='retrieval',questions=answered,provider_requests=0)),flush=True)
    assert answered==1013
    report=dict(state='retrieval_complete_gold_not_loaded',finished_at_utc=datetime.now(timezone.utc).isoformat(),
        source_count=len(records),source_summary=source_only,questions=answered,qtypes=dict(Counter(q['qtype'] for q in questions)),
        question_public_sha256=hashlib.sha256(qraw).hexdigest(),retrieval_result_sha256=hashlib.sha256((WORK/'retrieval-results.jsonl').read_bytes()).hexdigest(),
        source_index_sha256=hashlib.sha256((WORK/'source-index.sqlite').read_bytes()).hexdigest(),
        frozen_config_sha256=hashlib.sha256((WORK/'FROZEN_CONFIG.json').read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-start,query_latencies_ms=latencies,returned_counts=returned,no_match_query_count=len(zero),
        provider_requests=0,gold_loaded=False,raw_media_loaded=False,model_answering=False,price_cny='0',
        no_query_or_gold_written_into_source_index=True)
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'reports/atm-n0-progress.json').write_text(json.dumps(dict(stage='retrieval_complete_waiting_isolated_scorer',questions=answered,source_count=len(records),provider_requests=0)),encoding='utf-8')
    print(json.dumps(dict(stage=report['state'],questions=answered,elapsed_seconds=report['elapsed_seconds'],provider_requests=0)),flush=True)

if __name__=='__main__':main()
