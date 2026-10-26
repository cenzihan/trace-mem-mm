"""Read-only cumulative accounting after the separate corrected eight judges.

All three real chat scopes, the known rejected HTTP400 and local-only preflight
remain separately counted. No scoring, private mapping parsing, repair or API.
"""
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_memgallery_v71_chat as original
from scripts import ack_memgallery_v71_json400 as rejection
from scripts.audit_memgallery_v7_encoding import canonical,digest,sha,load,lines,ro,history,require
from agent_memory.memgallery_v7_reader import proxy_judge_payload,decode_proxy_judge
from agent_memory.visual_recovery import _strict_json

WORK=Path('runtime/memgallery-v7/judge-json-v71')
REPORT=Path('reports/memgallery-v71-judge-json-accounting-audit.json')


def hashes(root,values):
    require(isinstance(values,dict) and bool(values),'nonempty frozen hashes required')
    for relative,expected in values.items():
        path=(root/relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(root) and sha(path)==expected,'frozen artifact changed')


def bound_receipts(root,work,namespace,work_items,calls,ledger_first):
    """Validate one contiguous successful phase without inferring retry safety."""
    with closing(ro(work/'receipts/private-embedding.sqlite')) as db:
        rows=[dict(r) for r in db.execute('SELECT * FROM operations ORDER BY run_ordinal')]
        claims=[tuple(r) for r in db.execute('SELECT scope,chunk_key,fingerprint FROM claims')]
    if work==root/rejection.CHAT:
        require(len(rows)==65,'original65 receipts required');rows=rows[:64]
        claims=[c for c in claims if c[2] in {r['fingerprint'] for r in rows}]
    require(len(rows)==len(work_items),'all phase receipts required')
    expected_claims=[];usd_total=Decimal(0);invalid=0;attempts=[]
    with closing(ro(root/'runtime/development-budget.sqlite')) as ledger:
        for ordinal,(row,item) in enumerate(zip(rows,work_items),1):
            phase,rid,payload,pack,saved=item
            manifest=dict(keys=[rid],payload_sha256=digest(canonical(payload).encode()),run_path=str((work/'run.sqlite').resolve()))
            fingerprint=digest(canonical(dict(namespace=namespace,manifest=manifest)).encode())
            require(row['fingerprint']==fingerprint and row['scope']==namespace and row['manifest']==canonical(manifest)
                    and row['run_path']==manifest['run_path'] and row['run_ordinal']==ordinal and row['state']=='complete'
                    and row['status']==200 and row['redacted']==0 and row['transport_completed']==1,'completed bound HTTP200 required')
            require(saved['request_id']==rid and saved['run_ordinal']==ordinal and saved['attempt_id']==row['attempt_id']
                    and saved['receipt_sha256']==row['body_sha']==digest(row['body']),'saved output differs from receipt')
            response=_strict_json(row['body']);original.prior.original_provider_contract(response)
            require(response['model']==calls[ordinal]['returned_model'],'guard returned model differs')
            if phase=='reader':
                decoded=original.projection(response,pack)
                require(decoded==saved['answer'],'reader decoder output differs')
                invalid+=decoded['schema_status']!='valid'
            else:
                require(saved['response']==response,'judge response changed')
                decode_proxy_judge(response,pack['item_ids'])
            usage=response.get('usage',{});inp,out=usage.get('prompt_tokens'),usage.get('completion_tokens')
            require(type(inp) is int and type(out) is int and 0<=inp<=128000 and 0<=out<=payload['max_completion_tokens']
                    and usage.get('total_tokens')==inp+out,'invalid receipt token usage')
            usd=(Decimal(inp)*Decimal('.15')+Decimal(out)*Decimal('.60'))/1000000
            found=ledger.execute('SELECT rowid AS original_rowid,* FROM attempts WHERE id=?',(row['attempt_id'],)).fetchone()
            observed=ledger.execute('SELECT * FROM provider_observations WHERE attempt_id=?',(row['attempt_id'],)).fetchone()
            require(found is not None and observed is not None and found['original_rowid']==ledger_first+ordinal-1
                    and found['model']=='gpt-4o-mini' and found['status']=='uncertain'
                    and found['amount_kind']=='retained_upper_bound' and found['conversion_basis']=='budget_assumption_not_market_fx'
                    and found['reserved']==found['accounted']==250000 and found['max_input']==128000
                    and found['max_output']==payload['max_completion_tokens'] and found['input_tokens']==inp and found['output_tokens']==out
                    and Decimal(found['input_rate'])==Decimal('1.5') and Decimal(found['output_rate'])==Decimal('6'),'ledger full hold/usage differs')
            require(observed['mode']=='live' and observed['category']=='success' and observed['http_status']==200
                    and observed['returned_model']==response['model'] and observed['input_tokens']==inp and observed['output_tokens']==out
                    and Decimal(observed['estimated_usd'])==usd,'provider usage observation differs')
            expected_claims.append((namespace,rid,fingerprint));usd_total+=usd;attempts.append(row['attempt_id'])
    require(sorted(expected_claims)==sorted(claims) and len(set(attempts))==len(work_items),'receipt claims are not one-to-one')
    return usd_total,invalid,attempts


def audit(root=ROOT):
    root=Path(root).resolve();work=root/WORK;old=root/rejection.CHAT
    completion=load(work/'JUDGE_COMPLETED.json');progress_path=root/'reports/memgallery-v71-judge-json-progress.json'
    progress=load(progress_path)
    require(completion.get('state')=='complete' and completion.get('phase')=='judge' and completion.get('count')==8
            and progress.get('state')=='judge_complete' and progress.get('phase')=='judge','wait for eight completed corrected judges')
    evidence=rejection.inspect(root);report=load(root/rejection.REPORT)
    require(evidence['acknowledged'] and report['state']=='passed' and report['http_status']==400
            and report['guard_after_sha256']==evidence['guard_sha256'] and report['ledger_prefix_sha256']==evidence['ledger_history_sha256']
            and report['real_http_requests']==1 and report['model_grades_generated']==0,'known400 acknowledgement changed')
    hashes(root,report['artifact_sha256'])
    previous_report=load(root/original.OLD_AUDIT);hashes(root,previous_report['artifact_sha256'])
    previous=original.prior.audit_reader_only(root)
    for key in ('total_real_provider_requests','chat_usage_estimate_usd','embedding_usage_estimate_cny','combined_cumulative_preflight_micro_cny'):
        require(previous[key]==previous_report[key],'original failed64 accounting changed')
    require(previous['total_real_provider_requests']==91 and previous['combined_cumulative_preflight_micro_cny']==17350000,'original27+64 missing')
    frozen,live=load(work/'FROZEN_JUDGE.json'),load(work/'FROZEN_LIVE.json')
    require(frozen['judge_requests']==8 and frozen['reader_answers']==64 and frozen['grades']==[0,.25,.5,.75,1]
            and frozen['max_requests']==8 and frozen['max_preflight_cny']=='2' and frozen['fixed_hold_cny']=='.25'
            and live['baseline']['attempts']==1612 and live['judge_sha256']==sha(work/'FROZEN_JUDGE.json')
            and live['max_requests']==8 and live['max_preflight_cny']=='2' and live['automatic_retries'] is False
            and live['old_guard_reopened'] is False,'independent eight/CNY2 correction scope changed')
    hashes(root,frozen['input_hashes']);hashes(root,frozen['code_hashes']);hashes(root,live['code_hashes'])
    require(completion['frozen_live_sha256']==sha(work/'FROZEN_LIVE.json')
            and completion['judge_frozen_sha256']==sha(work/'FROZEN_JUDGE.json')
            and completion['output_sha256']==sha(work/'judge-results.jsonl'),'new completion not bound')
    require((root/frozen['requests_file']).resolve()==work/'judge-requests.jsonl'
            and frozen['requests_sha256']==sha(work/'judge-requests.jsonl'),'corrected frozen request hash changed')
    mapping=(root/frozen['mapping_file']).resolve()
    require(mapping.is_relative_to(work/'scorer-only') and sha(mapping)==frozen['mapping_sha256'],'mapping hash differs')
    old_requests=lines(old/'judge-requests.jsonl');requests=lines(work/'judge-requests.jsonl');results=lines(work/'judge-results.jsonl')
    require(len(old_requests)==len(requests)==len(results)==8,'exact eight original/corrected batches required')
    new_items=[]
    for before,current,saved in zip(old_requests,requests,results):
        expected=proxy_judge_payload(before['pack']);expected['messages'][0]['content']+='\nReturn JSON.'
        require(current['payload']==expected and current['request_id']!=before['request_id'],'only explicit JSON suffix may change')
        require(current['request_id']==digest(canonical(['judge-json-v71',before['request_id'],expected]).encode()),'new corrected request identity differs')
        new_items.append(('judge',current['request_id'],expected,before['pack'],saved))
    packs=lines(old/'reader-packs.jsonl');answers=lines(old/'reader-answers.jsonl')
    alias=original.same_inputs(lines(root/original.OLD/'reader-packs.jsonl'),packs)
    require(len(packs)==len(answers)==64,'original64 reader outputs required')
    old_items=[]
    for meta,saved in zip(packs,answers):
        require(all(meta[k]==saved[k] for k in ('query_id','index','point','variant')),'reader metadata differs')
        rid=digest((meta['query_id']+'\0'+meta['variant']).encode())
        old_items.append(('reader',rid,original.native.reader_payload(meta['pack']),meta['pack'],saved))
    with closing(ro(old/'run.sqlite')) as db:old_calls={r['ordinal']:dict(r) for r in db.execute('SELECT * FROM calls WHERE ordinal<=64')}
    calls=original.prior.complete_guard(work/'run.sqlite',8,2000000,'gpt-4o-mini',maximum=8)
    files=[work/n for n in ('FROZEN_JUDGE.json','FROZEN_LIVE.json','judge-requests.jsonl','judge-results.jsonl','JUDGE_COMPLETED.json','run.sqlite','receipts/private-embedding.sqlite')]
    files += [mapping,progress_path,root/rejection.REPORT,root/original.OLD_AUDIT]
    before={p.relative_to(root).as_posix():sha(p) for p in files}
    old_usd,invalid,old_attempts=bound_receipts(root,old,'memgallery-v7:'+sha(old/'FROZEN_LIVE.json'),old_items,old_calls,1548)
    judge_usd,_,new_attempts=bound_receipts(root,work,'memgallery-v71-judge-json:'+sha(work/'FROZEN_LIVE.json'),new_items,calls,1613)
    require(len(set(old_attempts+new_attempts+[evidence['operation']['attempt_id']]))==73,'cross-scope attempts reused')
    with closing(ro(root/'runtime/development-budget.sqlite')) as db:
        require(history(db,1612)==live['historical_sha256']==report['ledger_prefix_sha256'],'historical1612 including rejected400 changed')
        final_history=history(db,1620)
    require(before=={p:sha(root/p) for p in before},'new artifacts changed during read-only audit')
    hashes(root,report['artifact_sha256']);hashes(root,previous_report['artifact_sha256'])
    previous_usd=Decimal(previous['chat_usage_estimate_usd']);embedding=Decimal(previous['embedding_usage_estimate_cny'])
    return dict(schema='memgallery-v71-json-judge-accounting-audit-v1',state='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),
        original_reader_requests=64,new_alias_reader_requests=64,new_judge_requests=8,rejected_judge_http_requests=1,embedding_requests=27,
        total_real_provider_requests=164,cumulative_provider_hold_micro_cny=35600000,
        original_local_preflight_provider_requests=0,original_local_preflight_claims=1,
        nominal_claims_including_local_preflight=165,nominal_hold_including_local_preflight_micro_cny=35850000,shared_budget_cny='500',
        rejected_request_usage_available=False,rejected_request_actual_usage_cost=None,rejected_request_retained_hold_micro_cny=250000,
        old_reader_usage_estimate_usd=str(previous_usd),new_reader_usage_estimate_usd=str(old_usd),corrected_judge_usage_estimate_usd=str(judge_usd),
        known_successful_chat_usage_estimate_usd=str(previous_usd+old_usd+judge_usd),embedding_usage_estimate_cny=str(embedding),
        known_successful_usage_estimate_cny_budget_assumption=str((previous_usd+old_usd+judge_usd)*10+embedding),
        usage_note='Known successful usage only; HTTP400 returned no usage, not assumed free. All provider holds retained separately.',
        old_reader_invalid_outputs_preserved=previous['invalid_reader_outputs_retained'],new_reader_invalid_outputs_retained=invalid,
        alias_map_sha256=alias,original_calls_replayed=0,old_guard_reopened=False,original1547_and1612_history_unchanged=True,
        ledger_prefix_rows=1620,ledger_prefix_sha256=final_history,artifact_sha256=before,
        rejected_old_scope_artifact_sha256=report['artifact_sha256'],script_sha256=sha(Path(__file__)))


def main():
    value=audit();path=ROOT/REPORT
    with path.open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n')
    print(canonical(dict(state=value['state'],report=REPORT.as_posix(),total_real_provider_requests=164)))


if __name__=='__main__':main()
