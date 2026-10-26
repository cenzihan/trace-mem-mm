"""Source-only baseline preparation contracts. No encoders, HTTP or credential access.

These are authored adapters inspired by pinned references, not upstream reproductions.
No current frozen retrieval or reader imports this module.
"""
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from .retrieval import rrf,ranked


@dataclass(frozen=True)
class ComparisonBudget:
    candidates:int=100
    evidence:int=20
    text_tokens:int=8192
    utf8_text_bytes:int=24000
    images:int=2
    output_tokens:int=768
    def validate(self):
        if self!=ComparisonBudget():raise ValueError('comparison budget must be frozen and identical across methods')


def source_views(record,*,condition):
    """At most one joint + two fine views per native artifact, no invented sessions.

    short/detail are reviewed SOURCE fields, never query hypotheses or gold answers.
    The caller must finish privacy/license checks before any future external encoding.
    """
    allowed={'namespace','native_id','modality','timestamp','short_text','detail_text','occurrence_ids','preprocessing_models'}
    if not isinstance(record,dict) or set(record)!=allowed:raise ValueError('source-only artifact fields required; query/gold forbidden')
    for key in ('namespace','native_id','modality','timestamp','short_text','detail_text'):
        if not isinstance(record[key],str):raise ValueError('explicit native source strings required')
    if not record['namespace'] or not record['native_id'] or record['modality'] not in ('image','video','email'):
        raise ValueError('explicit namespace/native ID/modality required')
    occurrences=record['occurrence_ids'];models=record['preprocessing_models']
    if not isinstance(occurrences,list) or any(not isinstance(v,str) or not v for v in occurrences) or len(set(occurrences))!=len(occurrences):
        raise ValueError('unique media occurrence references required')
    if not isinstance(models,list) or any(not isinstance(v,str) or not v for v in models):raise ValueError('declared preprocessing provenance required')
    if condition not in ('strict_from_reviewed_source','external_sgm_research'):raise ValueError('explicit comparison condition required')
    if condition=='strict_from_reviewed_source' and any(v not in ('gpt-4o-mini','gpt-4o-mini-2024-07-18') for v in models):
        raise ValueError('external preprocessing cannot be relabeled strict')
    if condition=='external_sgm_research' and not models:raise ValueError('external SGM model provenance must be disclosed')
    pieces=[('short',record['short_text']),('detail',record['detail_text'])]
    pieces=[(kind,text) for kind,text in pieces if text.strip()]
    if not pieces:raise ValueError('empty native artifact; do not silently omit source')
    joint='\n'.join(text for _,text in pieces)
    if any(len(text.encode('utf-8'))>8192 for _,text in [('joint',joint)]+pieces):raise ValueError('explicit embedding admission exceeded; no silent clipping')
    output=[]
    for kind,text in [('joint',joint)]+pieces:
        identity=hashlib.sha256(('\0'.join((record['namespace'],record['native_id'],kind))).encode()).hexdigest()
        output.append(dict(unit_id='unit_'+identity,namespace=record['namespace'],native_id=record['native_id'],kind=kind,text=text,
            modality=record['modality'],timestamp=record['timestamp'],occurrence_ids=list(occurrences),condition=condition,
            preprocessing_models=list(models)))
    return output


def fuse_to_native_candidates(lanes,units,namespace,budget=ComparisonBudget()):
    """RRF fine units then max-score native parent collapse, one global pool cap.

    Neighbor/anchor generation and actual dense indexes are separate future hooks.
    Original native evidence IDs, never view IDs, define evaluation denominators.
    """
    budget.validate()
    byid={r['unit_id']:r for r in units}
    if len(byid)!=len(units):raise ValueError('duplicate unit identity')
    if not lanes or any(len(lane)!=len(set(lane)) for lane in lanes):raise ValueError('nonempty unique ranked lanes required')
    for lane in lanes:
        for identity in lane:
            if identity not in byid or byid[identity]['namespace']!=namespace:raise ValueError('cross-namespace or unknown unit')
    scores=rrf(lanes,[1.]*len(lanes),60);parents={}
    for uid,score in scores.items():
        native=byid[uid]['native_id'];parents[native]=max(parents.get(native,0),score)
    order=ranked(parents)[:budget.candidates]
    return dict(candidates=order,evidence=order[:budget.evidence],native_evidence_count=len(order),
                unit_count=len(scores),query_hypotheses_persisted=False,method='prepared_adapter_not_full_upstream_reproduction')


def estimate_atm_plan(*,native_artifacts,questions,methods=3,anchors=True):
    """Conservative planning only. Never reserves or dispatches a real request."""
    for value in (native_artifacts,questions,methods):
        if type(value) is not int or value<1:raise ValueError('positive bounded plan counts required')
    if native_artifacts>11034 or questions>1013 or methods>3:raise ValueError('metadata plan bounds exceeded')
    source=(3*native_artifacts+9)//10 # shared joint view + at most two fine views
    query=(questions+9)//10;anchor_embeddings=query if anchors else 0;anchor_calls=questions if anchors else 0
    reader=questions*methods;judge=(reader+4)//5
    embedding=source+query+anchor_embeddings;llm=anchor_calls+reader+judge
    held=Decimal('.05')*embedding+Decimal('.25')*llm
    return dict(source_embedding_calls=source,query_embedding_calls=query,anchor_embedding_calls=anchor_embeddings,
        query_anchor_calls=anchor_calls,reader_calls=reader,proxy_judge_calls=judge,provider_request_upper_bound=embedding+llm,
        cumulative_preflight_cny=str(held),actual_bill_estimate=False,planning_only=True,new_approval_required=True,
        existing_completed_guard_reusable=False,condition='external_sgm_research_not_strict_end_to_end',
        complete_corpus_required=True,oversize_or_missing_source_blocks_dispatch=True)
