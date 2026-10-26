"""Source-only ATM SGM adaptation. No disk reads, external calls or image-path fetches.

Inputs are native lists, as required by the pinned upstream MMRAG entrypoint.
This external-generated research condition is never relabeled strict end-to-end.
"""
from pathlib import PurePosixPath
from .baseline_preparation import source_views

FORBIDDEN={'question','answer','evidence_ids','niah_evidence_ids','clue','gold','qas'}

def text(row,key):
    value=row.get(key,'')
    if value is None:return ''
    if not isinstance(value,str):raise ValueError('declared source text field must be text')
    return value

def prepare_sgm_sources(images,videos,emails,*,namespace,expected_images,expected_videos,expected_email_count,preprocessing_models):
    if any(not isinstance(rows,list) for rows in (images,videos,emails)):raise ValueError('native batch results/email lists required')
    if not isinstance(namespace,str) or not namespace:raise ValueError('explicit task/user namespace required')
    if type(expected_email_count) is not int or expected_email_count<1:raise ValueError('explicit expected full email count required')
    if not isinstance(preprocessing_models,list) or not preprocessing_models:raise ValueError('external generator provenance required')
    records=[];seen=set();observed_images=set();observed_videos=set();email_count=0
    def add(identity,modality,timestamp,short,detail,occurrences,models):
        if not identity or identity in seen:raise ValueError('duplicate native evidence ID; no overwrite')
        seen.add(identity)
        record=dict(namespace=namespace,native_id=identity,modality=modality,timestamp=timestamp,short_text=short,detail_text=detail,
                    occurrence_ids=occurrences,preprocessing_models=models)
        # Fail before any future encoding if a source is empty/oversize or provenance ambiguous.
        source_views(record,condition='external_sgm_research');records.append(record)
    for modality,rows,path_key,expected,observed in (
        ('image',images,'image_path',set(expected_images),observed_images),
        ('video',videos,'video_path',set(expected_videos),observed_videos)):
        for row in rows:
            if not isinstance(row,dict) or FORBIDDEN&set(row):raise ValueError('QA/gold cannot enter SGM evidence')
            original=text(row,path_key)
            if not original or original.lower().startswith(('http:','https:')):raise ValueError('native source path label required, never remote URL')
            identity=PurePosixPath(original.replace('\\','/')).stem
            if identity not in expected:raise ValueError('source identity not in full pinned native inventory')
            tags=row.get('tags',[]) or []
            if not isinstance(tags,list) or any(not isinstance(v,str) for v in tags):raise ValueError('native tags must be text list')
            timestamp=text(row,'timestamp');location=text(row,'location_name') or text(row,'location')
            short='\n'.join(v for v in (text(row,'short_caption'),text(row,'caption')) if v.strip())
            detail='\n'.join(('Timestamp: '+timestamp,'Location: '+location,'OCR: '+text(row,'ocr_text'),'Tags: '+', '.join(tags)))
            if not short.strip() and not text(row,'ocr_text').strip() and not tags:raise ValueError('empty generated source; do not count metadata as visual understanding')
            # Preserve original media as one native source. Frames are separately materialized later.
            add(identity,modality,timestamp,short,detail,[identity+':original-media'],list(preprocessing_models))
            observed.add(identity)
        if observed!=expected:raise ValueError('partial native corpus is not a full evaluation corpus')
    for row in emails:
        if not isinstance(row,dict) or FORBIDDEN&set(row):raise ValueError('QA/gold cannot enter email evidence')
        identity=text(row,'id')
        if not identity.startswith('email'):raise ValueError('native email identity required')
        add(identity,'email',text(row,'timestamp'),text(row,'short_summary'),text(row,'detail'),[],['author-provided summarized email; generator unverified'])
        email_count+=1
    if email_count!=expected_email_count:raise ValueError('email corpus count mismatch; no silent omission')
    return dict(records=records,source_counts=dict(image=len(observed_images),video=len(observed_videos),email=email_count),
                condition='external_sgm_research',academic_strict_end_to_end=False,media_loaded=False,
                privacy_review_complete=False,external_encoding_enabled=False,gold_imported=False)
