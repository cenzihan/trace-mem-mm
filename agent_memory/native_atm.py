"""ATM native preparation boundary. No network, model calls, or service-memory writes."""
from pathlib import PurePosixPath
import unicodedata


def split_native_qa(value):
    """Preparation-only: public questions and isolated gold are separate results.

    Answers, evidence labels and NIAH pool labels are never memory payloads.
    The caller must keep the second return value outside the memory/reader process.
    """
    rows=value.get('qas') if isinstance(value,dict) else value
    if not isinstance(rows,list):raise ValueError('ATM native QA array required')
    questions=[];gold=[];seen=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('ATM QA object required')
        identity=row.get('id');question=row.get('question');answer=row.get('answer');evidence=row.get('evidence_ids')
        if not isinstance(identity,str) or not identity or identity in seen:raise ValueError('unique native question id required')
        if not isinstance(question,str) or not question.strip() or not isinstance(answer,str):raise ValueError('native question/answer text required')
        if not isinstance(evidence,list) or any(not isinstance(v,str) or not v for v in evidence) or len(evidence)!=len(set(evidence)):
            raise ValueError('unique native evidence labels required')
        qtype=row.get('qtype','unspecified')
        if not isinstance(qtype,str):raise ValueError('native qtype must be text')
        pool=row.get('niah_evidence_ids')
        if pool is not None and (not isinstance(pool,list) or any(not isinstance(v,str) or not v for v in pool) or len(pool)!=len(set(pool)) or not set(evidence)<=set(pool)):
            raise ValueError('NIAH fixed pool must contain all gold evidence')
        seen.add(identity);questions.append(dict(id=identity,question=question,qtype=qtype))
        labels=dict(id=identity,answer=answer,evidence_ids=list(evidence))
        if pool is not None:labels['niah_evidence_ids']=list(pool)
        gold.append(labels)
    return questions,gold


def corpus_inventory(paths):
    """Inventory all native media paths; unsupported modalities are explicit.

    Does not generate captions or use processed_memory/SGM. The email bundle is
    counted as one file until a separately reviewed loader reads its actual entries.
    """
    records=[];seen=set();ids=set();counts={k:0 for k in ('image','video','audio','email_bundle','geocoding','external_processed','unsupported')}
    for name in paths:
        if not isinstance(name,str) or '\\' in name or ':' in name:raise ValueError('native POSIX relative path required')
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or name in seen:raise ValueError('duplicate or escaping native path')
        seen.add(name)
        if name.startswith('data/raw_memory/image/'):
            modality='image' if path.suffix.lower() in ('.jpg','.jpeg','.png','.webp') else 'unsupported'
        elif name.startswith('data/raw_memory/video/'):
            modality='video' if path.suffix.lower()=='.mp4' else 'unsupported'
        elif name.startswith('data/raw_memory/audio/'):modality='audio'
        elif name=='data/raw_memory/email/emails.json':modality='email_bundle'
        elif name.startswith('data/raw_memory/geocoding_cache/'):modality='geocoding'
        elif name.startswith('data/processed_memory/'):modality='external_processed'
        else:continue
        counts[modality]+=1
        row=dict(path=name,modality=modality)
        if modality in ('image','video','audio'):
            identity=path.stem
            if identity in ids:raise ValueError('native media evidence-id collision; do not merge occurrences')
            ids.add(identity);row['evidence_id']=identity
        records.append(row)
    return dict(counts=counts,records=records,raw_media_complete=False,email_entries_not_read=True,
                external_sgm_never_used=True,unsupported_modalities=[k for k in ('video','audio','unsupported') if counts[k]])


def split_overlap(regular_questions,hard_questions,niah_questions):
    """Check IDs AND text: different IDs cannot establish an independent holdout."""
    regular={q['id']:q['question'] for q in regular_questions};hard={q['id']:q['question'] for q in hard_questions}
    normalize=lambda text:' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    regular_text=set(regular.values());regular_normalized={normalize(v) for v in regular.values()}
    result=dict(regular=len(regular),hard=len(hard),hard_ids_in_regular=len(set(hard)&set(regular)),
                hard_same_question_in_regular=sum(regular.get(k)==v for k,v in hard.items()),
                hard_exact_question_text_in_regular=sum(v in regular_text for v in hard.values()),
                hard_normalized_question_text_in_regular=sum(normalize(v) in regular_normalized for v in hard.values()),
                independence_not_established=True,niah={})
    for name,questions in niah_questions.items():
        current={q['id']:q['question'] for q in questions}
        result['niah'][name]=dict(count=len(current),ids_in_hard=len(set(current)&set(hard)),
                                same_question_in_hard=sum(hard.get(k)==v for k,v in current.items()))
    return result
