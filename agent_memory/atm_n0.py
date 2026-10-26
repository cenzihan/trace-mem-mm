"""Local-only native SGM flat BM25. No encoder, model, network or credential hooks."""
import heapq
import math
from collections import Counter,defaultdict
from pathlib import PurePosixPath
from .retrieval import tokenize

FORBIDDEN={'question','answer','evidence_ids','niah_evidence_ids','clue','gold','qas'}

def _text(row,key):
    value=row.get(key,'')
    if value is None:return ''
    if not isinstance(value,str):raise ValueError('source text field has unsupported type')
    return value

def native_sgm_corpus(images,videos,emails,*,expected_images,expected_videos,expected_emails):
    """Keep every native source, including metadata-only items; never clip local text.

    Selected fields follow the pinned ATM text formatter. No generated entities,
    QA, gold or arbitrary extra fields are incorporated. Source paths identify a
    whitelisted filename stem only and are never opened or fetched.
    """
    if any(not isinstance(rows,list) for rows in (images,videos,emails)):raise ValueError('native source lists required')
    output=[];seen=set();coverage={};empty={}
    for modality,rows,path_key,expected in (
        ('image',images,'image_path',set(expected_images)),('video',videos,'video_path',set(expected_videos)),('email',emails,'id',None)):
        actual=set();semantic_empty=0
        for row in rows:
            if not isinstance(row,dict) or FORBIDDEN&set(row):raise ValueError('source-only object required; QA/gold forbidden')
            label=_text(row,path_key)
            if not label:raise ValueError('native source identity missing')
            if modality=='email':identity=label
            else:
                if label.lower().startswith(('http:','https:')):raise ValueError('unexpected remote source path label')
                identity=PurePosixPath(label.replace('\\','/')).stem
                if identity not in expected:raise ValueError('SGM source outside pinned native inventory')
            if not identity or identity in seen:raise ValueError('duplicate native evidence ID; no overwrite')
            seen.add(identity);actual.add(identity);timestamp=_text(row,'timestamp')
            if modality=='email':
                summary=_text(row,'short_summary');detail=_text(row,'detail')
                body='\n'.join(('ID: '+identity,'Timestamp: '+timestamp,'Summary: '+summary,'Detail: '+detail))
                meaningful=summary.strip() or detail.strip()
            else:
                tags=row.get('tags',[]) or []
                if not isinstance(tags,list) or any(not isinstance(v,str) for v in tags):raise ValueError('native tags require text list')
                short=_text(row,'short_caption');caption=_text(row,'caption');ocr=_text(row,'ocr_text')
                body='\n'.join(('ID: '+identity,'Type: '+modality,'Timestamp: '+timestamp,'Location: '+_text(row,'location_name'),
                    'Short Caption: '+short,'Caption: '+caption,'OCR: '+ocr,'Tags: '+', '.join(tags)))
                meaningful=short.strip() or caption.strip() or ocr.strip() or tags
            semantic_empty+=not bool(meaningful)
            output.append(dict(id=identity,modality=modality,text=body,timestamp=timestamp,description_present=bool(meaningful)))
        if expected is not None and actual!=expected:raise ValueError('incomplete source inventory; no full-corpus claim')
        if modality=='email' and len(actual)!=expected_emails:raise ValueError('email count mismatch')
        coverage[modality]=len(actual);empty[modality]=semantic_empty
    return output,dict(source_counts=coverage,metadata_only_sources=empty,local_text_not_truncated=True,
        provider_calls=0,raw_media_loaded=False,entities_field_indexed=False)


class NativeBM25:
    """Precompute postings once; same k1/b/IDF formula as the existing sparse baseline."""
    def __init__(self,documents,*,k1=1.2,b=.75):
        if k1!=1.2 or b!=.75:raise ValueError('N0 parameters are frozen, not fitted')
        self.k1=k1;self.b=b;self.lengths={};self.postings=defaultdict(list);self.count=len(documents)
        for identity,text in documents.items():
            if not isinstance(identity,str) or not identity or not isinstance(text,str):raise ValueError('source identity/text required')
            counts=Counter(tokenize(text));self.lengths[identity]=sum(counts.values())
            for term,frequency in counts.items():self.postings[term].append((identity,frequency))
        self.average=sum(self.lengths.values())/max(1,self.count) or 1
    def search(self,query,top_k=100):
        if not isinstance(query,str) or type(top_k) is not int or not 1<=top_k<=100:raise ValueError('bounded original query required')
        scores=defaultdict(float)
        for term in sorted(set(tokenize(query))):
            posting=self.postings.get(term,());df=len(posting)
            if not df:continue
            idf=math.log(1+(self.count-df+.5)/(df+.5))
            for identity,frequency in posting:
                length=self.lengths[identity]
                scores[identity]+=idf*frequency*(self.k1+1)/(frequency+self.k1*(1-self.b+self.b*length/self.average))
        order=heapq.nsmallest(top_k,scores,key=lambda identity:(-scores[identity],identity))
        return [(identity,scores[identity]) for identity in order]
