"""Bounded image/text scoring views on the original hosted journal and ceiling."""
import io
import json

from .bailian_rerank import (HostedRerankTransport, RerankJSONTransport, MODEL,
                            INSTRUCTION, MAX_INPUT)
from .caption_image import caption_part, PREVIEW_BYTES, PREVIEW_EDGE
from .protocol import ContractError, image_bytes, text_of
from .provider_http import BAILIAN_RERANK_ENDPOINT

MAX_SOURCES = 40
MAX_VIEWS = 80
MAX_BATCHES = 24
MAX_REQUEST_BYTES = 3 * 1024 * 1024
TEXT_VIEW_BYTES = 6000


def text_prefix(value, limit):
    """Scoring-only UTF-8 prefix; returned originals are never truncated."""
    return value.encode()[:limit].decode('utf-8', errors='ignore')


def query_views(parts):
    text = text_of(parts)
    if len(text.encode()) > 4000:
        raise ContractError('visual rerank query capacity:4000 UTF-8 bytes', 413)
    result = [{'text': text}] if text.strip() else []
    first_image = next((p for p in parts if p['type'] == 'image_url'), None)
    if first_image:
        result.append({'image': caption_part(first_image)['image_url']['url']})
    if not result:raise ContractError('nonempty scoring query required')
    return result


def source_views(parts):
    text = text_prefix(text_of(parts), TEXT_VIEW_BYTES)
    result = [{'text': text}] if text.strip() else []
    first_image = next((p for p in parts if p['type'] == 'image_url'), None)
    if first_image:
        result.append({'image': caption_part(first_image)['image_url']['url']})
    return result


def view_bound(view, *, query=False):
    if not isinstance(view, dict) or len(view) != 1:
        raise ValueError('one explicit text or image scoring view required')
    if 'text' in view:
        value=view['text'];limit=4000 if query else TEXT_VIEW_BYTES
        if not isinstance(value,str) or not value.strip() or len(value.encode())>limit:
            raise ValueError('text scoring view exceeds reviewed bound')
        return len(value.encode())
    if set(view) != {'image'}:raise ValueError('unsupported scoring view')
    raw,mime=image_bytes(view['image'])
    if mime!='jpeg' or len(raw)>PREVIEW_BYTES:
        raise ValueError('bounded JPEG scoring view required')
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if max(image.size)>PREVIEW_EDGE or image.format!='JPEG' or getattr(image,'is_animated',False):
            raise ValueError('image scoring dimensions exceed reviewed bound')
        image.verify()
    # Conservative per-item API maximum, not an invented measured image usage.
    return 8000


def visual_payloads(query, documents):
    qbound=view_bound(query,query=True)
    if not isinstance(documents,list) or not 1<=len(documents)<=MAX_VIEWS:
        raise ValueError('bounded scoring documents required')
    batches=[];batch=[];units=0
    def payload(values):
        return dict(model=MODEL,input=dict(query=query,documents=values),
                    parameters=dict(top_n=len(values),return_documents=False,instruct=INSTRUCTION))
    for document in documents:
        cost=qbound+view_bound(document)+len(INSTRUCTION.encode())+256
        candidate=payload(batch+[document])
        if batch and (len(batch)>=8 or units+cost>MAX_INPUT or
                      len(json.dumps(candidate,ensure_ascii=False).encode())>MAX_REQUEST_BYTES):
            batches.append(payload(batch));batch=[];units=0
        batch.append(document);units+=cost
        if units>MAX_INPUT or len(json.dumps(payload(batch),ensure_ascii=False).encode())>MAX_REQUEST_BYTES:
            raise ValueError('single scoring pair exceeds reviewed bound')
    if batch:batches.append(payload(batch))
    return batches


class VisualRerankJSONTransport(RerankJSONTransport):
    def _bounds(self,payload):
        if not isinstance(payload,dict) or set(payload)!={'model','input','parameters'} or payload['model']!=MODEL:
            raise ValueError('explicit approved hosted model required')
        value=payload['input']
        if not isinstance(value,dict) or set(value)!={'query','documents'}:
            raise ValueError('explicit visual query and document views required')
        expected=visual_payloads(value['query'],value['documents'])
        if len(expected)!=1 or payload!=expected[0]:
            raise ValueError('visual batch or parameters exceed reviewed contract')
        return MAX_INPUT,0


class VisualHostedTransport(HostedRerankTransport):
    transport_type = VisualRerankJSONTransport

    def preflight(self, batches):
        if not 1<=len(batches)<=MAX_BATCHES:raise ValueError('whole-search scoring call cap exceeded')
        for payload in batches:VisualRerankJSONTransport._bounds(None,payload)
        with self.lock:
            self.ready()
            with self.journal.connect() as db:
                used=db.execute('SELECT count(*) FROM operations').fetchone()[0]
            if used+len(batches)>self.call_limit:
                raise ValueError('insufficient original hosted lifetime calls')
            # Existing fixed rate and120000-token full reservation per call.
            if self.ledger.totals()['remaining_micro_cny']<len(batches)*60000:
                raise ValueError('insufficient original money for whole scoring search')


class VisualScorer:
    model=MODEL
    def __init__(self,transport):self.transport=transport
    def ready(self):self.transport.ready()
    def score(self,parts,sources):
        if not 1<=len(sources)<=MAX_SOURCES:raise ValueError('bounded original source pool required')
        owners=[];documents=[]
        for index,source in enumerate(sources):
            for view in source_views(source['content']):
                owners.append(index);documents.append(view)
        if not documents:raise ValueError('source scoring views are empty')
        batches=[]
        for query in query_views(parts):batches.extend(visual_payloads(query,documents))
        self.transport.preflight(batches)
        scores=[0.0]*len(sources)
        # Each query traverses the same complete document-view list.
        cursor=0
        for payload in batches:
            response=self.transport(BAILIAN_RERANK_ENDPOINT,payload)
            RerankJSONTransport._contract(None,response,payload)
            for item in response['output']['results']:
                owner=owners[(cursor+item['index'])%len(owners)]
                scores[owner]=max(scores[owner],item['relevance_score'])
            cursor=(cursor+len(payload['input']['documents']))%len(owners)
        return scores

