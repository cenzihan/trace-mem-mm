"""Real, pinned Mem0 OSS baseline with budgeted academic-model adapters.

Mem0 owns fact extraction, deduplication, entity links and retrieval. This bridge
provides lossless input chunking, image captions, source archives and API safety.
It never modifies the native prompts or answers Search questions.
"""
from contextvars import ContextVar
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import threading
import uuid

from .protocol import ContractError,canonical,content_parts,text_of,validate_add,validate_search
from .provider_http import OPENAI_ENDPOINT
from .store import MemoryStore

VERSION='tracemem-mm-mem0-v3.0.1'
MEM0_VERSION='2.2.1'
_adapters=ContextVar('mem0_budget_adapters')
_factory_lock=threading.Lock()


def utf8_chunks(text, limit):
    if not isinstance(text,str) or not text.strip():raise ValueError('nonempty text required')
    chunk=[]; size=0
    for char in text:
        width=len(char.encode())
        if size+width>limit:
            yield ''.join(chunk); chunk=[]; size=0
        chunk.append(char); size+=width
    if chunk:yield ''.join(chunk)


def native_scope(uid):
    # SDK trims identity strings; hashing prevents merging distinct API identities.
    return 'user_'+hashlib.sha256(uid.encode()).hexdigest()


class BudgetEmbedding:
    def __init__(self,config):
        self.config=config; self.embedding=_adapters.get()['embedding']; self.failed=False

    def embed_batch(self,texts,memory_action=None):
        if self.failed:raise RuntimeError('previous provider failure; no SDK fallback replay')
        if not texts:return []
        groups=[list(utf8_chunks(t,8000)) for t in texts]
        flat=[chunk for group in groups for chunk in group]
        vectors=[]
        try:
            for start in range(0,len(flat),10):vectors.extend(self.embedding.embed(flat[start:start+10]))
        except BaseException:
            self.failed=True; raise
        out=[]; cursor=0
        for group in groups:
            rows=vectors[cursor:cursor+len(group)]; cursor+=len(group)
            mean=[sum(v[i]*len(t.encode()) for v,t in zip(rows,group)) for i in range(256)]
            norm=math.sqrt(sum(x*x for x in mean))
            if not norm or not math.isfinite(norm):self.failed=True;raise ValueError('invalid pooled vector')
            out.append([x/norm for x in mean])
        return out

    def embed(self,text,memory_action=None):return self.embed_batch([text],memory_action)[0]


class BudgetLLM:
    def __init__(self,config):
        self.config=config; self.transport=_adapters.get()['transport']; self.failed=False

    def generate_response(self,messages,response_format=None,**kwargs):
        if self.failed:raise RuntimeError('previous LLM failure; no automatic replay')
        if kwargs or response_format!={'type':'json_object'}:raise ValueError('native additive extraction only')
        payload=dict(model='gpt-4o-mini',messages=messages,response_format=response_format,
            max_completion_tokens=2048,store=False,n=1)
        try:return self.transport(OPENAI_ENDPOINT,payload)['choices'][0]['message']['content']
        except BaseException:
            self.failed=True;raise


def create_native(root,embedding,transport):
    if importlib.metadata.version('mem0ai')!=MEM0_VERSION:raise ValueError('pinned Mem0 version required')
    os.environ['MEM0_TELEMETRY']='false'
    os.environ['MEM0_DIR']=str(root/'sdk-config')
    # Native SDK exception logging can include source text. No such logs escape.
    logger=logging.getLogger('mem0');logger.setLevel(logging.CRITICAL+1);logger.propagate=False
    logger.addHandler(logging.NullHandler())
    from mem0 import Memory
    from mem0.memory import main
    if main.MEM0_TELEMETRY:raise ValueError('Mem0 was imported before telemetry was disabled')
    from mem0.utils.factory import EmbedderFactory,LlmFactory
    from mem0.configs.llms.base import BaseLlmConfig
    config=dict(version='v1.1',history_db_path=str(root/'history.sqlite'),
        embedder={'provider':'openai','config':{'model':'text-embedding-v4','embedding_dims':256}},
        llm={'provider':'openai','config':{'model':'gpt-4o-mini','max_tokens':2048,'enable_vision':False}},
        vector_store={'provider':'qdrant','config':{'collection_name':'tracemem_mem0',
            'path':str(root/'qdrant'),'embedding_model_dims':256,'on_disk':True}})
    # Factories expose replacement hooks, but config validators restrict labels.
    # Bind the existing provider label only while constructing dedicated adapters.
    with _factory_lock:
        old_e=EmbedderFactory.provider_to_class['openai'];old_l=LlmFactory.provider_to_class['openai']
        token=_adapters.set(dict(embedding=embedding,transport=transport))
        try:
            EmbedderFactory.provider_to_class['openai']='agent_memory.mem0_bridge.BudgetEmbedding'
            LlmFactory.provider_to_class['openai']=('agent_memory.mem0_bridge.BudgetLLM',BaseLlmConfig)
            return Memory.from_config(config)
        finally:
            EmbedderFactory.provider_to_class['openai']=old_e;LlmFactory.provider_to_class['openai']=old_l
            _adapters.reset(token)


class Mem0System(MemoryStore):
    def __init__(self,path,*,embedding,transport,card_adapter):
        super().__init__(path)
        self.transport=transport;self.cards=card_adapter;self.lock=threading.RLock()
        root=Path(path).parent/'mem0-native';root.mkdir(parents=True,exist_ok=True)
        self.native=create_native(root,embedding,transport)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS add_operations(
                user_id TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL,
                state TEXT NOT NULL, owner_token TEXT NOT NULL,
                PRIMARY KEY(user_id,request_id))''')

    def close(self):self.native.vector_store.client.close()

    def capabilities(self):
        return dict(version=VERSION,mode='native-mem0',mem0_version=MEM0_VERSION,
            models=['gpt-4o-mini','text-embedding-v4'],embedding_dimensions=256,
            search_evidence='derived-memory-facts',originals_archived=True,
            vision='gpt-4o-mini-image-caption',telemetry=False,concurrency=1,
            admission_capacity=16,queue_wait_seconds=30,receipt_content_logging=False)

    def ready(self):
        if self.native.embedding_model.failed or self.native.llm.failed:
            raise ValueError('SDK provider failure requires operator reconciliation')
        self.transport.ready()

    def _describe(self,parts):
        text=text_of(parts);images=[p for p in parts if p['type']=='image_url']
        captions=[]
        # One image per caption call: preserve every image regardless of message size.
        for part in images:captions.append(self.cards.extract([part])['summary'])
        if captions:text+='\nDerived image descriptions (may be uncertain):\n'+'\n'.join(captions)
        return text.strip()

    def add(self,payload):
        validate_add(payload)
        check=getattr(self.transport,'assert_source_allowed',None)
        if check:
            try:check(payload)
            except ValueError:raise ContractError('retired or invalid source scope',503) from None
        uid,req=payload['user_id'],payload['request_id']
        digest=hashlib.sha256(canonical(payload).encode()).hexdigest();owner=uuid.uuid4().hex
        with self.lock:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                old=db.execute('SELECT digest,state FROM add_operations WHERE user_id=? AND request_id=?',(uid,req)).fetchone()
                if old:
                    if old[0]!=digest:raise ContractError('request_id conflict',409)
                    if old[1]!='complete':raise ContractError('unresolved Add; no automatic replay',503)
                    return self._persist_add(db,payload)
                if db.execute("SELECT 1 FROM add_operations WHERE user_id=? AND state!='complete'",(uid,)).fetchone():
                    raise ContractError('user has unresolved Add',503)
                db.execute('INSERT INTO add_operations VALUES(?,?,?,?,?)',(uid,req,digest,'preparing',owner))
            try:
                messages=[]
                for msg in payload['messages']:
                    text=self._describe(content_parts(msg['content']))
                    if 'timestamp' in msg:text='Source timestamp (Unix milliseconds): '+str(msg['timestamp'])+'\n'+text
                    # Split losslessly, retaining role and ordering. No hidden truncation.
                    messages.extend(dict(role=msg['role'],content=t) for t in utf8_chunks(text,2000))
                batch=[];size=0
                for message in messages:
                    width=len(message['content'].encode())
                    if batch and (size+width>4000 or len(batch)>=10):
                        self.native.add(batch,user_id=native_scope(uid),infer=True);self.ready();batch=[];size=0
                    batch.append(message);size+=width
                if batch:self.native.add(batch,user_id=native_scope(uid),infer=True);self.ready()
                with self.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    response=self._persist_add(db,payload,operation_token=owner)
                    db.execute("UPDATE add_operations SET state='complete' WHERE user_id=? AND request_id=? AND owner_token=?",(uid,req,owner))
                return response
            except BaseException:
                with self.connect() as db:
                    db.execute("UPDATE add_operations SET state='uncertain' WHERE user_id=? AND request_id=? AND state='preparing'",(uid,req))
                raise

    def search(self,payload):
        parts=validate_search(payload)
        with self.lock:
            with self.connect() as db:
                if db.execute("SELECT 1 FROM add_operations WHERE user_id=? AND state!='complete'",(payload['user_id'],)).fetchone():
                    raise ContractError('user has unresolved Add',503)
                if not db.execute('SELECT 1 FROM memories WHERE user_id=?',(payload['user_id'],)).fetchone():return {'data':[]}
            query=self._describe(parts)
            result=self.native.search(query,filters={'user_id':native_scope(payload['user_id'])},top_k=min(payload['top_k'],100))
            self.ready()
            rows=[]
            for row in result['results']:
                score=row.get('score',0)
                if type(score) not in (float,int) or not math.isfinite(score):raise ValueError('invalid native score')
                rows.append(dict(id='mem0_'+row['id'],content='Derived memory fact: '+row['memory'],
                    score=float(score),created_at=row['created_at']))
            return {'data':rows}
