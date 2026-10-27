"""B1/B2 memory system, retaining the unchanged MemoryStore as rollback baseline."""
import hashlib
import json
import math
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from .cards import derive
from .protocol import (ContractError, MEDIA_LIMIT, canonical, content_parts, image_hashes,
                       media_size, text_of, validate_add, validate_search)
from .retrieval import bm25, cosine, ranked, rrf, tokenize
from .store import MemoryStore


@dataclass(frozen=True)
class RetrievalConfig:
    cards: bool = True
    temporal: bool = True
    links: bool = True
    image_identity: bool = True
    dense: bool = False
    rerank: bool = False
    candidate_limit: int = 200
    rrf_constant: int = 60
    max_link_hops: int = 2
    visual_query_cards: bool = False


class AgentMemorySystem(MemoryStore):
    model_add_message_limit = 16

    def __init__(self, path, retrieval=None, embedding=None, card_adapter=None, scoring_adapter=None):
        super().__init__(path)
        self.retrieval = retrieval or RetrievalConfig()
        self.embedding = embedding
        self.card_adapter = card_adapter
        self.scoring_adapter = scoring_adapter
        self._write_lock = threading.RLock()
        if self.retrieval.dense and (embedding is None or getattr(embedding, "model", None) != "text-embedding-v4"):
            raise ValueError("dense blocked: configured text-embedding-v4 adapter required")
        if card_adapter is not None and getattr(card_adapter, "model", None) != "gpt-4o-mini":
            raise ValueError("card adapter must use gpt-4o-mini")
        if self.retrieval.visual_query_cards and card_adapter is None:
            raise ValueError('visual query summaries require configured gpt-4o-mini')
        if self.retrieval.rerank and scoring_adapter is None:
            raise ValueError("rerank blocked: approved candidate-only scorer required")
        if self.retrieval.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if not 0<=self.retrieval.max_link_hops<=3:
            raise ValueError("link traversal must be bounded to 0..3 hops")
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY, memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL, session_id TEXT NOT NULL, content TEXT NOT NULL,
                indexed_text TEXT NOT NULL, vector TEXT, vector_model TEXT);
              CREATE INDEX IF NOT EXISTS cards_scope ON cards(user_id);
              CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY, card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL, entity TEXT NOT NULL, attribute TEXT NOT NULL,
                value TEXT NOT NULL, event_at TEXT NOT NULL, observed_at TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS fact_scope ON facts(user_id,entity,attribute,event_at);
              CREATE TABLE IF NOT EXISTS entity_links (
                card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL, entity TEXT NOT NULL, related TEXT NOT NULL,
                PRIMARY KEY(card_id,entity,related));
              CREATE TABLE IF NOT EXISTS add_operations (
                user_id TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('preparing','complete','uncertain')),
                owner_token TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id,request_id));
            ''')
            db.execute('PRAGMA user_version=3')

    def capabilities(self):
        adapters=[a for a in (self.embedding,self.card_adapter) if a is not None]
        fake=any(getattr(getattr(a,'transport',None),'mode',None)=='offline_fake' for a in adapters)
        return {"mode": "offline_fake_providers" if fake else "configured_adapters" if adapters else "local_deterministic", "retrieval": "BM25+RRF+exact-image-identity",
                "cards": "gpt-4o-mini" if self.card_adapter else "explicit-source-cards-no-LLM",
                "dense": "text-embedding-v4" if self.retrieval.dense else "disabled_not_simulated",
                "reranker": "configured_candidate_scorer" if self.retrieval.rerank else "disabled",
                "temporal": self.retrieval.temporal, "links": self.retrieval.links,
                "online_verified": False}

    def add(self, payload):
        with self._write_lock:
            return self._add_locked(payload)

    def _execute_add(self, payload, operation_token=None):
        # Build derived material before raw commit. A model/adapter error never acknowledges Add.
        plans = []
        for ordinal, message in enumerate(payload["messages"]):
            parts = content_parts(message["content"])
            text = text_of(parts)
            card = derive(text)
            for fact in card["facts"]:
                try:
                    datetime.fromisoformat(fact["event"].replace("Z","+00:00"))
                except ValueError as exc:
                    raise ContractError("invalid explicit fact event timestamp",422) from exc
            if self.card_adapter:
                # Adapter output is separately labelled; original quotes remain authoritative.
                generated = self.card_adapter.extract(json.loads(canonical(parts)))
                if not isinstance(generated, dict) or not isinstance(generated.get("summary"), str):
                    raise ValueError("invalid card adapter output")
                card["generated"] = generated
                card["generator"] = "gpt-4o-mini"
                if not isinstance(generated.get('evidence_quotes'),list) or any(not isinstance(q,str) or q not in text for q in generated['evidence_quotes']):
                    raise ValueError("generated card has ungrounded evidence quote")
            indexed = text
            if self.retrieval.cards:
                indexed += " " + " ".join(" ".join(f[k] for k in ("entity", "attribute", "value")) for f in card["facts"])
                if self.card_adapter:
                    indexed += " " + card["generated"]["summary"]
            mid = "mem_" + hashlib.sha256(canonical([payload["user_id"], payload["session_id"], payload["request_id"], ordinal]).encode()).hexdigest()
            plans.append((mid, parts, card, indexed))
        vectors = self.embedding.embed([plan[3] for plan in plans]) if self.retrieval.dense else [None]*len(plans)
        if len(vectors) != len(plans):
            raise ValueError("embedding batch length mismatch")
        if self.retrieval.dense:
            for vector in vectors:
                cosine(vector, vector)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_dense_scope(db,payload)
            response = self._persist_add(db,payload,operation_token=operation_token)
            for (mid, parts, card, indexed), vector in zip(plans, vectors):
                cid = "card_" + mid[4:]
                db.execute("INSERT OR IGNORE INTO cards VALUES (?,?,?,?,?,?,?,?)",
                           (cid,mid,payload["user_id"],payload["session_id"],canonical(card),indexed,
                            canonical(vector) if vector is not None else None,"text-embedding-v4" if vector is not None else None))
                observed = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
                for i, fact in enumerate(card["facts"]):
                    # Validate explicit event timestamps; never infer a date from persistence time.
                    datetime.fromisoformat(fact["event"].replace("Z","+00:00"))
                    db.execute("INSERT OR IGNORE INTO facts VALUES (?,?,?,?,?,?,?,?)",
                               (cid+":"+str(i),cid,payload["user_id"],fact["entity"].casefold(),fact["attribute"].casefold(),fact["value"],fact["event"],observed))
                for link in card["links"]:
                    db.execute("INSERT OR IGNORE INTO entity_links VALUES (?,?,?,?)",
                               (cid,payload["user_id"],link["entity"].casefold(),link["related"].casefold()))
            if operation_token is not None:
                changed=db.execute("UPDATE add_operations SET state='complete',updated_at=? WHERE user_id=? AND request_id=? AND owner_token=? AND state='preparing'",
                    (datetime.now(timezone.utc).isoformat(),payload['user_id'],payload['request_id'],operation_token)).rowcount
                if changed!=1:raise ContractError('model Add ownership changed; inspect outcome',503)
        return response

    def _add_locked(self,payload):
        validate_add(payload)
        payload=json.loads(canonical(payload))
        digest=hashlib.sha256(canonical(payload).encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN')
            if self._request_ready(db,payload,digest):return self._echo(payload)
            self._check_dense_scope(db,payload)
        if self.card_adapter is not None and len(payload['messages'])>self.model_add_message_limit:
            raise ContractError(f'local model-assisted Add capacity is {self.model_add_message_limit} messages',413)
        if self.retrieval.dense and len(payload['messages'])>100:
            raise ContractError('local dense Add capacity is 100 messages',413)
        # Validate every explicit timestamp before claiming or charging any model work.
        for message in payload['messages']:
            for fact in derive(text_of(content_parts(message['content'])))['facts']:
                try:datetime.fromisoformat(fact['event'].replace('Z','+00:00'))
                except ValueError:raise ContractError('invalid explicit fact event timestamp',422) from None
        with self._model_operation(payload,digest) as operation:
            if operation is False:return self._echo(payload)
            return self._execute_add(payload,operation_token=operation)

    def _echo(self,payload):
        return dict(success=True,**{key:payload[key] for key in ('request_id','user_id','session_id')})

    def _request_ready(self,db,payload,digest):
        uid,req=payload['user_id'],payload['request_id']
        self._guard_operation(db,uid,req,digest)
        existing=db.execute('SELECT digest FROM requests WHERE user_id=? AND request_id=?',(uid,req)).fetchone()
        if not existing:return False
        if existing[0]!=digest:raise ContractError('request_id reused with a different payload',409)
        counts=db.execute("SELECT count(m.id),count(c.id),sum(CASE WHEN c.vector IS NULL OR c.vector_model!='text-embedding-v4' THEN 1 ELSE 0 END) FROM memories m LEFT JOIN cards c ON c.memory_id=m.id WHERE m.user_id=? AND m.request_id=?",(uid,req)).fetchone()
        ready=counts[0]==len(payload['messages']) and counts[0]==counts[1]
        if ready and self.retrieval.dense and counts[2]:
            raise ContractError('existing source lacks configured dense index; explicit migration required',503)
        if ready:self._check_dense_scope(db,payload)
        return ready

    def _check_dense_scope(self,db,payload):
        if self.retrieval.dense and db.execute("SELECT 1 FROM memories m LEFT JOIN cards c ON c.memory_id=m.id WHERE m.user_id=? AND m.request_id!=? AND (c.id IS NULL OR c.vector IS NULL OR c.vector_model!='text-embedding-v4') LIMIT 1",(payload['user_id'],payload['request_id'])).fetchone():
            raise ContractError('user dense index incomplete; explicit migration required',503)

    @contextmanager
    def _model_operation(self,payload,digest):
        token=None;ready=False
        if self.card_adapter is not None or self.retrieval.dense:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                ready=self._request_ready(db,payload,digest)
                if not ready:
                    self._check_dense_scope(db,payload)
                    if db.execute('SELECT 1 FROM add_operations WHERE user_id=? AND request_id=?',(payload['user_id'],payload['request_id'])).fetchone():
                        raise ContractError('existing model Add needs explicit index recovery',503)
                    token=uuid.uuid4().hex
                    db.execute('INSERT INTO add_operations VALUES(?,?,?,?,?,?)',
                        (payload['user_id'],payload['request_id'],digest,'preparing',token,datetime.now(timezone.utc).isoformat()))
        if ready:
            yield False
            return
        try:yield token
        except BaseException:
            if token is not None:
                try:
                    with self.connect() as db:
                        db.execute("UPDATE add_operations SET state='uncertain',updated_at=? WHERE user_id=? AND request_id=? AND owner_token=? AND state='preparing'",
                            (datetime.now(timezone.utc).isoformat(),payload['user_id'],payload['request_id'],token))
                except Exception:pass
            raise

    def search(self, payload):
        result, _ = self.retrieve(payload)
        return result

    def retrieve(self, payload):
        query_parts = validate_search(payload)
        query = text_of(query_parts)
        uid = payload["user_id"]
        # Decode/hash potentially large query media before acquiring a database snapshot.
        query_images=image_hashes(query_parts)
        with self.connect() as db:
            db.execute('BEGIN')
            missing = db.execute("SELECT count(*) FROM memories m LEFT JOIN cards c ON m.id=c.memory_id WHERE m.user_id=? AND c.id IS NULL",(uid,)).fetchone()[0]
            if missing:
                raise ContractError("derived index incomplete; retry original Add before Search",503)
            # Rank small metadata first. Never materialize an entire user's image blobs.
            rows = db.execute("SELECT m.id,r.digest,m.created_at,c.id,c.indexed_text,c.content,c.vector FROM memories m JOIN cards c ON m.id=c.memory_id JOIN requests r ON r.user_id=m.user_id AND r.request_id=m.request_id WHERE m.user_id=? ORDER BY m.id",(uid,)).fetchall()
            facts = db.execute("SELECT card_id,entity,attribute,value,event_at FROM facts WHERE user_id=?",(uid,)).fetchall()
            links = db.execute("SELECT card_id,entity,related FROM entity_links WHERE user_id=?",(uid,)).fetchall()
            image_matches=[]
            if query_images:
                evidence=db.execute("SELECT e.memory_id,e.sha256 FROM evidence e JOIN memories m ON m.id=e.memory_id WHERE m.user_id=?",(uid,)).fetchall()
                image_matches=sorted({mid for mid,digest in evidence if digest in query_images})
        memories = {row[0]:dict(id=row[0],created_at=row[2]) for row in rows}
        query_summary=None
        query_unverified_quotes=0
        if rows and query_images and self.retrieval.visual_query_cards:
            generated=self.card_adapter.extract(json.loads(canonical(query_parts)))
            if (not isinstance(generated,dict) or not isinstance(generated.get('summary'),str) or
                not generated['summary'].strip() or len(generated['summary'].encode('utf-8'))>8192 or
                not isinstance(generated.get('evidence_quotes'),list) or
                any(not isinstance(q,str) for q in generated['evidence_quotes'])):
                raise ValueError('query_card_schema_invalid')
            # Image OCR/visual observations are not literal quotes from query text.
            # They may guide retrieval, but never become stored or verified facts.
            query_unverified_quotes=sum(q not in query for q in generated['evidence_quotes'])
            query_summary=generated['summary']
            query+=' '+query_summary
        source_digests={row[0]:row[1] for row in rows}
        loaded={}
        def original(mid):
            if mid not in loaded:
                with self.connect() as db:
                    value=db.execute("SELECT m.content,m.created_at,r.digest FROM memories m JOIN requests r ON r.user_id=m.user_id AND r.request_id=m.request_id WHERE m.id=? AND m.user_id=?",(mid,uid)).fetchone()
                if value is None or value[1]!=memories[mid]['created_at'] or value[2]!=source_digests[mid]:
                    raise ContractError("source changed during retrieval; retry Search",503)
                loaded[mid]=dict(memories[mid],content=json.loads(value[0]))
            return loaded[mid]
        docs = {row[0]: row[4] if self.retrieval.cards else json.loads(row[5])["source_quote"] for row in rows}
        lexical = bm25(query,docs)
        # Zero scores are not evidence; RRF must not manufacture positive relevance.
        rankings = [ranked({key:value for key,value in lexical.items() if value>0})]
        weights = [1.0]
        if self.retrieval.image_identity and image_matches:
            rankings.append(sorted(image_matches)); weights.append(4.0)
        if self.retrieval.dense and rows:
            if any(row[6] is None for row in rows):
                raise ContractError("dense index not complete; no silent lexical fallback",503)
            qv = self.embedding.embed([query])[0]
            dense = {row[0]:cosine(qv,json.loads(row[6])) for row in rows}
            rankings.append(ranked(dense)); weights.append(1.0)
        scores = rrf(rankings,weights,self.retrieval.rrf_constant)
        card_to_mid = {row[3]:row[0] for row in rows}
        qt = set(tokenize(query))
        asof_match = re.search(r"as of (\d{4}-\d{2}-\d{2})",query.casefold())
        wants_current = bool(qt & {"current","currently","latest","now","现在","最新"})
        temporal_picks = []
        if self.retrieval.temporal and (asof_match or wants_current):
            groups = {}
            for cid,entity,attribute,value,event in facts:
                if entity not in qt or attribute not in qt:
                    continue
                if asof_match and event[:10] > asof_match[1]:
                    continue
                groups.setdefault((entity,attribute),[]).append((event,cid))
            for group in groups.values():
                newest = max(event for event,_ in group)
                # Equal-time conflicts remain tied; never invent a winner.
                for event,cid in group:
                    if event == newest:
                        mid = card_to_mid[cid]
                        scores[mid] = scores.get(mid,0)+0.2
                        temporal_picks.append(mid)
        linked = []
        if self.retrieval.links:
            frontier=set(qt); seen=set(qt)
            for depth in range(self.retrieval.max_link_hops):
                next_frontier=set()
                for cid,entity,related in links:
                    if entity not in frontier:
                        continue
                    mid=card_to_mid[cid]
                    scores[mid]=scores.get(mid,0)+0.01/(depth+1)
                    for row in rows:
                        if related in set(tokenize(row[4])):
                            scores[row[0]]=scores.get(row[0],0)+0.02/(depth+1)
                            linked.append(row[0])
                    if related not in seen:
                        next_frontier.add(related)
                seen.update(next_frontier); frontier=next_frontier
        order = ranked(scores)[:max(payload["top_k"],self.retrieval.candidate_limit)]
        if self.retrieval.rerank:
            values = self.scoring_adapter.score(query_parts,[original(mid) for mid in order],[scores[mid] for mid in order])
            if len(values)!=len(order) or any(not math.isfinite(v) for v in values):
                raise ValueError("invalid reranker scores")
            scores = dict(zip(order,values)); order=ranked(scores)
        selected=[]; used=0; media_budget_omitted=0
        for mid in order[:payload["top_k"]]:
            memory=original(mid)
            size=media_size(memory["content"])
            if used+size>MEDIA_LIMIT:
                # top_k is a maximum. Choose a ranked prefix of whole memories;
                # never crop images/text or emit an oversized response.
                media_budget_omitted=len(order[:payload['top_k']])-len(selected)
                break
            used+=size
            selected.append(dict(memory,score=scores[mid]))
        # Trace is an explicit in-memory diagnostic return, never an automatic query log.
        trace=dict(candidate_count=len(memories),image_matches=image_matches,temporal_picks=temporal_picks,
                   linked_candidates=sorted(set(linked)),dense_enabled=self.retrieval.dense,
                   cards_enabled=self.retrieval.cards,media_budget_omitted_records=media_budget_omitted,
                   capabilities=self.capabilities(),query_summary=query_summary,
                   query_unverified_visual_quotes=query_unverified_quotes,
                   query_summary_is_verified_fact=False)
        return {"data":selected},trace

    def inspect_user(self,user_id):
        with self.connect() as db:
            return {name:db.execute("SELECT count(*) FROM "+name+" WHERE user_id=?",(user_id,)).fetchone()[0]
                    for name in ("memories","cards","facts","entity_links")}
