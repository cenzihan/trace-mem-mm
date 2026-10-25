"""Native round retrieval views over authoritative original message/image evidence."""
import hashlib
import json
import math
from .system import AgentMemorySystem,RetrievalConfig
from .native_memgallery import materialize_round
from .protocol import canonical,content_parts,text_of,image_bytes,media_size,MEDIA_LIMIT,ContractError
from .retrieval import bm25,cosine,ranked,rrf


class NativeRoundIndex:
    def __init__(self,path,dimensions=256):
        self.source=AgentMemorySystem(path,RetrievalConfig(cards=False,temporal=False,links=False,image_identity=False))
        self.dimensions=dimensions
        with self.source.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS native_units (
              uid TEXT,id TEXT,source_round TEXT,session_id TEXT,ordinal INTEGER,source_date TEXT,
              request_id TEXT,raw_text TEXT,content TEXT,memory_ids TEXT,image_hashes TEXT,provenance TEXT,
              text_a TEXT,text_b TEXT,vector_a TEXT,vector_b TEXT,entities TEXT,
              PRIMARY KEY(uid,id));
              CREATE TABLE IF NOT EXISTS native_index_meta(uid TEXT PRIMARY KEY,settings TEXT);''')

    def register_originals(self,plans,reader):
        for ordinal,plan in enumerate(plans):
            payload=materialize_round(plan,reader);self.source.add(payload)
            parts=[p for message in payload['messages'] for p in content_parts(message['content'])]
            hashes=[hashlib.sha256(image_bytes(p['image_url']['url'])[0]).hexdigest() for p in parts if p['type']=='image_url']
            unit='round_'+hashlib.sha256(canonical([payload['user_id'],payload['session_id'],plan['provenance']['source_round']]).encode()).hexdigest()
            with self.source.connect() as db:
                memory_ids=[row[0] for row in db.execute('SELECT id FROM memories WHERE user_id=? AND request_id=? ORDER BY ordinal',
                                                       (payload['user_id'],payload['request_id']))]
                db.execute('INSERT OR IGNORE INTO native_units VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (payload['user_id'],unit,plan['provenance']['source_round'],payload['session_id'],ordinal,
                     plan['provenance']['source_date'],payload['request_id'],text_of(parts),canonical(parts),canonical(memory_ids),
                     canonical(hashes),canonical(plan['provenance']),None,None,None,None,None))

    def units(self,uid):
        with self.source.connect() as db:
            db.row_factory=__import__('sqlite3').Row
            rows=[dict(row) for row in db.execute('SELECT * FROM native_units WHERE uid=? ORDER BY ordinal',(uid,))]
        return rows

    def documents(self,uid,visuals):
        output=[]
        for row in self.units(uid):
            descriptions=[visuals[value] for value in json.loads(row['image_hashes'])]
            globals_=' '.join(value['global_caption'] for value in descriptions)
            structure=[];entities=[]
            for value in descriptions:
                structure.extend(value['visible_text']);structure.extend(value['regions']);structure.extend(value['relations'])
                grounded=(row['raw_text']+' '+' '.join(value['visible_text'])).casefold()
                for entity in value['named_entities']:
                    if len(entity)>=3 and entity.casefold() in grounded:entities.append(entity.casefold())
            a=row['raw_text']+'\nVISUAL_GLOBAL\n'+globals_
            b=a+'\nSTRUCTURED_VISUAL\n'+' '.join(structure+entities)
            if any(len(value.encode('utf-8'))>8192 for value in (a,b)):
                raise ValueError('derived round text exceeds explicit embedding admission; no silent truncation')
            output.append(dict(id=row['id'],a=a,b=b,entities=sorted(set(entities))))
        return output

    def publish(self,uid,documents,vectors_a,vectors_b,settings):
        rows=self.units(uid)
        if len(rows)!=len(documents) or len(rows)!=len(vectors_a) or len(rows)!=len(vectors_b):
            raise ValueError('whole native index must be complete before publication')
        for vector in list(vectors_a)+list(vectors_b):
            if len(vector)!=self.dimensions:raise ValueError('embedding dimension mismatch')
            cosine(vector,vector)
        with self.source.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row,doc,va,vb in zip(rows,documents,vectors_a,vectors_b):
                if row['id']!=doc['id']:raise ValueError('source order mismatch')
                db.execute('UPDATE native_units SET text_a=?,text_b=?,vector_a=?,vector_b=?,entities=? WHERE uid=? AND id=?',
                           (doc['a'],doc['b'],canonical(va),canonical(vb),canonical(doc['entities']),uid,row['id']))
            db.execute('INSERT OR REPLACE INTO native_index_meta VALUES(?,?)',(uid,canonical(settings)))

    def search(self,uid,query,query_vector,variant='A',top_k=100):
        if variant not in ('A','B','C') or type(top_k) is not int or not 1<=top_k<=100:
            raise ValueError('explicit A/B/C and top_k within 100 required')
        if not isinstance(query,str) or not query.strip():raise ValueError('original text question required; image query needs explicit adapter')
        rows=self.units(uid)
        if not rows:return {'data':[]},dict(candidate_count=0,source_rounds=[],variant=variant)
        if len(query_vector)!=self.dimensions:raise ValueError('real query vector dimension mismatch')
        column='a' if variant=='A' else 'b'
        if any(row['vector_'+column] is None for row in rows):raise ContractError('native index incomplete',503)
        docs={row['id']:row['text_'+column] for row in rows};by_id={row['id']:row for row in rows}
        lexical=bm25(query,docs)
        scores=rrf([ranked({key:value for key,value in lexical.items() if value>0}),
                    ranked({row['id']:cosine(query_vector,json.loads(row['vector_'+column])) for row in rows})],
                   [1.,1.],60)
        base=ranked(scores);neighbors=[]
        if variant=='C':
            # No question/gold fitting: exact grounded entity + source-order neighbors only.
            for key in base[:5]:
                seed=by_id[key];grounded=set(json.loads(seed['entities']))
                adjacent=[row['id'] for row in rows if row['session_id']==seed['session_id'] and abs(row['ordinal']-seed['ordinal'])==1]
                entity_matches=[row for row in rows if row['id']!=key and grounded & set(json.loads(row['entities']))]
                entity_matches.sort(key=lambda row:(abs(row['ordinal']-seed['ordinal']),row['id']))
                for candidate in adjacent+[row['id'] for row in entity_matches[:2]]:
                    if candidate not in neighbors:neighbors.append(candidate)
            extra=rrf([neighbors[:20]],[.25],60)
            for key,value in extra.items():scores[key]=scores.get(key,0)+value
        order=ranked(scores);selected=[];source_rounds=[];used=0
        for key in order[:top_k]:
            row=by_id[key]
            # Original source must still exist under exactly this user after any purge.
            ids=json.loads(row['memory_ids'])
            with self.source.connect() as db:
                live=[db.execute('SELECT content FROM memories WHERE user_id=? AND id=?',(uid,mid)).fetchone() for mid in ids]
            if any(value is None for value in live):raise ContractError('source purged; derived index is stale',503)
            parts=[part for value in live for part in json.loads(value[0])]
            if canonical(parts)!=row['content']:raise ContractError('original source changed; index stale',503)
            size=media_size(parts)
            if used+size>MEDIA_LIMIT:break
            selected.append(dict(id=key,content=parts,score=scores[key]));source_rounds.append(row['source_round']);used+=size
        return {'data':selected},dict(candidate_count=len(rows),source_rounds=source_rounds,variant=variant,
            returned_decoded_media_bytes=used,media_prefix_omitted=len(order[:top_k])-len(selected),
            neighbors=neighbors[:20],natural_time_inference=False,query_persisted=False)
