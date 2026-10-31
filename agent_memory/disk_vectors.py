"""Bounded SQLite vector storage for the pinned dense-only Mem0 profile.

No corpus-sized Python arrays, payload cache, or ID map. Exact cosine scans
use 128 vectors at a time and retain at most 400 results (Mem0's existing
four-times over-fetch for API TopK100). Qdrant's pinned
filter evaluator and Mem0 adapter remain in use. Equal-score ties are ordered
by ID, rather than the local backend's unspecified numpy sort order.
"""
from contextlib import contextmanager
import heapq
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import QueryResponse
from qdrant_client.local.distances import cosine_similarity, EPSILON
from qdrant_client.local.payload_filters import check_filter
from qdrant_client.local.utils import normalize_point_id
from mem0.vector_stores.qdrant import Qdrant


class DiskClient(QdrantClient):
    """Only the dense operations used by this deployment are implemented.

Do not initialize QdrantClient's local backend: that would load the full DB.
Each transaction owns and closes its SQLite connection, including on failure.
"""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS collections(name TEXT PRIMARY KEY, dims INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS vectors(
                    collection TEXT NOT NULL, id TEXT NOT NULL, vector BLOB NOT NULL,
                    payload TEXT NOT NULL, user_id TEXT,
                    PRIMARY KEY(collection,id));
                CREATE INDEX IF NOT EXISTS vector_user ON vectors(collection,user_id);
                CREATE TABLE IF NOT EXISTS migration(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute('PRAGMA cache_size=-2048')
            db.execute('PRAGMA mmap_size=0')
            db.execute('PRAGMA temp_store=FILE')
            with db:yield db
        finally:db.close()

    def close(self, **kwargs):pass

    @staticmethod
    def key(point_id):return json.dumps(normalize_point_id(point_id), separators=(',', ':'))

    def get_collections(self, **kwargs):
        with self.connect() as db:
            return SimpleNamespace(collections=[SimpleNamespace(name=r[0]) for r in db.execute('SELECT name FROM collections')])

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None, **kwargs):
        cfg = vectors_config.get('') if isinstance(vectors_config, dict) else vectors_config
        if cfg.distance != models.Distance.COSINE or cfg.multivector_config is not None:
            raise ValueError('disk profile requires dense cosine')
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO collections VALUES(?,?)',(collection_name,cfg.size))
            if db.execute('SELECT dims FROM collections WHERE name=?',(collection_name,)).fetchone()[0] != cfg.size:
                raise ValueError('vector dimensions changed')
        return True

    def get_collection(self, collection_name, **kwargs):
        with self.connect() as db:
            row=db.execute('SELECT dims FROM collections WHERE name=?',(collection_name,)).fetchone()
            if row is None:raise ValueError('unknown collection')
            count=db.execute('SELECT count(*) FROM vectors WHERE collection=?',(collection_name,)).fetchone()[0]
        return SimpleNamespace(points_count=count, config=SimpleNamespace(params=SimpleNamespace(
            sparse_vectors={},vectors=models.VectorParams(size=row[0],distance=models.Distance.COSINE,on_disk=True))))

    def create_payload_index(self, **kwargs):
        # user_id already has an exact SQL index; other filters are streamed.
        if kwargs.get('field_name') not in {'user_id','agent_id','run_id','actor_id'}:
            raise ValueError('unsupported index')
        return True

    @staticmethod
    def encoded(vector, dims):
        if isinstance(vector, dict):
            if set(vector) != {''}:raise ValueError('sparse/named vectors outside frozen dense profile')
            vector=vector['']
        a=np.array(vector,dtype=np.float32)
        if a.shape != (dims,) or not np.isfinite(a).all():raise ValueError('invalid dense vector')
        norm=np.linalg.norm(a)
        if norm>EPSILON:a=a/norm
        return a.astype('<f4').tobytes()

    def upsert(self, collection_name, points, **kwargs):
        with self.connect() as db:
            dims=db.execute('SELECT dims FROM collections WHERE name=?',(collection_name,)).fetchone()[0]
            for point in points:
                payload=point.payload or {}
                db.execute('INSERT INTO vectors VALUES(?,?,?,?,?) ON CONFLICT(collection,id) DO UPDATE SET vector=excluded.vector,payload=excluded.payload,user_id=excluded.user_id',
                    (collection_name,self.key(point.id),self.encoded(point.vector,dims),json.dumps(payload,ensure_ascii=False,allow_nan=False),payload.get('user_id')))
        return models.UpdateResult(operation_id=0,status=models.UpdateStatus.COMPLETED)

    @staticmethod
    def user_constraint(f):
        if f is None:return None
        must=f.must or []
        if not isinstance(must,list):must=[must]
        for item in must:
            if isinstance(item,models.FieldCondition) and item.key=='user_id' and isinstance(item.match,models.MatchValue) and isinstance(item.match.value,str):
                return item.match.value
            if isinstance(item,models.Filter):
                value=DiskClient.user_constraint(item)
                if value is not None:return value
        return None

    def rows(self, db, collection, f=None):
        sql='SELECT id,vector,payload FROM vectors WHERE collection=?';args=[collection]
        user=self.user_constraint(f)
        if user is not None:sql+=' AND user_id=?';args.append(user)
        for key, vector, raw in db.execute(sql,args):
            payload=json.loads(raw);pid=json.loads(key)
            if f is None or check_filter(f,payload,pid,{'':True}):yield key,vector,payload

    def query_points(self, collection_name, query, query_filter=None, limit=10, using=None,
                     with_payload=True, with_vectors=False, **kwargs):
        if using not in (None,'') or kwargs or type(limit) is not int or not 1<=limit<=400:
            raise ValueError('unsupported dense search options')
        q=np.asarray(query)
        heap=[]
        with self.connect() as db:
            db.execute('BEGIN')  # one committed snapshot through final payload fetch
            dims=db.execute('SELECT dims FROM collections WHERE name=?',(collection_name,)).fetchone()[0]
            if q.shape!=(dims,) or not np.isfinite(q).all():raise ValueError('invalid query vector')
            batch=[]
            def consume():
                matrix=np.stack([np.frombuffer(row[1],dtype='<f4') for row in batch])
                scores=cosine_similarity(q,matrix)
                for row,score in zip(batch,scores):
                    item=(float(score),row[0])
                    if len(heap)<limit:heapq.heappush(heap,item)
                    elif item>heap[0]:heapq.heapreplace(heap,item)
                batch.clear()
            for row in self.rows(db,collection_name,query_filter):
                # Do not retain payloads across batches or in the result heap.
                batch.append((row[0],row[1]))
                if len(batch)==128:consume()
            if batch:consume()
            result=[]
            for score,key in sorted(heap,reverse=True):
                vector,raw=db.execute('SELECT vector,payload FROM vectors WHERE collection=? AND id=?',(collection_name,key)).fetchone()
                result.append(models.ScoredPoint(id=json.loads(key),version=0,score=score,
                    payload=json.loads(raw) if with_payload else None,
                    vector=np.frombuffer(vector,dtype='<f4').tolist() if with_vectors else None))
        return QueryResponse(points=result)

    def query_batch_points(self, collection_name, requests, **kwargs):
        return [self.query_points(collection_name,r.query,query_filter=r.filter,limit=r.limit,
                                  with_payload=r.with_payload,with_vectors=r.with_vector) for r in requests]

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False, **kwargs):
        out=[]
        with self.connect() as db:
            for pid in ids:
                row=db.execute('SELECT vector,payload FROM vectors WHERE collection=? AND id=?',(collection_name,self.key(pid))).fetchone()
                if row:out.append(models.Record(id=normalize_point_id(pid),payload=json.loads(row[1]) if with_payload else None,
                    vector=np.frombuffer(row[0],dtype='<f4').tolist() if with_vectors else None))
        return out

    def count(self, collection_name, count_filter=None, **kwargs):
        with self.connect() as db:
            n=(db.execute('SELECT count(*) FROM vectors WHERE collection=?',(collection_name,)).fetchone()[0]
               if count_filter is None else sum(1 for _ in self.rows(db,collection_name,count_filter)))
        return models.CountResult(count=n)

    def delete(self, collection_name, points_selector, **kwargs):
        with self.connect() as db:
            if isinstance(points_selector,models.PointIdsList):
                keys=(self.key(pid) for pid in points_selector.points)
            elif isinstance(points_selector,models.FilterSelector):
                keys=(row[0] for row in self.rows(db,collection_name,points_selector.filter))
            else:raise ValueError('unsupported point selector')
            for key in keys:db.execute('DELETE FROM vectors WHERE collection=? AND id=?',(collection_name,key))
        return models.UpdateResult(operation_id=0,status=models.UpdateStatus.COMPLETED)

    def scroll(self, collection_name, scroll_filter=None, limit=100, with_payload=True, with_vectors=False, **kwargs):
        if kwargs or not 1<=limit<=100:raise ValueError('bounded scroll required')
        records=[]
        with self.connect() as db:
            for key,vector,payload in self.rows(db,collection_name,scroll_filter):
                records.append(models.Record(id=json.loads(key),payload=payload if with_payload else None,
                    vector=np.frombuffer(vector,dtype='<f4').tolist() if with_vectors else None))
                if len(records)==limit:break
        return records,None


class DiskQdrant(Qdrant):
    def __init__(self, collection_name, embedding_model_dims, path=None, client=None, **kwargs):
        if client is None:
            path=Path(path)
            target=path.parent/'disk-vectors.sqlite'
            if (path/'meta.json').exists() and not target.exists():
                raise ValueError('existing Qdrant requires verified offline migration')
            client=DiskClient(target)
            if (path/'meta.json').exists():
                with client.connect() as db:
                    if not db.execute("SELECT 1 FROM migration WHERE key='verified'").fetchone():
                        raise ValueError('unverified vector migration')
        if not isinstance(client,DiskClient):raise ValueError('disk profile client mismatch')
        super().__init__(collection_name,embedding_model_dims,client=client,on_disk=True)
        # The frozen deployment has no sparse encoder or sparse vectors. Never
        # download one or silently replace dense retrieval with another model.
        self._has_bm25_slot=False

