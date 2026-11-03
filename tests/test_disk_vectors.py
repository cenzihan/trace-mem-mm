import importlib.util
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0')
class DiskVectorTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qdrant_client import QdrantClient,models
        from agent_memory.disk_vectors import DiskClient
        self.models=models;self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.old=QdrantClient(path=str(self.root/'qdrant'))
        self.new=DiskClient(self.root/'new.sqlite')
        for c in (self.old,self.new):
            c.create_collection('test',vectors_config=models.VectorParams(size=256,distance=models.Distance.COSINE))
        rng=np.random.default_rng(42)
        self.points=[models.PointStruct(id=str(uuid.UUID(int=i+1)),vector=v.tolist(),
            payload={'user_id':str(i%3),'data':'Synthetic datum '+str(i),'tracemem_source_id':str(i%9)})
            for i,v in enumerate(rng.normal(size=(650,256)))]
        self.old.upsert('test',self.points);self.new.upsert('test',self.points)
        self.addCleanup(self.tmp.cleanup);self.addCleanup(self.old.close)

    def test_exact_search_filter_scores_and_top400_match(self):
        from agent_memory.disk_vectors import DiskClient
        m=self.models
        filters=[None,m.Filter(must=[m.FieldCondition(key='user_id',match=m.MatchValue(value='1'))]),
            m.Filter(must=[m.FieldCondition(key='user_id',match=m.MatchValue(value='1'))],
                     must_not=[m.FieldCondition(key='tracemem_source_id',match=m.MatchValue(value='2'))]),
            m.Filter(must=[m.FieldCondition(key='user_id',match=m.MatchValue(value='absent'))])]
        reopened=DiskClient(self.root/'new.sqlite')
        for f in filters:
            a=self.old.query_points('test',self.points[17].vector,query_filter=f,limit=400).points
            b=reopened.query_points('test',self.points[17].vector,query_filter=f,limit=400).points
            self.assertEqual([x.id for x in a],[x.id for x in b])
            for x,y in zip(a,b):
                self.assertAlmostEqual(x.score,y.score,places=6);self.assertEqual(x.payload,y.payload)

    def test_atomic_upsert_rolls_back_and_no_cross_user_delete(self):
        m=self.models
        before=self.new.count('test').count
        points=[m.PointStruct(id=str(uuid.uuid4()),vector=[0.1]*256,payload={'user_id':'new'}),
                m.PointStruct(id=str(uuid.uuid4()),vector=[0.1]*2)]
        with self.assertRaises(ValueError):self.new.upsert('test',points)
        self.assertEqual(self.new.count('test').count,before)
        filt=m.Filter(must=[m.FieldCondition(key='user_id',match=m.MatchValue(value='1')),
            m.FieldCondition(key='tracemem_source_id',match=m.MatchAny(any=['1','4']))])
        for c in (self.old,self.new):c.delete('test',m.FilterSelector(filter=filt))
        self.assertEqual(self.old.count('test').count,self.new.count('test').count)
        self.assertEqual(self.old.retrieve('test',[self.points[0].id])[0].payload,self.new.retrieve('test',[self.points[0].id])[0].payload)

    def test_migrate_preserves_every_id_payload_vector_and_old_database(self):
        import hashlib
        from agent_memory.migrate_disk_vectors import migrate
        from agent_memory.disk_vectors import DiskClient
        self.old.close()
        src=self.root/'qdrant/collection/test/storage.sqlite'
        before=hashlib.sha256(src.read_bytes()).hexdigest()
        result=migrate(self.root/'qdrant',self.root/'migrated.sqlite')
        self.assertEqual(result['collections']['test']['verified'],650)
        self.assertEqual(before,hashlib.sha256(src.read_bytes()).hexdigest())
        migrated=DiskClient(self.root/'migrated.sqlite')
        for p in self.points:
            self.assertEqual(migrated.retrieve('test',[p.id])[0].payload,p.payload)
        with self.assertRaises(ValueError):migrate(self.root/'qdrant',self.root/'migrated.sqlite')

    def test_existing_store_without_verified_migration_fails_closed(self):
        from agent_memory.disk_vectors import DiskQdrant
        with self.assertRaises(ValueError):DiskQdrant('test',256,path=self.root/'qdrant')

    def test_payload_and_id_not_cached_in_python(self):
        self.assertEqual(set(vars(self.new)),{'path'})
        with self.new.connect() as c:
            self.assertEqual(c.execute('PRAGMA cache_size').fetchone()[0],-2048)
            self.assertEqual(c.execute('PRAGMA mmap_size').fetchone()[0],0)


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0')
class DiskNativeTests(unittest.TestCase):
    def test_real_mem0_add_restart_search_idempotence_and_isolation(self):
        from agent_memory.disk_options import DiskOptionsSystem
        from test_mem0_native import FakeEmbedding,FakeTransport,FakeCards
        from agent_memory.protocol import ContractError
        e=FakeEmbedding();t=FakeTransport()
        # The evidence-selection protocol is distinct from native fact extraction.
        def select(endpoint,payload):
            t.calls.append(payload)
            return {'choices':[{'message':{'content':json.dumps({'selected':[0]})}}]}
        class Transport:
            def ready(self):pass
            def __call__(self,*args):return select(*args)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'raw.sqlite'
            s=DiskOptionsSystem(p,embedding=e,transport=Transport(),card_adapter=FakeCards())
            payload=dict(user_id='a',request_id='r',session_id='s',messages=[dict(role='user',content='The synthetic teal notebook is in the workshop.')])
            try:
                s.add(payload);n=len(e.calls);s.add(payload);self.assertEqual(n,len(e.calls))
                with self.assertRaises(ContractError):s.add(dict(payload,messages=[dict(role='user',content='Changed')]))
                self.assertEqual(s.search(dict(user_id='b',query='notebook',top_k=100)),{'data':[]})
            finally:s.close();s.native.db.close()
            s=DiskOptionsSystem(p,embedding=e,transport=Transport(),card_adapter=FakeCards())
            try:
                s.add(payload);self.assertEqual(n,len(e.calls))
                result=s.search(dict(user_id='a',query='teal notebook',top_k=100))
                self.assertTrue(result['data'])
            finally:s.close();s.native.db.close()
