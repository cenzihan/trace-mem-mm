"""Independent synthetic evidence tests; no official data or provider traffic."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_memory.mem0_evidence import Mem0EvidenceSystem, pack_mixed, VERSION
from agent_memory.mem0_bridge import Mem0System
from agent_memory.protocol import ContractError, text_of
from test_mem0_native import FakeEmbedding, FakeTransport, FakeCards


def image():
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (40, 30), 'teal').save(buf, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
            base64.b64encode(buf.getvalue()).decode()}}


class PackingTests(unittest.TestCase):
    def test_two_channels_and_top_k(self):
        raw=[dict(id='s1',content='source1'),dict(id='s2',content='source2')]
        facts=[dict(id='f1',content='Derived memory fact: fact1')]
        rows=pack_mixed(iter(raw),facts,2)
        self.assertEqual([r['id'] for r in rows],['s1','f1'])
        self.assertEqual([r['score'] for r in rows],[1.0,.5])

    def test_whole_record_skip_keeps_later_small_source_and_facts(self):
        large=dict(id='large',content=[image()])
        small=dict(id='small',content='unchanged original')
        fact=dict(id='fact',content='Derived memory fact: uncertain')
        rows=pack_mixed(iter([large,small]),[fact],3,limit=0)
        self.assertEqual([r['id'] for r in rows],['fact','small'])
        self.assertEqual(rows[-1]['content'],small['content'])


@unittest.skipUnless(importlib.util.find_spec('mem0'),'optional pinned Mem0 dependency')
class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'raw.sqlite'
        self.e=FakeEmbedding();self.t=FakeTransport();self.c=FakeCards()
        self.store=self.open()
    def open(self):
        return Mem0EvidenceSystem(self.path,embedding=self.e,transport=self.t,card_adapter=self.c)
    def close(self):
        self.store.close()
        conn=getattr(self.store.native.db,'connection',None)
        if conn:conn.close()
    def tearDown(self):
        self.close();self.temp.cleanup()
    def payload(self,contents,uid='alice',req='r',session='s'):
        return dict(user_id=uid,request_id=req,session_id=session,
                    messages=[dict(role='user',content=c) for c in contents])
    def search(self,q,uid='alice',top=100):
        return self.store.search(dict(user_id=uid,query=q,top_k=top))['data']

    def test_exact_image_returns_complete_original_before_derived_fact(self):
        pic=image()
        parts=[dict(type='text',text='original before'),pic,dict(type='text',text='original after')]
        self.store.add(self.payload([parts]))
        calls=self.c.calls
        rows=self.search([dict(type='text',text='unrelated words'),pic])
        self.assertEqual(rows[0]['content'],parts)
        self.assertEqual(self.c.calls,calls+1)  # Only original query-caption call.
        self.assertTrue(any(isinstance(r['content'],str) and r['content'].startswith('Derived memory fact:') for r in rows))
        self.assertTrue(rows[0]['id'].startswith('mem_'))

    def test_source_index_reuses_add_description_without_extra_llm(self):
        self.store.add(self.payload([[image()]]))
        before=(self.c.calls,len(self.t.calls))
        rows=self.search('teal notebook')
        self.assertTrue(any(isinstance(r['content'],list) for r in rows))
        self.assertEqual((self.c.calls,len(self.t.calls)),before)
        with self.store.connect() as db:
            value=db.execute('select indexed_text from mem0_source_index').fetchone()[0]
        self.assertIn('teal notebook',value)

    def test_context_is_same_add_and_original_order_is_preserved(self):
        self.store.add(self.payload(['earlier support','rare lexical anchor','later support']))
        self.store.add(self.payload(['unrelated elsewhere'],req='other'))
        rows=self.search('rare lexical')
        originals=[text_of(r['content']) for r in rows if isinstance(r['content'],list)]
        self.assertEqual(originals,['rare lexical anchor','earlier support','later support'])

    def test_same_user_cross_session_and_other_user_isolation(self):
        self.store.add(self.payload(['match first session'],session='first'))
        self.store.add(self.payload(['match second session'],req='r2',session='second'))
        self.store.add(self.payload(['match private source'],uid='other'))
        rows=self.search('match')
        text=' '.join(text_of(r['content']) if isinstance(r['content'],list) else r['content'] for r in rows)
        self.assertIn('first session',text);self.assertIn('second session',text)
        self.assertNotIn('private source',text)
        before=(len(self.t.calls),len(self.e.calls),self.c.calls)
        self.assertEqual(self.search('anything',uid='absent'),[])
        self.assertEqual(before,(len(self.t.calls),len(self.e.calls),self.c.calls))

    def test_restart_idempotence_and_query_is_never_indexed(self):
        payload=self.payload(['unique source witness'])
        self.store.add(payload)
        self.close();self.store=self.open()
        before=(len(self.t.calls),len(self.e.calls),self.c.calls)
        self.store.add(payload)
        self.assertEqual(before,(len(self.t.calls),len(self.e.calls),self.c.calls))
        self.search('unique query-only-sentinel')
        with self.store.connect() as db:
            rows=db.execute('select indexed_text from mem0_source_index').fetchall()
        self.assertEqual(rows,[('unique source witness',)])

    def test_incomplete_old_index_blocks_before_paid_query(self):
        self.store.add(self.payload(['source']))
        with self.store.connect() as db:db.execute('delete from mem0_source_index')
        before=(len(self.t.calls),len(self.e.calls),self.c.calls)
        with self.assertRaises(ContractError):self.search('source')
        self.assertEqual(before,(len(self.t.calls),len(self.e.calls),self.c.calls))

    def test_index_failure_rolls_back_raw_archive_and_blocks_replay(self):
        with patch.object(self.store,'_persist_descriptions',side_effect=RuntimeError('synthetic')):
            with self.assertRaises(RuntimeError):self.store.add(self.payload(['source']))
        with self.store.connect() as db:
            self.assertEqual(db.execute('select count(*) from memories').fetchone()[0],0)
            self.assertEqual(db.execute('select count(*) from mem0_source_index').fetchone()[0],0)
        before=(len(self.t.calls),len(self.e.calls),self.c.calls)
        with self.assertRaises(ContractError):self.store.add(self.payload(['source']))
        with self.assertRaises(ContractError):self.search('source')
        self.assertEqual(before,(len(self.t.calls),len(self.e.calls),self.c.calls))

    def test_explicit_factory_and_original_receipt_scope(self):
        from agent_memory.mem0_app import create_evidence_app
        from agent_memory.mem0_transport import Mem0ServiceTransport
        with self.assertRaises(ValueError):create_evidence_app({})
        self.assertEqual(Mem0ServiceTransport.receipt_scope,'native-mem0-v3.0.3')
        self.assertEqual(self.store.capabilities()['version'],VERSION)
        self.assertEqual(self.store.capabilities()['additional_source_model_calls'],0)

