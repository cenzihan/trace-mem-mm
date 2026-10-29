"""Fresh synthetic visibility controls; no official inputs or network calls."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.extended_views import (extended_payload, text_bytes,
    ExtendedSelector, ExtendedSystem, create_extended_app)
from agent_memory.joint_selection import selection_payload, candidate_count, SelectionBudgetedTransport
from agent_memory.protocol import content_parts, text_of
from agent_memory.provider_http import ProviderFailure
from test_completeness_selection import SequenceTransport
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding, FakeCards


def header(payload):
    return json.loads(payload['messages'][1]['content'][0]['text'])


class ExtendedViewTests(unittest.TestCase):
    def test_short_inputs_are_byte_identical_to_baseline(self):
        sources = [source('Copper sailboat is called Juniper.', True), source('Juniper entered the cove.')]
        parts = content_parts('Where is the copper sailboat?')
        self.assertEqual(extended_payload(parts,sources,['cove','dock']),
                         selection_payload(parts,sources,['cove','dock']))

    def test_late_pure_name_link_and_temporal_evidence_become_visible(self):
        padding = 'Routine inventory entry; no location or ownership update. ' * 40
        sources = [source(padding+'The copper sailboat is called Juniper.'),
                   source(padding+'Juniper left the dock on Tuesday and entered the cove Wednesday.')]
        before = copy.deepcopy(sources)
        old = header(selection_payload(content_parts('Where did the copper sailboat move?'),sources))
        new = header(extended_payload(content_parts('Where did the copper sailboat move?'),sources))
        self.assertTrue(all('Juniper' not in r['text_prefix'] for r in old['candidates']))
        self.assertTrue(all('Juniper' in r['text_prefix'] for r in new['candidates']))
        self.assertTrue(all(not r['text_truncated'] for r in new['candidates']))
        self.assertEqual(sources,before)

    def test_many_escaped_unicode_sources_fit_original_transport_budget(self):
        sources=[source(('汉字\\\"\n' * 2500),True) for _ in range(32)]
        p=extended_payload([image()]+content_parts('q'*4000),sources,['o'*3500])
        rows=header(p)['candidates']
        self.assertLessEqual(text_bytes(p),94000)
        self.assertTrue(all(1497<=len(r['text_prefix'].encode())<=6000 for r in rows))
        self.assertEqual(len({r['text_prefix'] for r in rows}),1)
        self.assertEqual(sum(v['type']=='image_url' for v in p['messages'][1]['content']),8)
        self.assertLessEqual(len(json.dumps(p,ensure_ascii=False).encode()),3*1024*1024)
        transport=object.__new__(SelectionBudgetedTransport)
        transport.model='gpt-4o-mini';transport.output_token_cap=512
        self.assertEqual(transport._bounds(p),(128000,512))

    def test_truncation_remains_explicit_and_images_labels_unchanged(self):
        rows=[source('x'*8000,True) for _ in range(3)]
        old=selection_payload([image()],rows);new=extended_payload([image()],rows)
        self.assertEqual(old['messages'][1]['content'][1:],new['messages'][1]['content'][1:])
        self.assertEqual(old['messages'][0],new['messages'][0])
        self.assertTrue(all(len(r['text_prefix'])==6000 and r['text_truncated'] for r in header(new)['candidates']))
        self.assertEqual(candidate_count(new),3)

    def test_one_call_empty_selection_valid_and_failure_not_hidden(self):
        rows=[source('Unrelated clay pot stayed in the shed.')]
        t=SequenceTransport([[]])
        self.assertEqual(ExtendedSelector(t).select(content_parts('Where is the sailboat?'),rows),[])
        self.assertEqual(len(t.calls),1)
        with self.assertRaises(ProviderFailure):
            ExtendedSelector(SequenceTransport([ProviderFailure('network_error')])).select(content_parts('q'),rows)
        with self.assertRaises(ValueError):create_extended_app({})


@unittest.skipUnless(importlib.util.find_spec('mem0'),'pinned Mem0 required')
class ExtendedNativeTests(unittest.TestCase):
    def test_complete_originals_returned_and_foreign_user_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            t=SequenceTransport([[0,1]])
            s=ExtendedSystem(Path(directory)/'memory.sqlite',embedding=FakeEmbedding(),transport=t,card_adapter=FakeCards())
            texts=['Routine note. '*160+'Copper sailboat is called Juniper.',
                   'Routine note. '*160+'Juniper is now in the cove.']
            try:
                for uid,req,text in [('a','one',texts[0]),('a','two',texts[1]),('b','foreign','FOREIGN private source')]:
                    s.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=text)]))
                result=s.search(dict(user_id='a',query='Where is the copper sailboat?',top_k=100))['data']
                self.assertEqual({text_of(r['content']) for r in result},set(texts))
                self.assertNotIn('FOREIGN',json.dumps(t.calls))
                self.assertEqual(len(t.calls),1)
                self.assertTrue(all(not r['text_truncated'] for r in header(t.calls[0])['candidates']))
            finally:
                s.close()
                connection=getattr(s.native.db,'connection',None)
                if connection:connection.close()
