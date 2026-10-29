"""Independent synthetic source-link admission and original-source regressions."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from agent_memory.identifier_support import (identifiers, linked_payload,
    IdentifierSelector, IdentifierSystem, create_identifier_app)
from agent_memory.anchored_support import checked_anchored_payload
from agent_memory.joint_selection import candidate_count
from agent_memory.protocol import content_parts, text_of
from agent_memory.provider_http import ProviderFailure
from test_completeness_selection import SequenceTransport
from test_mem0_joint import source
from test_mem0_evidence import image
from test_mem0_native import FakeEmbedding, FakeCards


class IdentifierTests(unittest.TestCase):
    def test_direct_evidence_does_not_admit_different_identifiers(self):
        t = SequenceTransport([[0]])
        rows = [source('Cargo wagon Jora74 remains in the depot.'),
                source('Jora47 is at the hill station.'), source('A spare cart is nearby.')]
        self.assertEqual(IdentifierSelector(t).select(content_parts('Where is the cargo wagon?'), rows), [0])
        self.assertEqual(len(t.calls), 1)

    def test_missing_identity_source_is_only_selectable_support(self):
        rows = [source('Jora74 remains in the depot.'), source('A spare cart is nearby.'),
                source('Cargo wagon uses identifier Jora74.'), source('Jora47 is elsewhere.')]
        t = SequenceTransport([[0], [0]])
        self.assertEqual(IdentifierSelector(t).select(content_parts('Where is the cargo wagon?'), rows), [0, 2])
        header = json.loads(t.calls[1]['messages'][1]['content'][0]['text'])
        self.assertEqual(len(header['candidates']), 1)
        self.assertEqual(candidate_count(checked_anchored_payload(t.calls[1])), 1)

    def test_no_identity_from_dates_partial_tokens_or_case_folding(self):
        self.assertEqual(identifiers('2026-10-29 1024 warehouse 3.14 AB-72 ab-72 ABCD'), {'AB-72', 'ab-72'})
        for text in ['XJora74Y is elsewhere.', 'jora74 is elsewhere.', 'Jora740 is elsewhere.']:
            self.assertEqual(linked_payload(content_parts('q'), [source('Jora74 here.'),source(text)], [], [0]), (None, []))

    def test_query_options_and_metadata_do_not_create_source_links(self):
        rows = [source('Unnumbered vessel in dock.'), source('Unrelated Jora74 at hill.')]
        rows[0]['session_group'] = 'Jora74'
        self.assertEqual(linked_payload(content_parts('Jora74?'), rows, ['Jora74'], [0]), (None, []))

    def test_identifier_after_truncated_prefix_does_not_create_visible_link(self):
        rows = [source('Jora74 here.'), source('x' * 1600 + ' Jora74 elsewhere.')]
        self.assertEqual(linked_payload(content_parts('q'), rows, [], [0]), (None, []))

    def test_images_keep_exact_bytes_roles_and_correct_remapped_indices(self):
        rows = [source('Jora74 first.', True),source('Jora47 excluded.', True),source('Jora74 linked.', True)]
        before = copy.deepcopy(rows)
        p, mapping = linked_payload([image()], rows, [], [0])
        content = p['messages'][1]['content'];labels = [json.loads(v['text']) for v in content[1:] if v['type']=='text']
        self.assertEqual(labels, [{'image_for':'query'}, {'anchor_image_for':0}, {'image_for':0}])
        self.assertEqual(mapping, [2]);self.assertEqual(rows, before)
        self.assertEqual(sum(v['type']=='image_url' for v in content), 3)

    def test_empty_full_and_no_link_skip_second_call(self):
        for first, rows in [([], [source('Jora74')]*2), (list(range(8)), [source('Jora74')]*9), ([0], [source('Words only'), source('More words')])]:
            t = SequenceTransport([first]);self.assertEqual(IdentifierSelector(t).select(content_parts('q'), rows), first)
            self.assertEqual(len(t.calls), 1)

    def test_empty_support_keeps_primary_order_and_failure_propagates(self):
        rows = [source('Jora74 one'), source('Jora74 two'), source('Jora74 three')]
        self.assertEqual(IdentifierSelector(SequenceTransport([[2,0],[]])).select(content_parts('q'),rows), [2,0])
        with self.assertRaises(ProviderFailure):
            IdentifierSelector(SequenceTransport([[0],ProviderFailure('network_error')])).select(content_parts('q'),rows)
        with self.assertRaises(ValueError):create_identifier_app({})


@unittest.skipUnless(importlib.util.find_spec('mem0'), 'pinned Mem0 required')
class IdentifierNativeTests(unittest.TestCase):
    def test_linked_originals_returned_without_foreign_user_source(self):
        with tempfile.TemporaryDirectory() as directory:
            t = SequenceTransport([[1],[0]])
            s = IdentifierSystem(Path(directory)/'memory.sqlite',embedding=FakeEmbedding(),transport=t,card_adapter=FakeCards())
            try:
                for uid,req,text in [('a','one','Cargo wagon uses identifier Jora74.'),('a','two','Jora74 remains in the depot.'),('b','private','Jora74 FOREIGN')]:
                    s.add(dict(user_id=uid,request_id=req,session_id=req,messages=[dict(role='user',content=text)]))
                rows=s.search(dict(user_id='a',query='Where is the cargo wagon?',top_k=100))['data']
                self.assertEqual({text_of(r['content']) for r in rows},{'Cargo wagon uses identifier Jora74.','Jora74 remains in the depot.'})
                self.assertNotIn('FOREIGN',json.dumps(t.calls))
            finally:
                s.close()
                connection=getattr(s.native.db,'connection',None)
                if connection:connection.close()
