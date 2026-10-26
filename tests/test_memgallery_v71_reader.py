"""Synthetic V7.1 citation-interface tests; no native files, gold or providers."""
import copy
import json
import unittest

from agent_memory import memgallery_v7_reader as old
from agent_memory import memgallery_v71_reader as new


def question(point='FR'):
    return dict(index=7, question=' Original synthetic question?\n', point=point,
                question_image='private-synthetic-reference', image_caption='Original query caption.')


def sources():
    return {
        '["S2","R1"]': dict(text='Exact USER: blue.\nASSISTANT: 原文 unchanged.', source_session='S2',
                            source_round='R1', source_date='2025-01-02', image_ids=['IMG_2']),
        '["S1","R1"]': dict(text='Exact USER: green.\nASSISTANT: quotation " unchanged.', source_session='S1',
                            source_round='R1', source_date='2025-01-01', image_ids=['IMG_1']),
        '["S3","R9"]': dict(text='Outside this candidate pool but in the complete corpus.', source_session='S3',
                            source_round='R9', source_date='2025-01-03', image_ids=['IMG_3']),
    }


def response(answer='Original answer.', cited_ids=None, confidence='supported'):
    value = dict(summary='Free auxiliary text.', evidence_quotes=['Original quote.'], answer=answer,
                 cited_ids=['M0001'] if cited_ids is None else cited_ids, confidence=confidence)
    return dict(model='gpt-4o-mini-2024-07-18', choices=[dict(finish_reason='stop',
                message=dict(content=json.dumps(value, ensure_ascii=False)))])


class MemGalleryV71ReaderTests(unittest.TestCase):
    def pack(self, point='FR', **options):
        return new.pack_memgallery_evidence(question(point), list(sources())[:2], sources(), len, **options)

    def test_global_aliases_are_rank_and_dictionary_order_independent(self):
        mapping = sources()
        forward = new.pack_memgallery_evidence(question(), list(mapping)[:2], mapping, len)
        backward = new.pack_memgallery_evidence(question(), list(reversed(list(mapping)[:2])),
            dict(reversed(list(mapping.items()))), len)
        expected = {'["S1","R1"]': 'M0001', '["S2","R1"]': 'M0002', '["S3","R9"]': 'M0003'}
        self.assertEqual(forward['alias_by_native_id'], expected)
        self.assertEqual(backward['alias_by_native_id'], expected)
        self.assertEqual(forward['alias_map_sha256'], backward['alias_map_sha256'])
        self.assertEqual(forward['source_memory_ids'], list(mapping)[:2])
        self.assertEqual(forward['alias_pack']['source_memory_ids'], ['M0002', 'M0001'])

    def test_only_source_header_serialization_and_citation_appendix_change(self):
        q, rows = question(), sources()
        original_inputs = copy.deepcopy((q, rows))
        ranking = list(rows)[:2]
        baseline = old.pack_memgallery_evidence(q, ranking, rows, len)
        packed = new.pack_memgallery_evidence(q, ranking, rows, len)
        payload = new.reader_payload(packed)
        self.assertEqual((q, rows), original_inputs)
        self.assertEqual(payload['messages'][0]['content'], old.READER_PROMPT + new.CITATION_INSTRUCTION)
        expected = baseline['parts'][0]['text']
        for native, alias in packed['alias_by_native_id'].items():
            expected = expected.replace('"source_id":' + old._json(native), '"source_id":' + old._json(alias))
        self.assertEqual(payload['messages'][1]['content'][0]['text'], expected)
        self.assertEqual(packed['question_sha256'], baseline['question_sha256'])
        self.assertEqual(packed['image_ids'], baseline['image_ids'])
        self.assertEqual(packed['source_sha256'], baseline['source_sha256'])
        self.assertEqual(packed['source_memory_ids'], baseline['source_memory_ids'])
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['max_completion_tokens'], 768)
        self.assertEqual(packed['image_count'], 0)

    def test_complete_actual_payload_budget_and_whole_prefix(self):
        packed = self.pack()
        wire = new.reader_payload(packed)
        system = wire['messages'][0]['content']; user = wire['messages'][1]['content'][0]['text']
        self.assertEqual(packed['text_token_count'], len(system) + len(user) + 32)
        self.assertEqual(packed['utf8_text_bytes'], len(system.encode()) + len(user.encode()))
        self.assertLessEqual(packed['text_token_count'], 8192)
        self.assertLessEqual(packed['utf8_text_bytes'], 24000)
        rows = sources(); ranking = list(rows)[:2]
        rows[ranking[0]]['text'] = '原' * 10000
        result = new.pack_memgallery_evidence(question(), ranking, rows, lambda text: max(1, len(text)//4))
        self.assertEqual(result['source_memory_ids'], [])
        self.assertNotIn(rows[ranking[1]]['text'], new.reader_payload(result)['messages'][1]['content'][0]['text'])
        self.assertRaises(ValueError, self.pack, budget=new.MemGalleryReaderBudget(text_tokens=1))

    def test_fake_model_payload_and_native_decode_preserve_answer_and_context(self):
        packed = self.pack()
        answer = '  Original answer\n"quoted" 原文 and trailing  '
        def fake_model(payload):
            self.assertEqual(payload, new.reader_payload(packed))
            self.assertIn('M followed by exactly four digits', payload['messages'][0]['content'])
            self.assertIn('"source_id":"M0001"', payload['messages'][1]['content'][0]['text'])
            return response(answer, ['M0001', 'M0002'])
        raw = fake_model(new.reader_payload(packed))
        before = copy.deepcopy(raw)
        decoded = new.decode_reader(raw, packed)
        self.assertEqual(decoded['answer'], answer)
        self.assertEqual(decoded['cited_aliases'], ['M0001', 'M0002'])
        self.assertEqual(decoded['cited_ids'], ['["S1","R1"]', '["S2","R1"]'])
        self.assertEqual(decoded['context_memory_ids'], packed['source_memory_ids'])
        self.assertEqual(decoded['context_aliases'], ['M0002', 'M0001'])
        self.assertEqual(decoded['payload_sha256'], packed['payload_sha256'])
        self.assertEqual(decoded['pack_sha256'], packed['pack_sha256'])
        self.assertEqual(raw, before)

    def test_wrong_unknown_duplicate_and_out_of_context_aliases_are_not_repaired(self):
        packed = self.pack()
        for cited in (['S1'], ['R1'], ['IMG_1'], ['["S1","R1"]'], ['M0003'], ['M9999'],
                      ['M0001', 'M0001'], ['m0001'], ['M1'], [' M0001']):
            with self.subTest(kind=cited), self.assertRaises(ValueError):
                new.decode_reader(response(cited_ids=cited), packed)

    def test_typed_answer_and_completion_rules_are_unchanged(self):
        self.assertEqual(new.decode_reader(response('Yes.'), self.pack('CD'))['answer'], 'Yes.')
        self.assertRaises(ValueError, new.decode_reader, response('yes'), self.pack('CD'))
        vs = new.decode_reader(response('IMG_1, IMG_2'), self.pack('VS'))
        self.assertEqual(vs['answer_image_ids'], ['IMG_1', 'IMG_2'])
        self.assertRaises(ValueError, new.decode_reader, response('M0001'), self.pack('VS'))
        abstention = new.decode_reader(response('Not mentioned.', [], 'insufficient'), self.pack('AR'))
        self.assertEqual(abstention['answer'], 'Not mentioned.')
        raw = response(); raw['choices'][0]['finish_reason'] = 'length'
        self.assertRaises(ValueError, new.decode_reader, raw, self.pack())

    def test_tampered_mapping_payload_and_native_outer_context_fail_closed(self):
        packed = self.pack()
        for change in ('mapping', 'payload', 'context', 'tokens'):
            bad = copy.deepcopy(packed)
            if change == 'mapping':
                bad['native_id_by_alias']['M0001'] = '["S2","R1"]'
            elif change == 'payload':
                bad['payload']['messages'][0]['content'] += 'changed semantic rule'
            elif change == 'context':
                bad['source_memory_ids'].reverse()
            else:
                bad['text_token_count'] = 1
            # Even rehashing the outer artifact cannot bypass internal consistency.
            bad['pack_sha256'] = old._sha(old._json({key: value for key, value in bad.items() if key != 'pack_sha256'}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                new.reader_payload(bad)


if __name__ == '__main__':
    unittest.main()
