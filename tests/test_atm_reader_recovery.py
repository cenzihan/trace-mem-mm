"""Synthetic receipt-only recovery tests; no files, gold, credentials or HTTP."""
import copy
import json
import unittest

from agent_memory.atm_reader import ATMTextReader, pack_atm_evidence
from agent_memory.atm_reader_recovery import recover_decode


def pack(qtype='open_end'):
    return pack_atm_evidence(
        dict(id='synthetic-q', question='What does the synthetic record support?', qtype=qtype),
        ['email0001', 'email0002'],
        {'email0001': 'Synthetic fact one.', 'email0002': 'Synthetic fact two.'}, len)


def value(**changes):
    result = dict(summary='atm reader output', evidence_quotes=[], answer='Synthetic fact one.',
                  answer_ids=[], cited_ids=['email0001'], confidence='supported')
    result.update(changes)
    return result


def response(content=None):
    if content is None:
        content = value()
    return dict(model='gpt-4o-mini-2024-07-18',
                choices=[dict(finish_reason='stop', message=dict(
                    content=content if isinstance(content, str) else json.dumps(content)))])


class ATMReaderRecoveryTests(unittest.TestCase):
    def test_only_auxiliary_fields_normalized_answer_exact_and_inputs_unchanged(self):
        answer = '  原文\n"quoted" \\ literal\t trailing  '
        content = value(summary='A descriptive summary', evidence_quotes=['quote one', '引文二'], answer=answer)
        raw, packed = response(content), pack()
        original_raw, original_pack = copy.deepcopy(raw), copy.deepcopy(packed)
        with self.assertRaises(ValueError):
            ATMTextReader.decode(raw, packed)
        decoded, provenance = recover_decode(raw, packed)
        expected = response(value(answer=answer))
        self.assertEqual(decoded, ATMTextReader.decode(expected, packed))
        self.assertEqual(decoded['answer'], answer)
        self.assertEqual(provenance, dict(summary_changed=True, evidence_quotes_changed=True,
                                         evidence_quote_count_removed=2, auxiliary_fields_changed=2))
        self.assertEqual(raw, original_raw)
        self.assertEqual(packed, original_pack)

    def test_original_valid_response_unchanged_and_each_auxiliary_change_counted(self):
        packed = pack()
        raw = response()
        decoded, provenance = recover_decode(raw, packed)
        self.assertEqual(decoded, ATMTextReader.decode(raw, packed))
        self.assertEqual(provenance, dict(summary_changed=False, evidence_quotes_changed=False,
                                         evidence_quote_count_removed=0, auxiliary_fields_changed=0))
        for content, summary_changed, quotes_changed in (
            (value(summary=''), True, False),
            (value(evidence_quotes=['']), False, True),
        ):
            with self.subTest(content=content):
                _, provenance = recover_decode(response(content), packed)
                self.assertEqual(provenance['summary_changed'], summary_changed)
                self.assertEqual(provenance['evidence_quotes_changed'], quotes_changed)
                self.assertEqual(provenance['auxiliary_fields_changed'], 1)

    def test_exact_six_fields_and_auxiliary_types_required(self):
        invalid = []
        for field in value():
            content = value()
            del content[field]
            invalid.append(content)
        invalid.append(value(extra='not allowed'))
        invalid.extend(value(summary=bad) for bad in (None, 1, [], {}))
        invalid.extend(value(evidence_quotes=bad) for bad in (None, 'quote', {}, [1], [None], [{}], [['nested']]))
        invalid.extend(([], 1, None))
        for content in invalid:
            with self.subTest(content=content):
                # Encode None explicitly because response() defaults to valid data.
                with self.assertRaises(ValueError):
                    recover_decode(response(json.dumps(content)), pack())

    def test_duplicate_keys_nonfinite_and_invalid_json_rejected(self):
        valid = json.dumps(value(summary='descriptive'))
        for content in (
            '{"summary":"first",' + valid[1:],
            valid[:-1] + ',"answer":"second"}',
            valid.replace('"evidence_quotes": []', '"evidence_quotes": [NaN]'),
            valid.replace('"evidence_quotes": []', '"evidence_quotes": [Infinity]'),
            valid.replace('"evidence_quotes": []', '"evidence_quotes": [-Infinity]'),
            valid.replace('"evidence_quotes": []', '"evidence_quotes": [1e309]'),
            '```json\n' + valid + '\n```',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                recover_decode(response(content), pack())

    def test_model_choices_refusal_tools_and_truncation_remain_rejected(self):
        mutations = (
            lambda raw: raw.update(model='other-model'),
            lambda raw: raw.update(choices=[]),
            lambda raw: raw.update(choices=raw['choices'] * 2),
            lambda raw: raw.update(choices=[None]),
            lambda raw: raw['choices'][0].update(finish_reason='length'),
            lambda raw: raw['choices'][0].update(finish_reason='content_filter'),
            lambda raw: raw['choices'][0].update(message=None),
            lambda raw: raw['choices'][0]['message'].update(refusal='refused'),
            lambda raw: raw['choices'][0]['message'].update(tool_calls=[{'id': 'synthetic-tool'}]),
            lambda raw: raw['choices'][0]['message'].update(content={}),
        )
        for mutation in mutations:
            raw = response(value(summary='descriptive'))
            mutation(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                recover_decode(raw, pack())
        with self.assertRaises(ValueError):
            recover_decode(None, pack())

    def test_unknown_ids_preserved_and_flagged_never_repaired(self):
        ids = ['email0001', 'unseen-native-id']
        raw = response(value(summary='descriptive', evidence_quotes=['decorative'],
                             answer=', '.join(ids), answer_ids=ids, cited_ids=['unseen-native-id']))
        decoded, _ = recover_decode(raw, pack('list_recall'))
        self.assertEqual(decoded['answer'], ', '.join(ids))
        self.assertEqual(decoded['answer_ids'], ids)
        self.assertEqual(decoded['cited_ids'], ['unseen-native-id'])
        self.assertFalse(decoded['answer_ids_valid'])
        self.assertFalse(decoded['citations_valid'])

    def test_invalid_id_arrays_and_list_mismatch_cannot_bypass_original_decoder(self):
        for field in ('answer_ids', 'cited_ids'):
            for invalid_ids in ('email0001', [''], [' '], [1], ['email0001'] * 2,
                                ['email' + str(index) for index in range(21)]):
                content = value(summary='descriptive', evidence_quotes=['quote'])
                content[field] = invalid_ids
                with self.subTest(field=field, ids=invalid_ids), self.assertRaises(ValueError):
                    recover_decode(response(content), pack('list_recall'))
        with self.assertRaises(ValueError):
            recover_decode(response(value(summary='descriptive', answer='reformatted',
                                          answer_ids=['email0001'])), pack('list_recall'))
        with self.assertRaises(ValueError):
            recover_decode(response(value(summary='descriptive', answer_ids=['email0001'])), pack())

    def test_abstention_confidence_and_answer_types_cannot_be_relaxed(self):
        valid = value(summary='descriptive', evidence_quotes=['quote'], answer='INSUFFICIENT_EVIDENCE',
                      answer_ids=[], cited_ids=[], confidence='insufficient')
        decoded, _ = recover_decode(response(valid), pack())
        self.assertEqual(decoded['answer'], 'INSUFFICIENT_EVIDENCE')
        self.assertEqual(decoded['confidence'], 'insufficient')
        invalid = [dict(valid, answer='invented'), dict(valid, cited_ids=['email0001']),
                   dict(valid, answer_ids=['email0001']), dict(valid, confidence='supported'),
                   value(summary='descriptive', confidence='maybe'),
                   value(summary='descriptive', confidence=None), value(summary='descriptive', answer=None),
                   value(summary='descriptive', answer=42), value(summary='descriptive', answer=' ')]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                recover_decode(response(content), pack())

    def test_frozen_pack_tampering_still_rejected(self):
        for field in ('parts', 'source_memory_ids', 'payload_sha256', 'prompt_sha256', 'pack_sha256'):
            packed = pack()
            packed[field] = 'tampered'
            with self.subTest(field=field), self.assertRaises(ValueError):
                recover_decode(response(value(summary='descriptive')), packed)


if __name__ == '__main__':
    unittest.main()
