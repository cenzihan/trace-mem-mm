"""Synthetic-only reader/proxy contracts: no files, native QA, HTTP or ledger."""
import copy
import json
import unittest

from agent_memory.memgallery_v7_reader import (
    MODEL, POINTS, MemGalleryReaderBudget, build_proxy_judge, decode_proxy_judge,
    decode_reader, pack_memgallery_evidence, proxy_judge_payload, question_text,
    reader_payload,
)


def query(point='FR', **extra):
    return dict(index=7, question='What is stated in the synthetic conversation?', point=point, **extra)


def sources():
    return {
        'source-one': dict(text='USER: Synthetic original 原文。\nASSISTANT: A blue object.',
                           source_session='S1', source_round='R1', source_date='2020-01-01',
                           image_ids=['D2:IMG_003', 'D10:IMG_002']),
        'source-two': dict(text='USER: A later synthetic update.\nASSISTANT: A green object.',
                           source_session='S2', source_round='R1', source_date='2020-01-03',
                           image_ids=['D2:IMG_010']),
    }


def packed(point='FR', **extra):
    return pack_memgallery_evidence(query(point, **extra), list(sources()), sources(), len)


def value(**changes):
    result = dict(summary='Any auxiliary text.', evidence_quotes=['Synthetic quote.'],
                  answer='A blue object.', cited_ids=['source-one'], confidence='supported')
    result.update(changes)
    return result


def response(content):
    return dict(model='gpt-4o-mini-2024-07-18', choices=[dict(finish_reason='stop', message=dict(
        content=content if isinstance(content, str) else json.dumps(content)))])


def judge_items():
    return [dict(item_id=i, point=POINTS[i], question='Synthetic question ' + str(i),
                 reference='Synthetic reference.', prediction='Synthetic prediction.') for i in range(8)]


def judge_response(scores=(0, .25, .5, .75, 1)):
    return response(dict(summary='Independent scores.', evidence_quotes=[], judgments=[
        dict(item_id=i, score=score, reason='Synthetic reason.') for i, score in enumerate(scores)]))


class MemGalleryV7ReaderTests(unittest.TestCase):
    def test_complete_source_prefix_preserves_native_metadata_and_input(self):
        q, rows = query(), sources()
        before = copy.deepcopy((q, rows))
        result = pack_memgallery_evidence(q, list(rows), rows, len)
        payload = reader_payload(result)
        self.assertEqual((q, rows), before)
        text = payload['messages'][1]['content'][0]['text']
        for row in rows.values():
            self.assertIn(row['text'], text)
            self.assertIn(row['source_date'], text)
            for image_id in row['image_ids']:
                self.assertIn(image_id, text)
        self.assertEqual(result['source_memory_ids'], list(rows))
        self.assertEqual(result['image_count'], 0)
        self.assertEqual(payload['model'], MODEL)
        self.assertEqual(payload['max_completion_tokens'], 768)
        self.assertEqual(payload['temperature'], 0)
        self.assertFalse(result['context_gold_access'])
        payload['messages'][1]['content'][0]['text'] = 'changed copy'
        self.assertNotEqual(payload, reader_payload(result))

    def test_query_image_captions_are_present_but_paths_not_exposed(self):
        q = query('VR', question_image=['private/path1', 'private/path2'],
                  image_caption=['完整 caption one.', 'Second exact caption.'])
        text = question_text(q)
        self.assertTrue(text.startswith(q['question']))
        self.assertIn('完整 caption one.', text)
        self.assertNotIn('private/path', text)
        result = pack_memgallery_evidence(q, list(sources()), sources(), len)
        self.assertEqual(result['query_image_count'], 2)
        self.assertEqual(result['query_image_condition'], 'author-caption-only')
        self.assertIn(text, result['parts'][0]['text'])
        alias = query(query_image='path', image_caption='Caption.')
        self.assertIn('Caption.', question_text(alias))
        self.assertEqual(question_text(query(question_image=None, image_caption=None)), query()['question'])

    def test_missing_unaligned_ambiguous_image_captions_fail_closed(self):
        cases = (
            dict(question_image='image'), dict(question_image='image', image_caption=' '),
            dict(question_image=['a', 'b'], image_caption=['one']),
            dict(question_image='a', query_image='b', image_caption='one'),
            dict(image_caption='orphan caption'), dict(question_image=['a', ''], image_caption=['a', 'b']),
            dict(question_image='a', image_caption=[123]),
        )
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                packed(**extra)

    def test_gold_fields_unknown_point_and_invalid_index_rejected(self):
        for extra in ({'answer': 'gold'}, {'clue': ['R1']}, {'profile': {}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                packed(**extra)
        for q in (dict(query(), point='number'), dict(query(), index=True), dict(query(), index=-1)):
            with self.assertRaises(ValueError):
                pack_memgallery_evidence(q, list(sources()), sources(), len)
        rows = sources(); rows['source-one']['answer'] = 'gold'
        with self.assertRaises(ValueError):
            pack_memgallery_evidence(query(), list(rows), rows, len)

    def test_whole_prefix_does_not_crop_or_skip_large_first_round(self):
        rows = sources(); rows['source-one']['text'] = '原' * 9000
        result = pack_memgallery_evidence(query(), list(rows), rows, lambda s: max(1, len(s)//4))
        self.assertEqual(result['source_memory_ids'], [])
        self.assertEqual(result['dropped_native_ids'], list(rows))
        self.assertNotIn(rows['source-two']['text'], result['parts'][0]['text'])
        # A later oversized round preserves the complete first round only.
        rows = sources(); rows['source-two']['text'] = '原' * 9000
        result = pack_memgallery_evidence(query(), list(rows), rows, lambda s: max(1, len(s)//4))
        self.assertEqual(result['source_memory_ids'], ['source-one'])
        self.assertIn(rows['source-one']['text'], result['parts'][0]['text'])

    def test_top20_and_all_candidate_validation(self):
        rows = {str(i): dict(sources()['source-one'], text='short', image_ids=[]) for i in range(25)}
        result = pack_memgallery_evidence(query(), list(rows), rows, lambda s: max(1, len(s)//4))
        self.assertEqual(result['source_records_retained'], 20)
        self.assertEqual(len(result['ranked_native_ids']), 20)
        rows['24']['text'] = ''
        with self.assertRaises(ValueError):
            pack_memgallery_evidence(query(), list(rows), rows, len)
        with self.assertRaises(ValueError):
            pack_memgallery_evidence(query(), ['source-one'] * 2, sources(), len)
        with self.assertRaises(ValueError):
            pack_memgallery_evidence(query(), ['unknown'], sources(), len)

    def test_budget_type_limits_and_question_overflow(self):
        for kwargs in (dict(images=1), dict(images=False), dict(candidate_k=21), dict(candidate_k=True),
                       dict(text_tokens=8193), dict(utf8_bytes=24001), dict(output_tokens=1536)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MemGalleryReaderBudget(**kwargs)
        for counter in (lambda _: 0, lambda _: True, lambda _: 1.5, lambda _: 5000):
            with self.assertRaises(ValueError):
                pack_memgallery_evidence(query(), list(sources()), sources(), counter)
        q = dict(query(), question='原' * 9000)
        with self.assertRaises(ValueError):
            pack_memgallery_evidence(q, [], {}, lambda s: max(1, len(s)//4))

    def test_image_id_type_collision_and_duplicates_are_rejected(self):
        for values in (([1], ['1']), (['same', 'same'], []), (['bad,id'], []), ([True], [])):
            rows = sources()
            rows['source-one']['image_ids'], rows['source-two']['image_ids'] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                pack_memgallery_evidence(query(), list(rows), rows, len)

    def test_auxiliary_types_and_substantive_answer_preserved_exactly(self):
        answer = '  原文\n"quoted" \\ tab\t and trailing  '
        raw = response(value(answer=answer, summary='', evidence_quotes=['', 'arbitrary']))
        before = copy.deepcopy(raw)
        result = decode_reader(raw, packed())
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['schema_status'], 'valid')
        self.assertEqual(raw, before)

    def test_native_vs_uses_image_ids_and_natural_numeric_order(self):
        result = decode_reader(response(value(answer='D2:IMG_003, D2:IMG_010, D10:IMG_002')), packed('VS'))
        self.assertEqual(result['answer_image_ids'], ['D2:IMG_003', 'D2:IMG_010', 'D10:IMG_002'])
        for answer in ('source-one', 'D99:IMG_001', 'D2:IMG_003, D2:IMG_003',
                       'D10:IMG_002, D2:IMG_003', 'D2:IMG_003,D2:IMG_010', 'D2:IMG_003.'):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                decode_reader(response(value(answer=answer)), packed('VS'))
        rows = sources(); rows['source-one']['image_ids'] = [1, 2, 10]
        p = pack_memgallery_evidence(query('VS'), list(rows), rows, len)
        self.assertEqual(decode_reader(response(value(answer='1, 2, 10')), p)['answer_image_ids'], ['1', '2', '10'])

    def test_cd_exact_polarity_and_ar_native_abstention(self):
        for answer in ('Yes.', 'No.'):
            self.assertEqual(decode_reader(response(value(answer=answer)), packed('CD'))['answer'], answer)
        for answer in ('Yes', 'yes.', 'Yes. Explanation.', 'Not mentioned.'):
            with self.assertRaises(ValueError):
                decode_reader(response(value(answer=answer)), packed('CD'))
        raw = response(value(answer='Not mentioned.', cited_ids=[], confidence='insufficient'))
        self.assertEqual(decode_reader(raw, packed('AR'))['answer'], 'Not mentioned.')
        # A point label never forces abstention when evidence supports an answer.
        self.assertEqual(decode_reader(response(value()), packed('AR'))['answer'], value()['answer'])
        for point in POINTS:
            answer = 'Not mentioned.' if point == 'AR' else 'INSUFFICIENT_EVIDENCE'
            self.assertEqual(decode_reader(response(value(answer=answer, cited_ids=[], confidence='insufficient')),
                                          packed(point))['confidence'], 'insufficient')

    def test_unknown_citations_and_inconsistent_abstention_rejected(self):
        for changes in (dict(cited_ids=['invented']), dict(cited_ids=['source-one'] * 2),
                        dict(cited_ids=[1]), dict(cited_ids='source-one'),
                        dict(answer='INSUFFICIENT_EVIDENCE'), dict(confidence='insufficient'),
                        dict(answer='INSUFFICIENT_EVIDENCE', confidence='insufficient')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decode_reader(response(value(**changes)), packed())

    def test_exact_reader_fields_and_types(self):
        bad = [dict(value(), extra='x'), value(summary=1), value(evidence_quotes=[1]),
               value(evidence_quotes='quote'), value(answer=None), value(confidence='maybe')]
        for key in value():
            row = value(); del row[key]; bad.append(row)
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                decode_reader(response(row), packed())

    def test_strict_response_envelope_and_json(self):
        mutations = (lambda r: r.update(model='another-model'), lambda r: r.update(choices=[]),
                     lambda r: r.update(choices=r['choices'] * 2),
                     lambda r: r['choices'][0].update(finish_reason='length'),
                     lambda r: r['choices'][0]['message'].update(refusal='refused'),
                     lambda r: r['choices'][0]['message'].update(tool_calls=[{}]),
                     lambda r: r['choices'][0]['message'].update(function_call={"name": "x"}))
        for change in mutations:
            raw = response(value()); change(raw)
            with self.assertRaises(ValueError):
                decode_reader(raw, packed())
        valid = json.dumps(value())
        for raw in ('{"summary":"duplicate",' + valid[1:], '```json\n' + valid + '\n```',
                    valid.replace('"confidence": "supported"', '"confidence": NaN'),
                    valid.replace('"confidence": "supported"', '"confidence": Infinity'),
                    valid.replace('"confidence": "supported"', '"confidence": 1e309')):
            with self.assertRaises(ValueError):
                decode_reader(response(raw), packed())

    def test_pack_tampering_blocks_payload_and_decode(self):
        p = packed(); p['parts'][0]['text'] += 'modified'
        with self.assertRaises(ValueError): reader_payload(p)
        with self.assertRaises(ValueError): decode_reader(response(value()), p)


class MemGalleryV7ProxyTests(unittest.TestCase):
    def test_blind_bounded_payload_and_immutable_copy(self):
        rows = judge_items(); before = copy.deepcopy(rows)
        pack = build_proxy_judge(rows, len)
        payload = proxy_judge_payload(pack)
        self.assertEqual(rows, before)
        self.assertEqual(pack['item_ids'], list(range(8)))
        self.assertIn('not-official', pack['metric'])
        self.assertEqual(payload['max_completion_tokens'], 1536)
        self.assertEqual(payload['model'], MODEL)
        self.assertEqual(payload['temperature'], 0)
        payload['model'] = 'changed'
        self.assertEqual(proxy_judge_payload(pack)['model'], MODEL)

    def test_five_grades_are_preserved_without_upstream_rounding(self):
        self.assertEqual(decode_proxy_judge(judge_response(), list(range(5))),
                         {0: 0., 1: .25, 2: .5, 3: .75, 4: 1.})

    def test_judge_input_blinding_identity_types_and_bounds(self):
        cases = [[], judge_items() + [judge_items()[0]], judge_items()[:1] * 2]
        for changes in (dict(method='dense'), dict(item_id=True), dict(point='number'),
                        dict(reference=None), dict(question=''), dict(prediction=12)):
            rows = judge_items(); rows[0].update(changes); cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                build_proxy_judge(rows, len)
        rows = judge_items(); rows[0]['reference'] = '原' * 9000
        with self.assertRaises(ValueError):
            build_proxy_judge(rows, lambda s: max(1, len(s)//4))
        with self.assertRaises(ValueError): build_proxy_judge(judge_items(), lambda _: 5000)

    def test_judge_wrong_duplicate_missing_ids_and_invalid_scores(self):
        for score in (True, .1, .8, -1, 2, '1', None):
            with self.subTest(score=score), self.assertRaises(ValueError):
                decode_proxy_judge(judge_response([score]), [0])
        base = json.loads(judge_response()['choices'][0]['message']['content'])
        cases = []
        for changes in (dict(item_id=99), dict(item_id=True), dict(extra='x'), dict(reason='x' * 401)):
            row = copy.deepcopy(base); row['judgments'][0].update(changes); cases.append(row)
        row = copy.deepcopy(base); row['judgments'][1]['item_id'] = 0; cases.append(row)
        row = copy.deepcopy(base); row['judgments'].pop(); cases.append(row)
        row = copy.deepcopy(base); row['evidence_quotes'] = [3]; cases.append(row)
        for row in cases:
            with self.assertRaises(ValueError): decode_proxy_judge(response(row), list(range(5)))
        raw = json.dumps(base).replace('"score": 0', '"score": NaN', 1)
        with self.assertRaises(ValueError): decode_proxy_judge(response(raw), list(range(5)))
        with self.assertRaises(ValueError): decode_proxy_judge(judge_response(), [0, 0, 1, 2, 3])
        raw = judge_response(); raw['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError): decode_proxy_judge(raw, list(range(5)))

    def test_judge_pack_tampering_is_rejected(self):
        pack = build_proxy_judge(judge_items(), len)
        pack['payload']['max_completion_tokens'] = 1600
        with self.assertRaises(ValueError): proxy_judge_payload(pack)


if __name__ == '__main__':
    unittest.main()
