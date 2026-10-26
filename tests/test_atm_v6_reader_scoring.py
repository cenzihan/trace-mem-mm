"""Temporary synthetic 16x4 reader outcomes; never native gold or provider calls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_memory.atm_reader import ATMTextReader, VARIANTS, pack_atm_evidence, query_id
from agent_memory.atm_answer_metrics import deterministic_accuracy, list_jaccard_score, UPSTREAM_PROVENANCE
from scripts import score_atm_v6_reader as scoring


class ATMV6ReaderScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (*scoring.CODE, 'agent_memory/atm_reader.py', 'scripts/run_atm_v6_reader.py'):
            self.write(name, b'# Synthetic immutable code fixture\n')
        queries, selected, packs, answers, specs = [], [], [], [], []
        for dataset in ('regular', 'hard'):
            gold = []
            for i in range(8):
                qtype = ('number', 'list_recall', 'open_end')[i % 3]
                identity = query_id(dataset, 'q'+str(i))
                question = dict(id='q'+str(i), qtype=qtype, question='SYNTHETIC_PRIVATE_QUESTION_'+dataset+str(i))
                query = dict(query_id=identity, dataset=dataset, **question)
                queries.append(query)
                selected.append({key: query[key] for key in ('query_id', 'dataset', 'id', 'qtype')})
                reference = '1,000' if qtype == 'number' else 'email0001, email0002' if qtype == 'list_recall' else 'SYNTHETIC_PRIVATE_REFERENCE_BLUE'
                gold.append(dict(question, answer=reference, evidence_ids=['email0001']))
                source = {'email0001': 'Synthetic source one', 'email0002': 'Synthetic source two'}
                packed = pack_atm_evidence(question, list(source), source, len)
                for variant in VARIANTS:
                    packs.append(dict(query_id=identity, dataset=dataset, id=question['id'], qtype=qtype, variant=variant, pack=packed))
                    text = '1000' if qtype == 'number' else reference
                    result = dict(summary='atm reader output', evidence_quotes=[], answer=text,
                        answer_ids=['email0001', 'email0002'] if qtype == 'list_recall' else [],
                        cited_ids=['email0001'], confidence='supported')
                    decoded = ATMTextReader.decode(dict(model='gpt-4o-mini-2024-07-18', choices=[dict(
                        finish_reason='stop', message=dict(content=json.dumps(result)))]), packed)
                    ordinal = len(answers)+1
                    answers.append(dict(query_id=identity, dataset=dataset, id=question['id'], qtype=qtype,
                        variant=variant, answer=decoded,
                        request_id=hashlib.sha256((identity+'\0'+variant).encode()).hexdigest(),
                        receipt_sha256=hashlib.sha256(str(ordinal).encode()).hexdigest(), attempt_id='reader-'+str(ordinal),
                        run_ordinal=ordinal))
            gold_file = f'data/scorer-only/{dataset}.json'
            self.dump(gold_file, gold)
            specs.append(dict(name=dataset, count=8, gold_file=gold_file, gold_sha256=self.hash(gold_file)))
        self.dump(scoring.V6_FROZEN, dict(datasets=specs))
        self.lines(scoring.QUERIES, queries)
        self.dump(scoring.SELECTION, dict(selection=dict(selected=selected)))
        self.lines(scoring.PACKS, packs)
        self.lines(scoring.ANSWERS, answers)
        inputs = {name: self.hash(name) for name in (scoring.V6_FROZEN, scoring.QUERIES, scoring.SELECTION)}
        self.dump(scoring.READER_FROZEN, dict(state='reader_inputs_frozen_no_calls', input_hashes=inputs,
            code_hashes={'agent_memory/atm_reader.py': self.hash('agent_memory/atm_reader.py')},
            questions=16, planned_reader_answers=64, variants=list(VARIANTS), packs_file=scoring.PACKS,
            packs_sha256=self.hash(scoring.PACKS), selection_sha256=self.hash(scoring.SELECTION),
            exposure='synthetic test only', source_condition='synthetic text only'))
        self.dump(scoring.LIVE_FROZEN, dict(reader_frozen_sha256=self.hash(scoring.READER_FROZEN),
            model='gpt-4o-mini', max_requests=80, max_cumulative_preflight_cny='20',
            code_hashes={'scripts/run_atm_v6_reader.py': self.hash('scripts/run_atm_v6_reader.py')}))
        self.dump(scoring.READER_RUN, dict(state='reader_complete_waiting_isolated_scoring', completed_answers=64,
            questions=16, reader_frozen_file=scoring.READER_FROZEN, reader_frozen_sha256=self.hash(scoring.READER_FROZEN),
            answers_file=scoring.ANSWERS, answers_sha256=self.hash(scoring.ANSWERS),
            live_frozen_sha256=self.hash(scoring.LIVE_FROZEN)))

    def write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def dump(self, name, value):
        self.write(name, json.dumps(value).encode())

    def lines(self, name, values):
        self.write(name, ('\n'.join(json.dumps(row) for row in values)+'\n').encode())

    def hash(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def read(self, name):
        return json.loads((self.root / name).read_bytes())

    def no_gold(self):
        read = Path.read_bytes
        def guard(path):
            if 'scorer-only' in path.parts:
                raise AssertionError('Gold was opened before all immutable reader outcomes were verified')
            return read(path)
        return patch.object(Path, 'read_bytes', guard)

    def rebind_answers(self):
        run = self.read(scoring.READER_RUN)
        run['answers_sha256'] = self.hash(scoring.ANSWERS)
        self.dump(scoring.READER_RUN, run)

    def judge_outputs(self, duplicate=False):
        mapping = self.read(scoring.JUDGE_MAPPING)
        requests = scoring.lines((self.root / scoring.JUDGE_REQUESTS).read_bytes())
        result = []
        for ordinal, request in enumerate(requests, 65):
            items = []
            for row in mapping:
                if row['request_id'] != request['request_id']:
                    continue
                score = 0 if row['variant'] == 'dense' else 1
                items.append(dict(item_id=row['item_id'], score=score,
                    label='correct' if score else 'incorrect', reason='Synthetic test grade'))
            body = dict(summary='atm proxy grading', evidence_quotes=[], judgments=items)
            response = dict(model='gpt-4o-mini-2024-07-18', choices=[dict(finish_reason='stop', message=dict(content=json.dumps(body)))])
            result.append(dict(request_id=request['request_id'], response=response,
                receipt_sha256=hashlib.sha256(str(ordinal).encode()).hexdigest(), attempt_id='judge-'+str(ordinal), run_ordinal=ordinal))
        if duplicate:
            result.append(result[0])
        self.lines(scoring.JUDGE_RESULTS, result)
        self.dump(scoring.JUDGE_RUN, dict(state='judge_complete', judge_frozen_sha256=self.hash(scoring.JUDGE_FROZEN),
            results_file=scoring.JUDGE_RESULTS, results_sha256=self.hash(scoring.JUDGE_RESULTS),
            completed_requests=len(requests)))

    def test_pinned_number_and_list_semantics(self):
        self.assertTrue(deterministic_accuracy('1,000', '1000'))
        self.assertTrue(deterministic_accuracy('4pm', '16:00'))
        self.assertFalse(deterministic_accuracy('12', '13'))
        self.assertAlmostEqual(list_jaccard_score('email0001, email0002', 'email0001'), .5)
        self.assertAlmostEqual(list_jaccard_score('email0001, email0002', 'email0001, email0003'), 1/3)
        self.assertTrue(UPSTREAM_PROVENANCE)

    def test_missing_64th_answer_never_opens_gold_even_if_report_hash_updated(self):
        rows = scoring.lines((self.root / scoring.ANSWERS).read_bytes())[:-1]
        self.lines(scoring.ANSWERS, rows); self.rebind_answers()
        with self.no_gold(), self.assertRaisesRegex(ValueError, '64 reader outcomes'):
            scoring.prepare_judge(self.root)
        self.assertFalse((self.root / scoring.JUDGE_REQUESTS).exists())

    def test_duplicate_answer_and_changed_pack_fail_before_gold(self):
        rows = scoring.lines((self.root / scoring.ANSWERS).read_bytes())
        rows[-1] = rows[0]
        self.lines(scoring.ANSWERS, rows); self.rebind_answers()
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'duplicate'):
            scoring.prepare_judge(self.root)

    def test_changed_input_hash_fails_before_gold(self):
        self.write(scoring.PACKS, (self.root / scoring.PACKS).read_bytes()+b' ')
        with self.no_gold(), self.assertRaisesRegex(ValueError, 'hash mismatch'):
            scoring.prepare_judge(self.root)

    def test_blinded_requests_native_metrics_and_final_proxy_report(self):
        with self.no_gold():
            verified = scoring.verify_reader_complete(self.root)
        self.assertEqual(len(verified['answers']), 64)
        prep = scoring.prepare_judge(self.root)
        self.assertEqual(prep['judge_requests'], 4)
        requests = scoring.lines((self.root / scoring.JUDGE_REQUESTS).read_bytes())
        for row in requests:
            self.assertEqual(set(row), {'request_id', 'payload'})
            encoded = json.dumps(row['payload'])
            for variant in VARIANTS:
                self.assertNotIn(variant, encoded)
            body = json.loads(row['payload']['messages'][1]['content'])
            self.assertEqual({item['item_id'] for item in body['items']}, set(range(4)))
        self.judge_outputs()
        report = scoring.finalize(self.root, bootstrap_samples=50)
        all_group = report['groups']['all']
        self.assertAlmostEqual(all_group['variants']['dense']['mixed_proxy_mean'], .75)
        self.assertAlmostEqual(all_group['variants']['dense_reranked']['mixed_proxy_mean'], 1.)
        self.assertAlmostEqual(all_group['paired']['dense_reranked_minus_dense']['mean_delta'], .25)
        self.assertFalse(report['official_QS'])
        self.assertFalse(report['official_score'])
        content = (self.root / scoring.SCORES).read_text()
        self.assertNotIn('SYNTHETIC_PRIVATE_QUESTION', content)
        self.assertNotIn('SYNTHETIC_PRIVATE_REFERENCE', content)
        with self.assertRaises(FileExistsError):
            scoring.prepare_judge(self.root)
        with self.assertRaises(FileExistsError):
            scoring.finalize(self.root)

    def test_invalid_reader_schema_is_zero_not_a_retry(self):
        rows = scoring.lines((self.root / scoring.ANSWERS).read_bytes())
        rows[0]['answer'] = dict(answer='', answer_ids=[], cited_ids=[], schema_status='invalid',
            citations_valid=False, answer_ids_valid=False, model='gpt-4o-mini')
        self.lines(scoring.ANSWERS, rows); self.rebind_answers()
        scoring.prepare_judge(self.root)
        self.judge_outputs()
        result = scoring.finalize(self.root, bootstrap_samples=20)
        self.assertEqual(result['groups']['all']['variants']['dense']['invalid_answers'], 1)

    def test_duplicate_judge_response_cannot_finalize(self):
        scoring.prepare_judge(self.root)
        self.judge_outputs(duplicate=True)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            scoring.finalize(self.root, bootstrap_samples=20)
        self.assertFalse((self.root / scoring.SCORES).exists())

    def test_judge_label_score_and_item_id_validation(self):
        items = [dict(item_id=i, label='correct', score=1, reason='Synthetic') for i in range(4)]
        def response():
            return dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
                dict(summary='atm proxy grading', evidence_quotes=[], judgments=items))))])
        items[0]['score'] = .5
        with self.assertRaises(ValueError): scoring.decode_judge(response())
        items[0]['score'] = 1; items[-1]['item_id'] = 0
        with self.assertRaises(ValueError): scoring.decode_judge(response())


if __name__ == '__main__':
    unittest.main()
