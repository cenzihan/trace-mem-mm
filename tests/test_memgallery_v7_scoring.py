"""Synthetic scoring contracts only; never loads a native dataset or provider."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_memory.memgallery_v7 import native_id, VARIANTS
from scripts import score_memgallery_v7 as score
from agent_memory.memgallery_v7_reader import pack_memgallery_evidence, reader_payload, decode_reader


def mapping():
    return {'by_native_id': {native_id('S1', 'R1'): {'source_session': 'S1', 'source_round': 'R1'},
                            native_id('S2', 'R1'): {'source_session': 'S2', 'source_round': 'R1'},
                            native_id('S2', 'R2'): {'source_session': 'S2', 'source_round': 'R2'}}}


def questions(n=16):
    return [dict(index=i, point='AR' if i % 2 else 'FR', question='Synthetic question?',
                 question_image=None, image_caption=None) for i in range(n)]


def gold(n=16):
    return [dict(index=i, answer='Synthetic reference', clue=[] if i % 2 else ['R1'],
                 session_id=[] if i % 2 else ['S1']) for i in range(n)]


def rankings(n=16):
    ids = list(mapping()['by_native_id'])
    return [dict(query_id='memgallery-v7:'+str(i), index=i, point='AR' if i % 2 else 'FR',
                 methods={v: dict(ranked_ids=list(ids), scores=[3., 2., 1.]) for v in VARIANTS}) for i in range(n)]


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(score.canonical(value)+'\n', encoding='utf-8')


def append_receipt(root, request_id, payload, response, ordinal):
    namespace = 'memgallery-v7:' + score.sha(root, score.LIVE_FROZEN)
    manifest = dict(keys=[request_id], payload_sha256=score.digest(score.canonical(payload).encode()),
                    run_path=str(score.path_at(root, score.GUARD)))
    fingerprint = score.digest(score.canonical(dict(namespace=namespace, manifest=manifest)).encode())
    body = score.canonical(response).encode(); attempt = 'synthetic-attempt-'+str(ordinal)
    with closing(sqlite3.connect(root / score.JOURNAL)) as db, db:
        db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                   (fingerprint, namespace, score.canonical(manifest), 'complete', attempt, 200, body,
                    score.digest(body), 0, 1, manifest['run_path'], ordinal))
        db.execute('INSERT INTO claims VALUES(?,?,?)', (namespace, request_id, fingerprint))
    with closing(sqlite3.connect(root / score.GUARD)) as db, db:
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?)',
                   (ordinal, 'gpt-4o-mini', 250000, 'complete', response['model']))
    return dict(request_id=request_id, attempt_id=attempt, receipt_sha256=score.digest(body), run_ordinal=ordinal)


def fixture(root):
    for name in score.CODE:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic code hash fixture')
    put(root, score.QUERIES, questions())
    put(root, score.SELECTION, {'selected_indices': list(range(16))})
    put(root, score.GOLD, gold())
    original_prep = {'scorer_only_gold_sha256': score.sha(root, score.GOLD)}
    source_id = native_id('S1', 'R1')
    sources = {source_id: dict(text='Synthetic original complete source.', source_session='S1',
                              source_round='R1', source_date='2020-01-01', image_ids=[])}
    packs = []
    for query in questions():
        for variant in VARIANTS:
            pack = pack_memgallery_evidence(query, [source_id], sources, lambda s: max(1, len(s)//4))
            packs.append(dict(query_id='memgallery-v7:'+str(query['index']), index=query['index'],
                              point=query['point'], variant=variant, pack=pack))
    score.write_lines(root, score.PACKS, packs)
    frozen = dict(state='reader_inputs_frozen_no_calls', planned_reader_answers=64, max_requests=72,
                  max_preflight_cny='18', packs_file=score.PACKS, packs_sha256=score.sha(root, score.PACKS),
                  input_hashes={p: score.sha(root, p) for p in (score.PACKS, score.SELECTION, score.QUERIES)},
                  code_hashes={p: score.sha(root, p) for p in score.CODE})
    put(root, score.READER_FROZEN, frozen)
    put(root, score.LIVE_FROZEN, dict(reader_sha256=score.sha(root, score.READER_FROZEN)))
    p = root / score.JOURNAL; p.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(p)) as db, db:
        db.executescript('CREATE TABLE operations(fingerprint TEXT,scope TEXT,manifest TEXT,state TEXT,attempt_id TEXT,status INTEGER,body BLOB,body_sha TEXT,redacted INTEGER,transport_completed INTEGER,run_path TEXT,run_ordinal INTEGER); CREATE TABLE claims(scope TEXT,chunk_key TEXT,fingerprint TEXT);')
    with closing(sqlite3.connect(root / score.GUARD)) as db, db:
        db.executescript("CREATE TABLE run(max_requests INTEGER,cap INTEGER,state TEXT); INSERT INTO run VALUES(72,18000000,'active'); CREATE TABLE calls(ordinal INTEGER,model TEXT,held INTEGER,state TEXT,returned_model TEXT);")
    answers = []
    for ordinal, row in enumerate(packs, 1):
        value = dict(summary='', evidence_quotes=[], answer='Synthetic prediction', cited_ids=[source_id],
                     confidence='invalid' if ordinal == 1 else 'supported')
        response = dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message=dict(content=score.canonical(value)))])
        request_id = score.digest((row['query_id']+'\0'+row['variant']).encode())
        saved = append_receipt(root, request_id, reader_payload(row['pack']), response, ordinal)
        saved.update({k: row[k] for k in ('query_id', 'index', 'point', 'variant')})
        if ordinal == 1:
            saved['answer'] = dict(answer='', cited_ids=[], confidence='invalid', schema_status='invalid',
                                   citations_valid=False, answer_ids_valid=False, model='gpt-4o-mini')
        else:
            saved['answer'] = decode_reader(response, row['pack'])
        answers.append(saved)
    score.write_lines(root, score.ANSWERS, answers)
    put(root, score.READER_COMPLETE, dict(state='complete', phase='reader', count=64,
        output_sha256=score.sha(root, score.ANSWERS), frozen_live_sha256=score.sha(root, score.LIVE_FROZEN)))
    put(root, score.CHAT_PROGRESS, dict(phase='reader', state='reader_complete'))
    return dict(questions=questions(), preparation=original_prep,
                input_hashes={score.QUERIES: score.sha(root, score.QUERIES)})


class MemGalleryV7ScoringTests(unittest.TestCase):
    def test_same_round_label_resolves_only_with_correct_session(self):
        self.assertEqual(score.resolve_evidence({'clue': ['R1'], 'session_id': ['S2']}, mapping()), {native_id('S2', 'R1')})
        self.assertEqual(score.resolve_evidence({'clue': ['R1', 'R2'], 'session_id': ['S2']}, mapping()),
                         {native_id('S2', 'R1'), native_id('S2', 'R2')})

    def test_ambiguous_unresolved_or_missing_sessions_block(self):
        for reference in ({'clue': ['R1'], 'session_id': ['S1', 'S2']},
                          {'clue': ['R2'], 'session_id': ['S1']}, {'clue': ['R1'], 'session_id': []}):
            with self.assertRaises(ValueError):
                score.resolve_evidence(reference, mapping())

    def test_empty_gold_excluded_from_recall_not_zero(self):
        groups, details = score.retrieval_statistics(questions(2), gold(2), mapping(), rankings(2))
        self.assertEqual(groups['all']['questions'], 2)
        self.assertEqual(groups['all']['recall_denominator'], 1)
        self.assertEqual(groups['AR']['empty_gold_questions'], 1)
        self.assertIsNone(groups['AR']['variants']['dense']['20']['fractional_recall'])
        self.assertIsNone(details[1]['methods']['dense']['20'])
        self.assertEqual(groups['all']['variants']['dense']['20']['fractional_recall'], 1)

    def test_incomplete_ranking_and_duplicate_source_block(self):
        with self.assertRaises(ValueError):
            score.validate_rankings(rankings(1), score.query_map(questions(2)), mapping())
        rows = rankings(1); rows[0]['methods']['dense']['ranked_ids'][1] = rows[0]['methods']['dense']['ranked_ids'][0]
        with self.assertRaises(ValueError):
            score.validate_rankings(rows, score.query_map(questions(1)), mapping())

    def test_macro_recall_math(self):
        ids = ['a', 'b', 'c']
        self.assertEqual(score.recall(ids, {'a', 'z'}, 1),
                         {'fractional_recall': .5, 'any_hit': 1., 'all_evidence_coverage': 0.})
        self.assertIsNone(score.recall(ids, set(), 100))

    def test_five_grade_pairing_and_invalid_zero(self):
        rows = [dict(index=i, point='AR' if i else 'FR', variant=v, score=.75 if v.endswith('_reranked') else .25,
                     invalid_schema=False) for i in range(2) for v in VARIANTS]
        groups = score.aggregate_reader(rows, samples=100)
        self.assertEqual(groups['all']['paired']['dense_reranked_minus_dense']['ci95'], [.5, .5])
        rows[0]['invalid_schema'] = True
        with self.assertRaises(ValueError):
            score.aggregate_reader(rows)
        rows[0]['score'] = 0
        self.assertEqual(score.aggregate_reader(rows, samples=100)['all']['variants']['dense']['invalid_answers'], 1)

    def test_blind_batches_are_complete_deterministic_and_no_method_labels(self):
        answers = [dict(index=i, variant=v, answer=dict(schema_status='valid', answer='Synthetic prediction'))
                   for i in range(16) for v in VARIANTS]
        answers[0]['answer'] = dict(schema_status='invalid', answer='')
        first = score.judge_batches(questions(), list(range(16)), answers, gold(), lambda s: max(1, len(s)//4))
        second = score.judge_batches(questions(), list(range(16)), list(reversed(answers)), gold(), lambda s: max(1, len(s)//4))
        self.assertEqual(first, second)
        requests, private = first
        self.assertEqual(len(requests), 8)
        self.assertEqual(sum(len(row['items']) for row in private), 64)
        for row in requests:
            self.assertEqual(sorted(row['pack']['item_ids']), list(range(8)))
            payload = json.dumps(row['pack']['payload'])
            for variant in VARIANTS:
                self.assertNotIn(variant, payload)
        with self.assertRaises(ValueError):
            score.judge_batches(questions(), list(range(16)), answers[:-1], gold(), lambda _: 1)

    def test_partial_tail_and_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):
            score.lines(b'{"a":1}')
        with self.assertRaises(ValueError):
            score.lines(b'{"a":1,"a":2}\n')

    def test_stage_gate_never_opens_gold_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root, score.RERANK_RUN, {'state': 'incomplete'})
            put(root, score.READER_COMPLETE, {'state': 'incomplete'})
            with patch.object(score, 'read_gold_after_completion', side_effect=AssertionError('gold opened')):
                with self.assertRaises(ValueError): score.score_retrieval(root)
                with self.assertRaises(ValueError): score.prepare_judge(root, count_tokens=lambda _: 1)
            self.assertFalse((root / score.JUDGE_FROZEN).exists())

    def test_raw_receipt_and_saved_answer_must_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); complete = fixture(root)
            with patch.object(score, 'verify_retrieval_complete', return_value=complete):
                self.assertEqual(len(score.verify_reader_complete(root)['answers']), 64)
                rows = score.lines((root / score.ANSWERS).read_bytes())
                rows[1]['answer']['answer'] = 'Tampered saved answer'
                (root / score.ANSWERS).write_text(''.join(score.canonical(r)+'\n' for r in rows), encoding='utf-8')
                completion = score.load(root, score.READER_COMPLETE)
                completion['output_sha256'] = score.sha(root, score.ANSWERS)
                put(root, score.READER_COMPLETE, completion)
                with self.assertRaises(ValueError): score.verify_reader_complete(root)

    def test_fake_complete_reader_and_eight_batch_final_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); complete = fixture(root)
            with patch.object(score, 'verify_retrieval_complete', return_value=complete):
                prepared = score.prepare_judge(root, count_tokens=lambda s: max(1, len(s)//4))
                self.assertEqual(prepared['judge_requests'], 8)
                self.assertEqual(prepared['invalid_reader_answers'], 1)
                requests = score.lines((root / score.JUDGE_REQUESTS).read_bytes()); results = []
                for ordinal, request in enumerate(requests, 65):
                    value = dict(summary='', evidence_quotes=[], judgments=[dict(item_id=i, score=1, reason='Synthetic.')
                        for i in request['pack']['item_ids']])
                    response = dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message=dict(content=score.canonical(value)))])
                    saved = append_receipt(root, request['request_id'], score.proxy_judge_payload(request['pack']), response, ordinal)
                    saved['response'] = response; results.append(saved)
                score.write_lines(root, score.JUDGE_RESULTS, results)
                with closing(sqlite3.connect(root / score.GUARD)) as db, db:
                    db.execute("UPDATE run SET state='complete'")
                put(root, score.JUDGE_COMPLETE, dict(state='complete', phase='judge', count=8,
                    output_sha256=score.sha(root, score.JUDGE_RESULTS), frozen_live_sha256=score.sha(root, score.LIVE_FROZEN)))
                put(root, score.CHAT_PROGRESS, dict(phase='judge', state='judge_complete'))
                result = score.finalize(root, bootstrap_samples=100)
                self.assertEqual(result['groups']['all']['questions'], 16)
                self.assertEqual(result['groups']['all']['variants']['dense']['proxy_mean'], 15/16)
                self.assertEqual(result['groups']['all']['variants']['dense_reranked']['proxy_mean'], 1)
                self.assertEqual(result['groups']['all']['variants']['dense']['invalid_answers'], 1)
                self.assertNotIn('Synthetic prediction', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
