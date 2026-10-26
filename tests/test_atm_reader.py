"""Synthetic-only ATM reader preparation tests; never use native QA or HTTP."""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent_memory.atm_reader import (
    ATM_READER_PROMPT, ATMReaderBudget, ATMTextReader, MODEL, VARIANTS,
    pack_atm_evidence, query_id, reader_payload, select_pilot,
)


def metadata():
    return [dict(dataset=dataset, id='q'+str(i), qtype=('number', 'open_end', 'list_recall')[i % 3],
                 query_id=query_id(dataset, 'q'+str(i)))
            for dataset in ('regular', 'hard') for i in range(20)]


def question(qtype='open_end'):
    return dict(id='synthetic-q', question='What is supported by these synthetic records?', qtype=qtype)


def sources():
    return {identity: 'Original synthetic evidence ' + identity + '\nunchanged Unicode 原文。'
            for identity in ('20250101_120000', 'email0001', 'email0002', 'email0003')}


def response(answer='A supported synthetic fact', answer_ids=None, cited_ids=None, confidence='supported'):
    value=dict(summary='atm reader output', evidence_quotes=[], answer=answer,
               answer_ids=answer_ids or [], cited_ids=cited_ids or [], confidence=confidence)
    return dict(model='gpt-4o-mini-2024-07-18', choices=[dict(finish_reason='stop', message=dict(content=json.dumps(value)))])


class ATMReaderTests(unittest.TestCase):
    def pack(self, qtype='open_end', **kwargs):
        return pack_atm_evidence(question(qtype), list(sources()), sources(), len, **kwargs)

    def test_selection_uses_only_ids_is_order_invariant_and_extends(self):
        rows=metadata()
        first=select_pilot(rows)
        self.assertEqual(first, select_pilot(list(reversed(rows))))
        self.assertEqual(first['planned_reader_answers'], 64)
        self.assertEqual(first['variants'], list(VARIANTS))
        self.assertFalse(first['qtype_used_for_selection'])
        extended=select_pilot(rows, per_dataset=16)
        for dataset in ('regular','hard'):
            ids=lambda result:[row['id'] for row in result['selected'] if row['dataset']==dataset]
            self.assertEqual(ids(first),ids(extended)[:8])
        changed=copy.deepcopy(rows)
        for row in changed:row['qtype']='number'
        self.assertEqual([r['query_id'] for r in first['selected']],
                         [r['query_id'] for r in select_pilot(changed)['selected']])

    def test_selection_rejects_questions_gold_duplicates_and_unknown_dataset(self):
        for key in ('question','answer','evidence_ids','gold'):
            rows=metadata();rows[0][key]='PRIVATE_SENTINEL'
            self.assertRaises(ValueError,select_pilot,rows)
        rows=metadata();rows.append(rows[0])
        self.assertRaises(ValueError,select_pilot,rows)
        rows=metadata();rows[0]['query_id']='changed'
        self.assertRaises(ValueError,select_pilot,rows)
        rows=metadata();rows[0]['dataset']='other'
        self.assertRaises(ValueError,select_pilot,rows)
        self.assertRaises(ValueError,select_pilot,metadata(),seed=1)
        self.assertRaises(ValueError,select_pilot,metadata(),per_dataset=21)

    def test_native_ids_exact_original_text_and_zero_media(self):
        packed=self.pack()
        content=packed['parts'][0]['text']
        for identity,text in sources().items():
            self.assertIn(text,content)
            self.assertIn(identity,content)
            self.assertEqual(packed['source_sha256'][identity],hashlib.sha256(text.encode()).hexdigest())
        self.assertNotIn('source_round=',content)
        self.assertNotIn('source_session=',content)
        self.assertEqual(packed['source_memory_ids'],list(sources()))
        self.assertEqual(packed['image_count'],0)
        self.assertFalse(packed['context_gold_access'])
        self.assertEqual(packed['budget'],asdict(ATMReaderBudget()))

    def test_budget_is_a_complete_ranked_prefix_never_crop_or_skip(self):
        baseline=pack_atm_evidence(question(),['email0001'],sources(),len)
        budget=ATMReaderBudget(text_tokens=baseline['text_token_count'])
        mapping=dict(sources());mapping['email0002']='big '*4000
        packed=pack_atm_evidence(question(),['email0001','email0002','email0003'],mapping,len,budget)
        self.assertEqual(packed['source_memory_ids'],['email0001'])
        self.assertEqual(packed['dropped_native_ids'],['email0002','email0003'])
        self.assertNotIn('big ',packed['parts'][0]['text'])
        self.assertNotIn(mapping['email0003'],packed['parts'][0]['text'])
        too_small=ATMReaderBudget(text_tokens=1)
        self.assertRaises(ValueError,self.pack,budget=too_small)

    def test_utf8_limit_is_independent_and_whole_source(self):
        mapping={'email0001':'原'*10000}
        packed=pack_atm_evidence(question(),['email0001'],mapping,lambda text:max(1,len(text)//20))
        self.assertEqual(packed['source_records_retained'],0)
        self.assertEqual(packed['dropped_records'],1)
        self.assertLessEqual(packed['utf8_text_bytes'],24000)

    def test_top20_and_all_candidate_ids_are_validated(self):
        mapping={'email'+str(i):'x' for i in range(30)}
        packed=pack_atm_evidence(question(),list(mapping),mapping,lambda text:max(1,len(text)//4))
        self.assertEqual(len(packed['source_memory_ids']),20)
        self.assertEqual(packed['ranked_native_ids'],list(mapping)[:20])
        self.assertRaises(ValueError,pack_atm_evidence,question(),list(mapping)+['unknown'],mapping,len)
        self.assertRaises(ValueError,pack_atm_evidence,question(),['email0','email0'],mapping,len)
        self.assertRaises(ValueError,pack_atm_evidence,question(),['email0'],{'email0':{'answer':'gold'}},len)

    def test_question_schema_and_budget_reject_nonpermitted_inputs(self):
        bad=question();bad['answer']='GOLD_SENTINEL'
        self.assertRaises(ValueError,pack_atm_evidence,bad,[],{},len)
        self.assertRaises(ValueError,ATMReaderBudget,images=1)
        self.assertRaises(ValueError,ATMReaderBudget,text_tokens=8193)
        self.assertRaises(ValueError,ATMReaderBudget,output_tokens=256)
        self.assertRaises(ValueError,pack_atm_evidence,question(),[],{},lambda _:False)

    def test_payload_is_exact_frozen_and_requires_no_transport(self):
        packed=self.pack();payload=reader_payload(packed)
        self.assertEqual(payload['model'],MODEL)
        self.assertEqual(payload['max_completion_tokens'],768)
        self.assertFalse(payload['store'])
        self.assertEqual(payload['messages'][0]['content'],ATM_READER_PROMPT)
        payload['messages'][1]['content'][0]['text']='mutated copy'
        self.assertNotEqual(packed['parts'][0]['text'],'mutated copy')
        calls=[]
        def transport(endpoint,payload):calls.append((endpoint,payload));return response()
        reader=ATMTextReader(transport)
        reader.decode(reader.request(packed),packed)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1],reader_payload(packed))

    def test_pack_tampering_or_changed_runtime_budget_blocks_before_call(self):
        calls=[]
        reader=ATMTextReader(lambda *args:calls.append(args))
        for key in ('parts','source_memory_ids','payload_sha256','prompt_sha256'):
            packed=self.pack();packed[key]='tamper'
            self.assertRaises(ValueError,reader.request,packed)
        self.assertFalse(calls)
        self.assertRaises(ValueError,ATMTextReader(lambda *args:None,ATMReaderBudget(candidate_k=10)).request,self.pack())

    def test_list_answer_keeps_more_than_three_native_ids(self):
        packed=self.pack('list_recall');ids=list(sources())
        decoded=ATMTextReader.decode(response(', '.join(ids),ids,ids),packed)
        self.assertEqual(decoded['answer_ids'],ids)
        self.assertTrue(decoded['answer_ids_valid'])
        self.assertTrue(decoded['citations_valid'])
        self.assertEqual(decoded['answer'],', '.join(ids))
        self.assertRaises(ValueError,ATMTextReader.decode,response('E1',ids),packed)

    def test_unknown_predictions_are_flagged_not_repaired(self):
        packed=self.pack('list_recall')
        decoded=ATMTextReader.decode(response('email9999',['email9999'],['email9999']),packed)
        self.assertEqual(decoded['answer'],'email9999')
        self.assertFalse(decoded['answer_ids_valid'])
        self.assertFalse(decoded['citations_valid'])

    def test_schema_model_abstention_and_duplicate_json_rejected(self):
        packed=self.pack()
        good=response('INSUFFICIENT_EVIDENCE',confidence='insufficient')
        self.assertEqual(ATMTextReader.decode(good,packed)['confidence'],'insufficient')
        self.assertRaises(ValueError,ATMTextReader.decode,response('invented',confidence='insufficient'),packed)
        for mutate in (
            lambda r:r.update(model='other'),
            lambda r:r['choices'][0].update(finish_reason='length'),
            lambda r:r['choices'][0]['message'].update(content='{"answer": "a", "answer": "b"}'),
        ):
            bad=response();mutate(bad)
            self.assertRaises(ValueError,ATMTextReader.decode,bad,packed)
        bad=response(answer_ids=['email0001'])
        self.assertRaises(ValueError,ATMTextReader.decode,bad,packed)

    def test_real_transport_contract_and_guard_with_fake_http(self):
        from agent_memory.budget import BudgetLedger, VerifiedRate
        from agent_memory.development_run import DevelopmentRun, ScopedTransport
        from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply, OPENAI_ENDPOINT
        class FakeHTTP:
            is_fake_http=True
            calls=0
            def exchange(self,endpoint,body,headers,timeout,limit):
                self.calls+=1
                payload=json.loads(body)
                assert payload['max_completion_tokens']==768 and payload['store'] is False
                result=response()
                result['usage']=dict(prompt_tokens=200,completion_tokens=30,total_tokens=230)
                return HTTPReply(200,json.dumps(result).encode())
        with tempfile.TemporaryDirectory() as directory:
            ledger=BudgetLedger(Path(directory)/'budget.sqlite')
            guard=DevelopmentRun(Path(directory)/'run.sqlite',create=True,max_requests=1,max_reservation_cny='.25')
            http=FakeHTTP()
            transport=BudgetedJSONTransport(model=MODEL,endpoint=OPENAI_ENDPOINT,ledger=ledger,
                rate=VerifiedRate(MODEL,'1.5','6','synthetic',datetime.now(timezone.utc).isoformat()),
                http=http,mode='offline_fake',fixed_reserve_cny='.25',output_token_cap=768,
                credential_reader=lambda: self.fail('fake integration must never read credentials'))
            packed=self.pack()
            reader=ATMTextReader(ScopedTransport(transport,guard,'.25'))
            result=reader.decode(reader.request(packed),packed)
            self.assertEqual(result['schema_status'],'valid')
            self.assertEqual(http.calls,1)
            self.assertEqual(guard.summary()['request_claims'],1)
            self.assertEqual(ledger.totals()['attempts'],1)
            guard.close()


class ATMReaderPreparationTests(unittest.TestCase):
    def setUp(self):
        from scripts import prepare_atm_v6_reader as prep
        self.prep=prep
        self.temporary=tempfile.TemporaryDirectory(prefix='atm-reader-prepare-')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        for name in prep.CODE:self.write(name,b'# synthetic code identity\n')
        self.queries=[dict(row,question='Original synthetic question, no gold.') for row in metadata()]
        self.lines(prep.QUERIES,self.queries)
        self.lines(prep.SOURCES,[dict(source_id=identity,text=text) for identity,text in sources().items()])
        artifacts={name:self.sha(name) for name in (prep.QUERIES,prep.SOURCES)}
        self.dump(prep.V6_FROZEN,dict(artifact_hashes=artifacts,query_count=len(self.queries),
            source_count=len(sources()),methods=list(VARIANTS)))
        self.dump(prep.V6_PREP,dict(state='prepared_gold_not_loaded',frozen_file=prep.V6_FROZEN,
            frozen_sha256=self.sha(prep.V6_FROZEN),artifact_hashes=artifacts))
        self.write('data/scorer-only/never-read.json',b'GOLD_SENTINEL')

    def write(self,name,raw):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)

    def dump(self,name,value):self.write(name,json.dumps(value,ensure_ascii=False).encode())

    def lines(self,name,rows):self.write(name,b''.join(json.dumps(row,ensure_ascii=False).encode()+b'\n' for row in rows))

    def sha(self,name):return hashlib.sha256((self.root/name).read_bytes()).hexdigest()

    def rank(self):
        prep=self.prep
        records=[]
        for q in self.queries:
            methods={v:dict(ranked_ids=list(sources()),scores=[4.,3.,2.,1.]) for v in VARIANTS}
            records.append(dict(dataset=q['dataset'],id=q['id'],qtype=q['qtype'],methods=methods))
        self.lines(prep.RANKINGS,records)
        self.dump(prep.V6_RUN,dict(state='retrieval_complete_gold_not_loaded',ranking_file=prep.RANKINGS,
            ranking_sha256=self.sha(prep.RANKINGS),frozen_file=prep.V6_FROZEN,
            frozen_sha256=self.sha(prep.V6_FROZEN),preparation_sha256=self.sha(prep.V6_PREP)))

    def test_select_before_rank_and_four_variant_packing_never_read_gold(self):
        prep=self.prep
        original=Path.read_bytes
        def guarded(path):
            if 'scorer-only' in path.parts:raise AssertionError('gold opened by reader preparation')
            return original(path)
        with patch.object(Path,'read_bytes',guarded):
            selected=prep.prepare_selection(self.root)
            self.assertEqual(selected['planned_reader_answers'],64)
            self.assertFalse((self.root/prep.RANKINGS).exists())
            self.assertNotIn('Original synthetic question',json.dumps(selected))
            self.assertNotIn('GOLD_SENTINEL',json.dumps(selected))
            self.rank()
            frozen=prep.prepare_packs(self.root,count_tokens=len,offline_test=True)
            self.assertEqual(frozen['planned_reader_answers'],64)
            self.assertEqual(frozen['questions'],16)
            self.assertEqual(frozen['provider_requests'],0)
            self.assertFalse(frozen['gold_loaded'])
            self.assertEqual(frozen['packs_sha256'],self.sha(prep.PACKS))
            rows=[json.loads(line) for line in original(self.root/prep.PACKS).splitlines()]
            self.assertEqual(len(rows),64)
            self.assertEqual({r['variant'] for r in rows},set(VARIANTS))
            self.assertTrue(all(reader_payload(row['pack']) for row in rows))
        self.assertRaises(FileExistsError,prep.prepare_selection,self.root)
        self.assertRaises(FileExistsError,prep.prepare_packs,self.root,count_tokens=len,offline_test=True)

    def test_selection_does_not_inspect_question_values(self):
        prep=self.prep
        before=select_pilot(metadata())['selected']
        for row in self.queries:row['question']={'not_read_for_selection':'synthetic sentinel'}
        self.lines(prep.QUERIES,self.queries)
        frozen=json.loads((self.root/prep.V6_FROZEN).read_bytes())
        frozen['artifact_hashes'][prep.QUERIES]=self.sha(prep.QUERIES)
        self.dump(prep.V6_FROZEN,frozen)
        preparation=json.loads((self.root/prep.V6_PREP).read_bytes())
        preparation.update(artifact_hashes=frozen['artifact_hashes'],frozen_sha256=self.sha(prep.V6_FROZEN))
        self.dump(prep.V6_PREP,preparation)
        self.assertEqual(prep.prepare_selection(self.root)['selected_ids'],before)
        self.rank()
        self.assertRaises(ValueError,prep.prepare_packs,self.root,count_tokens=len,offline_test=True)
        self.assertFalse((self.root/prep.PACKS).exists())
        self.assertFalse((self.root/prep.FROZEN).exists())

    def test_changed_ranking_source_or_code_fails_before_packs(self):
        prep=self.prep
        prep.prepare_selection(self.root);self.rank()
        for name in (prep.RANKINGS,prep.SOURCES,prep.CODE[0]):
            original=(self.root/name).read_bytes()
            self.write(name,original+b' ')
            self.assertRaises(ValueError,prep.prepare_packs,self.root,count_tokens=len,offline_test=True)
            self.assertFalse((self.root/prep.PACKS).exists())
            self.write(name,original)

    def test_fake_tokenizer_requires_explicit_temporary_test_mode(self):
        prep=self.prep
        prep.prepare_selection(self.root);self.rank()
        self.assertRaises(ValueError,prep.prepare_packs,self.root,count_tokens=len)
        self.assertFalse((self.root/prep.PACKS).exists())


if __name__ == '__main__':
    unittest.main()
