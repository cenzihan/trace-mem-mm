"""Synthetic contract/fake HTTP tests only. No dataset media or live providers."""
import base64
import copy
import json
import io
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

from agent_memory.visual_batch8 import GPT4oMiniVisualBatch8 as Batch, VisualInput, VisualBatchFailure
from agent_memory.budget import BudgetLedger, VerifiedRate
from agent_memory.provider_http import BudgetedJSONTransport, HTTPReply, ProviderFailure, OPENAI_ENDPOINT

ROOT = Path(__file__).resolve().parents[1]


def sources(count=8):
    uri = 'data:image/png;base64,' + base64.b64encode((ROOT/'fixtures/red.png').read_bytes()).decode()
    return [VisualInput('native-'+str(i//2), 'occurrence-'+str(i), uri, uri) for i in range(count)]


def response(count=8):
    value = dict(summary='unit batch', evidence_quotes=[], images=[dict(index=i, description='visible red square', uncertainties=[]) for i in reversed(range(count))])
    return dict(model='gpt-4o-mini-2024-07-18', usage=dict(prompt_tokens=22680,completion_tokens=120,total_tokens=22800),
                choices=[dict(finish_reason='stop', message=dict(content=json.dumps(value)))])


def change_value(data, fn):
    value = json.loads(data['choices'][0]['message']['content']); fn(value)
    data['choices'][0]['message']['content'] = json.dumps(value)


class FakeHTTP:
    is_fake_http=True
    def __init__(self, result):self.result=result;self.calls=0
    def exchange(self, *args):
        self.calls+=1
        if isinstance(self.result, Exception):raise self.result
        return self.result


class Batch8Tests(unittest.TestCase):
    def setUp(self):self.prepared=Batch.prepare(sources(), 'unit-user:unit-dataset')

    def test_order_original_occurrences_and_request_variant_preserved(self):
        captured=[]
        result=Batch(lambda endpoint,payload: captured.append((endpoint,payload)) or response()).analyze(self.prepared)
        self.assertEqual([v['index'] for v in result], list(range(8)))
        self.assertEqual(len({v['occurrence_id'] for v in result}),8)
        self.assertEqual(result[0]['original_sha256'],result[0]['request_sha256'])
        payload=captured[0][1];self.assertEqual(payload['model'],'gpt-4o-mini');self.assertEqual(payload['max_completion_tokens'],2048)
        self.assertEqual([v['text'] for v in payload['messages'][1]['content'] if v['type']=='text'], ['IMAGE_'+str(i) for i in range(8)])
        self.assertTrue(all(v['image_url']['detail']=='low' for v in payload['messages'][1]['content'] if v['type']=='image_url'))
        self.assertNotIn('native-',json.dumps(payload));self.assertFalse(payload['store'])

    def test_namespace_order_variant_change_fingerprint(self):
        a=Batch.prepare(sources(),'other');b=Batch.prepare(list(reversed(sources())),'unit-user:unit-dataset')
        c=sources();s=c[0];c[0]=VisualInput(s.native_id,s.occurrence_id,s.original_uri,s.request_uri,'normalized_png')
        self.assertEqual(len({p.request_fingerprint for p in (self.prepared,a,b,Batch.prepare(c,'unit-user:unit-dataset'))}),4)
        self.assertNotIn('base64',repr(self.prepared));self.assertNotIn('base64',repr(sources()[0]))

    def test_mutation_fails_before_dispatch(self):
        self.prepared.payload['model']='other';calls=[]
        with self.assertRaises(VisualBatchFailure):Batch(lambda *v:calls.append(v)).analyze(self.prepared)
        self.assertFalse(calls)

    def test_batch_duplicate_occurrence_and_url_reject(self):
        for items in ([],sources(9),[sources()[0]]*2):
            self.assertRaises(ValueError,Batch.prepare,items,'unit')
        s=sources()[0];self.assertRaises(ValueError,Batch.prepare,[VisualInput(s.native_id,s.occurrence_id,s.original_uri,'https://unit.invalid/image')],'unit')

    def test_original_variant_bytes_and_dimension_admission(self):
        s=sources()[0];buffer=io.BytesIO();Image.new('RGB',(513,1),'red').save(buffer,format='PNG')
        original='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        source=VisualInput('unit','unit-occ',original,s.request_uri,'resize512-unit')
        result=Batch.prepare([source],'unit')
        self.assertNotEqual(result.manifest[0]['original_sha256'],result.manifest[0]['request_sha256'])
        self.assertEqual(result.original_sources[0].original_uri,original)
        self.assertRaises(VisualBatchFailure,Batch.prepare,[VisualInput('unit','occ',original,original)],'unit')

    def test_truncation_refusal_missing_and_extra_fields_reject(self):
        variants=[]
        a=response();a['choices'][0]['finish_reason']='length';variants.append(a)
        a=response();a['choices'][0]['message']['refusal']='unit refusal';variants.append(a)
        a=response();a['model']='other';variants.append(a)
        a=response();a['choices'][0]['message']['content']='{';variants.append(a)
        for mutation in (lambda v:v.pop('images'),lambda v:v['images'][0].pop('description'),lambda v:v.update(gold='unit')):
            a=response();change_value(a,mutation);variants.append(a)
        for value in variants:
            with self.assertRaises(VisualBatchFailure) as caught:Batch.validate(value,self.prepared)
            self.assertIs(caught.exception.private_response,value)
            self.assertNotIn('unit refusal',repr(caught.exception))

    def test_count_index_and_unicode_byte_bounds(self):
        for mutation in (lambda v:v['images'].pop(),lambda v:v['images'][0].update(index=0),
                         lambda v:v['images'][0].update(index=True),lambda v:v['images'][0].update(description='界'*171),
                         lambda v:v['images'][0].update(uncertainties=['x'*81]),
                         lambda v:v['images'][0].update(description='\ud800')):
            value=response();change_value(value,mutation)
            self.assertRaises(VisualBatchFailure,Batch.validate,value,self.prepared)

    def test_known_response_offline_recovery_never_redispatches(self):
        data=response();calls=[];adapter=Batch(lambda *args:calls.append(1) or data)
        adapter.analyze(self.prepared);self.assertEqual(len(calls),1)
        recovered=Batch.validate(data,self.prepared);self.assertEqual(len(calls),1);self.assertEqual(len(recovered),8)

    def test_fake_transport_budget_holds_timeout_5xx_refusal(self):
        replies=[TimeoutError(),HTTPReply(503,b'unit private error'),HTTPReply(429,b'unit limited'),
                 HTTPReply(200,json.dumps(dict(response(),choices=[dict(finish_reason='length',message=dict(content='private partial'))])).encode()),
                 HTTPReply(200,json.dumps(dict(response(),choices=[dict(finish_reason='stop',message=dict(refusal='private refusal'))])).encode())]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'unit.sqlite';ledger=BudgetLedger(path)
            rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat(),conversion_basis='budget_assumption_not_market_fx')
            for reply in replies:
                fake=FakeHTTP(reply)
                client=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,rate=rate,
                    mode='offline_fake',http=fake,fixed_reserve_cny='.25',output_token_cap=2048,
                    credential_reader=lambda:self.fail('no credentials in fake mode'))
                self.assertRaises(ProviderFailure,Batch(client).analyze,self.prepared);self.assertEqual(fake.calls,1)
            with ledger.connect() as db:
                rows=db.execute('SELECT reserved,accounted FROM attempts').fetchall()
            self.assertEqual(rows,[(250000,250000)]*len(replies))
            self.assertEqual(BudgetLedger(path).totals()['attempts'],len(replies))

    def test_paid_shape_failure_retains_hold_and_known_body_for_local_recovery(self):
        value=response();change_value(value,lambda v:v['images'].pop())
        with tempfile.TemporaryDirectory() as folder:
            ledger=BudgetLedger(Path(folder)/'unit.sqlite');fake=FakeHTTP(HTTPReply(200,json.dumps(value).encode()))
            rate=VerifiedRate('gpt-4o-mini','1.5','6','synthetic',datetime.now(timezone.utc).isoformat(),conversion_basis='budget_assumption_not_market_fx')
            client=BudgetedJSONTransport(model='gpt-4o-mini',endpoint=OPENAI_ENDPOINT,ledger=ledger,rate=rate,
                mode='offline_fake',http=fake,fixed_reserve_cny='.25',output_token_cap=2048)
            with self.assertRaises(VisualBatchFailure) as caught:Batch(client).analyze(self.prepared)
            self.assertEqual(caught.exception.category,'item_count');self.assertIsNotNone(caught.exception.private_response)
            self.assertEqual(fake.calls,1)
            with ledger.connect() as db:self.assertEqual(db.execute('SELECT accounted FROM attempts').fetchone()[0],250000)
            self.assertRaises(VisualBatchFailure,Batch.validate,caught.exception.private_response,self.prepared)
            self.assertEqual(fake.calls,1)


if __name__=='__main__':unittest.main()
