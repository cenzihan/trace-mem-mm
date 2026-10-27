"""Current AML/provider wire contract checks. Synthetic data; zero real calls."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from agent_memory.bailian_app import BailianServingSystem
from agent_memory.bailian_rerank import BailianScorer, RerankJSONTransport, payloads
from agent_memory.online_cards import VerifiedQuoteCards
from agent_memory.provider_http import BAILIAN_BEIJING_ENDPOINT, ProviderFailure
from agent_memory.providers import GPT4oMiniCards, TextEmbeddingV4, source_card_response_format
from agent_memory.service_app import ServiceApplication
from agent_memory.service_transport import ServiceBudgetedTransport, omit_nontext_quote_metadata

ROOT=Path(__file__).resolve().parent.parent


class SyntheticWire:
    mode='offline_fake'
    def __init__(self): self.calls=[]
    def ready(self): pass
    def __call__(self,endpoint,payload):
        model=payload['model']
        if model=='qwen3-vl-rerank':
            count=len(payload['input']['documents'])
            value={'output':{'results':[{'index':i,'relevance_score':(i+1)/(count+1)} for i in reversed(range(count))]},'usage':{'total_tokens':10}}
            validator=object.__new__(RerankJSONTransport)
        else:
            validator=object.__new__(ServiceBudgetedTransport)
            if model=='gpt-4o-mini':
                value={'model':'gpt-4o-mini-2024-07-18','usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},
                    'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'summary':'Synthetic source summary','evidence_quotes':[]})}}]}
            else:
                value={'model':model,'usage':{'total_tokens':10},'data':[
                    {'index':i,'embedding':[1.]+[0.]*(payload['dimensions']-1)} for i in reversed(range(len(payload['input'])))]}
        validator.model=model;validator.output_token_cap=512
        validator._bounds(payload);validator._response_model(value);validator._usage(value);validator._contract(value,payload)
        self.calls.append(copy.deepcopy(payload))
        return value


class OfficialFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'memory.sqlite';self.wire=SyntheticWire()
        self.cards=VerifiedQuoteCards(self.wire,image_detail='low',structured_output=True)
        self.store=BailianServingSystem(self.path,embedding=TextEmbeddingV4(BAILIAN_BEIJING_ENDPOINT,self.wire,256),
            card_adapter=self.cards,scoring_adapter=BailianScorer(self.wire))
        self.app=ServiceApplication(self.store,self.wire,'synthetic-auth',scorer_health=lambda:None,
            admission_capacity=16,queue_wait_seconds=1)
        self.add=json.loads((ROOT/'fixtures/add.json').read_text(encoding='utf-8'))
        self.query={'user_id':self.add['user_id'],'query':'synthetic source','top_k':100}

    def call(self,path,payload=None,*,auth=True,method='POST'):
        raw=json.dumps(payload).encode('utf-8');seen=[]
        env={'PATH_INFO':path,'REQUEST_METHOD':method,'CONTENT_TYPE':'application/json; charset=utf-8',
            'CONTENT_LENGTH':str(len(raw)),'wsgi.input':io.BytesIO(raw)}
        if auth:env['HTTP_AUTHORIZATION']='Bearer synthetic-auth'
        body=b''.join(self.app(env,lambda status,headers:seen.append((int(status.split()[0]),dict(headers)))))
        return seen[0],json.loads(body)

    def test_strict_gpt_schema_keeps_prompt_and_ordered_original_media(self):
        parts=self.add['messages'][0]['content'];original=copy.deepcopy(parts)
        self.cards.extract(parts);strict=self.wire.calls[-1]
        GPT4oMiniCards(self.wire,image_detail='low').extract(parts);legacy=self.wire.calls[-1]
        self.assertEqual(strict['messages'],legacy['messages'])
        self.assertEqual(strict['response_format']['type'],'json_schema')
        schema=strict['response_format']['json_schema']
        self.assertIs(schema['strict'],True);self.assertIs(schema['schema']['additionalProperties'],False)
        self.assertEqual(schema['schema']['required'],['summary','evidence_quotes'])
        self.assertEqual(schema['schema']['properties']['evidence_quotes'],{'type':'array','items':{'type':'string'}})
        self.assertFalse(strict['store']);self.assertEqual(strict['n'],1);self.assertEqual(parts,original)

    def test_unreviewed_chat_schema_is_rejected_before_wire_dispatch(self):
        self.cards.extract([{'type':'text','text':'Synthetic source'}]);request=self.wire.calls[-1]
        count=len(self.wire.calls)
        for fmt in ({'type':'text'},{'type':'json_schema','json_schema':{'strict':False}},None):
            bad=copy.deepcopy(request);bad['response_format']=fmt
            with self.assertRaises(ValueError):self.wire('',bad)
        self.assertEqual(len(self.wire.calls),count)
        mutated=source_card_response_format();mutated['json_schema']['schema']['required'].clear()
        self.assertEqual(source_card_response_format()['json_schema']['schema']['required'],['summary','evidence_quotes'])

    def test_twenty_messages_commit_once_and_embedding_batches_preserve_order(self):
        self.add['messages']=[{'role':'user' if i%2==0 else 'assistant','content':f'Synthetic source {i}',
                               'timestamp':1704067200000+i} for i in range(20)]
        status,result=self.call('/add',self.add)
        self.assertEqual(status[0],200)
        self.assertEqual(result,dict(success=True,**{k:self.add[k] for k in ('request_id','session_id','user_id')}))
        self.assertEqual([len(p['input']) for p in self.wire.calls if p['model']=='text-embedding-v4'],[10,10])
        count=len(self.wire.calls);self.assertEqual(self.call('/add',self.add)[1],result)
        self.assertEqual(len(self.wire.calls),count)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM memories').fetchone()[0],20)

    def test_over_declared_message_capacity_rejects_before_models(self):
        self.add['messages']=[{'role':'user','content':'synthetic'}]*21
        self.assertEqual(self.call('/add',self.add)[0][0],413)
        self.assertEqual(self.wire.calls,[])

    def test_multimodal_response_source_order_time_topk_and_restart(self):
        self.assertEqual(self.call('/add',self.add)[0][0],200)
        self.query['top_k']=1;status,result=self.call('/search',self.query)
        self.assertEqual(status[0],200);self.assertEqual(len(result['data']),1)
        row=result['data'][0]
        self.assertTrue(isinstance(row['id'],str) and row['id']);self.assertIsInstance(row['score'],float)
        self.assertIn(row['content'],[m['content'] for m in self.add['messages']])
        self.assertNotIn('Synthetic source summary',json.dumps(row))
        self.assertTrue(row['created_at'].endswith('Z'))
        reopened=BailianServingSystem(self.path,embedding=self.store.embedding,card_adapter=self.cards,scoring_adapter=self.store.scoring_adapter)
        self.assertEqual(reopened.search(self.query),result)

    def test_empty_optional_options_matches_absent_without_persistence(self):
        self.store.add(self.add);before=self.path.read_bytes()
        first=self.store.search(self.query);self.query['options']=[]
        self.assertEqual(self.store.search(self.query),first)
        self.query['options']=['PRIVATE_OPTION_SENTINEL'];self.query['query']='PRIVATE_QUERY_SENTINEL'
        self.store.search(self.query);self.assertEqual(before,self.path.read_bytes())

    def test_image_only_query_has_valid_evidence_without_invented_caption(self):
        self.store.add(self.add);count=len(self.wire.calls)
        self.query['query']=[self.add['messages'][0]['content'][1]]
        result=self.store.search(self.query)
        self.assertTrue(result['data']);self.assertEqual(len(self.wire.calls),count)
        self.assertIn(self.query['query'][0],result['data'][0]['content'])

    def test_auth_health_isolation_and_conflict_do_not_dispatch_models(self):
        self.assertEqual(self.call('/health',method='GET',auth=False)[0][0],200)
        self.assertEqual(self.call('/add',self.add,auth=False)[0][0],401)
        self.assertEqual(self.wire.calls,[])
        self.store.add(self.add);count=len(self.wire.calls)
        self.add['messages'][0]['content']='Different synthetic source'
        self.assertEqual(self.call('/add',self.add)[0][0],409)
        self.query['user_id']='different-user';self.assertEqual(self.store.search(self.query),{'data':[]})
        self.assertEqual(len(self.wire.calls),count)

    def test_optional_metadata_is_omitted_but_substantive_invalid_outputs_fail(self):
        self.cards.extract([{'type':'text','text':'Synthetic source'}])
        validator=object.__new__(ServiceBudgetedTransport);validator.model='gpt-4o-mini'
        def response(card,finish='stop',**extra):
            return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(card),**extra}}]}
        for quote in ({'quote':'untrusted'},'untrusted',None,[None,42,'synthetic']):
            value=response({'summary':'valid summary','evidence_quotes':quote});validator._contract(value,{})
            card=json.loads(omit_nontext_quote_metadata(value)['choices'][0]['message']['content'])
            self.assertTrue(all(isinstance(q,str) for q in card['evidence_quotes']))
        for value in [response({'summary':'','evidence_quotes':[]}),response({'summary':None}),
                      response({'summary':'valid'},'length'),response({'summary':'valid'},refusal='refused')]:
            with self.assertRaises(ProviderFailure):validator._contract(value,{})

    def test_embedding_index_dimension_and_finiteness_contract(self):
        validator=object.__new__(ServiceBudgetedTransport);validator.model='text-embedding-v4'
        payload={'input':['synthetic'],'dimensions':256}
        good={'data':[{'index':0,'embedding':[1.]+[0.]*255}]};validator._contract(good,payload)
        for mutation in [{'index':True,'embedding':[1.]*256},{'index':0,'embedding':[0.]*256},
                         {'index':0,'embedding':[float('nan')]*256},{'index':0,'embedding':[1.]*255}]:
            with self.assertRaises(ProviderFailure):validator._contract({'data':[mutation]},payload)

    def test_rerank_native_shape_and_complete_permutation(self):
        request=payloads('synthetic',['one','two'])[0]
        self.assertEqual(request['input'],{'query':'synthetic','documents':['one','two']})
        self.assertFalse(request['parameters']['return_documents']);self.assertEqual(request['parameters']['top_n'],2)
        for rows in [[{'index':0,'relevance_score':.5}]*2,
                     [{'index':0,'relevance_score':1.1},{'index':1,'relevance_score':.2}],
                     [{'index':0,'relevance_score':float('inf')},{'index':1,'relevance_score':.2}]]:
            with self.assertRaises(ProviderFailure):RerankJSONTransport._contract(None,{'output':{'results':rows}},request)


if __name__=='__main__':unittest.main()
