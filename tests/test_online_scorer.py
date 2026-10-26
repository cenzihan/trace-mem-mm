import json
import unittest
from unittest.mock import patch
from agent_memory.online_app import PrivateQwenScorer

class Reply:
    status=200
    def __init__(self,value):self.value=value
    def read(self,*args):return json.dumps(self.value).encode()

class Conn:
    def __init__(self,value):self.value=value;self.body=None
    def request(self,method,path,body,headers):self.body=json.loads(body)
    def getresponse(self):return Reply(self.value)
    def close(self):pass

class Tests(unittest.TestCase):
    def value(self):return dict(model=PrivateQwenScorer.model,revision=PrivateQwenScorer.revision,
                                weights_sha256=PrivateQwenScorer.weights_sha256,scores=[1.5])
    def run_score(self,value):
        conn=Conn(value)
        with patch('agent_memory.online_app.http.client.HTTPConnection',return_value=conn):
            result=PrivateQwenScorer('test').score([dict(type='text',text='query')],
                  [dict(retrieval_text='source',content=[{'not_sent':'media'}])],[.1])
        self.assertEqual(conn.body,dict(query='query',documents=['source']))
        return result
    def test_exact_model_identity_and_text_only_payload(self):self.assertEqual(self.run_score(self.value()),[1.5])
    def test_hosted_alias_is_rejected(self):
        v=self.value();v['model']='qwen3-vl-rerank'
        with self.assertRaises(ValueError):self.run_score(v)
    def test_revision_or_weights_mismatch_rejected(self):
        for field in ['revision','weights_sha256']:
            v=self.value();v[field]='wrong'
            with self.assertRaises(ValueError):self.run_score(v)
