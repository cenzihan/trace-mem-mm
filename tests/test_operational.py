import copy
import http.client
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from datetime import datetime,timedelta,timezone
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer,WSGIRequestHandler,make_server

from agent_memory.app import Application
from agent_memory.budget import BudgetLedger,VerifiedRate
from agent_memory.evaluation_ledger import EvaluationLedger
from agent_memory.maintenance import backup_database,integrity,repair_missing_indexes
from agent_memory.store import MemoryStore
from agent_memory.system import AgentMemorySystem


class OperationsTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def payload(self,user='u'):return dict(request_id='r',user_id=user,session_id='same-session',messages=[dict(role='user',content='source evidence')])

    def test_zero_evidence_abstains(self):
        store=AgentMemorySystem(self.root/'m.sqlite');store.add(self.payload())
        self.assertEqual(store.search(dict(query='quasar neutrino spectroscopy',user_id='u',top_k=5)),{'data':[]})

    def test_migration_source_only_recovery_and_backup(self):
        path=self.root/'m.sqlite';raw=MemoryStore(path);payload=self.payload();raw.add(payload)
        before=raw.search(dict(query='source',user_id='u',top_k=1))
        report=repair_missing_indexes(path);self.assertEqual(report['repaired'],1)
        self.assertEqual(report['integrity']['schema_version'],3)
        self.assertEqual(AgentMemorySystem(path).search(dict(query='source',user_id='u',top_k=1))['data'][0]['content'],before['data'][0]['content'])
        self.assertEqual(repair_missing_indexes(path)['repaired'],0)
        backup=self.root/'snapshot.sqlite';backup_database(path,backup)
        restored=self.root/'restored.sqlite';backup_database(backup,restored)
        self.assertTrue(integrity(restored)['integrity_ok'])
        self.assertRaises(ValueError,backup_database,backup,restored)
        self.assertRaises(ValueError,BudgetLedger,path)
        self.assertRaises(ValueError,EvaluationLedger,path)

    def test_unknown_future_schema_not_downgraded(self):
        path=self.root/'future.sqlite';store=MemoryStore(path)
        with store.connect() as db:db.execute('PRAGMA user_version=99')
        self.assertRaises(ValueError,AgentMemorySystem,path)
        self.assertEqual(integrity(path)['schema_version'],99)

    def test_smoke_persistent_cap_hourly_and_full_forbidden(self):
        path=self.root/'eval.sqlite';ledger=EvaluationLedger(path)
        self.assertRaises(ValueError,ledger.reserve_smoke)
        self.assertEqual(ledger.summary()['attempts_including_uncertain'],0)
        approved=dict(access_approved=True,credential_established=True,endpoints_authorized=True,submission_approved=True)
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        first=ledger.reserve_smoke(now=start,**approved);ledger.record_result(first,'uncertain')
        self.assertRaises(ValueError,ledger.reserve_smoke,now=start+timedelta(minutes=59),**approved)
        for i in range(1,5):EvaluationLedger(path).reserve_smoke(now=start+timedelta(hours=i),**approved)
        self.assertEqual(EvaluationLedger(path).summary()['remaining'],0)
        self.assertRaises(ValueError,ledger.reserve_smoke,now=start+timedelta(hours=5),**approved)
        self.assertRaises(ValueError,ledger.reserve_full)

    def test_concurrent_budget_and_smoke_initialization(self):
        budget=self.root/'budget.sqlite';rate=VerifiedRate('gpt-4o-mini','1','1','synthetic','synthetic')
        def reserve(_):
            try:return BudgetLedger(budget,cap_cny='1').reserve(rate,1,fixed_reserve_cny='1')
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=4) as pool:accepted=[v for v in pool.map(reserve,range(4)) if v]
        self.assertEqual(len(accepted),1)
        path=self.root/'eval.sqlite';now=datetime(2026,1,1,tzinfo=timezone.utc)
        def smoke(_):
            try:return EvaluationLedger(path).reserve_smoke(access_approved=True,credential_established=True,endpoints_authorized=True,submission_approved=True,now=now)
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=4) as pool:accepted=[v for v in pool.map(smoke,range(4)) if v]
        self.assertEqual(len(accepted),1)


class LoopbackHTTPTests(unittest.TestCase):
    def test_real_http_concurrent_scope_and_no_body_or_key_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=AgentMemorySystem(Path(tmp)/'m.sqlite')
            class Server(ThreadingMixIn,WSGIServer):daemon_threads=True
            class Quiet(WSGIRequestHandler):
                def log_message(self,*args):pass
            app=Application(store,auth_mode='Bearer',key='SYNTHETIC_LOCAL_TRANSPORT_TOKEN')
            log=io.StringIO()
            with redirect_stderr(log):
                server=make_server('127.0.0.1',0,app,server_class=Server,handler_class=Quiet)
                port=server.server_port;thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                def request(path,payload=None,auth=True):
                    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
                    headers={'Content-Type':'application/json'}
                    if auth:headers['Authorization']='Bearer SYNTHETIC_LOCAL_TRANSPORT_TOKEN'
                    connection.request('POST' if payload is not None else 'GET',path,json.dumps(payload).encode() if payload is not None else None,headers)
                    response=connection.getresponse();status=response.status;body=json.loads(response.read());connection.close();return status,body
                try:
                    self.assertEqual(request('/health',auth=False)[0],200)
                    self.assertEqual(request('/search',dict(query='x',user_id='u',top_k=1),auth=False)[0],401)
                    def roundtrip(i):
                        uid='user'+str(i);sentinel='PRIVATE_BODY_SENTINEL_'+str(i)
                        payload=dict(request_id='same',user_id=uid,session_id='same',messages=[dict(role='user',content=sentinel)])
                        status,echo=request('/add',payload)
                        self.assertEqual(status,200)
                        self.assertEqual(echo,dict(success=True,request_id='same',user_id=uid,session_id='same'))
                        self.assertEqual(request('/add',payload),(200,echo))
                        conflict=copy.deepcopy(payload);conflict['messages'][0]['content']='different source'
                        self.assertEqual(request('/add',conflict)[0],409)
                        status,found=request('/search?private=QUERY_URL_SENTINEL',dict(query=sentinel,user_id=uid,top_k=100))
                        self.assertEqual(status,200);self.assertEqual(len(found['data']),1)
                        self.assertEqual(found['data'][0]['content'][0]['text'],sentinel)
                    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(roundtrip,range(8)))
                    malformed=dict(request_id='invalid',user_id='malformed',session_id='same',messages=[dict(role='user',content=[
                        {'type':'text','text':'must not commit'},
                        {'type':'image_url','image_url':{'url':'data:image/png;base64,INVALID!!'}}])])
                    self.assertEqual(request('/add',malformed)[0],422)
                    self.assertEqual(request('/search',dict(query='commit',user_id='malformed',top_k=100)),(200,{'data':[]}))
                    self.assertEqual(request('/search',dict(query='anything',user_id='unknown-user',top_k=100)),(200,{'data':[]}))
                    self.assertEqual(request('/search',dict(query='x',user_id='user0',top_k=0))[0],422)
                finally:server.shutdown();server.server_close();thread.join(5)
            self.assertNotIn('PRIVATE_BODY_SENTINEL',log.getvalue());self.assertNotIn('QUERY_URL_SENTINEL',log.getvalue())
            self.assertNotIn('SYNTHETIC_LOCAL_TRANSPORT_TOKEN',log.getvalue())
