"""Verify documented CLI and real loopback HTTP with new temporary DBs, no model access."""
import argparse
import contextlib
import http.client
import hashlib
import io
import ipaddress
import json
import runpy
import socket
import sys
import tempfile
import threading
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
from wsgiref.simple_server import make_server as real_make_server

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def verify():
    import agent_memory
    from agent_memory import cli
    if not Path(agent_memory.__file__).resolve().is_relative_to(ROOT):raise AssertionError('wrong source checkout imported')
    attempts={'external_network':0,'credentials':0,'model_transport':0}
    ledger_paths=[ROOT/'runtime/development-budget.sqlite',ROOT/'runtime/evaluation-ledger.sqlite']
    ledger_before={p:hashlib.sha256(p.read_bytes()).digest() if p.exists() else None for p in ledger_paths}
    original_connect=socket.create_connection
    def connect(address,*args,**kwargs):
        try:allowed=ipaddress.ip_address(address[0]).is_loopback
        except ValueError:allowed=False
        if not allowed:
            attempts['external_network']+=1;raise AssertionError('external network forbidden in offline delivery')
        return original_connect(address,*args,**kwargs)
    def credentials(*args,**kwargs):
        attempts['credentials']+=1;raise AssertionError('credential lookup forbidden in offline delivery')
    def model(*args,**kwargs):
        attempts['model_transport']+=1;raise AssertionError('model transport forbidden in offline delivery')
    def command(args):
        output=io.StringIO()
        with patch.object(sys,'argv',['agent_memory']+args),contextlib.redirect_stdout(output):
            runpy.run_module('agent_memory',run_name='__main__',alter_sys=True)
        return json.loads(output.getvalue())
    with contextlib.ExitStack() as guard:
        guard.enter_context(patch('socket.create_connection',connect))
        guard.enter_context(patch('agent_memory.windows_credentials.read_key',credentials))
        guard.enter_context(patch('agent_memory.provider_http.NativeHTTPS.exchange',model))
        with tempfile.TemporaryDirectory(prefix='aml-offline-delivery-') as temporary:
            root=Path(temporary).resolve()
            if root==Path(tempfile.gettempdir()).resolve() or not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
                raise AssertionError('temporary cleanup boundary invalid')
            cli_db=root/'cli.sqlite';http_db=root/'http.sqlite'
            empty_cli=not cli_db.exists();empty_http=not http_db.exists()
            demonstration=command([])
            capabilities=command(['--db',str(cli_db),'capabilities'])
            add=command(['--db',str(cli_db),'add','fixtures/add.json'])
            found=command(['--db',str(cli_db),'search','fixtures/search.json'])
            assert demonstration['mode']=='mock' and capabilities['mode']=='local_deterministic'
            assert capabilities['dense']=='disabled_not_simulated' and capabilities['online_verified'] is False
            source=json.loads((ROOT/'fixtures/add.json').read_text(encoding='utf-8'))
            query=json.loads((ROOT/'fixtures/search.json').read_text(encoding='utf-8'))
            expected=dict(success=True,**{key:source[key] for key in ('request_id','user_id','session_id')})
            assert add==expected and found['data']
            captured=[];ready=threading.Event();errors=[];server_output=io.StringIO()
            def server_factory(*args,**kwargs):
                instance=real_make_server(*args,**kwargs);captured.append(instance);ready.set();return instance
            def serve():
                try:
                    with patch('wsgiref.simple_server.make_server',server_factory),contextlib.redirect_stdout(server_output):
                        cli.main(['--db',str(http_db),'serve','--development-loopback','--host','127.0.0.1','--port','0'])
                except BaseException as exc:errors.append(type(exc).__name__);ready.set()
            worker=threading.Thread(target=serve,daemon=True);worker.start()
            if not ready.wait(5):raise AssertionError('loopback startup timed out')
            if errors or not captured:raise AssertionError('loopback startup failed')
            server=captured[0];port=server.server_port
            def request(path,payload=None):
                connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
                try:
                    connection.request('GET' if payload is None else 'POST',path,
                        body=None if payload is None else json.dumps(payload).encode(),
                        headers={'Content-Type':'application/json'})
                    response=connection.getresponse();return response.status,json.loads(response.read())
                finally:connection.close()
            try:
                health_status,health=request('/health')
                assert health_status==200 and health['mode']=='local_deterministic'
                empty_status,empty=request('/search',query)
                assert empty_status==200 and empty=={'data':[]}
                add_status,echo=request('/add',source)
                search_status,retrieved=request('/search',query)
                replay_status,replay=request('/add',source)
                originals=[message['content'] for message in source['messages']]
                preserved=bool(retrieved['data']) and all(row['content'] in originals for row in retrieved['data'])
                assert add_status==200 and echo==expected and search_status==200 and preserved
                assert replay_status==200 and replay==expected
            finally:
                server.shutdown();worker.join(5)
            assert not worker.is_alive() and not errors and server.socket.fileno()==-1
            assert all(value==0 for value in attempts.values())
            assert all((hashlib.sha256(p.read_bytes()).digest() if p.exists() else None)==before for p,before in ledger_before.items())
            return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),source='extracted local release candidate',
                mode='offline local_deterministic; no model simulation used by HTTP service',
                cli_noargs_demo='mock-labelled original B0 demo',cli_add_echo_correct=add==expected,
                cli_search_result_count=len(found['data']),fresh_cli_database=empty_cli,fresh_http_database=empty_http,
                http_health=health_status,http_add=add_status,http_search=search_status,http_retry_add=replay_status,
                empty_before_add=True,original_ordered_image_parts_preserved=preserved,
                http_search_result_count=len(retrieved['data']),host='127.0.0.1',port=port,
                port_choice='OS-assigned ephemeral port; same documented CLI entry point',
                server_closed=True,server_thread_joined=True,public_binding=False,
                external_network_attempts=attempts['external_network'],credential_reader_calls=attempts['credentials'],
                model_transport_calls=attempts['model_transport'],real_api_calls=0,gpu_calls=0,official_evaluations=0,
                persistent_budget_or_evaluation_ledger_created=False,temporary_databases_removed_on_return=True,
                full_105_test_suite_rerun=False,heldout_rerun=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=verify()
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
