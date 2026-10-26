"""Synthetic container-only HTTP checks. Run only against a disposable test volume.

seed writes fixed synthetic data; verify checks persistence after recreation.
No model calls are enabled. Only status summaries, never credentials or bodies,
are printed. Defaults target the API from inside its own container.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII='
CONTENT = [dict(type='text', text='Synthetic cobalt memo 173 before image.'),
           dict(type='image_url', image_url=dict(url='data:image/png;base64,'+PNG)),
           dict(type='text', text='Synthetic cobalt memo 173 after image.')]
ADD = dict(request_id='container-smoke-request', user_id='container-smoke-user',
           session_id='container-smoke-session', messages=[dict(role='user',content=CONTENT)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('seed','verify'))
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    target=urlsplit(args.base_url)
    if target.scheme!='http' or target.hostname not in ('127.0.0.1','localhost') or target.path not in ('','/'):
        raise ValueError('synthetic smoke is restricted to local container HTTP')
    auth_mode=os.environ.get('TRACEMEM_AUTH_MODE','Bearer')
    if auth_mode not in ('Bearer','Token','X-Api-Key'):
        raise ValueError('synthetic smoke requires a supported authenticated mode')
    token=Path(os.environ['TRACEMEM_AUTH_TOKEN_FILE']).read_text(encoding='ascii').strip()
    def request(method,path,payload=None,authenticated=True):
        connection=http.client.HTTPConnection(target.hostname,target.port or 8000,timeout=10)
        body=None if payload is None else json.dumps(payload).encode()
        headers={'Content-Type':'application/json'}
        if authenticated:
            if auth_mode=='X-Api-Key': headers['X-Api-Key']=token
            else: headers['Authorization']=auth_mode+' '+token
        try:
            connection.request(method,path,body,headers)
            response=connection.getresponse()
            return response.status,json.loads(response.read())
        finally: connection.close()
    status,health=request('GET','/health',authenticated=False)
    assert status==200 and health['mode']=='local_deterministic' and health['experiment_v6_online'] is False
    query=dict(user_id=ADD['user_id'],query='Synthetic cobalt memo 173',top_k=5)
    assert request('POST','/search',query,False)[0]==401
    assert request('POST','/add',ADD,False)[0]==401
    if args.phase=='seed':
        assert request('POST','/add',ADD)[0]==200
        assert request('POST','/add',ADD)[0]==200
        changed=dict(ADD,messages=[dict(role='user',content='different synthetic body')])
        assert request('POST','/add',changed)[0]==409
    status,result=request('POST','/search',query)
    assert status==200 and len(result['data'])==1 and result['data'][0]['content']==CONTENT
    other=dict(query,user_id='container-smoke-other-user')
    assert request('POST','/search',other)==(200,{'data':[]})
    print(json.dumps(dict(state='passed',phase=args.phase,synthetic=True,model_calls=0,
                         multimodal_order_preserved=True,user_isolation=True,official_evaluation=False)))


if __name__=='__main__': main()
