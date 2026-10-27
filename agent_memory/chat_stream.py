"""Strict Chat Completions SSE assembly; no partial answers or automatic retry."""
import json

from .provider_http import ProviderFailure


def assemble_chat_stream(raw):
    def fail():raise ProviderFailure('invalid_contract')
    try:text=raw.decode('utf-8').replace('\r\n','\n')
    except UnicodeError:fail()
    done=False;identity=None;model=None;finish=None;usage=None;content=[]
    for block in text.split('\n\n'):
        lines=[line for line in block.split('\n') if line and not line.startswith(':')]
        if not lines:continue
        if done:fail()
        if any(not line.startswith('data:') for line in lines):fail()
        data='\n'.join(line[5:].lstrip(' ') for line in lines)
        if data=='[DONE]':
            done=True;continue
        try:chunk=json.loads(data,parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
        except (ValueError,TypeError):fail()
        if not isinstance(chunk,dict) or 'error' in chunk:fail()
        if not isinstance(chunk.get('id'),str) or not chunk['id']:fail()
        if chunk.get('model') not in ('gpt-4o-mini','gpt-4o-mini-2024-07-18'):
            raise ProviderFailure('model_mismatch')
        if identity is None:identity=chunk['id'];model=chunk['model']
        if chunk['id']!=identity or chunk['model']!=model:fail()
        choices=chunk.get('choices')
        if not isinstance(choices,list) or len(choices)>1:fail()
        if choices:
            row=choices[0]
            if not isinstance(row,dict) or type(row.get('index')) is not int or row['index']!=0 or finish is not None:fail()
            delta=row.get('delta')
            if not isinstance(delta,dict) or delta.get('tool_calls') or delta.get('function_call') or delta.get('refusal'):fail()
            if delta.get('role') not in (None,'assistant'):fail()
            piece=delta.get('content')
            if piece is not None:
                if not isinstance(piece,str):fail()
                content.append(piece)
            reason=row.get('finish_reason')
            if reason is not None:
                if reason!='stop':fail()
                finish=reason
        if chunk.get('usage') is not None:
            if usage is not None or finish!='stop' or not isinstance(chunk['usage'],dict):fail()
            usage=chunk['usage']
    if not done or finish!='stop' or usage is None or identity is None:fail()
    return {'id':identity,'model':model,'choices':[{'index':0,'finish_reason':finish,
        'message':{'role':'assistant','content':''.join(content)}}],'usage':usage}

