"""Finite content-free diagnostics; optional private cache for approved public inputs."""
import json


def chat_contract_branch(data):
    """Same acceptance conditions as the frozen transport, with a finite failure code."""
    if not isinstance(data,dict):return 'response_not_object'
    choices=data.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):return 'choices_shape'
    choice=choices[0];message=choice.get('message')
    if choice.get('finish_reason')!='stop':
        reason=choice.get('finish_reason')
        return 'finish_'+reason if reason in ('length','content_filter','tool_calls','function_call') else 'finish_other'
    if not isinstance(message,dict):return 'message_shape'
    if message.get('refusal'):return 'message_refusal'
    if message.get('tool_calls'):return 'message_tool_calls'
    try:card=json.loads(message['content'])
    except (KeyError,TypeError,ValueError):return 'content_not_json'
    if not isinstance(card,dict):return 'content_not_object'
    if not isinstance(card.get('summary'),str):return 'summary_type_or_missing'
    quotes=card.get('evidence_quotes')
    if not isinstance(quotes,list):return 'quotes_type_or_missing'
    if any(not isinstance(q,str) for q in quotes):return 'quote_item_type'
    return 'accepted_frozen_chat_contract'


def describe_reply(model,reply):
    """Finite values and bounded counts, no arbitrary strings from a response."""
    status=getattr(reply,'status',None);raw=getattr(reply,'body',None)
    if type(status) is not int or not 100<=status<=599:status=None
    result=dict(http_status=status,response_bytes=len(raw) if isinstance(raw,bytes) else None,
                branch='uninspected',includes_response_content=False)
    if not isinstance(raw,bytes) or len(raw)>8*1024*1024:
        result['branch']='body_type_or_limit';return result
    try:data=json.loads(raw.decode('utf-8'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
    except (ValueError,UnicodeError):result['branch']='response_not_json';return result
    if model=='gpt-4o-mini':result['branch']=chat_contract_branch(data)
    elif model=='text-embedding-v4':result['branch']='embedding_diagnostics_not_chat'
    else:result['branch']='unapproved_model'
    return result


class DiagnosticHTTP:
    """Wrap an existing HTTP exchange without another request or changed validation.

    sink receives structural fields only. A failed sink blocks returning the reply;
    callers retain the dispatched request's budget and must not automatically retry.
    """
    def __init__(self,http,model,sink,response_sink=None):
        self.http=http;self.model=model;self.sink=sink;self.response_sink=response_sink
        self.is_fake_http=getattr(http,'is_fake_http',False)
    def exchange(self,endpoint,body,headers,timeout,max_response_bytes):
        reply=self.http.exchange(endpoint,body,headers,timeout,max_response_bytes)
        self.sink(describe_reply(self.model,reply))
        # Optional private evaluation cache, never a log. Opt in only for reviewed
        # public source inputs. Never retain an HTTP error body or credentials.
        if self.response_sink is not None and reply.status==200 and self.model=='gpt-4o-mini':
            try:data=json.loads(reply.body.decode('utf-8'))
            except (ValueError,UnicodeError):data=None
            if isinstance(data,dict) and data.get('model') in ('gpt-4o-mini','gpt-4o-mini-2024-07-18'):
                self.response_sink({key:data[key] for key in ('model','usage','choices') if key in data})
        return reply
