"""Budgeted official HTTPS preparation. Disabled live calls; explicit fake HTTP injection.

No key/environment/file access occurs unless the supplied reader is called after gates.
No SDK retries, redirects, proxy changes, error bodies or prompts are logged/persisted.
"""
import http.client
import json
import math
import queue
import re
import ssl
import threading
import time
from dataclasses import dataclass,field
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from .budget import BudgetLedger,VerifiedRate
from .protocol import MEDIA_LIMIT,content_parts,media_size
from .providers import ProviderBlocked

OPENAI_ENDPOINT='https://api.openai.com/v1/chat/completions'
BAILIAN_BEIJING_ENDPOINT='https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'
SHARED_LEDGER=Path(__file__).resolve().parent.parent/'runtime/development-budget.sqlite'


class ProviderFailure(RuntimeError):
    def __init__(self,category,status=None,attempt_id=None):
        self.category=category;self.status=status;self.attempt_id=attempt_id
        super().__init__('provider '+category+((' HTTP '+str(status)) if status is not None else ''))


@dataclass(frozen=True)
class HTTPReply:
    status:int
    body:bytes=field(repr=False)
    headers:dict=field(default_factory=dict,repr=False)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts:int=1
    retry_429:bool=False
    max_delay_seconds:float=2.0

    def validate(self,purpose):
        if type(self.max_attempts) is not int or not 1<=self.max_attempts<=2:
            raise ValueError('at most two explicitly enabled 429 attempts')
        if type(self.retry_429) is not bool or not math.isfinite(self.max_delay_seconds) or not 0<=self.max_delay_seconds<=5:
            raise ValueError('invalid bounded retry policy')
        if self.max_attempts>1 and not self.retry_429:raise ValueError('retry must explicitly target 429 only')
        if purpose=='connectivity_check' and (self.max_attempts!=1 or self.retry_429):
            raise ValueError('single connectivity check has no retry')


def validate_endpoint(model,endpoint):
    if not isinstance(endpoint,str):raise ValueError('explicit endpoint required')
    parsed=urlsplit(endpoint)
    if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port not in (None,443):
        raise ValueError('official HTTPS endpoint required')
    if model=='gpt-4o-mini' and endpoint==OPENAI_ENDPOINT:return
    if model=='text-embedding-v4' and endpoint==BAILIAN_BEIJING_ENDPOINT:return
    if (model=='text-embedding-v4' and parsed.hostname and
        re.fullmatch(r'[a-zA-Z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com',parsed.hostname) and
        parsed.path=='/compatible-mode/v1/embeddings'):return
    raise ValueError('endpoint/model policy forbids substitution or implicit region change')


class NativeHTTPS:
    """One direct TLS exchange, bounded caller deadline, no automatic retries.

    A daemon worker bounds even OS DNS waiting. On timeout the outcome is uncertain;
    its slot stays busy until exit and it checks cancellation before sending the POST.
    """
    is_fake_http=False

    def __init__(self):self._slot=threading.Lock()

    def exchange(self,endpoint,body,headers,timeout,max_response_bytes):
        if not self._slot.acquire(blocking=False):raise ProviderFailure('network_error')
        result=queue.Queue(maxsize=1);cancel=threading.Event();deadline=time.monotonic()+timeout
        def worker():
            conn=None
            try:
                parsed=urlsplit(endpoint)
                conn=http.client.HTTPSConnection(parsed.hostname,port=443,timeout=timeout,context=ssl.create_default_context())
                conn.connect()
                if cancel.is_set() or time.monotonic()>=deadline:raise TimeoutError()
                conn.sock.settimeout(max(.001,deadline-time.monotonic()))
                conn.request('POST',parsed.path,body=body,headers=headers)
                response=conn.getresponse();chunks=[];size=0
                while True:
                    if cancel.is_set() or time.monotonic()>=deadline:raise TimeoutError()
                    if conn.sock:conn.sock.settimeout(max(.001,deadline-time.monotonic()))
                    chunk=response.read1(min(65536,max_response_bytes+1-size))
                    if not chunk:break
                    chunks.append(chunk);size+=len(chunk)
                    if size>max_response_bytes:raise ProviderFailure('response_limit')
                value=HTTPReply(response.status,b''.join(chunks),{'retry-after':response.getheader('Retry-After','')})
            except ProviderFailure as exc:value=exc
            except TimeoutError:value=ProviderFailure('timeout')
            except ssl.SSLError:value=ProviderFailure('tls_error')
            except Exception:value=ProviderFailure('network_error')
            finally:
                try:
                    if conn:conn.close()
                except Exception:pass
                self._slot.release()
            result.put(value)
        thread=threading.Thread(target=worker,daemon=True);thread.start()
        try:value=result.get(timeout=timeout)
        except queue.Empty:
            cancel.set();raise ProviderFailure('timeout') from None
        if isinstance(value,Exception):raise value from None
        return value


class BudgetedJSONTransport:
    """Callable for TextEmbeddingV4/GPT4oMiniCards. Real network off by default."""
    def __init__(self,*,model,endpoint,ledger,rate,credential_reader=None,http=None,
                 mode='live',enabled=False,prices_confirmed=False,retry=None,purpose='pipeline',
                 timeout_seconds=30,max_response_bytes=4*1024*1024,usd_rates=None,sleep=time.sleep,
                 fixed_reserve_cny=None,output_token_cap=512):
        validate_endpoint(model,endpoint)
        if not isinstance(ledger,BudgetLedger) or not isinstance(rate,VerifiedRate) or rate.model!=model:
            raise ValueError('matching verified model price and persistent ledger required')
        rate.validate()
        if rate.conversion_basis=='budget_assumption_not_market_fx' and fixed_reserve_cny is None:
            raise ValueError('assumed FX requires an explicit fixed hold')
        if fixed_reserve_cny is not None:
            from .budget import micro_cny
            if not 0<micro_cny(fixed_reserve_cny)<=10000000:
                raise ValueError('fixed per-call hold must be within CNY 10')
        self.fixed_reserve_cny=fixed_reserve_cny
        if type(output_token_cap) is not int or not 1<=output_token_cap<=2048:
            raise ValueError('explicit output cap must be within 2048')
        self.output_token_cap=output_token_cap
        if mode not in ('live','offline_fake') or purpose not in ('pipeline','connectivity_check'):
            raise ValueError('invalid dispatch mode/purpose')
        if type(enabled) is not bool or type(prices_confirmed) is not bool:
            raise ValueError('explicit boolean activation/price gates required')
        self.http=http or NativeHTTPS()
        if mode=='offline_fake' and getattr(self.http,'is_fake_http',False) is not True:
            raise ValueError('offline mode requires an explicitly fake HTTP exchange')
        if mode=='offline_fake' and Path(ledger.path).resolve()==SHARED_LEDGER.resolve():
            raise ValueError('fake HTTP must never consume the real shared budget ledger')
        if mode=='live' and getattr(self.http,'is_fake_http',False):raise ValueError('fake HTTP cannot be labelled live')
        if mode=='live' and Path(ledger.path).resolve()!=SHARED_LEDGER.resolve():
            raise ValueError('live calls must share the project development ledger')
        if not isinstance(timeout_seconds,(int,float)) or isinstance(timeout_seconds,bool) or not math.isfinite(timeout_seconds) or not 0<timeout_seconds<=120:
            raise ValueError('timeout must be bounded to 120 seconds')
        if type(max_response_bytes) is not int or not 1024<=max_response_bytes<=8*1024*1024:
            raise ValueError('response size must be bounded')
        self.retry=retry or RetryPolicy();self.retry.validate(purpose)
        self.model=model;self.endpoint=endpoint;self.ledger=ledger;self.rate=rate
        self.credential_reader=credential_reader;self.mode=mode;self.enabled=enabled
        self.prices_confirmed=prices_confirmed;self.timeout=timeout_seconds;self.limit=max_response_bytes
        self.sleep=sleep;self.usd_rates=None
        if usd_rates is not None:
            if len(usd_rates)!=2:raise ValueError('two USD price rates required')
            self.usd_rates=tuple(Decimal(str(v)) for v in usd_rates)
            if any(not v.is_finite() or v<0 for v in self.usd_rates):raise ValueError('invalid USD prices')

    def _bounds(self,payload):
        if not isinstance(payload,dict) or payload.get('model')!=self.model:
            raise ValueError('Academic model name mismatch')
        if self.model=='text-embedding-v4':
            if set(payload)-{'model','input','dimensions','encoding_format'}:raise ValueError('unsupported embedding request parameter')
            texts=payload.get('input')
            if not isinstance(texts,list) or not 1<=len(texts)<=10:raise ValueError('embedding batch must be 1..10')
            if any(not isinstance(t,str) or not t.strip() or len(t.encode('utf-8'))>8192 for t in texts):
                raise ValueError('local embedding input UTF-8 byte admission limit exceeded')
            if payload.get('dimensions') not in (64,128,256,512,768,1024,1536,2048) or payload.get('encoding_format')!='float':
                raise ValueError('unsupported embedding encoding/dimensions')
            # Reserve each full accepted provider context, independent of local tokenizer estimates.
            return 8192*len(texts),0
        allowed={'model','messages','max_completion_tokens','store','n','response_format'}
        if set(payload)-allowed or payload.get('store') is not False or type(payload.get('n')) is not int or payload.get('n')!=1:
            raise ValueError('unsupported chat parameters or persistence')
        output=payload.get('max_completion_tokens')
        if type(output) is not int or not 1<=output<=self.output_token_cap:raise ValueError('local completion token cap exceeded')
        messages=payload.get('messages')
        if not isinstance(messages,list) or len(messages)!=2 or any(not isinstance(m,dict) for m in messages) or [m.get('role') for m in messages]!=['system','user']:
            raise ValueError('source-card request requires system plus one user source')
        if not isinstance(messages[0].get('content'),str):raise ValueError('text system instruction required')
        raw_parts=messages[1].get('content')
        if isinstance(raw_parts,list):
            raw_parts=json.loads(json.dumps(raw_parts))
            for part in raw_parts:
                if isinstance(part,dict) and part.get('type')=='image_url' and isinstance(part.get('image_url'),dict):
                    detail=part['image_url'].pop('detail',None)
                    if detail is not None and detail not in ('low','high','auto'):
                        raise ValueError('unsupported image detail')
        parts=content_parts(raw_parts)
        text_size=len(messages[0]['content'].encode('utf-8'))+sum(len(p['text'].encode('utf-8')) for p in parts if p['type']=='text')
        if text_size>32768 or sum(p['type']=='image_url' for p in parts)>8 or media_size(parts)>MEDIA_LIMIT:
            raise ValueError('local card text/image input admission exceeded')
        return 128000,output

    def _usage(self,data):
        usage=data.get('usage')
        if not isinstance(usage,dict):raise ProviderFailure('invalid_usage')
        if self.model=='text-embedding-v4':
            inp=usage.get('total_tokens');out=0
            if 'prompt_tokens' in usage and usage['prompt_tokens']!=inp:raise ProviderFailure('invalid_usage')
        else:
            inp=usage.get('prompt_tokens');out=usage.get('completion_tokens')
            if type(inp) is not int or type(out) is not int or type(usage.get('total_tokens')) is not int or usage['total_tokens']!=inp+out:
                raise ProviderFailure('invalid_usage')
        if type(inp) is not int or type(out) is not int or inp<0 or out<0:raise ProviderFailure('invalid_usage')
        return inp,out

    def _contract(self,data,payload):
        if self.model=='text-embedding-v4':
            rows=data.get('data');batch=payload['input'];dim=payload['dimensions']
            if not isinstance(rows,list) or len(rows)!=len(batch):raise ProviderFailure('invalid_contract')
            indices=[r.get('index') for r in rows if isinstance(r,dict)]
            if len(indices)!=len(batch) or any(type(i) is not int for i in indices) or set(indices)!=set(range(len(batch))):
                raise ProviderFailure('invalid_contract')
            for row in rows:
                vector=row.get('embedding')
                if not isinstance(vector,list) or len(vector)!=dim or any(type(v) not in (int,float) or not math.isfinite(v) for v in vector) or not any(vector):
                    raise ProviderFailure('invalid_contract')
        else:
            choices=data.get('choices')
            if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):raise ProviderFailure('invalid_contract')
            choice=choices[0];message=choice.get('message')
            if choice.get('finish_reason')!='stop' or not isinstance(message,dict) or message.get('refusal') or message.get('tool_calls'):
                raise ProviderFailure('invalid_contract')
            try:card=json.loads(message['content'])
            except (KeyError,TypeError,ValueError):raise ProviderFailure('invalid_contract') from None
            if not isinstance(card,dict) or not isinstance(card.get('summary'),str) or not isinstance(card.get('evidence_quotes'),list) or any(not isinstance(q,str) for q in card['evidence_quotes']):
                raise ProviderFailure('invalid_contract')

    def __call__(self,endpoint,payload):
        if endpoint!=self.endpoint:raise ValueError('endpoint changed after configuration')
        validate_endpoint(self.model,endpoint)
        bounds=self._bounds(payload)
        if self.mode=='live':
            if not self.enabled:raise ProviderBlocked('real HTTP is disabled; no credential read or dispatch')
            if not self.prices_confirmed:raise ProviderBlocked('verified prices not confirmed')
            if self.rate.conversion_basis=='budget_assumption_not_market_fx' and self.fixed_reserve_cny is None:
                raise ProviderBlocked('conservative FX assumption requires a retained fixed hold')
            source=urlsplit(self.rate.price_source)
            official={'developers.openai.com','platform.openai.com'} if self.model=='gpt-4o-mini' else {'help.aliyun.com'}
            if source.scheme!='https' or source.hostname not in official or source.username or source.password:
                raise ProviderBlocked('official price source required for live preparation')
            try:checked=datetime.fromisoformat(self.rate.checked_at.replace('Z','+00:00'))
            except (ValueError,TypeError):raise ProviderBlocked('price verification date required') from None
            if checked.tzinfo is None or not 0<=(datetime.now(timezone.utc)-checked).total_seconds()<=172800:
                raise ProviderBlocked('price verification is stale or future-dated')
        if self.mode=='offline_fake':
            # Fake exchanges never invoke a supplied real reader, even if wired by mistake.
            key='SYNTHETIC_KEY_NEVER_LOG_THIS'
        else:
            if self.credential_reader is None:raise ProviderBlocked('explicit safe credential reader required')
            try:key=self.credential_reader()
            except Exception:raise ProviderBlocked('credential unavailable') from None
        if not isinstance(key,str) or not key or len(key)>16384 or '\r' in key or '\n' in key:
            raise ProviderBlocked('credential unavailable')
        raw=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
        if len(raw)>44*1024*1024:raise ValueError('local serialized request cap exceeded')
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'}
        previous=None
        for index in range(self.retry.max_attempts):
            attempt=self.ledger.reserve(self.rate,*bounds,retry_of=previous,fixed_reserve_cny=self.fixed_reserve_cny)
            self.ledger.mark_dispatched(attempt);tick=time.monotonic()
            status=None;returned=None;usage=None;data=None;category='network_error';reply=None
            try:
                reply=self.http.exchange(endpoint,raw,headers,self.timeout,self.limit)
                if not isinstance(reply,HTTPReply) or type(reply.status) is not int or not 100<=reply.status<=599:
                    raise ProviderFailure('invalid_contract')
                status=reply.status
                if not 200<=status<300:
                    category=('authentication' if status==401 else 'permission_or_region' if status==403 else
                              'rate_limited' if status==429 else 'server_error' if status>=500 else
                              'redirect' if 300<=status<400 else 'client_error')
                    raise ProviderFailure(category)
                if not isinstance(reply.body,bytes) or len(reply.body)>self.limit:raise ProviderFailure('response_limit')
                try:data=json.loads(reply.body.decode('utf-8'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
                except (ValueError,UnicodeError):raise ProviderFailure('invalid_json') from None
                if not isinstance(data,dict):raise ProviderFailure('invalid_contract')
                usage=self._usage(data)
                allowed={self.model} if self.model=='text-embedding-v4' else {'gpt-4o-mini','gpt-4o-mini-2024-07-18'}
                if data.get('model') not in allowed:raise ProviderFailure('model_mismatch')
                returned=data['model'];self._contract(data,payload)
                category='success'
            except ProviderFailure as exc:category=exc.category
            except TimeoutError:category='timeout'
            except ssl.SSLError:category='tls_error'
            except Exception:category='network_error'
            usd=None
            if usage and (usage[0]>bounds[0] or usage[1]>bounds[1]):category='invalid_usage'
            if usage is not None and self.usd_rates is not None and category not in ('model_mismatch','invalid_usage'):
                usd=str((self.usd_rates[0]*usage[0]+self.usd_rates[1]*usage[1])/1000000)
            self.ledger.observe_provider(attempt,mode=self.mode,category=category,http_status=status,returned_model=returned,
                                         input_tokens=usage[0] if usage else None,output_tokens=usage[1] if usage else None,
                                         elapsed_ms=int((time.monotonic()-tick)*1000),estimated_usd=usd)
            if usage and (usage[0]>bounds[0] or usage[1]>bounds[1]):raise ProviderFailure('invalid_usage',status,attempt)
            if category=='success':return data
            if category=='rate_limited' and self.retry.retry_429 and index+1<self.retry.max_attempts:
                try:delay=float(reply.headers.get('retry-after','1'))
                except (ValueError,TypeError,AttributeError):delay=1.0
                if not math.isfinite(delay):delay=1.0
                self.sleep(max(0,min(delay,self.retry.max_delay_seconds)));previous=attempt;continue
            raise ProviderFailure(category,status,attempt) from None
