"""One serialized, pooled HTTP/2 POST per GPT request; no model retry."""
import json
import queue
import ssl
import threading
import time

from .provider_http import (BAILIAN_BEIJING_ENDPOINT, OPENAI_ENDPOINT,
                            HTTPReply, NativeHTTPS, ProviderFailure)


class HTTP2AcademicHTTPS:
    """Explicit evidence-only client; the existing embedding transport is kept.

    The worker bounds total elapsed time, including DNS. A timed-out worker owns
    its slot until exit and checks cancellation before sending request headers.
    The trace callback sees event names only; it never logs event arguments.
    """
    is_fake_http = False

    def __init__(self):
        import httpx
        self._httpx = httpx
        transport = httpx.HTTPTransport(verify=ssl.create_default_context(),
            http1=False, http2=True, trust_env=False, retries=0,
            limits=httpx.Limits(max_connections=1,max_keepalive_connections=1,
                               keepalive_expiry=60))
        self._client = httpx.Client(transport=transport,trust_env=False,
                                    follow_redirects=False)
        self._embedding = NativeHTTPS(handshake_attempts=3)
        self._slot = threading.Lock()
        self._closed = False
        self._failed = False
        self._counts = dict(gpt_posts_started=0,http2_responses=0,
                            complete_sse_responses=0,new_connections=0)

    def counters(self):
        return dict(self._counts)

    def close(self):
        self._closed = True
        if self._slot.acquire(blocking=False):
            try:self._client.close()
            finally:self._slot.release()

    def exchange(self,endpoint,body,headers,timeout,max_response_bytes):
        if self._closed or self._failed:
            raise ProviderFailure('network_error',phase='tls_setup')
        if endpoint == BAILIAN_BEIJING_ENDPOINT:
            return self._embedding.exchange(endpoint,body,headers,timeout,max_response_bytes)
        if endpoint != OPENAI_ENDPOINT:
            raise ValueError('fixed official academic endpoint required')
        requested=json.loads(body)
        if (requested.get('model')!='gpt-4o-mini' or requested.get('stream') is not True
                or requested.get('stream_options')!={'include_usage':True}):
            raise ValueError('fixed usage-bearing GPT stream required')
        if not 0 < timeout <= 120 or not 1024 <= max_response_bytes <= 8*1024*1024:
            raise ValueError('bounded timeout and response size required')
        if not self._slot.acquire(blocking=False):
            raise ProviderFailure('network_error',phase='deadline_unknown')
        deadline=time.monotonic()+timeout
        cancel=threading.Event();result=queue.Queue(maxsize=1)
        state=dict(phase='tls_setup',received_bytes=0,response_status=None,
                   sse_done_frame=False,sse_complete=False,http2=False)

        def check_deadline():
            if cancel.is_set() or self._closed or time.monotonic()>=deadline:
                raise ProviderFailure('timeout',phase=state['phase'])

        def worker():
            from .chat_stream import CompleteChatStream
            tracker=CompleteChatStream();posts=0;reply=None;failure=None
            def trace(event,info):
                # HTTPcore can reassign an unavailable pooled connection. Guard
                # every send event so even an internal reassignment cannot send
                # this POST twice. No request/exception/header info is retained.
                nonlocal posts
                # Cancel at stages whose owner already has a cleanup path. In
                # particular, raising between TCP completion and TLS ownership
                # can leak a freshly connected socket in the client library.
                if event in ('connection.connect_tcp.started','http2.send_connection_init.started',
                             'http2.send_request_headers.started','http11.send_request_headers.started',
                             'http2.send_request_body.started','http2.receive_response_headers.started',
                             'http2.receive_response_body.started'):
                    check_deadline()
                if event == 'connection.connect_tcp.started':
                    self._counts['new_connections']+=1
                if event in ('http2.send_request_headers.started','http11.send_request_headers.started'):
                    if event.startswith('http11.') or posts:
                        raise ProviderFailure('network_error',phase=state['phase'])
                    posts+=1;self._counts['gpt_posts_started']+=1
                    state['phase']='post_started'
                elif event == 'http2.receive_response_headers.started':
                    state['phase']='response_headers'
            try:
                check_deadline()
                sent_headers=dict(headers,Accept='text/event-stream',**{'Accept-Encoding':'identity'})
                with self._client.stream('POST',endpoint,content=body,headers=sent_headers,
                        timeout=self._httpx.Timeout(timeout,connect=min(15,timeout),pool=min(1,timeout)),
                        extensions={'trace':trace},follow_redirects=False) as response:
                    state['response_status']=response.status_code
                    state['http2']=response.http_version=='HTTP/2'
                    if not state['http2'] or posts!=1:
                        raise ProviderFailure('network_error')
                    self._counts['http2_responses']+=1
                    state['phase']='response_body';chunks=[]
                    if response.headers.get('content-encoding','identity').lower() not in ('','identity'):
                        raise ProviderFailure('invalid_contract')
                    sse=(response.status_code==200 and response.headers.get('content-type','').split(';')[0].strip().lower()=='text/event-stream')
                    for chunk in response.iter_raw():
                        check_deadline();state['received_bytes']+=len(chunk)
                        if state['received_bytes']>max_response_bytes:
                            raise ProviderFailure('response_limit')
                        chunks.append(chunk)
                        if sse:
                            tracker.feed(chunk)
                            state['sse_done_frame']=tracker.done_frame
                            state['sse_complete']=tracker.complete
                    check_deadline()
                    # Require HTTP/2 END_STREAM as well as the model's strict
                    # SSE contract. An interrupted frame is never recovered from
                    # partial bytes. Completed connections may be reused safely.
                    if sse and tracker.complete:self._counts['complete_sse_responses']+=1
                    reply=HTTPReply(response.status_code,b''.join(chunks),{
                        'content-type':response.headers.get('content-type',''),
                        'retry-after':response.headers.get('retry-after','')})
            except ProviderFailure as exc:failure=exc
            except Exception as exc:
                kind='OtherTransportError';category='network_error'
                for cls,label in ((self._httpx.TimeoutException,'HTTPXTimeout'),
                                  (self._httpx.RemoteProtocolError,'HTTPXRemoteProtocolError'),
                                  (self._httpx.ConnectError,'HTTPXConnectError'),
                                  (self._httpx.ReadError,'HTTPXReadError'),
                                  (self._httpx.WriteError,'HTTPXWriteError')):
                    if isinstance(exc,cls):kind=label;break
                if isinstance(exc,self._httpx.TimeoutException):category='timeout'
                failure=ProviderFailure(category);failure.transport_kind=kind
            finally:
                if failure is not None or cancel.is_set() or self._closed:
                    self._failed=True
                    try:self._client.close()
                    except Exception:pass
                if failure is not None:
                    failure.phase=state['phase']
                    failure.response_diagnostics={k:v for k,v in state.items() if k!='phase'}
                self._slot.release()
                result.put(failure if failure is not None else reply)
        threading.Thread(target=worker,daemon=True).start()
        try:value=result.get(timeout=timeout)
        except queue.Empty:
            cancel.set();self._failed=True
            failure=ProviderFailure('timeout',phase='deadline_unknown')
            failure.response_diagnostics={k:v for k,v in state.items() if k!='phase'}
            raise failure from None
        if isinstance(value,Exception):raise value from None
        return value
