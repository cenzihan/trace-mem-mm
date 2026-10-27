"""Native Mem0 fact-extraction response contract, on the original paid journal."""
import json

from .provider_http import ProviderFailure
from .service_transport import ServiceBudgetedTransport, ServiceTransport
from .chat_stream import assemble_chat_stream


class Mem0BudgetedTransport(ServiceBudgetedTransport):
    # UTF-8 bytes bound tokens conservatively; reserve the full 128k input window.
    text_byte_limit = 96000

    def _bounds(self,payload):
        if self.model=='gpt-4o-mini' and 'stream' in payload:
            if payload.get('stream') is not True or payload.get('stream_options')!={'include_usage':True}:
                raise ValueError('complete usage-bearing stream required')
            checked=dict(payload);checked.pop('stream');checked.pop('stream_options')
            return super()._bounds(checked)
        return super()._bounds(payload)

    def decode_response(self,reply,payload):
        if self.model=='gpt-4o-mini' and payload.get('stream') is True:
            if reply.headers.get('content-type','').split(';')[0].strip().lower()!='text/event-stream':
                raise ProviderFailure('invalid_contract')
            return assemble_chat_stream(reply.body)
        return super().decode_response(reply,payload)

    def _contract(self, data, payload):
        if self.model != 'gpt-4o-mini' or payload.get('response_format') != {'type':'json_object'}:
            return super()._contract(data, payload)
        choices = data.get('choices')
        if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):
            raise ProviderFailure('invalid_contract')
        choice = choices[0]; message = choice.get('message')
        if choice.get('finish_reason')!='stop' or not isinstance(message,dict) or message.get('refusal') or message.get('tool_calls'):
            raise ProviderFailure('invalid_contract')
        try: value=json.loads(message['content'])
        except (KeyError,TypeError,ValueError): raise ProviderFailure('invalid_contract') from None
        if not isinstance(value,dict) or not isinstance(value.get('memory'),list):
            raise ProviderFailure('invalid_contract')
        if any(not isinstance(row,dict) or not isinstance(row.get('text'),str) or not row['text'].strip() for row in value['memory']):
            raise ProviderFailure('invalid_contract')


class Mem0ServiceTransport(ServiceTransport):
    transport_type = Mem0BudgetedTransport
    output_token_cap = 2048
    receipt_scope = 'native-mem0-v3.0.3'

    def __call__(self,endpoint,payload):
        if payload.get('model')=='gpt-4o-mini':
            payload=dict(payload,stream=True,stream_options={'include_usage':True})
        return super().__call__(endpoint,payload)

    def normalize_result(self, value, model):
        if model=='gpt-4o-mini' and 'memory' in json.loads(value['choices'][0]['message']['content']):
            return value
        return super().normalize_result(value,model)


class Mem0EvidenceTransport(Mem0ServiceTransport):
    # Same global journal and money ledger. A new execution scope cannot bypass
    # any unknown without a separately audited permanent retirement proof.
    receipt_scope = 'native-mem0-v3.1.1'
