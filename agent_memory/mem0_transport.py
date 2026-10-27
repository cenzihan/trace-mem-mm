"""Native Mem0 fact-extraction response contract, on the original paid journal."""
import json

from .provider_http import ProviderFailure
from .service_transport import ServiceBudgetedTransport, ServiceTransport


class Mem0BudgetedTransport(ServiceBudgetedTransport):
    # UTF-8 bytes bound tokens conservatively; reserve the full 128k input window.
    text_byte_limit = 96000

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

    def normalize_result(self, value, model):
        if model=='gpt-4o-mini' and 'memory' in json.loads(value['choices'][0]['message']['content']):
            return value
        return super().normalize_result(value,model)
