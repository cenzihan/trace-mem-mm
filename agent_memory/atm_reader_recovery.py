"""Offline compatibility decoding of completed ATM reader receipts.

Only the two auxiliary display fields are normalized. Answer text, native IDs,
confidence, abstention, response admission and frozen evidence validation retain
the original decoder's rules. This function neither reads gold nor performs I/O;
the caller must independently verify receipt provenance before calling it.
"""
import json

from .atm_reader import ATMTextReader, MODEL, _strict_json, reader_payload


def recover_decode(response, pack):
    """Return ``(decoded, auxiliary_change_counts)`` without mutating inputs.

    Unknown native IDs retain the original decoder's false validity flags; they
    are never removed, replaced or silently made valid. Provenance contains no
    response text, predictions or evidence contents.
    """
    reader_payload(pack)
    if not isinstance(response, dict) or response.get('model') not in (MODEL, 'gpt-4o-mini-2024-07-18'):
        raise ValueError('specified reader model required')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('one reader response required')
    choice = choices[0]
    message = choice.get('message')
    if (choice.get('finish_reason') != 'stop' or not isinstance(message, dict)
            or message.get('refusal') or message.get('tool_calls') or not isinstance(message.get('content'), str)):
        raise ValueError('incomplete or refused reader response')
    value = _strict_json(message['content'])
    if (not isinstance(value, dict)
            or set(value) != {'summary', 'evidence_quotes', 'answer', 'answer_ids', 'cited_ids', 'confidence'}
            or not isinstance(value['summary'], str)
            or not isinstance(value['evidence_quotes'], list)
            or any(not isinstance(quote, str) for quote in value['evidence_quotes'])):
        raise ValueError('ATM reader recovery auxiliary schema')
    summary_changed = value['summary'] != 'atm reader output'
    quotes_changed = bool(value['evidence_quotes'])
    provenance = dict(summary_changed=summary_changed, evidence_quotes_changed=quotes_changed,
                      evidence_quote_count_removed=len(value['evidence_quotes']),
                      auxiliary_fields_changed=int(summary_changed) + int(quotes_changed))
    normalized = dict(value, summary='atm reader output', evidence_quotes=[])
    normalized_message = dict(message, content=json.dumps(normalized, ensure_ascii=False, allow_nan=False))
    normalized_response = dict(response, choices=[dict(choice, message=normalized_message)])
    return ATMTextReader.decode(normalized_response, pack), provenance
