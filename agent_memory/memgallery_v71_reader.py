"""V7.1 source-reference interface, wrapping the unchanged V7 reader semantics.

Aliases are stable over the entire source inventory, not assigned by rank.
Only source_id serialization and an appended citation instruction change.
Source text/metadata/image IDs, native question, point rules and decoded answer
are never rewritten. No I/O, provider, gold or scoring operations exist here.
"""
from dataclasses import asdict, replace
import json

from . import memgallery_v7_reader as original


MODEL, ENDPOINT = original.MODEL, original.ENDPOINT
MemGalleryReaderBudget = original.MemGalleryReaderBudget
CITATION_INSTRUCTION = '''
SOURCE REFERENCE FORMAT: cited_ids may contain only the supplied source_id aliases
M0001, M0002, and so on (M followed by exactly four digits). Copy each cited alias
as one complete string. Do not cite session labels, round labels, image IDs, or
the separate components of an identity. Those remain evidence metadata, not
source references. This changes only citation format, not any answer rule.'''
READER_PROMPT = original.READER_PROMPT + CITATION_INSTRUCTION


def source_aliases(source_native_ids):
    ids = list(source_native_ids)
    if any(not isinstance(identity, str) for identity in ids) or len(set(ids)) != len(ids) or len(ids) > 9999:
        raise ValueError('at most 9999 distinct original source identities required')
    for identity in ids:
        original._identity(identity)
    return {identity: 'M' + str(index).zfill(4) for index, identity in enumerate(sorted(ids), 1)}


def pack_memgallery_evidence(query, ranked_ids, sources_by_id, count_tokens,
                            budget=MemGalleryReaderBudget()):
    if not isinstance(sources_by_id, dict) or not isinstance(ranked_ids, list):
        raise ValueError('complete source inventory and original ranking required')
    if not isinstance(budget, MemGalleryReaderBudget) or not callable(count_tokens):
        raise ValueError('fixed reader budget and local tokenizer required')
    aliases = source_aliases(sources_by_id)
    if any(not isinstance(identity, str) or identity not in aliases for identity in ranked_ids):
        raise ValueError('all ranked original identities must exist in the complete inventory')
    inverse = {alias: identity for identity, alias in aliases.items()}
    old_system = original._count(count_tokens, original.READER_PROMPT)
    new_system = original._count(count_tokens, READER_PROMPT)
    added_tokens = max(0, new_system - old_system)
    added_bytes = len(READER_PROMPT.encode('utf-8')) - len(original.READER_PROMPT.encode('utf-8'))
    if budget.text_tokens <= added_tokens or budget.utf8_bytes <= added_bytes:
        raise ValueError('citation instruction exceeds fixed reader budget')
    inner_budget = replace(budget, text_tokens=budget.text_tokens - added_tokens,
                           utf8_bytes=budget.utf8_bytes - added_bytes)
    # The existing packer generates the same complete evidence prefix and headers;
    # supplying alternate source-map keys changes only their source_id field.
    alias_pack = original.pack_memgallery_evidence(query, [aliases[key] for key in ranked_ids],
        {aliases[key]: value for key, value in sources_by_id.items()}, count_tokens, inner_budget)
    original.reader_payload(alias_pack)
    text = alias_pack['parts'][0]['text']
    user_tokens = original._count(count_tokens, text)
    tokens = new_system + user_tokens + 32
    byte_count = len(READER_PROMPT.encode('utf-8')) + len(text.encode('utf-8'))
    if tokens > budget.text_tokens or byte_count > budget.utf8_bytes:
        raise ValueError('complete actual V7.1 payload exceeds fixed budget')
    payload = original._payload(READER_PROMPT, text, budget.output_tokens)
    retained = [inverse[alias] for alias in alias_pack['source_memory_ids']]
    result = dict(schema='memgallery-v71-reader-pack-v1', alias_pack=alias_pack, payload=payload,
        alias_by_native_id=aliases, native_id_by_alias=inverse,
        alias_map_sha256=original._sha(original._json(aliases)),
        alias_rule='ascending original native ID over complete source inventory; M plus four decimal digits',
        question_index=alias_pack['question_index'], point=alias_pack['point'],
        question_sha256=alias_pack['question_sha256'], query_image_count=alias_pack['query_image_count'],
        query_image_condition=alias_pack['query_image_condition'], budget=asdict(budget),
        parts=payload['messages'][1]['content'], source_memory_ids=retained,
        ranked_native_ids=[inverse[alias] for alias in alias_pack['ranked_native_ids']],
        dropped_native_ids=[inverse[alias] for alias in alias_pack['dropped_native_ids']],
        source_sha256={inverse[alias]: digest for alias, digest in alias_pack['source_sha256'].items()},
        image_ids=list(alias_pack['image_ids']), source_records_retained=len(retained),
        text_token_count=tokens, utf8_text_bytes=byte_count, image_count=0, context_gold_access=False,
        token_components=dict(system=new_system, user=user_tokens, framing=32, legacy_system=old_system),
        internal_token_reserve=added_tokens, internal_byte_reserve=added_bytes,
        prompt_sha256=original._sha(READER_PROMPT),
        payload_sha256=original._sha(original._json(payload)))
    result['pack_sha256'] = original._sha(original._json(result))
    reader_payload(result)
    return result


def reader_payload(packed):
    original._verify_hash(packed, 'memgallery-v71-reader-pack-v1', 'pack_sha256')
    aliases, inverse = packed.get('alias_by_native_id'), packed.get('native_id_by_alias')
    if (not isinstance(aliases, dict) or not isinstance(inverse, dict)
            or aliases != source_aliases(aliases) or inverse != {value: key for key, value in aliases.items()}
            or packed.get('alias_map_sha256') != original._sha(original._json(aliases))):
        raise ValueError('complete deterministic bijective source alias map required')
    alias_pack = packed['alias_pack']
    old_payload = original.reader_payload(alias_pack)
    budget = MemGalleryReaderBudget(**packed['budget'])
    inner_budget = MemGalleryReaderBudget(**alias_pack['budget'])
    components = packed.get('token_components')
    if (not isinstance(components, dict) or set(components) != {'system', 'user', 'framing', 'legacy_system'}
            or any(type(value) is not int or value <= 0 for value in components.values())
            or components['framing'] != 32):
        raise ValueError('exact positive local token accounting required')
    added_tokens = max(0, components['system'] - components['legacy_system'])
    added_bytes = len(READER_PROMPT.encode('utf-8')) - len(original.READER_PROMPT.encode('utf-8'))
    if (packed.get('internal_token_reserve') != added_tokens or packed.get('internal_byte_reserve') != added_bytes
            or asdict(inner_budget) != asdict(replace(budget, text_tokens=budget.text_tokens-added_tokens,
                                                    utf8_bytes=budget.utf8_bytes-added_bytes))
            or alias_pack['text_token_count'] != components['legacy_system'] + components['user'] + 32):
        raise ValueError('legacy prefix must reserve the complete added prompt budget')
    if any(alias not in inverse for alias in alias_pack['ranked_native_ids']):
        raise ValueError('alias candidates must belong to the full inventory')
    for field in ('source_memory_ids', 'ranked_native_ids', 'dropped_native_ids'):
        if packed.get(field) != [inverse[alias] for alias in alias_pack[field]]:
            raise ValueError('native and alias ranking identities differ')
    for field in ('question_index', 'point', 'question_sha256', 'query_image_count',
                  'query_image_condition', 'image_ids', 'source_records_retained'):
        if packed.get(field) != alias_pack[field]:
            raise ValueError('V7.1 changed original question or evidence metadata')
    if packed.get('source_sha256') != {inverse[alias]: digest for alias, digest in alias_pack['source_sha256'].items()}:
        raise ValueError('V7.1 source records differ from original evidence')
    text = old_payload['messages'][1]['content'][0]['text']
    expected = original._payload(READER_PROMPT, text, budget.output_tokens)
    tokens = components['system'] + components['user'] + 32
    byte_count = len(READER_PROMPT.encode('utf-8')) + len(text.encode('utf-8'))
    if (packed.get('payload') != expected or packed.get('parts') != expected['messages'][1]['content']
            or packed.get('prompt_sha256') != original._sha(READER_PROMPT)
            or packed.get('payload_sha256') != original._sha(original._json(expected))
            or packed.get('text_token_count') != tokens or tokens > budget.text_tokens
            or packed.get('utf8_text_bytes') != byte_count or byte_count > budget.utf8_bytes
            or packed.get('image_count') != 0 or packed.get('context_gold_access') is not False):
        raise ValueError('exact V7.1 payload and fixed budget required')
    return json.loads(original._json(expected))


def decode_reader(response, packed):
    reader_payload(packed)
    value = original.decode_reader(response, packed['alias_pack'])
    aliases = list(value['cited_ids'])
    native = [packed['native_id_by_alias'][alias] for alias in aliases]
    return dict(value, cited_ids=native, cited_aliases=aliases,
                context_aliases=list(value['context_memory_ids']),
                context_memory_ids=list(packed['source_memory_ids']),
                text_token_count=packed['text_token_count'], utf8_text_bytes=packed['utf8_text_bytes'],
                payload_sha256=packed['payload_sha256'], pack_sha256=packed['pack_sha256'],
                alias_pack_sha256=packed['alias_pack']['pack_sha256'],
                alias_map_sha256=packed['alias_map_sha256'])
