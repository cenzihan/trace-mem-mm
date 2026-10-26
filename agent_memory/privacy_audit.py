"""Count lexical sensitivity signals locally. Never return matched text or identities.

No-hit is NOT clearance; detectors miss names, context, images and obfuscation.
No automatic redaction, model transfer or per-record allowlist is produced.
"""
import re
from collections import Counter

PATTERNS = {
    'email_address_pattern': r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
    'web_link_pattern': r'https?://[^\s<>]+',
    'phone_like_pattern': r'(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-])\d{3}[ .-]\d{4}(?!\w)',
    'credential_marker_pattern': r'\b(?:api[_ -]?key|access[_ -]?token|password|secret[_ -]?key|verification code|one.time (?:password|code))\b',
    'finance_or_transaction_terms': r'\b(?:bank|account number|credit card|invoice|payment|paid|salary|booking|reservation)\b',
    'health_terms': r'\b(?:medical|diagnosis|hospital|doctor|therapy|prescription|patient|disease)\b',
    'child_or_family_terms': r'\b(?:child|children|kid|kids|baby|son|daughter|school|family)\b',
    'identity_or_document_terms': r'\b(?:passport|identity card|student id|date of birth|full name|license plate)\b',
    'address_terms': r'\b(?:street|avenue|postal|postcode|home address|apartment|dormitory)\b',
    'redaction_marker': r'\[(?:email_address|phone_number|link|redacted)\]',
}
COMPILED = {key: re.compile(pattern, re.I) for key, pattern in PATTERNS.items()}


def text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from text_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from text_values(item)


def sensitivity_counts(rows):
    counts = Counter(); keys = Counter(); total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('source objects required')
        total += 1
        # Full row scanned in memory; keys and matched values are never emitted.
        text = '\n'.join(text_values(row))
        hits = {key for key, regex in COMPILED.items() if regex.search(text)}
        counts.update(hits); counts['any_lexical_signal'] += bool(hits)
        counts['location_field_nonempty'] += isinstance(row.get('location_name'), str) and bool(row['location_name'].strip())
        counts['timestamp_field_nonempty'] += isinstance(row.get('timestamp'), str) and bool(row['timestamp'].strip())
        for name in ('latitude','longitude','gps','coordinates','location'):
            keys['geolocation_named_field_present'] += name in row
    return dict(records=total, records_with_signals={key: counts[key] for key in COMPILED},
                any_lexical_signal=counts['any_lexical_signal'],
                location_field_nonempty=counts['location_field_nonempty'],
                timestamp_field_nonempty=counts['timestamp_field_nonempty'],
                top_level_geolocation_named_fields=keys['geolocation_named_field_present'],
                no_hit_does_not_mean_safe=True, transfer_cleared_records=0)
