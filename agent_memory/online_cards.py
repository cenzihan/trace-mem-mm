"""Keep only source-verifiable text quotations in online card metadata.

Raw model receipts are retained by the transport. A generated quote that is not
an exact substring is omitted, never rewritten or promoted to source evidence.
The model, prompt, summary and returned original source remain unchanged.
"""
from .protocol import content_parts, text_of
from .providers import GPT4oMiniCards


class VerifiedQuoteCards(GPT4oMiniCards):
    def extract(self, parts):
        value = super().extract(parts)
        text = text_of(content_parts(parts))
        quotes = value['evidence_quotes']
        if any(not isinstance(q, str) for q in quotes):
            raise ValueError('invalid quote type')
        verified = [q for q in quotes if q and q in text]
        return dict(value, evidence_quotes=verified,
                    omitted_unverified_quote_count=len(quotes)-len(verified))
