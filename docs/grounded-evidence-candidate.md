# Grounded evidence candidate v4.5.0-rc1

An independent control returned a source for an unrelated query even though all
provider calls completed successfully. The previous selector returned only source
indices; its original response was not retained. We cannot reconstruct its reason
or claim this explains any official score.

This separate candidate asks for a bounded witness with each source selection.
Text witnesses must match an exact substring of the provided source preview.
Image witnesses require an attached preview. Cross-session links may select
multiple supporting or contradicting original sources. Choices remain untrusted
alternatives, not facts or answers. The service still returns complete original
evidence, never the generated witness or a final answer.

The model remains gpt-4o-mini with exact dated return validation. One Search uses
at most one GPT call, now bounded to 1024 output tokens. The candidate pool, image
previews, raw Add path, embedding model, accounting, recovery policy and original
receipt scope are unchanged. Invalid witnesses, incomplete responses and provider
errors fail explicitly; there is no retry or hidden fallback.

Exact quotation verifies source attribution, not semantic relevance. Image
descriptions are model claims, not verified visual facts. This is a hypothesis
requiring new independent controls; it has no real preflight or official score.
It is not deployed. The stable v4.3.2 recovery service remains unchanged.
