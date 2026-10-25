"""Source-grounded offline cards. No claim of LLM extraction or visual recognition."""
import re

FACT = re.compile(r"FACT\s+entity=(?P<entity>[\w-]+)\s+attribute=(?P<attribute>[\w-]+)\s+value=(?P<value>[\w-]+)\s+event=(?P<event>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
LINK = re.compile(r"LINK\s+entity=(?P<entity>[\w-]+)\s+related=(?P<related>[\w-]+)")


def derive(text):
    facts = [match.groupdict() for match in FACT.finditer(text)]
    links = [match.groupdict() for match in LINK.finditer(text)]
    # Exact source text, never an invented caption/summary. Ordinary prose remains prose.
    return dict(kind="source_card", extraction="deterministic_explicit_templates",
                source_quote=text, facts=facts, links=links, uncertain=False)
