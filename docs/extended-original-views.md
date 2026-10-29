# Bounded original text visibility candidate

The selector previously saw only the first 1500 UTF-8 bytes of each retrieved
source. A relationship stated later in an original was therefore absent from
the selection input even though the service retained the complete original.
Fresh synthetic visibility tests reproduce that limitation without benchmark
content. This is a code-level limitation, not a diagnosis of official errors.

The opt-in `tracemem-mm-extended-views-v4.7.0-rc1` candidate uses one selection
call, the original flat-index contract and the original selection prompt.
It raises the per-source prefix ceiling to 6000 bytes while allocating a common
prefix limit across all sources. The entire prompt's text stays at or below
94000 bytes, below the existing 96000-byte transport limit; the serialized
request stays below 3 MiB. Escaping, UTF-8 boundaries, metadata, query, options
and image labels count against those limits. Short inputs remain identical to
the original selector. Truncated views remain explicitly marked.

No identifier gate or second-pass expansion is used by this factory. Retrieval
pool ordering, image previews, model identities, original evidence return,
user isolation, retry authorization, original billing and journals are retained.
The candidate does not generate an answer, summary or new source facts.

Longer views may add noise and latency. The 32-source candidate pool, eight-image
budget and truncation beyond the new bound still limit recall. Offline tests do
not establish improved model selection or an official score. Independent real
controls and a correctly bound official version are still required before
claiming an improvement.

Factory: `agent_memory.extended_views:create_extended_app()` with explicit
`TRACEMEM_EXTENDED_VIEWS_ENABLED=approved-bounded-original-views` activation.
