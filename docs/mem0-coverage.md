# Bounded candidate-image coverage candidate

`tracemem-mm-mem0-coverage-v4.4.0` retains the v4.3.1 raw Mem0 storage,
choice-context representation, candidate pool and fixed model identities.
Previously, the selector could consider 32 sources while attaching only eight
images including the query image. Later image-only sources could therefore lack
a visible preview. This is a code limitation, not an established explanation of
any official score.

The candidate groups image-bearing sources so that every candidate's first image
is attached once. Text-only candidates accompany each group. Each request keeps
the same eight-image, 3 MiB, 32-source and eight-selected-index bounds. A Search
uses at most five sequential `gpt-4o-mini` calls. Single-batch requests retain the
previous payload and selection order. For multiple batches, selected original
indices are merged by their best local rank with original candidate order as the
tie-breaker; repeated text candidates receive no extra votes. The union contains
at most 32 sources. Full original evidence is returned, never preview bytes.

All payloads and conservative aggregate funds are checked before the first GPT
call. Every actual call still reserves funds in the original ledger and uses the
original provider journal and raw scope. No retry, partial batch result, hidden
fallback or hosted-reranker unlock is introduced. Later images within a single
source and later query images remain outside the preview view. Independent batch
decisions also do not provide joint comparison between different image groups.

838 offline checks pass, including 16 new coverage, mapping, capacity, budget,
failure, isolation and real-Mem0-SDK regressions with mocked providers. These
checks make no paid calls and do not establish live reliability or score gains.
The fixed deployed v4.3.1 release remains unchanged while this candidate is
validated.
