# Evidence-completeness review candidate

v4.6.0-rc1 adds a second bounded selection pass on top of the v4.3.10 retry runtime.
An independent synthetic control previously placed both required sources in the
candidate pool but the selector returned only one. This motivates checking the
selection for missing links; it does not establish the cause of an official score.

The first pass is unchanged. The second sees the same original query, choices,
source prefixes and image previews plus the provisional source indices. It may
add needed linking sources, remove unrelated selections, or return an empty list.
Its final selection replaces the draft. Both passes return at most eight unique
indices. There are no quotes, dependency graphs, generated answers or source facts
in the response contract. Exceptions propagate without a draft fallback.

The existing pool and image limits remain in force: 32 sources, 1500-byte text
prefixes, eight image previews including the query, and a 3 MiB request limit.
Each pass allows 512 output tokens. Search uses two logical GPT calls; with the
explicitly enabled retry policy that can mean up to 12 accounted provider attempts.
This increases potential cost and latency. Retrieval quality, including negative
controls, still requires independent real validation before any score claim.

The new factory is opt-in. Add, original evidence storage, user isolation, ledger,
bounded retries and failed-Add cleanup are inherited from the successful runtime.
The existing production factory is unchanged. This candidate has not yet passed
real validation and has no official Smoke score.

Local offline suite: 897 checks, 65 skipped for optional platform/dependency reasons,
zero failures or errors. Ten new controls cover link completion, removal of a
misleading draft, empty drafts, failure propagation, strict response validation,
and native storage isolation. The native SDK control requires the pinned image.
