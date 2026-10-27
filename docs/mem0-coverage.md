# Bounded candidate-image coverage candidate

`tracemem-mm-mem0-coverage-v4.4.1` retains the v4.3.1 raw Mem0 storage,
choice-context representation, candidate pool and fixed model identities.
Previously, the selector could consider 32 sources while attaching only eight
images including the query image. Later image-only sources could therefore lack
a visible preview. This is a code limitation, not an established explanation of
any official score.

The initial v4.4.0 candidate groups image-bearing sources so that every candidate's first image
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

Its independent nine-original-image preflight failed the positive first-result
control even though all provider calls completed. Successful response bodies were
minimized, so the exact historical selection sequence cannot be reconstructed.
No deployment or official test followed that failure.

v4.4.1 adds one global comparison after multiple image batches. Image survivors
are admitted round-robin by local rank, up to the original image bound; all
selected text-only sources accompany them. The final GPT response supplies the
global source order, capped at eight sources. This limits Search to six GPT
calls, reserves conservative funds for all six before dispatch, and never returns
partial survivors if the final comparison fails. Single-batch behavior is
unchanged. The survivor cap can still discard useful lower-ranked images, and
more calls increase latency and connection exposure. This is a new hypothesis,
not a confirmed explanation or repair of the earlier independent failure.

Offline checks include the original 16 coverage regressions and ten new global
comparison, budget, index-mapping and failure controls. They use mocked providers
and do not establish live reliability or score gains. The fixed deployed v4.3.1
release remains unchanged while this candidate is validated.

The v4.4.1 independent live preflight also failed the positive first-result
control despite 14 successful model calls. Its remaining controls were not run;
the unused finite allowance was closed. The exact returned ordering was not
retained, and no historical detail is inferred from the boolean failure. This
candidate stays a draft: no deployment or official evaluation is authorized by
these failed controls. Further work needs prospective independent-control rank
diagnostics and evidence of improvement, not repeated official submissions.

## Independent-control diagnostics

`scripts/independent_control_probe.py` adds prospective diagnostics for a new,
preplanned independent control in an isolated local pilot. It is not imported by
service factories and must never be attached to official traffic or a store with
official data. Its provenance argument is a declaration, not a license or overlap
audit; that audit and a separate finite call plan are still required before any
real input is used. Closed or uncertain inputs must not be replayed for diagnostics.

The caller fixes the complete target content before running the control, wraps
the existing selector once, then observes the actual packed original results.
Output includes only candidate counts, one-based target ranks, image counts and
fixed stage/outcome labels. It distinguishes a target absent from the candidate
pool, omitted by an initial batch, excluded from the final comparison, selected
but absent/reordered in the returned results, and a failed model call. Absence
from final results alone does not identify whether identity ordering, neighbor
expansion or media packing caused it. Duplicate target matches are disclosed.

The transport wrapper dispatches exactly once through the original transport and
delegates all admission/ledger checks. It neither reserves extra calls nor
changes the algorithm, prompt, response validation or return order. Failures still
raise the original error without partial results. Mapping/diagnostic failures
invalidate stage observations rather than inventing ranks. The bounded private
JSON output has no source IDs, query, options, images, raw provider responses or
exception messages, and cannot overwrite an earlier record. Failure to write it
does not turn a failed selection into success or change a successful selection.

Sixteen synthetic offline controls exercise stage separation, omission, output
minimization, unchanged payloads and call counts, original budget denial, invalid
responses and write failure. This diagnostic helper has not been used for a new
live preflight, and cannot recover the missing historical R13/R14 ranks or make
either failed candidate ready for deployment.
