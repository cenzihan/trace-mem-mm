# Disjoint primary anchors and support candidates

Candidate `tracemem-mm-anchored-v4.6.2-rc1` changes only the optional second selection stage. The first selection is unchanged and remains an immutable prefix. Its original source views move into a reference-only anchor list; the remaining sources form a disjoint selectable list with local indices. Returned local indices map back to their original source indices. Primary evidence cannot be selected again in the second pass. Empty or full primary selections and exhausted support pools skip the extra call.

Text prefixes, complete original returns, total candidate capacity, query options as untrusted hypotheses, preview image bytes and the total eight-image budget are preserved. Image role labels are remapped between query, reference anchor and selectable candidate; missing previews stay explicitly missing. Invalid indices and failed/unfinished responses propagate. This is not a silent fallback or answer generator. Existing factories and accounting remain unchanged; activation requires `TRACEMEM_ANCHORED_ENABLED=approved-disjoint-support`.

The prior v4.6.1 independent control supplied both alias and location sources at pool ranks1/2, but both model passes selected only the location. It stopped at the first quality failure and closed unused allowance. This new design tests whether distinct roles prevent the second pass from merely reselecting the primary. It does not establish the cause of any protected official question failure. Added evidence can still be irrelevant; retaining primary selections can retain first-pass mistakes. No quality improvement is claimed until independent validation and official evaluation.

New synthetic checks exercise local-to-original mapping, non-contiguous primary order, empty abstention, full capacity, query/anchor/candidate image roles, unchanged image bytes, strict contracts and native original returns with user isolation. Real validation and deployment are pending.


Independent rc1 validation stopped on an HTTP 400: the standalone support prompt omitted the explicit JSON instruction required by JSON object mode. Five embedding calls and the first selection succeeded; support selection did not execute successfully, so quality remains unvalidated. Historical failed input and its plan are closed. rc2 adds the explicit instruction and an offline API-contract regression; a fresh independent control is required before deployment.


The corrected rc2 control completed both model stages, but failed the direct-recall control: support added an unrelated location associated with a different identifier. No quality qualification or deployment is claimed. A preceding isolated control accidentally retained the rc1 image digest in its compose file and reproduced the old API error; those receipts are preserved separately and are not attributed to rc2 runtime.

The v4.6.3 candidate keeps first-pass evidence and allows support review only for sources sharing an exact case-sensitive mixed letter/digit identifier in their visible text. Matching is an eligibility filter, not semantic proof. Pure names, dates, metadata, query/options and image-only relationships do not create edges; these rely on the unchanged first pass. This narrow hypothesis seeks to avoid speculative additions and does not claim general multi-session reasoning improvement.
