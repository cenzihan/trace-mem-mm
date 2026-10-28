# Linked evidence candidate

`tracemem-mm-linked-v4.5.2-rc1` is an unsubmitted candidate. An independent
control for its predecessor retrieved both needed sources into the candidate
pool but returned only one after model selection. That observation identifies
a selection-stage omission in that control, not the cause of any official score.

The new selector asks the same approved model for source witnesses, direct
evidence roots and explicit prerequisite indices. The service returns the
bounded dependency closure, including prerequisites from other Add sessions.
It rejects missing node witnesses, cycles, duplicate indices and more than eight
sources. All entries are validated, but valid entries outside the roots' dependency
closure are omitted from retrieval. The earlier v4.5.1 control exposed a complete
HTTP200 response with an unused downstream entry that its stricter validator
rejected. The new behavior does not replay or reclassify that failed request.
Text witnesses must occur in the provided
source prefix. Image witnesses require an attached preview. These checks prove
source attribution and graph consistency, not semantic truth or completeness:
the model can still omit a necessary edge or select unrelated evidence.

The candidate keeps real Mem0 infer=False Add, the existing source pool,
text-embedding-v4, one gpt-4o-mini Search call and the exact dated model check.
The output cap is 1536 tokens. Full original sources are returned, not generated
witnesses or answers. Same-Add context, exact-image priority and existing output
packing limits remain. Source prefixes and image coverage are still bounded;
packing or a small requested top_k can limit the evidence returned. No calibrated
abstention or universal multi-hop completeness is claimed.

It retains the same journal, raw receipt scope, original money accounting and
operator recovery policy. It neither uses nor unlocks the blocked hosted
reranker. New uncertainty still stops service, and POST calls are not retried.
It is not deployed and has no official score. Deployment requires a new frozen
independent real control to pass, exact image/source verification and an idle
official platform. Earlier failed controls and closed grants remain closed.
