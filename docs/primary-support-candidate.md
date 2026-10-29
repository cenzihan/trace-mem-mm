# Primary evidence with bounded supporting sources

Candidate `tracemem-mm-support-v4.6.1-rc1` keeps the baseline first selection and its order. When that selection contains one to seven sources, one additional GPT-4o-mini call examines the same bounded original candidate views and may append explicit supporting links. It cannot remove or reorder primary indices. An empty primary stays empty; an eight-source primary needs no expansion. The final selection remains at most eight unique indices.

This trades away second-pass pruning of an incorrect primary selection. Added sources may also distract the downstream answerer; model-selected links are not semantic proofs. A failure or invalid second response still propagates rather than silently returning a draft. The explicitly authorized bounded retries, original budget, model restrictions, input capacities and original evidence return remain unchanged. The new factory requires `TRACEMEM_SUPPORT_ENABLED=approved-primary-support`; old factories and frozen versions are unchanged.

The prior replacement-review candidate completed its private 13-question Smoke at23.08, versus30.77 for the successful baseline. Direct recall was0, atomic retrieval100, cross-session reasoning0 and abstention11.11. All26 GPT and235 embedding calls succeeded. This aggregate result motivates testing primary retention, but does not identify the lost source or establish a causal or statistically significant effect. Protected evaluation text and answers were not inspected for this change.

New synthetic regressions cover destructive review, stable primary order, unique bounded expansion, empty-primary abstention, complete-response failure, cross-user isolation and original-source return. Real independent quality validation and official score are pending. Do not deploy or advertise an improvement based only on these offline tests.

## Independent control result

The first real control failed the predeclared cross-session requirement after direct recall passed. Both required source roles were present at pool ranks1 and2. The first pass selected only the location source, and the support pass selected that same source again, so the final return lacked the alias link. This is evidence about this independent synthetic control, not an explanation of protected official questions.

Four Add and two Search requests returned HTTP200; all four GPT and six embedding calls succeeded. The finite plan stopped on its first quality failure and closed the remaining22 attempts and1.999942CNY conservative allowance. The source remains a draft, not deployed, tagged, bound or submitted for official Smoke. The exact tested runtime source is2a3d31c6138ecf3f7e1554fb95a9b43a08785af0. A future experiment must use a new finite plan and new control inputs.
