# Evidence-context retrieval candidate

`tracemem-mm-context-v2.1.0-rc1` is an opt-in library implementation in
`agent_memory/context_candidate.py`. The explicit `agent_memory.context_app`
factory is now prepared for an authorized official Smoke. It requires both
the existing hosted activation and `TRACEMEM_CONTEXT_RETRIEVAL_ENABLED` set to
`approved-context-v2.1.0-rc1`. The default `bailian_app` factory stays on v2.0.4.
The candidate reuses the original model journals, lifetime call limit, financial
ledger and retention schedule; none are reset. A fixed released tag and exact
endpoint version must be checked before submission. No improved score is claimed.

## Confirmed implementation limits and changes

| Existing limitation | Candidate behavior | Remaining uncertainty |
|---|---|---|
| Mixed text/image queries did not use exact image identity. | Exact image matches enter the bounded pool and contribute a separate final rank channel even when query text is present. | This only recognizes identical image bytes; it is not visual semantic retrieval. |
| Dense-only top100 can discard a literal name or identifier. | Equal-weight rank fusion combines dense, positive BM25 and exact-image rankings before the same hosted scorer. | Extra lexical matches can add noise; older research did not establish that hybrid always beats dense. |
| Individually retrieved turns can omit the preceding question or following response. | The first12 anchors can include their immediate original neighbours from the same user, session and Add request. Each message keeps its original content and identity. | This is bounded source context, not demonstrated cross-session reasoning. Context can displace useful anchors. |
| The first over-budget image record stopped the response entirely. | Skip that whole record, then consider later smaller records until top_k or100 is reached. | Skipping can still remove important evidence; source parts are never silently cropped or rewritten. |

Original evidence remains authoritative. Derived source descriptions are used
only for ranking. The scorer still receives text, even though its model name is
`qwen3-vl-rerank`; no visual-reranking improvement is claimed. Returned scores
are reciprocal final rank, not provider probabilities or answerability scores.

The existing Add path, fixed `gpt-4o-mini`, `text-embedding-v4` at256 dimensions,
and approved hosted reranker identity are inherited. Retrieval introduces no
additional provider stage. Existing batching can still change request count
when candidate texts differ; only the original guarded transports may dispatch.
Search questions/options are not stored. Diagnostics contain counts and sizes,
not source text, questions, answers or credentials.

## Evaluation interpretation

The [AML guide](https://agentmemoryleaderboard.ai/api-guide) assigns answering
and scoring to the platform. Search ordering therefore affects which original
evidence reaches the answerer, but our retrieval metrics alone are not the
official final-answer score. Successful Smoke confirms completion of a small
compatibility run; it does not establish high accuracy or public eligibility.

The official catalog lists [ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench)
and [Mem-Gallery](https://github.com/YuanchenBei/Mem-Gallery). ATM-Bench emphasizes
long-term referential, multi-source evidence; Mem-Gallery contains long-term
multimodal conversations. A private Smoke's exact dataset/category sample sizes
must come from its own receipt, not guesses from the catalog or opaque IDs.

[Bailian's native API](https://help.aliyun.com/zh/model-studio/text-rerank-api)
also accepts image inputs for qwen3-vl-rerank. Wiring original image views into
that API requires separate input-limit, batching, accounting and real-provider
validation. This candidate deliberately does not claim that work is complete.

## Validation and next gate

Synthetic regression tests cover image/lexical matches beyond dense top100,
bounded neighbouring context, scope isolation, intact source parts, media
packing, fixed model identity, malformed indices/scores, query privacy and
restart/idempotence. These are engineering checks using test adapters, not real
model calls or a quality benchmark. Run `python scripts/run_checks.py`.

Before promotion, freeze an independent development comparison with separate
source-only calibration and held-out questions, explicitly audit overlap with
previously consumed public scenes, and use a finite dispatch plan inside the
existing server-owned budget. Compare dense baseline, each retrieval change
alone, and the combined candidate. Measure evidence recall, answer F1/accuracy,
unanswerable precision/coverage, response-prefix evidence and cost/latency.
Do not tune on protected official questions/answers or reopen old exhausted
experimental grants. A calibrated abstention rule is still absent; an arbitrary
threshold fitted to a small private Smoke would not be valid evidence.

The operator separately authorized one new Multimodal Smoke for this fixed
candidate. It uses the final remaining slot in the original task-wide five-run
limit, shared with Coding; it does not grant an additional retry or a Full run.
The candidate remains experimental until that run completes and quality can be
assessed. Coding diagnosis remains unresolved and its previous job is not resumed.
