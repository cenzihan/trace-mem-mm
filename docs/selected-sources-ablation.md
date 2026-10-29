# Selected-source output ablation

This candidate starts from fixed v4.3.10 and disables only the automatic same-Add
neighbor expansion after selection. The candidate pool, original memory storage,
one GPT selection, prompt, options, exact-image identity inclusion, accounting and
bounded retries remain unchanged. Limits remain 32 candidates, 8 preview images,
8 model-selected indices and 1500 bytes per source view.

The hypothesis is that unselected neighbors can introduce irrelevant context.
This differs from primary-first packing, which changes order but retains those
neighbors. It is not a claimed repair for cross-session reasoning or abstention.
Removing neighbors can lose useful context when the selector omits it; an
explicit test demonstrates that tradeoff. Exact-image hits still survive empty
model selection, so this is not a fully selective abstention policy.

No official protected content is used in tests. New independent controls must
show that required direct and cross-session sources are actually selected, not
recovered incidentally through neighboring positions. Official results remain
the criterion for improvement over the fixed 30.77 private Smoke baseline.
