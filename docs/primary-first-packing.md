# Primary original sources before neighbors

Candidate v4.8.0-rc1 starts from the successful v4.3.10-rc1 source. It changes only packing order: keep every primary returned by the original selector, including exact-image identities, in its original order; then append the same eligible same-Add neighbors. The selector, prompt, candidate pool, image views, original contents, call accounting, retry policy and media limit are unchanged.

Independent offline reproduction: at top_k=100 a first selected image, its unselected neighbor and a later selected image compete for a two-image media budget. The old interleaving retains the neighbor and skips the later selection. The candidate places both primary images first. This is a packing mechanism, not a diagnosis of official protected examples. With ample media capacity the member set is unchanged, but order and reciprocal scores can change. Primary images can still exceed the total media limit themselves. Exact-image prioritization remains unchanged.

The role-selection family reached an official private Smoke score of15.38, below the30.77 baseline, despite passing its synthetic controls. This candidate does not include that family's prompt, role contract or identity-only abstention rule. No new official score is claimed here.
