# Export provenance

This publication copy excludes native benchmark questions, answers, source records,
per-question identifiers and results, private receipts, ledgers, credentials and
model weights. Aggregate results, selection rules and method limitations remain.
Detailed historical reports stay in the original experiment project's private
`reports/` directory and are not part of this export.

The exported `agent_memory/atm_answer_metrics.py` removes upstream function
docstrings because some examples overlap with benchmark answers. Its MIT license,
upstream revision, original upstream file hashes and provenance are retained.
Python ASTs before and after removal are identical after stripping docstrings;
the deterministic scoring algorithm is unchanged.

The original frozen experiment module remains unchanged at SHA256
`be2ef7c9cb2777fa9587c50cea2bd71be075396c86ae395bc775dc366a2a96a9`.
Historical experiment code hashes refer to that original frozen copy, not to this
documentation-cleaned export. The exported module has SHA256
`804a4f6bb0232453c6a8fa17cd3f076971b77294f7cf4850121d3a70e7dc58b4`.

The publication snapshot is prepared directly on the existing remote base rather
than publishing the earlier local development commits containing those examples.
The original local experiment history is preserved separately.

## V7 credential binding

V7 code and synthetic tests are copied from the original experiment project.
The export alone replaces `scripts/bailian_save_and_check.py` with a minimal
read-only Windows Credential Manager binding. Its `read_bailian` function has
the same Python AST as the source function. The source opaque-credential
validator is inlined into `validate_bailian` with the same checks and error
semantics; the wrapper continues to return `None` after successful validation.

The export binding contains no credential saving, desktop-file input, one-time
connectivity checks, network calls or command-line entry point. It reads the
existing dedicated Bailian target only when `read_bailian()` is explicitly
called by a live runner. No credential value or desktop file was accessed while
preparing this export. The setup-script dependency chain is not distributed.

The original source helper SHA256 is
`55a1daa2a04486af6eb8e7eb79aab7714d78135af0011282b0ed46b93f6fe357`;
its validator source SHA256 is
`9a39edc9e5c80f263392178a1c9b9d8cdcd26ff7834c32efcc1c57d071de89b0`.
The exported read-only binding SHA256 is
`17e689eb938fe794498ba7f8d0816564535c9e47bb0fa873df14df07d0a99d10`.
The original experiment source and frozen run artifacts remain unchanged.

V7 runtime data, raw containers, question/gold projections, receipts, ledgers,
tokenizer caches, model weights and dynamic reports are excluded. Historical
run hashes refer to the original frozen experiment files; this export is not a
copy of that private runtime or of the original credential setup program.
