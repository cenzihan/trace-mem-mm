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
