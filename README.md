# TraceMem-MM

Private development repository for the Academic Multimodal Agent Memory Leaderboard.
This is a local prototype, not an official submission or a competition score.

Original ordered text/images and per-occurrence provenance remain authoritative.
SQLite stores user-isolated memories, source cards and optional vectors. Retrieval
supports raw-text BM25, source-card BM25, text-embedding-v4 and reciprocal-rank fusion.
Cross-session retrieval stays within a user namespace. Explicit temporal facts and
bounded links are optional; natural-time reasoning is not established.

All LLM roles use gpt-4o-mini. Embeddings use text-embedding-v4; model substitution
fails closed. Actual development used the gpt-4o-mini-2024-07-18 response version and
256-dimensional text-embedding-v4. Image originals are returned with evidence; byte
identity is not visual understanding. A small live integration found weak visual
cards, and its first image query failed local validation. Full visual-query and
answer accuracy remain unverified. Qwen scoring-only reranking is permitted by the current announcement; it stays disabled in this controlled comparison.

## Run without credentials

Python 3.11 or newer, standard library only:

```sh
python -m agent_memory --db runtime/demo.sqlite capabilities
python -m agent_memory --db runtime/demo.sqlite add fixtures/add.json
python -m agent_memory --db runtime/demo.sqlite search fixtures/search.json
python scripts/verify_offline_delivery.py
python scripts/run_checks.py
```

The CLI defaults to deterministic offline mode and cannot silently call paid APIs.
The delivery verifier creates temporary databases and an ephemeral loopback server,
then closes it. Tests use synthetic unit fixtures and fake HTTP, not benchmark gold.
Explicit live library wiring requires official HTTPS, separate secure provider
readers, verified prices, a shared hard budget and a durable bounded run guard.
Do not add credentials, user data, logs, ledgers or benchmark data to this repository.

Native Mem-Gallery/ATM-Bench and popular public datasets are under source/split/license
and overlap review. Native benchmark questions/gold belong solely to an external
scoring harness. No invented dialogues are offered as a native benchmark. The
dataset boundary rejects gold/question ingestion and unreviewed provenance. No
official Smoke or Full has run, and no public endpoint or enrollment is configured.

## Research references

- Mem0: https://arxiv.org/abs/2504.19413
- A-MEM: https://arxiv.org/abs/2502.12110
- Zep: https://arxiv.org/abs/2501.13956
- M3-Agent: https://arxiv.org/abs/2508.09736
- Qwen3-VL reranking: https://arxiv.org/abs/2601.04720

These are design references. This repository does not vendor their code, model
weights or datasets. See NOTICE.md for the current licensing status.

Detailed design and pinned research implementations: [METHODS.md](METHODS.md).

Current Academic model/data policy: [eligibility](docs/eligibility.md).
