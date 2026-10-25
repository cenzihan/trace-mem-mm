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
identity is not visual understanding. Earlier weak-card and image-query failures are retained in local reports. The
image-only query validation is now fixed, and one frozen 185-round native image
study returned hash-verified original evidence. Answer accuracy, abstention and
public deployment remain unverified. Qwen scoring-only reranking is permitted by the current announcement; it stays disabled in this controlled comparison.

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

Native image milestone: the image-only query validation is fixed. A/B/C ran on one complete public scenario (185 rounds,31 originals); structural/neighbor expansion did not give stable gains. Original response images were hash-verified; same-vector restart order is stable, while provider re-encoding can exchange tail ranks. No reader/official score is claimed. See [native results and limitations](docs/native-v2-results.md). Dataset-specific paid runners and runtime artifacts remain local and are not included.


## Independent scene and reader preparation

The next native scene was selected by recorded filename ordering, keeping the prior
A/B/C core and visual prompt unchanged. All 177 rounds and 45 media were preserved.
The run stopped after 12 visual requests: 11 validated HTTP200 responses and one
HTTP200 local `invalid_contract`. No embedding/reader/proxy-judge requests followed,
and no new retrieval or answer score is claimed. It was not retried.

New evidence-reader and isolated-scoring modules have offline tests. They match A/C
image counts and text/output caps, validate citations, and keep references outside
the reader and memory. These tests do not establish live reader quality.
ATM native schema/inventory preparation now separates public questions from gold;
its data license is CC BY-NC 4.0, distinct from MIT code. Video ingestion is explicitly
unsupported; no ATM media or model evaluation has run.
See [scene status](docs/native-v3-results.md), [frozen plan](docs/native-v3-plan.md),
and [ATM integration](docs/atm-native-integration.md). Data, answers, logs, credentials,
budget databases and paid runner scripts are excluded from this repository.
