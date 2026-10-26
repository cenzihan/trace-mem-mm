# Online service v1.0.0

Application version: `tracemem-mm-online-v1.0.0`. This is an API admission
candidate for Academic Multimodal, not an official score or accepted entry.
The retrieval pipeline is the one validated in the complete earlier native-data
acceptance: GPT source cards, 256-dimensional Dense Top100, fixed Qwen candidate
scoring, and original ordered source messages returned to the evaluator.

## Fixed models

- LLM: `gpt-4o-mini` through the official OpenAI Chat Completions endpoint;
  accepted returned snapshot `gpt-4o-mini-2024-07-18`. No answer generation.
- Embedding: `text-embedding-v4`, Beijing official endpoint, 256 dimensions.
- Reranker: `Qwen/Qwen3-VL-Reranker-2B`, revision
  `4bd860ac4f15ad1897a214615cccc700f8f71818`, weight SHA256
  `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`.
  Candidate-only yes/no logit difference, text input, BF16 SDPA. The hosted
  `qwen3-vl-rerank` alias is not used. Identity mismatch fails closed.

Source images inform GPT source summaries. Query images use supplied text and
exact source-image identity; there is no general image-query encoder. Reranking
uses source text plus GPT summary, while returned evidence remains original.

## API and operation

`POST /add` and `POST /search` require the registered Bearer key. `GET /health`
is unauthenticated and checks storage, bounded accounting, price freshness, and
the exact private scorer identity. Only a successful durable, searchable Add
returns 200. Identical request IDs are idempotent; conflicting payloads return
409. Queries and options are not inserted into the memory store.

Capacity: one model-backed operation at a time; concurrent calls receive 429
with `Retry-After: 5`. HTTP body limit 64 MiB; decoded media 30 MiB per Add or
query, 10 MiB per image. Add supports at most 16 messages, 8 images per message,
and 6000 UTF-8 source-text bytes including extracted explicit facts per message.
Query text is capped at 8192 UTF-8 bytes. Oversize supported-schema inputs receive
clear 4xx errors. Embedding input including generated metadata is limited to
8192 UTF-8 bytes; source evidence is never silently truncated. At most 100
candidates are scored; returned images stay within 30 MiB.

The Compose profile uses one non-root worker, two threads, a 1 GiB memory cap,
read-only root filesystem, private loopback binding, persistent SQLite, and
file-mounted credentials. A TLS reverse proxy exposes only health/Add/Search.
GPU inference is on a separate supervised process through a restricted SSH
reverse tunnel, not this small CPU host. It loads only the pinned local model.

The original CNY500 development ledger is migrated as the single active writer;
the former desktop file is retired read-only. Historical accounting is verified
against its row-prefix hashes. No old experiment grant is reopened and no second
budget is created. Unknown outcomes stop future dispatch and are never replayed.
Prices require official-source verification within 48 hours. The USD conversion
uses the explicitly conservative 10 CNY/USD budget assumption, not market FX.

The deployment lease ends 2026-11-05 16:30 UTC; extending it requires a reviewed
budget, retention deadline and GPU allocation. A separate automatic cleanup
stops the service and deletes all submitted memory/derived databases by
2026-11-07, within 30 days of receipt. No official-data backups are enabled.
Successful provider bodies are deleted immediately after durable usage checks;
accounting retains amounts, usage and hashes only. Request bodies, query text,
API keys and retrieved evidence are not written to access logs. Evaluation data
is used only to serve evaluation, not training, tuning, analytics or sharing.

## Reproduction

Run `python scripts/run_checks.py` for offline unit and loopback checks. Use
`compose.service.yaml` beside a `source/` checkout and private `config/`,
`runtime/`, and `secrets/` directories. Provision the original ledger ownership
manifest and currently verified prices; this profile deliberately refuses to
create a fresh budget. Supply your own credentials and the fixed scorer service
on loopback port 18978. Private runtime data and model weights are excluded from
the source repository. The default Compose profile remains a separate offline
baseline. Full native acceptance evidence is in
`reports/online-native-acceptance.json`; local source tests alone do not establish
real provider behavior or official acceptance.

The exact deployed GPU entry points are `scripts/qwen_service.py`,
`scripts/gpu_supervisor.py`, and `scripts/tunnel_runner.py`; the frozen scoring
helper is `scripts/run_atm_v6_remote.py`. These reflect the actual deployment
paths. Reproduction on another host requires changing only installation, Python,
key-file, and tunnel paths while preserving the model hashes, inference settings,
and scorer code. The serving pair input preserves the query and template and may
shorten the document ranking view to fit 4096 tokens; returned evidence is never
shortened by this scorer. Keys and the pinned model snapshot must be provisioned
separately. Supervision preserves cumulative counters across process restarts.

The service adapter omits non-string auxiliary quote entries, and the source-card
adapter keeps only exact source-text quotes. Neither operation changes the GPT
summary, source messages, source images, model identity, prompt, or ranking
settings. A real preflight initially exposed object-valued quote metadata; its
original HTTP200 receipt and invalid-contract observation were preserved, then
the known response was reused locally without repeating the provider request.
