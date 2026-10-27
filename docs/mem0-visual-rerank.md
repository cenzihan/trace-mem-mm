# Native Mem0 with bounded visual reranking

`tracemem-mm-mem0-visual-v4.1.0` is a separate, explicitly activated candidate.
It keeps [raw Mem0](mem0-raw-source.md) Add and its original embedding journal:
Mem0 2.2.1 `infer=False`, local Qdrant, `text-embedding-v4` at256dimensions,
complete original source archives. It does not call an LLM or generate captions.
Images are decoded and validated before any Add provider call.

Search uses the user-scoped native/BM25 ranking and reserves20of40scoring
anchors for image-containing sources. Image anchors follow the existing ranking,
then stable source order;20other anchors come from the existing ranking.
The remaining original ranking is retained after the reranked prefix.
Exact image matches remain first, and bounded same-Add neighbours are retained.

Each scoring source has a UTF-8-safe6000-byte text prefix and its first image,
if present. The query contributes its complete text (local4000-byte limit)
and first image. Images are separate JPEG512/q85 views, at most256KiB each.
All additional source/query images remain preserved but are not scoring views.
Each query-view pass scores every selected document view; source ordering uses
the maximum valid relevance score across its views. These scores are not
calibrated answerability. No abstention threshold is fitted to official results.

The hosted `qwen3-vl-rerank` is the previously approved model identity,
**not** the frozen local Qwen2B weights. Its [official API](https://help.aliyun.com/zh/model-studio/text-rerank-api)
accepts text/image query and document objects with Base64 data URIs.
Our batches are stricter: at most8views,3MiB serialized bytes and the reviewed
120000-token input ceiling; at most24requests per Search. A complete batch plan
is checked against the existing512lifetime hosted-call ceiling and original
money ledger before scoring. Each actual call still reserves through that
same ledger and original receipt journal. There is no new allowance, retry,
endpoint change, TLS relaxation or hidden fallback after unknown responses.
Incomplete/invalid responses fail closed; no partial ranking is returned.

Returned evidence always contains the complete original ordered parts and image
bytes. Scoring prefixes/previews never replace originals or become facts.
Successful provider receipts retain usage metadata only; queries are not saved.

The [AML guide](https://agentmemoryleaderboard.ai/api-guide) fixes academic
embeddings and LLM components while leaving rerankers unrestricted. This
no-LLM ablation is explicitly disclosed; it is not represented as a GPT Add
system or an approved Full submission.

Offline tests use synthetic images and the real pinned Mem0 SDK with fake
providers. Independent bounded real-provider validation, immutable release,
CI/container source verification and fresh platform eligibility are required
before an official run. No quality improvement is assumed.

