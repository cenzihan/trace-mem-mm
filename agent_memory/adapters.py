"""No external service is called. Interfaces reserve future model integrations."""
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "mock"
    embedding_model: str = "text-embedding-v4"
    llm_model: str = "gpt-4o-mini"
    reranker: str = "identity"
    candidate_reranker_rules_confirmed: bool = False

    def validate(self):
        if self.embedding_model != "text-embedding-v4" or self.llm_model != "gpt-4o-mini":
            raise ValueError("Academic model policy forbids model substitution")
        if self.mode not in ("mock", "configured"):
            raise ValueError("unknown model mode")
        if self.reranker != "identity":
            raise ValueError("Candidate reranker is not enabled; rule confirmation and implementation required")


class EmbeddingAdapter(Protocol):
    model: str
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class CaptionAdapter(Protocol):
    model: str
    def caption(self, original_parts: list[dict]) -> str: ...


class Reranker(Protocol):
    def score(self, query: list[dict], candidates: list[dict], base_scores: list[float]) -> list[float]: ...


class IdentityMockReranker:
    """Offline scoring pass-through. No claim of model-based reranking."""
    def score(self, query, candidates, base_scores):
        return list(base_scores)


class ConfiguredModelBoundary:
    """Fail closed until reviewed provider/schema/billing configuration is implemented."""
    def __init__(self, config: ModelConfig):
        config.validate()
        if config.mode != "mock":
            raise RuntimeError("Real adapters are unconfigured; no network call was made")


class MemoryUpdater(Protocol):
    """Future Add-only update policy: retain raw provenance; never accept Search questions."""
    def derive(self, source_messages: list[dict]) -> list[dict]: ...
