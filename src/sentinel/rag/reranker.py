"""Second-stage rerankers. `lexical` is offline; `cross-encoder` needs sentence-transformers."""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Protocol

from sentinel.config import Settings, get_settings
from sentinel.rag.text import tokenize


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class LexicalReranker:
    """IDF-weighted query-term coverage + bigram bonus, with IDF computed over the candidate pool."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = tokenize(query)
        if not q or not passages:
            return [0.0] * len(passages)
        toks = [tokenize(p) for p in passages]
        n = len(passages)
        df = Counter(t for ts in toks for t in set(ts))
        idf = {t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in set(q)}
        q_bi = set(zip(q, q[1:]))
        out = []
        for ts in toks:
            tset = set(ts)
            cover = sum(idf[t] for t in set(q) if t in tset)
            total = sum(idf.values()) or 1.0
            bi = len(q_bi & set(zip(ts, ts[1:])))
            out.append(cover / total + 0.15 * bi)
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'sentence-transformers not installed. Run: pip install "sentinel[embeddings]" '
                "or set SENTINEL_RERANKER=lexical"
            ) from e
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(x) for x in self.model.predict([(query, p) for p in passages])]


@lru_cache(maxsize=2)
def _make(provider: str, model: str) -> Reranker:
    if provider == "lexical":
        return LexicalReranker()
    if provider == "cross-encoder":
        return CrossEncoderReranker(model)
    raise ValueError(f"Unknown reranker provider: {provider!r}")


def get_reranker(settings: Settings | None = None) -> Reranker:
    s = settings or get_settings()
    return _make(s.reranker_provider, s.reranker_model)
