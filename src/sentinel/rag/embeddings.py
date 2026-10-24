"""Embedding backends. `hash` is an offline feature-hashing embedder (no downloads, lower quality)."""
from __future__ import annotations

import zlib
from functools import lru_cache
from typing import Protocol

import numpy as np

from sentinel import gpu
from sentinel.config import Settings, get_settings
from sentinel.rag.text import tokenize

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Signed feature hashing over unigrams + bigrams, L2-normalised. Deterministic and offline."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = "hash"

    def _vec(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = zlib.crc32(f.encode())
            sign = 1.0 if zlib.crc32(("s" + f).encode()) & 1 else -1.0
            v[h % self.dim] += sign
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'sentence-transformers not installed. Run: pip install "sentinel[embeddings]" '
                "or set SENTINEL_EMBEDDINGS=hash"
            ) from e
        self.model = SentenceTransformer(model_name, device="cpu")  # moved to GPU only inside a ZeroGPU window
        self.name = model_name
        self._bge = "bge" in model_name.lower()
        self._key = f"embed:{model_name}"
        gpu.register(self._key, self._encode)

    def _encode(self, texts: list[str], on_gpu: bool) -> list[list[float]]:
        return gpu.with_device(
            self.model, on_gpu,
            lambda: self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist(),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return gpu.run(self._key, texts)

    def embed_query(self, text: str) -> list[float]:
        t = (_BGE_QUERY_PREFIX + text) if self._bge else text
        return gpu.run(self._key, [t])[0]


@lru_cache(maxsize=4)
def _make(provider: str, model: str) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    return _make(s.embeddings_provider, s.embeddings_model)
