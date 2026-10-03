"""Hybrid retrieval: dense (Chroma) + BM25, fused with Reciprocal Rank Fusion, then reranked.

Modes (used for ablations):  dense | bm25 | hybrid
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from sentinel.config import Settings, get_settings
from sentinel.rag import store
from sentinel.rag.embeddings import get_embedder
from sentinel.rag.ingest import KB
from sentinel.rag.reranker import get_reranker
from sentinel.rag.text import tokenize


@dataclass
class Hit:
    id: str
    text: str
    source: str
    score: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source, "score": round(self.score, 4), "meta": self.meta}


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, settings: Settings | None = None, collection: str = KB) -> None:
        self.s = settings or get_settings()
        self.base = collection
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._docs: dict[str, tuple[str, dict]] = {}
        self._built_version = -1

    def _context(self, doc_id: str) -> str:
        """Body text prefixed with its title/section (used for BM25 and reranking)."""
        text, meta = self._docs[doc_id]
        head = " ".join(x for x in (meta.get("title", ""), meta.get("section", "")) if x)
        return f"{head}. {text}" if head else text

    # -- BM25 index (rebuilt lazily when the collection changes) --
    def _ensure_bm25(self) -> None:
        v = store.index_version(self.base, self.s)
        if self._bm25 is not None and v == self._built_version:
            return
        data = store.get_collection(self.base, self.s).get(include=["documents", "metadatas"])
        self._ids = list(data["ids"])
        self._docs = {i: (d, m or {}) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        corpus = [tokenize(self._context(i)) or ["_"] for i in self._ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._built_version = v

    def _dense_rank(self, query: str, n: int, where: dict | None) -> list[str]:
        col = store.get_collection(self.base, self.s)
        if col.count() == 0:
            return []
        q = get_embedder(self.s).embed_query(query)
        res = col.query(
            query_embeddings=[q],
            n_results=min(n, col.count()),
            where=where,
            include=["documents", "metadatas"],
        )
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
            self._docs.setdefault(i, (d, m or {}))
        return list(res["ids"][0])

    def _bm25_rank(self, query: str, n: int, where: dict | None) -> list[str]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        qt = tokenize(query)
        if not qt:
            return []
        scores = self._bm25.get_scores(qt)
        order = sorted(range(len(self._ids)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            doc_id = self._ids[i]
            if where and any(self._docs[doc_id][1].get(k) != v for k, v in where.items()):
                continue
            out.append(doc_id)
            if len(out) >= n:
                break
        return out

    def search(
        self,
        query: str | list[str],
        k: int | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
        where: dict | None = None,
    ) -> list[Hit]:
        """`query` may be a list (multi-query / HyDE): all rankings are fused together;
        reranking always scores against the first (original) query."""
        queries = [q for q in ([query] if isinstance(query, str) else query) if q and q.strip()]
        if not queries:
            return []
        k = k or self.s.top_k_final
        self._ensure_bm25()
        rankings: list[list[str]] = []
        for q in queries:
            if mode in ("dense", "hybrid"):
                rankings.append(self._dense_rank(q, self.s.top_k_dense, where))
            if mode in ("bm25", "hybrid"):
                rankings.append(self._bm25_rank(q, self.s.top_k_bm25, where))
        fused = rrf_fuse([r for r in rankings if r], self.s.rrf_k)
        if not fused:
            return []
        pool_ids = sorted(fused, key=fused.get, reverse=True)[: max(k * 3, k)]
        if rerank and len(pool_ids) > 1:
            texts = [self._context(i) for i in pool_ids]
            rr = get_reranker(self.s).score(queries[0], texts)
            ordered = sorted(zip(pool_ids, rr), key=lambda t: t[1], reverse=True)
            final = [(i, sc) for i, sc in ordered[:k]]
        else:
            final = [(i, fused[i]) for i in pool_ids[:k]]
        hits = []
        for i, sc in final:
            text, meta = self._docs[i]
            hits.append(Hit(id=i, text=text, source=meta.get("source", i), score=float(sc), meta=meta))
        return hits
