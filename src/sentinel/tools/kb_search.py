"""Knowledge-base search tool: query expansion -> hybrid retrieval (dense + BM25 + RRF) -> rerank."""
from __future__ import annotations

from functools import lru_cache

from sentinel.config import Settings, get_settings
from sentinel.guardrails.injection_filter import sanitize
from sentinel.llm import get_llm
from sentinel.rag.query_rewrite import expand_query
from sentinel.rag.retriever import HybridRetriever
from sentinel.tools.base import EvidenceItem, Tool, ToolResult


@lru_cache(maxsize=8)
def _retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(settings)


def kb_search(query: str, k: int = 4) -> ToolResult:
    s = get_settings()
    queries = expand_query(get_llm(s), query, n=2, hyde=False)
    hits = _retriever(s).search(queries, k=int(k), mode="hybrid", rerank=True)
    if not hits:
        return ToolResult("No relevant passages found in the knowledge base. Try web_search or rephrase.")
    ev, lines = [], []
    for h in hits:
        clean, flagged = sanitize(h.text)
        title = " - ".join(x for x in (h.meta.get("title", h.source), h.meta.get("section", "")) if x)
        ev.append(EvidenceItem(text=clean, source=h.source, kind="kb", title=title, flagged=flagged))
        lines.append(f"- [{h.source}] {clean[:400]}")
    return ToolResult("\n".join(lines), ev)


KB_TOOL = Tool(
    name="kb_search",
    description="Search the internal knowledge base (ingested documents) using hybrid retrieval.",
    args="query: str, k: int = 4",
    fn=kb_search,
)
