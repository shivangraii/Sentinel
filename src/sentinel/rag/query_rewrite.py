"""Query expansion: LLM-written alternative queries (multi-query) and an optional HyDE passage."""
from __future__ import annotations

from sentinel.llm import LLMClient

_SYSTEM = (
    "You rewrite search queries for a retrieval system. Return ONLY a JSON array of strings. "
    "Each string is an alternative phrasing of the user's question that uses different keywords or "
    "synonyms. Do not answer the question."
)
_HYDE_SYSTEM = (
    "Write a short factual paragraph (max 60 words) that would plausibly answer the question. "
    "It is used only for retrieval. Return plain text."
)


def expand_query(llm: LLMClient, query: str, n: int = 2, hyde: bool = False) -> list[str]:
    """Returns [original, *rewrites, (hyde passage)]. Never raises: falls back to [original]."""
    out = [query]
    try:
        alts = llm.complete_json("rewriter", _SYSTEM, f"Question: {query}\nReturn {n} alternatives.", retries=0)
        if isinstance(alts, list):
            out += [str(a) for a in alts[:n] if isinstance(a, str) and a.strip() and a != query]
    except Exception:
        pass
    if hyde:
        try:
            out.append(llm.complete("hyde", _HYDE_SYSTEM, query).strip())
        except Exception:
            pass
    return out
