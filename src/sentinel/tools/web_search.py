"""Web search with pluggable providers: duckduckgo (ddgs package), tavily (API key) or none."""
from __future__ import annotations

import os

import requests

from sentinel.config import get_settings
from sentinel.guardrails.injection_filter import sanitize
from sentinel.tools.base import EvidenceItem, Tool, ToolResult


def _ddg(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError as e:
            raise RuntimeError('web search needs: pip install "sentinel[search]"') from e
    results = DDGS().text(query, max_results=n)
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in results]


def _tavily(query: str, n: int) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": n, "search_depth": "basic"},
        timeout=20,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", "")}
        for x in r.json().get("results", [])
    ]


def web_search(query: str, max_results: int | None = None) -> ToolResult:
    s = get_settings()
    n = int(max_results or s.search_max_results)
    if s.search_provider == "none":
        return ToolResult("Web search is disabled (SENTINEL_SEARCH=none). Use kb_search instead.", error=True)
    results = _tavily(query, n) if s.search_provider == "tavily" else _ddg(query, n)
    if not results:
        return ToolResult("No web results.")
    ev, lines = [], []
    for r in results:
        clean, flagged = sanitize(r["content"])
        ev.append(EvidenceItem(text=f"{r['title']}. {clean}", source=r["url"], kind="web", title=r["title"], flagged=flagged))
        lines.append(f"- {r['title']} [{r['url']}]: {clean[:300]}")
    return ToolResult("\n".join(lines), ev)


WEB_TOOL = Tool(
    name="web_search",
    description="Search the public web for current information; returns titles, URLs and snippets.",
    args="query: str, max_results: int = 5",
    fn=web_search,
)
