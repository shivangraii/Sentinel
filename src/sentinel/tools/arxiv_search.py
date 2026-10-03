"""arXiv search via the public Atom API (stdlib only)."""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from sentinel.guardrails.injection_filter import sanitize
from sentinel.tools.base import EvidenceItem, Tool, ToolResult

_NS = {"a": "http://www.w3.org/2005/Atom"}


def _text(entry: ET.Element, tag: str) -> str:
    return (entry.findtext(f"a:{tag}", default="", namespaces=_NS) or "").strip()


def parse_arxiv(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", _NS):
        out.append(
            {
                "title": " ".join(_text(e, "title").split()),
                "summary": " ".join(_text(e, "summary").split()),
                "url": _text(e, "id"),
                "published": _text(e, "published")[:10],
                "authors": [(a.findtext("a:name", default="", namespaces=_NS) or "").strip() for a in e.findall("a:author", _NS)],
            }
        )
    return out


def arxiv_search(query: str, max_results: int = 3) -> ToolResult:
    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": int(max_results), "sortBy": "relevance"}
    )
    req = urllib.request.Request(f"http://export.arxiv.org/api/query?{params}", headers={"User-Agent": "sentinel/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (fixed https/http host)
        papers = parse_arxiv(resp.read().decode("utf-8"))
    if not papers:
        return ToolResult("No arXiv results.")
    ev, lines = [], []
    for p in papers:
        clean, flagged = sanitize(p["summary"])
        authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        text = f"{p['title']} ({authors}, {p['published']}). {clean}"
        ev.append(EvidenceItem(text=text, source=p["url"], kind="arxiv", title=p["title"], flagged=flagged))
        lines.append(f"- {p['title']} [{p['url']}]: {clean[:300]}")
    return ToolResult("\n".join(lines), ev)


ARXIV_TOOL = Tool(
    name="arxiv_search",
    description="Search arXiv for research papers; returns titles, authors and abstracts.",
    args="query: str, max_results: int = 3",
    fn=arxiv_search,
)
