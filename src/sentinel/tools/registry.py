from __future__ import annotations

from sentinel.tools.arxiv_search import ARXIV_TOOL
from sentinel.tools.base import Tool
from sentinel.tools.calculator import CALCULATOR
from sentinel.tools.kb_search import KB_TOOL
from sentinel.tools.python_sandbox import PYTHON_TOOL
from sentinel.tools.sql_tool import SQL_TOOL
from sentinel.tools.web_search import WEB_TOOL

ALL_TOOLS: dict[str, Tool] = {
    t.name: t for t in (KB_TOOL, WEB_TOOL, ARXIV_TOOL, SQL_TOOL, PYTHON_TOOL, CALCULATOR)
}

RESEARCH_TOOLS = ["kb_search", "web_search", "arxiv_search"]
ANALYST_TOOLS = ["sql_query", "python", "calculator", "kb_search"]


def get_tools(names: list[str]) -> dict[str, Tool]:
    return {n: ALL_TOOLS[n] for n in names}
