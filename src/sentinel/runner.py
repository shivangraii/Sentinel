"""High-level helpers used by the CLI and API: run, resume (HITL) and stream a research job."""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from sentinel.config import get_settings
from sentinel.graph import build_graph
from sentinel.llm import usage
from sentinel.observability.tracing import get_callbacks

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def set_graph(g) -> None:
    global _graph
    _graph = g


def _config(thread_id: str) -> dict:
    cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    cbs = get_callbacks()
    if cbs:
        cfg["callbacks"] = cbs
    return cfg


def _result(thread_id: str) -> dict:
    snap = get_graph().get_state(_config(thread_id))
    vals = snap.values or {}
    awaiting = bool(snap.next)
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval" if awaiting else "done",
        "plan": vals.get("tasks", []),
        "report": vals.get("report", ""),
        "report_path": vals.get("report_path", ""),
        "critique": vals.get("critique", {}),
        "events": vals.get("events", []),
        "usage": vals.get("usage", {}),
    }


def _initial(query: str, thread_id: str, require_approval: bool | None) -> dict:
    s = get_settings()
    return {
        "run_id": thread_id,
        "query": query.strip(),
        "require_approval": s.require_approval if require_approval is None else bool(require_approval),
        "loops": 0,
        "passed": False,
    }


def run_research(query: str, thread_id: str | None = None, require_approval: bool | None = None) -> dict:
    thread_id = thread_id or uuid.uuid4().hex[:12]
    usage.reset()
    get_graph().invoke(_initial(query, thread_id, require_approval), _config(thread_id))
    return _result(thread_id)


def resume_research(thread_id: str, approved: bool = True, tasks: list | None = None) -> dict:
    from langgraph.types import Command

    decision: dict[str, Any] = {"approved": approved}
    if tasks:
        decision["tasks"] = tasks
    get_graph().invoke(Command(resume=decision), _config(thread_id))
    return _result(thread_id)


def stream_research(
    query: str | None = None,
    thread_id: str | None = None,
    require_approval: bool | None = None,
    resume: dict | None = None,
) -> Iterator[dict]:
    """Yield progress events node-by-node, then a final {'type': 'result', ...} event."""
    from langgraph.types import Command

    thread_id = thread_id or uuid.uuid4().hex[:12]
    if resume is not None:
        inp: Any = Command(resume=resume)
    else:
        usage.reset()
        inp = _initial(query or "", thread_id, require_approval)
    yield {"type": "start", "thread_id": thread_id}
    for chunk in get_graph().stream(inp, _config(thread_id), stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                yield {"type": "interrupt", "node": "approve_plan"}
                continue
            for ev in (update or {}).get("events", []):
                yield {"type": "node", **ev}
    yield {"type": "result", **_result(thread_id)}
