"""Supervisor: deterministic router. Control flow is code (reliable, testable, no wasted LLM calls);
judgement lives inside the LLM-driven nodes (planner, researcher, analyst, writer, critic)."""
from __future__ import annotations

from sentinel.observability.tracing import traced
from sentinel.state import AgentState


@traced("supervisor")
def supervisor(state: AgentState) -> dict:
    pending = [t for t in state.get("tasks", []) if t["status"] == "pending"]
    if pending:
        nxt = "analyst" if pending[0]["type"] == "analysis" else "researcher"
        return {"next": nxt, "_detail": f"route -> {nxt} (task {pending[0]['id']}, {len(pending)} pending)"}
    return {"next": "writer", "_detail": "all tasks done -> writer"}


def route_after_supervisor(state: AgentState) -> str:
    return state.get("next", "writer")
