"""Researcher and Analyst nodes: each handles the first pending task of its type."""
from __future__ import annotations

from sentinel.agents.tool_loop import run_tool_loop
from sentinel.config import get_settings
from sentinel.guardrails.budget import Budget
from sentinel.llm import get_llm
from sentinel.memory.semantic import remember_evidence
from sentinel.observability.tracing import traced
from sentinel.prompts import render
from sentinel.state import AgentState
from sentinel.tools.registry import ANALYST_TOOLS, RESEARCH_TOOLS, get_tools
from sentinel.tools.sql_tool import schema_text

_budgets: dict[str, Budget] = {}


def _budget(state: AgentState) -> Budget:
    """One budget per run_id, shared by all workers in that run."""
    s = get_settings()
    rid = state.get("run_id") or "default"
    if rid not in _budgets:
        if len(_budgets) > 200:
            _budgets.clear()
        _budgets[rid] = Budget(s.max_tool_calls_per_run, s.max_runtime_s)
    return _budgets[rid]


def reset_budget(run_id: str) -> None:
    _budgets.pop(run_id, None)


def _work(state: AgentState, role: str, task_type: str, tool_names: list[str], prompt: str) -> dict:
    s = get_settings()
    tasks = [dict(t) for t in state.get("tasks", [])]
    task = next((t for t in tasks if t["status"] == "pending" and t["type"] == task_type), None)
    if task is None:
        return {"_detail": "no pending task"}
    tools = get_tools(tool_names)
    system = render(prompt, tools="\n".join(t.spec() for t in tools.values()), schema=schema_text())
    res = run_tool_loop(
        role=role, llm=get_llm(s), system=system, task_id=task["id"], task_question=task["question"],
        tools=tools, existing_evidence=state.get("evidence", []), budget=_budget(state),
        max_steps=s.max_tool_steps,
    )
    task["status"] = "done"
    try:
        remember_evidence(res.evidence, s)  # semantic memory write-back (web/arXiv only)
    except Exception:
        pass
    finding = {
        "task_id": task["id"], "question": task["question"], "answer": res.answer,
        "evidence_ids": [e["id"] for e in res.evidence], "steps": res.steps,
    }
    used = " -> ".join(st["action"] for st in res.steps)
    return {
        "tasks": tasks, "findings": [finding], "evidence": res.evidence,
        "_detail": f"task {task['id']}: {used}; +{len(res.evidence)} evidence",
    }


@traced("researcher")
def researcher(state: AgentState) -> dict:
    return _work(state, "researcher", "research", RESEARCH_TOOLS, "researcher")


@traced("analyst")
def analyst(state: AgentState) -> dict:
    return _work(state, "analyst", "analysis", ANALYST_TOOLS, "analyst")
