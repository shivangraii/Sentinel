from __future__ import annotations

from sentinel.llm import get_llm
from sentinel.observability.tracing import traced
from sentinel.prompts import load_prompt
from sentinel.state import AgentState, Task
from sentinel.tools.sql_tool import schema_text

MAX_TASKS = 5


def normalize_tasks(raw, query: str) -> list[Task]:
    tasks: list[Task] = []
    items = raw.get("tasks") if isinstance(raw, dict) else raw
    for it in items or []:
        if isinstance(it, str):
            q, typ = it, "research"
        elif isinstance(it, dict):
            q, typ = str(it.get("question", "")).strip(), str(it.get("type", "research")).lower()
        else:
            continue
        if not q:
            continue
        typ = "analysis" if typ.startswith("analy") else "research"
        if any(t["question"].lower() == q.lower() for t in tasks):
            continue
        tasks.append({"id": len(tasks) + 1, "question": q, "type": typ, "status": "pending"})
        if len(tasks) >= MAX_TASKS:
            break
    if not tasks:
        tasks = [{"id": 1, "question": query, "type": "research", "status": "pending"}]
    return tasks


@traced("planner")
def planner(state: AgentState) -> dict:
    query = state["query"]
    user = (
        f"QUESTION: {query}\n\nDATABASE SCHEMA:\n{schema_text()}\n\n"
        f"PRIOR WORK:\n{state.get('memory_context') or 'none'}"
    )
    try:
        raw = get_llm().complete_json("planner", load_prompt("planner"), user)
    except Exception:
        raw = None
    tasks = normalize_tasks(raw, query)
    return {"tasks": tasks, "_detail": f"{len(tasks)} task(s): " + "; ".join(t["type"] for t in tasks)}
