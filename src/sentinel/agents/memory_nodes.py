"""Graph nodes that touch long-term memory: recall (start) and finalize (end)."""
from __future__ import annotations

import re
import time

from sentinel.agents.critic import cited_ids
from sentinel.agents.workers import _budget, reset_budget
from sentinel.config import get_settings
from sentinel.llm import usage
from sentinel.memory import episodic
from sentinel.observability.tracing import traced
from sentinel.state import AgentState


@traced("recall")
def recall(state: AgentState) -> dict:
    try:
        eps = episodic.recall(state["query"], k=2)
    except Exception:
        eps = []
    ctx = "\n".join(f"- Earlier question: {e['query']}\n  Summary: {e['summary'][:300]}" for e in eps)
    return {"memory_context": ctx, "_detail": f"{len(eps)} related past report(s)"}


@traced("approve_plan")
def approve_plan(state: AgentState) -> dict:
    """Human-in-the-loop gate. With require_approval the graph pauses here (LangGraph interrupt)
    until the API/CLI resumes it with {"approved": bool, "tasks": [...optional edits...]}."""
    if not state.get("require_approval"):
        return {"approved": True, "_detail": "auto-approved"}
    from langgraph.types import interrupt

    from sentinel.agents.planner import normalize_tasks

    decision = interrupt({"type": "plan_approval", "query": state["query"], "tasks": state["tasks"]})
    if not isinstance(decision, dict) or not decision.get("approved", False):
        return {"approved": False, "_detail": "plan rejected by user"}
    out: dict = {"approved": True, "_detail": "plan approved"}
    if decision.get("tasks"):
        out["tasks"] = normalize_tasks(decision["tasks"], state["query"])
        out["_detail"] = f"plan approved with edits ({len(out['tasks'])} tasks)"
    return out


def route_after_approval(state: AgentState) -> str:
    return "supervisor" if state.get("approved") else "rejected"


@traced("rejected")
def rejected(state: AgentState) -> dict:
    return {"report": "# Research cancelled\n\nThe proposed plan was rejected, so no research was run.", "_detail": "cancelled"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "report"


@traced("finalize")
def finalize(state: AgentState) -> dict:
    s = get_settings()
    draft = state.get("draft", "").strip()
    crit = state.get("critique", {})
    valid = {e["id"]: e for e in state.get("evidence", []) if not e.get("flagged")}
    used = [i for i in dict.fromkeys(cited_ids(draft)) if i in valid]

    parts = [draft]
    if not state.get("passed") and crit:
        n = len(crit.get("unsupported", []))
        parts.insert(
            0,
            f"> **Verification warning:** the critic could not fully verify this report "
            f"({n} claim(s) flagged after {state.get('loops', 0)} attempt(s)). Treat flagged statements with caution.\n",
        )
    if used:
        parts.append("\n## Sources")
        for i in used:
            e = valid[i]
            label = e["title"] or e["source"]
            parts.append(f"- **[{i}]** {label} — `{e['source'][:120]}` ({e['kind']})")
    snap = usage.snapshot()
    budget = _budget(state).snapshot()
    parts.append(
        "\n## Verification\n"
        f"- Claim support score: **{crit.get('score', 0)}** "
        f"({crit.get('supported', 0)}/{crit.get('claims_checked', 0)} claims verified)\n"
        f"- Critic rounds used: {state.get('loops', 0)} / {s.max_loops}\n"
        f"- Tool calls: {budget['tool_calls']} | LLM calls: {snap['llm_calls']} "
        f"(~{snap['approx_input_tokens']} in / {snap['approx_output_tokens']} out tokens)"
    )
    if s.offline:
        parts.append("- Mode: **offline demo** (extractive report; no real LLM was used)")
    report = "\n".join(parts).strip() + "\n"

    s.reports_dir.mkdir(parents=True, exist_ok=True)
    path = s.reports_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(state['query'])}.md"
    path.write_text(report, encoding="utf-8")
    try:
        summary = re.sub(r"\[E\d+\]", "", draft)[:700]
        episodic.save_episode(state.get("run_id") or path.stem, state["query"], summary, str(path), s)
    except Exception:
        pass
    reset_budget(state.get("run_id") or "default")
    return {"report": report, "report_path": str(path), "usage": {**snap, **budget}, "_detail": f"saved {path.name}"}
