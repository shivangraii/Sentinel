import json

from sentinel.agents import writer as writer_mod
from sentinel.agents.tool_loop import run_tool_loop
from sentinel.agents.writer import evidence_block
from sentinel.fake_llm import fake_respond
from sentinel.guardrails.budget import Budget
from sentinel.runner import resume_research, run_research
from sentinel.tools.base import EvidenceItem, Tool, ToolResult

Q_DOC = "Why does IMU-only dead reckoning drift?"
Q_BOTH = "Why does IMU-only dead reckoning drift, and how many missions are in the missions database?"


def test_full_run_offline_is_cited_and_verified(kb):
    res = run_research(Q_DOC, require_approval=False)
    assert res["status"] == "done"
    assert "## Sources" in res["report"] and "gps_denied_navigation.md" in res["report"]
    assert res["critique"]["score"] >= 0.8 and not res["report"].startswith("> **Verification warning")
    nodes = [e["node"] for e in res["events"]]
    assert nodes[:3] == ["recall", "planner", "approve_plan"] and nodes[-1] == "finalize"
    assert res["report_path"].endswith(".md")


def test_analysis_task_uses_sql_and_evidence_ids_are_unique(kb):
    res = run_research(Q_BOTH, require_approval=False)
    kinds = {e["node"] for e in res["events"]}
    assert {"researcher", "analyst"} <= kinds
    assert "80" in res["report"] and "sql" in res["report"].lower()
    assert [t["status"] for t in res["plan"]] == ["done", "done"]
    ids = [line for line in res["report"].splitlines() if line.startswith("- **[E")]
    assert len(ids) == len(set(ids))


def test_human_in_the_loop_approve_edit_reject(kb):
    first = run_research(Q_DOC, require_approval=True)
    assert first["status"] == "awaiting_approval" and first["plan"]
    done = resume_research(first["thread_id"], approved=True)
    assert done["status"] == "done" and "## Sources" in done["report"]

    paused = run_research(Q_DOC, require_approval=True)
    edited = resume_research(paused["thread_id"], approved=True, tasks=["What does SLAM stand for?"])
    assert edited["status"] == "done"
    assert [t["question"] for t in edited["plan"]] == ["What does SLAM stand for?"]
    assert "Simultaneous Localization and Mapping" in edited["report"]

    rej = run_research(Q_DOC, require_approval=True)
    out = resume_research(rej["thread_id"], approved=False)
    assert out["status"] == "done" and "cancelled" in out["report"].lower()


class ScriptedWriter:
    """Stub LLM for the writer: first draft is bad (fabricated id + uncited claim), then it is fixed."""

    def __init__(self):
        self.calls = 0
        self.saw_feedback = False

    def complete(self, role, system, user):
        self.calls += 1
        if "CRITIC FEEDBACK" in user:
            self.saw_feedback = True
            return fake_respond("writer", system, user)
        bad = (
            "# Report\n\n## Executive summary\n"
            "Dead reckoning drift is caused by cosmic rays hitting the accelerometer [E99].\n"
            "Another long sentence that makes a claim without any citation at all.\n\n"
            "## Findings\nplaceholder\n\n## Limitations\nnone"
        )
        return bad


def test_critic_rejects_bad_draft_and_loop_recovers(kb, monkeypatch):
    stub = ScriptedWriter()
    monkeypatch.setattr(writer_mod, "get_llm", lambda *a, **k: stub)
    res = run_research(Q_DOC, require_approval=False)
    assert stub.calls >= 2 and stub.saw_feedback, "writer should be re-invoked with critic feedback"
    assert res["critique"]["score"] >= 0.8
    nodes = [e["node"] for e in res["events"]]
    assert nodes.count("writer") >= 2 and nodes.count("critic") >= 2
    assert any("FAIL" in e["detail"] for e in res["events"] if e["node"] == "critic")


def test_max_loops_ends_with_warning_banner(kb, monkeypatch):
    class AlwaysBad:
        def complete(self, role, system, user):
            return "# R\n\n## Executive summary\nA fabricated claim without any evidence behind it whatsoever [E42].\n\n## Limitations\nx"

    monkeypatch.setattr(writer_mod, "get_llm", lambda *a, **k: AlwaysBad())
    res = run_research(Q_DOC, require_approval=False)
    assert res["report"].startswith("> **Verification warning")
    assert sum(1 for e in res["events"] if e["node"] == "critic") == 3  # max_loops


def test_flagged_evidence_never_reaches_writer():
    ev = [
        {"id": "E1", "text": "good fact", "source": "a", "kind": "kb", "flagged": False},
        {"id": "E2", "text": "evil", "source": "b", "kind": "web", "flagged": True},
    ]
    state = {"evidence": ev}
    assert [e["id"] for e in writer_mod.usable_evidence(state)] == ["E1"]
    assert "E2" not in evidence_block(writer_mod.usable_evidence(state))


class LoopingLLM:
    """Never finishes on its own: always asks for the same tool."""

    def complete_json(self, role, system, user, retries=1):
        if "Tool budget" in user:
            return {"action": "finish", "answer": "stopped"}
        return {"action": "echo", "args": {"text": "hi"}}


def test_tool_loop_respects_step_and_call_budget():
    calls = []

    def echo(text):
        calls.append(text)
        return ToolResult("echoed", [EvidenceItem(text=f"echo {len(calls)}", source="echo", kind="kb")])

    tools = {"echo": Tool("echo", "echo", "text: str", echo)}
    res = run_tool_loop("researcher", LoopingLLM(), "sys", 1, "task", tools, [], Budget(max_tool_calls=100), max_steps=3)
    assert len(calls) == 3 and res.answer == "stopped"
    assert [e["id"] for e in res.evidence] == ["E1", "E2", "E3"]

    calls.clear()
    run_tool_loop("researcher", LoopingLLM(), "sys", 1, "task", tools, [], Budget(max_tool_calls=1), max_steps=5)
    assert len(calls) == 1  # budget stops further tool use


def test_tool_loop_handles_unknown_tool_and_garbage():
    class Garbage:
        def __init__(self):
            self.n = 0

        def complete_json(self, role, system, user, retries=1):
            self.n += 1
            if self.n == 1:
                return {"action": "nonexistent"}
            if self.n == 2:
                return ["not", "a", "dict"]
            return {"action": "finish", "answer": "unreachable"}

    res = run_tool_loop("researcher", Garbage(), "sys", 1, "t", {}, [], Budget(), max_steps=4)
    assert res.answer == "No conclusion reached within the tool budget."


def test_fake_llm_roles_are_json():
    plan = json.loads(fake_respond("planner", "", "QUESTION: How many missions failed?\n\nDATABASE SCHEMA:\nTABLE missions (...)"))
    assert [t["type"] for t in plan["tasks"]] == ["research", "analysis"]
