from __future__ import annotations

from sentinel.llm import get_llm
from sentinel.observability.tracing import traced
from sentinel.prompts import load_prompt
from sentinel.state import AgentState

MAX_EVIDENCE_CHARS = 14000
PER_EVIDENCE = 900


def usable_evidence(state: AgentState) -> list[dict]:
    """Evidence the writer may cite: everything not flagged by the injection filter."""
    return [e for e in state.get("evidence", []) if not e.get("flagged")]


def evidence_block(evidence: list[dict]) -> str:
    lines, used = [], 0
    for e in evidence:
        line = f"[{e['id']}] ({e['kind']}: {e['source'][:80]}) {e['text'][:PER_EVIDENCE]}"
        if used + len(line) > MAX_EVIDENCE_CHARS:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) or "(none)"


@traced("writer")
def writer(state: AgentState) -> dict:
    findings = "\n".join(
        f"- (task {f['task_id']}) {f['question']}\n  ANSWER: {f['answer']}" for f in state.get("findings", [])
    )
    ev = usable_evidence(state)
    user = (
        f"QUESTION: {state['query']}\n\nFINDINGS:\n{findings or '(none)'}\n\n"
        f"EVIDENCE:\n{evidence_block(ev)}\n"
    )
    if state.get("feedback"):
        user += f"\nCRITIC FEEDBACK:\n{state['feedback']}\n\nPREVIOUS DRAFT:\n{state.get('draft', '')}\n"
    draft = get_llm().complete("writer", load_prompt("writer"), user).strip()
    return {"draft": draft, "_detail": f"{len(draft.split())} words, {len(ev)} evidence items available"}
