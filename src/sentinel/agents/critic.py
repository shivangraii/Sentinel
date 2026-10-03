"""Critic: verifies every claim in the draft against the evidence it cites.

Deterministic checks (citation ids exist, sentences are cited) + LLM entailment judging.
On failure it produces feedback and optional follow-up research tasks, closing the reflection loop.
"""
from __future__ import annotations

import json
import re

from sentinel.config import get_settings
from sentinel.llm import get_llm
from sentinel.observability.tracing import traced
from sentinel.prompts import load_prompt
from sentinel.state import AgentState

_BRACKET = re.compile(r"\[([^\]]*)\]")
_ID = re.compile(r"\bE\d+\b")
_SKIP_SECTIONS = re.compile(r"limitation|uncertaint|caveat|method|scope|note", re.IGNORECASE)
MIN_WORDS = 6
MAX_JUDGED = 30


def cited_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in _BRACKET.finditer(text):
        ids += _ID.findall(m.group(1))
    return ids


def extract_claims(draft: str) -> list[dict]:
    """Sentences from Summary/Findings sections with the evidence ids they cite."""
    claims: list[dict] = []
    section = ""
    for raw in draft.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        if _SKIP_SECTIONS.search(section):
            continue
        line = re.sub(r"^([-*\u2022]|\d+[.)])\s+", "", line)
        parts = re.split(r"(?<=[.!?])\s+", line)
        merged: list[str] = []
        for p in parts:
            if merged and not re.sub(r"\[[^\]]*\]|[\s.,;]", "", p):  # citation-only fragment
                merged[-1] += " " + p
            else:
                merged.append(p)
        for sent in merged:
            clean = _BRACKET.sub("", sent).strip()
            if len(clean.split()) < MIN_WORDS or clean.endswith("?"):
                continue
            claims.append({"text": sent.strip(), "ids": cited_ids(sent)})
    return claims


@traced("critic")
def critic(state: AgentState) -> dict:
    s = get_settings()
    llm = get_llm(s)
    draft = state.get("draft", "")
    valid = {e["id"]: e for e in state.get("evidence", []) if not e.get("flagged")}
    claims = extract_claims(draft)

    invalid: list[str] = []
    uncited = 0
    judged: list[tuple[int, dict]] = []
    unsupported: list[dict] = []
    for c in claims:
        if not c["ids"]:
            uncited += 1
            unsupported.append({"claim": c["text"], "reason": "no citation"})
            continue
        bad = [i for i in c["ids"] if i not in valid]
        if bad:
            invalid += bad
            unsupported.append({"claim": c["text"], "reason": f"cites unknown/blocked evidence {', '.join(bad)}"})
            continue
        judged.append((len(judged), c))

    supported = 0
    to_judge = judged[:MAX_JUDGED]
    if to_judge:
        payload = [
            {"i": i, "claim": c["text"], "evidence": [valid[x]["text"][:600] for x in dict.fromkeys(c["ids"])]}
            for i, c in to_judge
        ]
        try:
            verdicts = llm.complete_json("critic_judge", load_prompt("critic"), json.dumps(payload, ensure_ascii=False))
            vmap = {int(v["i"]): v for v in verdicts.get("verdicts", []) if isinstance(v, dict) and "i" in v}
        except Exception:
            vmap = {}
        for i, c in to_judge:
            v = vmap.get(i)
            if v is None:  # judge unavailable for this claim: do not silently pass it
                unsupported.append({"claim": c["text"], "reason": "could not be verified"})
            elif bool(v.get("supported")):
                supported += 1
            else:
                unsupported.append({"claim": c["text"], "reason": str(v.get("reason", "not supported"))[:200]})

    structural = sum(1 for u in unsupported if u["reason"] == "no citation" or u["reason"].startswith("cites unknown"))
    total_checked = len(to_judge) + structural
    score = (supported / total_checked) if total_checked else 0.0
    passed = bool(total_checked) and score >= s.critic_threshold and not invalid

    critique = {
        "score": round(score, 3), "claims_found": len(claims), "claims_checked": total_checked,
        "supported": supported, "unsupported": unsupported[:12], "invalid_citations": sorted(set(invalid)),
        "uncited": uncited,
    }
    out: dict = {"critique": critique, "passed": passed, "loops": state.get("loops", 0)}

    if passed:
        out["feedback"] = ""
        out["_detail"] = f"PASS score={critique['score']} ({supported}/{total_checked} claims supported)"
        return out

    tasks = [dict(t) for t in state.get("tasks", [])]
    feedback = "Fix these unsupported claims (remove, reword to match evidence, or re-source):\n" + "\n".join(
        f"- {u['claim']} ({u['reason']})" for u in unsupported[:8]
    )
    follow: list[str] = []
    try:
        fu = llm.complete_json(
            "critic_followup", load_prompt("critic_followup"),
            json.dumps({"query": state["query"], "unsupported": unsupported[:8]}, ensure_ascii=False),
        )
        if isinstance(fu, dict):
            feedback = str(fu.get("feedback") or feedback)
            follow = [str(q) for q in fu.get("followup_questions", []) if str(q).strip()][:2]
    except Exception:
        pass
    existing = {t["question"].lower() for t in tasks}
    for q in follow:
        if q.lower() not in existing:
            tasks.append({"id": max((t["id"] for t in tasks), default=0) + 1, "question": q, "type": "research", "status": "pending"})
    out.update(feedback=feedback, tasks=tasks, loops=state.get("loops", 0) + 1)
    out["_detail"] = f"FAIL score={critique['score']} ({supported}/{total_checked}); +{len(follow)} follow-up task(s)"
    return out


def route_after_critic(state: AgentState) -> str:
    if state.get("passed") or state.get("loops", 0) >= get_settings().max_loops:
        return "finalize"
    return "supervisor"
