"""Deterministic offline stand-in for an LLM (SENTINEL_OFFLINE=1 / provider "fake").

It lets the whole pipeline (graph, tools, retrieval, critic loop, API, tests, CI) run with no API
key and no network. Output is *extractive* (copies retrieved sentences with their citations); it is
a harness for plumbing and tests, not a substitute for a real model's reasoning.
"""
from __future__ import annotations

import json
import re

from sentinel.rag.text import tokenize

_ANALYSIS_HINT = re.compile(
    r"\b(average|mean|how many|count|total|percent|percentage|rate|success|compare|battery|duration|missions?)\b",
    re.IGNORECASE,
)
_EV = re.compile(
    r"\[(E\d+)\] \((\w+): [^)]*\)(?: \(flagged[^)]*\))? (.*?)(?=\n\[E\d+\] \(|\n\nOBSERVATION|\Z)",
    re.DOTALL,
)


def _line(user: str, prefix: str) -> str:
    m = re.search(rf"^{prefix}:\s*(.+)$", user, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _first_sentence(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return s[:limit].rstrip()


def _tool_step(role: str, user: str) -> str:
    task = _line(user, "TASK")
    if "OBSERVATION" not in user:
        if role == "analyst_step":
            sql = (
                "SELECT COUNT(*) AS missions, ROUND(AVG(duration_min),1) AS avg_duration_min, "
                "ROUND(AVG(battery_used_pct),1) AS avg_battery_pct, "
                "ROUND(100.0*SUM(success)/COUNT(*),1) AS success_rate_pct FROM missions"
            )
            return json.dumps({"action": "sql_query", "args": {"query": sql}})
        return json.dumps({"action": "kb_search", "args": {"query": task}})
    answers = []
    for m in _EV.finditer(user):
        eid, kind, text = m.group(1), m.group(2), m.group(3).strip()
        if kind == "sql":
            lines = text.split("Result:\n", 1)[-1].strip().splitlines()
            if len(lines) >= 2:
                answers.append(f"The missions database query returned {lines[0]} = {lines[1]} [{eid}].")
        else:
            answers.append(f"{_first_sentence(text).rstrip('.')} [{eid}].")
        if len(answers) >= 2:
            break
    answer = " ".join(answers) or "No relevant passages were found for this task."
    return json.dumps({"action": "finish", "answer": answer})


def _plan(user: str) -> str:
    q = _line(user, "QUESTION")
    tasks = [{"question": q, "type": "research"}]
    if _ANALYSIS_HINT.search(q) and "TABLE missions" in user:
        tasks.append({"question": f"Quantify from the missions database: {q}", "type": "analysis"})
    return json.dumps({"tasks": tasks})


def _write(user: str) -> str:
    q = _line(user, "QUESTION")
    findings = re.findall(r"- \(task \d+\) (.*)\n  ANSWER: (.*)", user)
    summary = " ".join(re.match(r"(.+?\.)(\s|$)", a).group(1) if re.match(r"(.+?\.)(\s|$)", a) else a for _, a in findings[:3])
    out = [f"# Research report: {q}", "", "## Executive summary", summary or "No findings were produced.", "", "## Findings"]
    for question, answer in findings:
        out += [f"### {question}", answer, ""]
    out += [
        "## Limitations",
        "Offline demo mode: this report is extractive (retrieved sentences copied with citations) "
        "and was not written by a language model.",
    ]
    return "\n".join(out)


def _judge(user: str) -> str:
    try:
        items = json.loads(user)
    except json.JSONDecodeError:
        return json.dumps({"verdicts": []})
    verdicts = []
    for it in items:
        claim = set(tokenize(re.sub(r"\[[^\]]*\]", "", it["claim"])))
        ev = set(tokenize(" ".join(it["evidence"])))
        overlap = len(claim & ev) / max(1, len(claim))
        verdicts.append({"i": it["i"], "supported": overlap >= 0.3, "reason": f"lexical overlap {overlap:.2f}"})
    return json.dumps({"verdicts": verdicts})


def _naive(user: str) -> str:
    passages = re.findall(r"^\[(\d+)\] (.+)$", user, re.MULTILINE)
    out = [_first_sentence(text) for _, text in passages[:2]]
    return " ".join(out) or "The context does not contain the answer."


def fake_respond(role: str, system: str, user: str) -> str:
    if role == "naive_rag":
        return _naive(user)
    if role == "rewriter":
        return "[]"
    if role == "hyde":
        return ""
    if role == "planner":
        return _plan(user)
    if role in ("researcher_step", "analyst_step"):
        return _tool_step(role, user)
    if role == "writer":
        return _write(user)
    if role == "critic_judge":
        return _judge(user)
    if role == "critic_followup":
        return json.dumps({"feedback": "Remove or re-source the unsupported claims.", "followup_questions": []})
    return ""
