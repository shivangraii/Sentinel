"""Shared graph state. List fields with reducers are append-only; others are replaced by nodes."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Task(TypedDict):
    id: int
    question: str
    type: str  # research | analysis
    status: str  # pending | done


class AgentState(TypedDict, total=False):
    run_id: str
    query: str
    require_approval: bool
    memory_context: str
    tasks: list[Task]
    approved: bool
    evidence: Annotated[list[dict[str, Any]], operator.add]
    findings: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    next: str
    draft: str
    feedback: str
    critique: dict[str, Any]
    passed: bool
    loops: int
    report: str
    report_path: str
    usage: dict[str, Any]
