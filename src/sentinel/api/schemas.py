from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    require_approval: bool | None = Field(default=None, description="Pause for plan approval (HITL)")
    thread_id: str | None = None


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool = True
    tasks: list[Any] | None = Field(default=None, description="Optional edited plan: strings or {question,type}")


class IngestRequest(BaseModel):
    reset: bool = False
