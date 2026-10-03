"""Run budget: caps tool calls and wall-clock time so an agent can never loop forever."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_tool_calls: int = 14
    max_seconds: float = 300.0
    started: float = field(default_factory=time.time)
    tool_calls: int = 0

    def charge_tool(self) -> None:
        self.tool_calls += 1
        self.check()

    def check(self) -> None:
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(f"tool-call budget exceeded ({self.max_tool_calls})")
        if time.time() - self.started > self.max_seconds:
            raise BudgetExceeded(f"time budget exceeded ({self.max_seconds}s)")

    def snapshot(self) -> dict:
        return {"tool_calls": self.tool_calls, "elapsed_s": round(time.time() - self.started, 1)}
