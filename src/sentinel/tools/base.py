"""Tool abstraction. Every tool returns a ToolResult: a text observation for the LLM plus
structured evidence items that the writer can cite."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    """One citable piece of evidence. `id` (E1, E2, ...) is assigned by the agent loop."""

    text: str
    source: str  # file name, URL, arXiv id, or "sql:<query>" / "python"
    kind: str  # kb | web | arxiv | sql | python | calc
    title: str = ""
    flagged: bool = False  # prompt-injection heuristic fired on this content


@dataclass
class ToolResult:
    observation: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    args: str  # human-readable arg spec shown to the LLM
    fn: Callable[..., ToolResult]

    def spec(self) -> str:
        return f"- {self.name}({self.args}): {self.description}"

    def run(self, **kwargs) -> ToolResult:
        try:
            return self.fn(**kwargs)
        except TypeError as e:
            return ToolResult(f"Bad arguments for {self.name}: {e}. Expected: {self.args}", error=True)
        except Exception as e:  # tools must never crash the graph
            return ToolResult(f"{self.name} failed: {type(e).__name__}: {e}", error=True)
