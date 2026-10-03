"""Heuristic prompt-injection filter for *retrieved* content (web pages, documents, tool output).

Retrieved text is untrusted data. We (1) strip sentences that look like instructions aimed at the
model and (2) flag the evidence so the critic/UI can see it. This is a defence-in-depth heuristic,
not a guarantee; the agents additionally treat tool output as data in their prompts.
"""
from __future__ import annotations

import re

_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|messages?)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"forget (everything|all|your) (above|previous|instructions)",
    r"you are now\b",
    r"\bnew (system )?instructions?:",
    r"\bsystem prompt\b",
    r"reveal (your|the) (system|hidden|secret) (prompt|instructions)",
    r"act as (an? )?(?!analyst)",
    r"do not (tell|inform) the user",
    r"\bjailbreak\b",
    r"<\s*/?\s*(system|assistant|tool)\s*>",
    r"(send|post|upload|exfiltrate) .{0,40}(api[_ -]?key|password|secret|token)",
]
_RX = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
REMOVED = "[removed: possible prompt injection]"


def sanitize(text: str) -> tuple[str, bool]:
    """Returns (clean_text, flagged). Offending sentences are replaced with a marker."""
    flagged = False
    parts = _SENT.split(text)
    cleaned = []
    for part in parts:
        if any(rx.search(part) for rx in _RX):
            flagged = True
            cleaned.append(REMOVED)
        else:
            cleaned.append(part)
    return (" ".join(p for p in cleaned if p), flagged)
