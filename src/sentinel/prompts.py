from __future__ import annotations

from functools import lru_cache

from sentinel.config import ROOT


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    path = ROOT / "configs" / "prompts" / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def render(name: str, **values: str) -> str:
    """Fill {{placeholders}} (double braces avoid clashing with JSON examples in prompts)."""
    text = load_prompt(name)
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    return text
