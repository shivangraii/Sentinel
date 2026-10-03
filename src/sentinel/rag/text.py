"""Small text utilities shared by retrieval components."""
from __future__ import annotations

import re

_STOP = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was "
    "were what when where which who why will with does do did can could should would about into "
    "than then there these those their they them you your we our not no but if so such also".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [t for t in toks if not (drop_stop and t in _STOP)]


def first_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""
