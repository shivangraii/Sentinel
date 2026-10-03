"""Tracing: (1) every node appends a timing event to state and a JSONL trace file;
(2) optional Langfuse callback handler when LANGFUSE_* keys are set (best effort)."""
from __future__ import annotations

import functools
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime

from sentinel.config import get_settings


def _write(record: dict) -> None:
    try:
        d = get_settings().traces_dir
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"trace-{datetime.now(UTC):%Y%m%d}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def traced(name: str) -> Callable:
    """Decorator for graph nodes: times the node, logs it, and adds an `events` entry."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state):
            t0 = time.time()
            try:
                out = fn(state) or {}
            except Exception as e:
                if type(e).__name__ not in ("GraphInterrupt", "NodeInterrupt", "GraphBubbleUp"):
                    _write({"run_id": state.get("run_id"), "node": name, "error": f"{type(e).__name__}: {e}"})
                raise
            detail = out.pop("_detail", "")
            ev = {"node": name, "seconds": round(time.time() - t0, 2), "detail": detail}
            _write({"run_id": state.get("run_id"), **ev, "ts": datetime.now(UTC).isoformat()})
            out["events"] = list(out.get("events", [])) + [ev]
            return out

        return wrapper

    return deco


def get_callbacks() -> list:
    """Langfuse handler if configured; empty list otherwise. Handles v2 and v3 import paths."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    for path in ("langfuse.langchain", "langfuse.callback"):
        try:
            mod = __import__(path, fromlist=["CallbackHandler"])
            return [mod.CallbackHandler()]
        except Exception:
            continue
    return []
