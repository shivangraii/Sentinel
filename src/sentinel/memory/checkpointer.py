"""Short-term memory: LangGraph checkpointer (resume runs, HITL interrupts, conversation threads)."""
from __future__ import annotations

import sqlite3

from sentinel.config import Settings, get_settings


def get_checkpointer(settings: Settings | None = None):
    s = settings or get_settings()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        s.storage_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(s.checkpoint_db), check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
