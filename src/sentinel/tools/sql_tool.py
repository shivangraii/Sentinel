"""Read-only Text-to-SQL execution against a SQLite database.

Defence in depth: (1) statement must start with SELECT/WITH and be a single statement,
(2) forbidden keywords are rejected, (3) the DB is opened read-only (mode=ro) and
(4) PRAGMA query_only=ON, (5) results are row-capped.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from sentinel.config import get_settings
from sentinel.tools.base import EvidenceItem, Tool, ToolResult

MAX_ROWS = 50
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate|grant)\b",
    re.IGNORECASE,
)


class UnsafeSQL(ValueError):
    pass


def validate_sql(query: str) -> str:
    q = query.strip().rstrip(";").strip()
    if not q:
        raise UnsafeSQL("empty query")
    if ";" in q:
        raise UnsafeSQL("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise UnsafeSQL("only SELECT queries are allowed")
    # strip string literals before keyword scan so 'delete' inside a string is fine
    stripped = re.sub(r"'(?:[^']|'')*'", "''", q)
    m = _FORBIDDEN.search(stripped)
    if m:
        raise UnsafeSQL(f"forbidden keyword: {m.group(0).upper()}")
    return q


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def schema_text(db_path: Path | None = None) -> str:
    """Human-readable schema (tables, columns, a sample row) for the analyst prompt."""
    path = Path(db_path or get_settings().db_path)
    if not path.exists():
        return "(no database found)"
    conn = _connect(path)
    try:
        out = []
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
            col_s = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            out.append(f"TABLE {t} ({col_s})  -- {n} rows")
        return "\n".join(out)
    finally:
        conn.close()


def run_sql(query: str, db_path: Path | None = None) -> ToolResult:
    path = Path(db_path or get_settings().db_path)
    if not path.exists():
        return ToolResult(f"Database not found at {path}", error=True)
    try:
        q = validate_sql(query)
    except UnsafeSQL as e:
        return ToolResult(f"Rejected unsafe SQL: {e}", error=True)
    conn = _connect(path)
    try:
        cur = conn.execute(q)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return ToolResult(f"SQL error: {e}", error=True)
    finally:
        conn.close()
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v) for v in r) for r in rows]
    body = "\n".join(lines) + ("\n(truncated to 50 rows)" if truncated else "")
    ev = EvidenceItem(text=f"Query: {q}\nResult:\n{body}", source=f"sql: {q}", kind="sql", title="SQL result")
    return ToolResult(f"{len(rows)} row(s)\n{body}", [ev])


SQL_TOOL = Tool(
    name="sql_query",
    description="Run a read-only SQLite SELECT query on the missions database and return rows.",
    args="query: str",
    fn=run_sql,
)
