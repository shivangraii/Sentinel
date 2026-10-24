"""Sentinel on Streamlit Community Cloud: runs the research graph IN-PROCESS (no separate API server).

Main file path on share.streamlit.io:  deploy/streamlit/streamlit_app.py
Dependencies come from deploy/streamlit/requirements.txt (the entrypoint's folder wins over the repo root).

Light profile for the free tier's small RAM: hash embeddings + lexical reranker (no torch). The agent graph, SQL tool,
critic and human-approval flow are unchanged; only retrieval quality is lower than with bge models.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Community Cloud ships SQLite 3.34 but ChromaDB needs >= 3.35: swap in the bundled modern build when available.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st  # noqa: E402

# 1. Streamlit secrets -> environment variables (Sentinel reads its config from the environment). Real env vars win.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str | int | float | bool) and _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:  # no secrets configured (e.g. local run without .streamlit/secrets.toml)
    pass

# 2. Light profile + writable scratch dirs (the repo checkout may be read-only; /tmp is always writable).
_tmp = Path(tempfile.gettempdir()) / "sentinel"
for _k, _v in {
    "SENTINEL_EMBEDDINGS": "hash",
    "SENTINEL_RERANKER": "lexical",
    "SENTINEL_STORAGE_DIR": str(_tmp / "storage"),
    "SENTINEL_REPORTS_DIR": str(_tmp / "reports"),
    "SENTINEL_TRACES_DIR": str(_tmp / "traces"),
}.items():
    os.environ.setdefault(_k, _v)

# 3. No API key for the chosen provider -> keep real retrieval + critic, but use the deterministic stand-in LLM.
_KEY_FOR = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_provider = os.getenv("SENTINEL_LLM_PROVIDER", "gemini")
if _provider in _KEY_FOR and not os.getenv(_KEY_FOR[_provider]) and not os.getenv("SENTINEL_OFFLINE"):
    os.environ["SENTINEL_LLM_PROVIDER"] = "fake"

from sentinel.config import get_settings  # noqa: E402
from sentinel.guardrails.rate_limit import RateLimiter  # noqa: E402
from sentinel.memory import episodic  # noqa: E402
from sentinel.rag.ingest import ensure_kb  # noqa: E402
from sentinel.runner import stream_research  # noqa: E402

EXAMPLES = [
    "Why does IMU-only dead reckoning drift?",
    "Compare CBBA and leader-follower control, and how do their mission success rates differ?",
    "Why does IMU-only dead reckoning drift, and how many missions are in the missions database?",
]


@st.cache_resource(show_spinner="Building the knowledge base…")
def bootstrap() -> dict:
    """Once per server process: knowledge base + a lock (runs are serialised, so memory stays bounded and the
    process-wide usage counters stay correct) + the shared rate limiter."""
    import threading

    try:
        chunks = ensure_kb(get_settings())
    except Exception as e:  # never stop the UI from rendering
        print(f"[sentinel] KB bootstrap skipped: {e}")
        chunks = 0
    limit = int(os.getenv("SENTINEL_MAX_RUNS_PER_HOUR", "30"))
    return {"chunks": chunks, "lock": threading.Lock(), "limiter": RateLimiter(limit)}


def plan_to_text(plan: list[dict]) -> str:
    return "\n".join(("analysis: " if t["type"] == "analysis" else "") + t["question"] for t in plan)


def text_to_plan(text: str) -> list[dict]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("analysis:"):
            out.append({"question": line.split(":", 1)[1].strip(), "type": "analysis"})
        else:
            out.append({"question": line, "type": "research"})
    return out


def banner() -> str:
    s = get_settings()
    llm = "offline stand-in LLM (no API key configured: answers are extractive)" if s.llm_provider == "fake" else f"{s.llm_provider}:{s.llm_model}"
    return f"LLM: **{llm}** · embeddings: **{s.embeddings_provider}** · reranker: **{s.reranker_provider}**"


def run_job(make_events) -> dict | None:
    """Drive a stream_research() generator inside a status box. Returns the final result event, or None on failure."""
    boot = bootstrap()
    final = None
    with st.status("Working…", expanded=True) as box:
        try:
            with boot["lock"]:
                for ev in make_events():
                    kind = ev.get("type")
                    if kind == "node":
                        box.write(f"**{ev['node']}** · {ev['seconds']}s · {ev['detail']}")
                    elif kind == "interrupt":
                        box.write("⏸️ Waiting for plan approval")
                    elif kind == "result":
                        final = ev
        except Exception as e:  # missing key / provider package / network: show it, don't crash the page
            box.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return None
        box.update(label="Done" if final and final["status"] == "done" else "Paused", state="complete")
    return final


def main() -> None:
    st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="wide")
    st.title("🛰️ Sentinel")
    st.caption("Multi-agent deep research: planner → researcher / analyst → writer → critic (self-verifying)")
    boot = bootstrap()
    st.caption(banner() + f" · knowledge base: **{boot['chunks']}** chunks")

    with st.sidebar:
        st.subheader("About")
        st.write(
            "Every sentence in the report must cite evidence; a critic checks each claim against its sources "
            "and sends failures back for re-research."
        )
        require_approval = st.toggle("Ask me to approve the plan first", value=False)
        st.caption("Public demo: the knowledge base is read-only (5 UAV / defence-analytics notes + a synthetic missions DB).")

    tab_run, tab_hist = st.tabs(["Research", "Past runs (this session's server)"])

    with tab_run:
        example = st.selectbox("Try an example", ["(write your own)"] + EXAMPLES)
        query = st.text_area(
            "Research question",
            value="" if example == "(write your own)" else example,
            placeholder=EXAMPLES[1], height=90, max_chars=2000,
        )
        if st.button("Run research", type="primary"):
            if len(query.strip()) < 3:
                st.warning("Please enter a research question (at least 3 characters).")
            else:
                ok, wait = boot["limiter"].acquire()
                if not ok:
                    st.warning(f"This demo is rate-limited to protect its API quota. Try again in ~{int(wait // 60) + 1} min.")
                else:
                    st.session_state["result"] = run_job(
                        lambda: stream_research(query.strip(), require_approval=require_approval)
                    )

        res = st.session_state.get("result")
        if res and res["status"] == "awaiting_approval":
            st.subheader("Proposed plan")
            st.caption("One task per line. Prefix a line with `analysis:` for database / calculation tasks.")
            edited = st.text_area("Plan", plan_to_text(res["plan"]), height=140, key="plan_edit")
            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button("Approve", type="primary"):
                resume: dict = {"approved": True}
                if (tasks := text_to_plan(edited)):
                    resume["tasks"] = tasks
                st.session_state["result"] = run_job(lambda: stream_research(thread_id=res["thread_id"], resume=resume))
                st.rerun()
            if c2.button("Reject"):
                st.session_state["result"] = run_job(
                    lambda: stream_research(thread_id=res["thread_id"], resume={"approved": False})
                )
                st.rerun()
        elif res and res["status"] == "done":
            st.markdown(res["report"])
            crit = res.get("critique") or {}
            m1, m2, m3 = st.columns(3)
            m1.metric("Claim support score", crit.get("score", "n/a"))
            m2.metric("Claims verified", f"{crit.get('supported', 0)}/{crit.get('claims_checked', 0)}")
            m3.metric("LLM calls", (res.get("usage") or {}).get("llm_calls", "n/a"))
            with st.expander("Agent trace"):
                st.dataframe(res.get("events", []))
            with st.expander("Critic details"):
                st.json(crit)
            st.download_button("Download report (.md)", res["report"], file_name="sentinel-report.md")

    with tab_hist:
        st.caption("Runs recorded by this server since it last started (the free tier's disk is wiped on restart).")
        try:
            eps = episodic.list_episodes(20)
        except Exception:
            eps = []
        if not eps:
            st.info("No finished runs yet.")
        for e in eps:
            with st.expander(e.get("query", "(untitled)")):
                st.write(e.get("summary", ""))


main()
