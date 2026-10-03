"""Streamlit front-end for the Sentinel API.   streamlit run ui/app.py"""
from __future__ import annotations

import json
import os

import requests
import streamlit as st

DEFAULT_API = os.getenv("SENTINEL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="wide")
st.title("🛰️ Sentinel")
st.caption("Multi-agent deep research: planner → researcher / analyst → writer → critic (self-verifying)")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    api = st.text_input("API URL", DEFAULT_API).rstrip("/")
    try:
        health = requests.get(f"{api}/health", timeout=3).json()
        st.success(f"API online · {health['llm']}")
        if health.get("offline"):
            st.warning("Offline demo mode: extractive answers, no real LLM.")
    except Exception:
        st.error("API not reachable. Start it with `sentinel serve`.")
    require_approval = st.toggle("Ask me to approve the plan first", value=False)
    st.divider()
    st.subheader("Knowledge base")
    up = st.file_uploader("Add a document (PDF / MD / TXT)", type=["pdf", "md", "txt"])
    if up is not None and st.button("Upload & ingest"):
        try:
            r = requests.post(f"{api}/upload", files={"file": (up.name, up.getvalue())}, timeout=300)
            r.raise_for_status()
            st.success(f"Ingested: {r.json()}")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    if st.button("Re-ingest data/raw"):
        try:
            st.success(requests.post(f"{api}/ingest", json={"reset": False}, timeout=300).json())
        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------------------- helpers
def sse(path: str, payload: dict):
    with requests.post(f"{api}{path}", json=payload, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


def run_job(path: str, payload: dict) -> dict | None:
    final = None
    with st.status("Working…", expanded=True) as box:
        try:
            for ev in sse(path, payload):
                if ev["type"] == "node":
                    box.write(f"**{ev['node']}** · {ev['seconds']}s · {ev['detail']}")
                elif ev["type"] == "interrupt":
                    box.write("⏸️ Waiting for plan approval")
                elif ev["type"] == "error":
                    box.update(label="Failed", state="error")
                    st.error(ev["message"])
                    return None
                elif ev["type"] == "result":
                    final = ev
            box.update(label="Done" if final and final["status"] == "done" else "Paused", state="complete")
        except Exception as e:
            box.update(label="Failed", state="error")
            st.error(f"Request failed: {e}")
    return final


def plan_to_text(plan: list[dict]) -> str:
    return "\n".join(("analysis: " if t["type"] == "analysis" else "") + t["question"] for t in plan)


def text_to_plan(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("analysis:"):
            out.append({"question": line.split(":", 1)[1].strip(), "type": "analysis"})
        else:
            out.append({"question": line, "type": "research"})
    return out


# ---------------------------------------------------------------- main
tab_run, tab_hist = st.tabs(["Research", "Past runs"])

with tab_run:
    query = st.text_area(
        "Research question",
        value=st.session_state.get("query", ""),
        placeholder="e.g. Compare leader-follower and CBBA coordination, and how do their mission success rates differ?",
        height=90,
    )
    if st.button("Run research", type="primary"):
        if len(query.strip()) < 3:
            st.warning("Please enter a research question (at least 3 characters).")
        else:
            st.session_state["query"] = query
            st.session_state["result"] = run_job(
                "/research/stream", {"query": query, "require_approval": require_approval}
            )

    res = st.session_state.get("result")
    if res and res["status"] == "awaiting_approval":
        st.subheader("Proposed plan")
        st.caption("One task per line. Prefix a line with `analysis:` for database / calculation tasks.")
        edited = st.text_area("Plan", plan_to_text(res["plan"]), height=140, key="plan_edit")
        c1, c2, _ = st.columns([1, 1, 6])
        if c1.button("Approve", type="primary"):
            st.session_state["result"] = run_job(
                "/approve/stream", {"thread_id": res["thread_id"], "approved": True, "tasks": text_to_plan(edited)}
            )
            st.rerun()
        if c2.button("Reject"):
            st.session_state["result"] = run_job(
                "/approve/stream", {"thread_id": res["thread_id"], "approved": False}
            )
            st.rerun()
    elif res and res["status"] == "done":
        st.markdown(res["report"])
        crit = res.get("critique") or {}
        col1, col2, col3 = st.columns(3)
        col1.metric("Claim support score", crit.get("score", "n/a"))
        col2.metric("Claims verified", f"{crit.get('supported', 0)}/{crit.get('claims_checked', 0)}")
        col3.metric("LLM calls", (res.get("usage") or {}).get("llm_calls", "n/a"))
        with st.expander("Agent trace"):
            st.dataframe(res.get("events", []))
        with st.expander("Critic details"):
            st.json(crit)
        st.download_button("Download report (.md)", res["report"], file_name="sentinel-report.md")

with tab_hist:
    try:
        eps = requests.get(f"{api}/episodes", timeout=5).json()
        if not eps:
            st.info("No finished runs yet.")
        for e in eps:
            with st.expander(e.get("query", "(untitled)")):
                st.write(e.get("summary", ""))
                st.caption(e.get("report_path", ""))
    except Exception:
        st.info("Connect to the API to see past runs.")
