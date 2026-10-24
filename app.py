"""Gradio front-end for Sentinel. Entry point for Hugging Face Spaces (Gradio SDK / ZeroGPU):   python app.py

Runs the research graph in-process (no separate API server). Works locally, on a CPU Space and on ZeroGPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Spaces install requirements.txt, not the package

# --- no API key configured -> keep real retrieval + critic, but use the deterministic stand-in LLM -------------------
_KEY_FOR = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_provider = os.getenv("SENTINEL_LLM_PROVIDER", "gemini")
if _provider in _KEY_FOR and not os.getenv(_KEY_FOR[_provider]) and not os.getenv("SENTINEL_OFFLINE"):
    os.environ["SENTINEL_LLM_PROVIDER"] = "fake"

import gradio as gr  # noqa: E402

from sentinel import gpu  # noqa: E402
from sentinel.config import get_settings  # noqa: E402
from sentinel.memory import episodic  # noqa: E402
from sentinel.rag.embeddings import get_embedder  # noqa: E402
from sentinel.rag.ingest import ensure_kb  # noqa: E402
from sentinel.rag.reranker import get_reranker  # noqa: E402
from sentinel.runner import stream_research  # noqa: E402

# Load models + build the knowledge base once at startup, on CPU (a ZeroGPU GPU window is per-request quota).
# Loading here also registers the models with the GPU dispatcher before the first request.
with gpu.cpu_only():
    try:
        _s = get_settings()
        get_embedder(_s)
        get_reranker(_s)
        ensure_kb(_s)
    except Exception as e:  # never stop the UI from starting
        print(f"[sentinel] startup warm-up skipped: {e}")

EXAMPLES = [
    "Why does IMU-only dead reckoning drift?",
    "Compare CBBA and leader-follower control, and how do their mission success rates differ?",
    "Why does IMU-only dead reckoning drift, and how many missions are in the missions database?",
]


def banner() -> str:
    s = get_settings()
    llm = "offline stand-in LLM (extractive answers, no API key set)" if s.llm_provider == "fake" else f"{s.llm_provider}:{s.llm_model}"
    hw = "ZeroGPU" if gpu.enabled() else "CPU"
    return f"**LLM:** {llm} · **Embeddings:** {s.embeddings_provider} · **Reranker:** {s.reranker_provider} · **Compute:** {hw}"


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


def _out(status, thread="", awaiting=False, plan="", res=None, trace=None):
    """Tuple matching the `outputs` list of the event handlers below."""
    res = res or {}
    done = res.get("status") == "done"
    crit = res.get("critique") or {}
    metrics = ""
    if done:
        metrics = (
            f"**Claim support score:** {crit.get('score', 'n/a')} · "
            f"**Claims verified:** {crit.get('supported', 0)}/{crit.get('claims_checked', 0)} · "
            f"**LLM calls:** {(res.get('usage') or {}).get('llm_calls', 'n/a')}"
        )
    return (
        status, gr.Group(visible=awaiting), plan, res.get("report", "") if done else "",
        metrics, trace or [], crit if done else {}, thread,
    )


def _drive(events):
    trace: list[list] = []
    thread = ""
    final = None
    yield _out("⏳ Starting…")
    try:
        for ev in events:
            kind = ev.get("type")
            if kind == "start":
                thread = ev["thread_id"]
            elif kind == "node":
                trace.append([ev["node"], ev["seconds"], ev["detail"]])
                yield _out(f"⏳ **{ev['node']}** finished in {ev['seconds']}s", thread=thread, trace=trace)
            elif kind == "interrupt":
                yield _out("⏸️ Waiting for plan approval", thread=thread, trace=trace)
            elif kind == "result":
                final = ev
    except Exception as e:  # surface provider/key/network errors in the UI instead of a stack trace
        yield _out(f"❌ {type(e).__name__}: {e}", thread=thread, trace=trace)
        return
    if final is None:
        yield _out("❌ The run ended without a result.", thread=thread, trace=trace)
    elif final["status"] == "awaiting_approval":
        yield _out("⏸️ Review the plan, edit it if you like, then approve.", thread=thread, awaiting=True,
                   plan=plan_to_text(final["plan"]), trace=trace)
    else:
        yield _out("✅ Done", thread=thread, res=final, trace=trace)


def start(query: str, require_approval: bool):
    q = (query or "").strip()
    if len(q) < 3:
        yield _out("⚠️ Please enter a research question (at least 3 characters).")
        return
    yield from _drive(stream_research(q, require_approval=bool(require_approval)))


def decide(thread: str, plan_text: str, approved: bool):
    if not thread:
        yield _out("⚠️ Nothing to approve. Run a research question first.")
        return
    resume: dict = {"approved": approved}
    tasks = text_to_plan(plan_text) if approved else []
    if tasks:
        resume["tasks"] = tasks
    yield from _drive(stream_research(thread_id=thread, resume=resume))


def approve(thread: str, plan_text: str):
    yield from decide(thread, plan_text, True)


def reject(thread: str, plan_text: str):
    yield from decide(thread, plan_text, False)


def past_runs():
    try:
        return [[e.get("query", ""), e.get("summary", ""), e.get("report_path", "")] for e in episodic.list_episodes(20)]
    except Exception:
        return []


with gr.Blocks(title="Sentinel") as demo:
    gr.Markdown(
        "# 🛰️ Sentinel\nMulti-agent deep research: planner → researcher / analyst → writer → **critic that verifies "
        "every claim against its evidence**."
    )
    gr.Markdown(banner())
    thread_state = gr.State("")

    with gr.Tab("Research"):
        query = gr.Textbox(label="Research question", lines=3, placeholder=EXAMPLES[1])
        gr.Examples(EXAMPLES, inputs=query)
        require_approval = gr.Checkbox(label="Ask me to approve the plan first (human-in-the-loop)", value=False)
        run_btn = gr.Button("Run research", variant="primary")
        status = gr.Markdown()
        with gr.Group(visible=False) as approval_box:
            plan_box = gr.Textbox(
                label="Proposed plan: one task per line (prefix a line with `analysis:` for database / calculation tasks)",
                lines=6,
            )
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                reject_btn = gr.Button("Reject")
        report = gr.Markdown()
        metrics = gr.Markdown()
        with gr.Accordion("Agent trace", open=False):
            trace = gr.Dataframe(headers=["node", "seconds", "detail"], interactive=False)
        with gr.Accordion("Critic details", open=False):
            critic = gr.JSON()

    with gr.Tab("Past runs"):
        refresh = gr.Button("Refresh")
        history = gr.Dataframe(headers=["query", "summary", "report"], interactive=False)
        refresh.click(past_runs, outputs=history)

    outs = [status, approval_box, plan_box, report, metrics, trace, critic, thread_state]
    run_btn.click(start, [query, require_approval], outs)
    approve_btn.click(approve, [thread_state, plan_box], outs)
    reject_btn.click(reject, [thread_state, plan_box], outs)

demo.queue(max_size=16)  # runs are serialised per handler: protects the shared (process-wide) usage counters

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
