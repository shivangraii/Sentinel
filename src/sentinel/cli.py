"""Command line: `sentinel ingest | ask | eval | serve | demo`."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_ingest(a: argparse.Namespace) -> None:
    from sentinel.rag.ingest import ingest_directory

    print(json.dumps(ingest_directory(reset=a.reset), indent=2))


def _cmd_ask(a: argparse.Namespace) -> None:
    from sentinel.rag.ingest import ensure_kb
    from sentinel.runner import resume_research, run_research

    ensure_kb()
    res = run_research(a.query, require_approval=a.approve)
    while res["status"] == "awaiting_approval":
        print("\nProposed plan:")
        for t in res["plan"]:
            print(f"  {t['id']}. [{t['type']}] {t['question']}")
        ans = input("Approve? [y]es / [n]o / [e]dit: ").strip().lower()
        if ans.startswith("n"):
            res = resume_research(res["thread_id"], approved=False)
        elif ans.startswith("e"):
            print("Enter one task per line (prefix 'analysis:' for SQL/calculation tasks). Empty line to finish.")
            tasks = []
            while (line := input("> ").strip()):
                tasks.append({"question": line.split(":", 1)[1].strip(), "type": "analysis"} if line.lower().startswith("analysis:") else {"question": line, "type": "research"})
            res = resume_research(res["thread_id"], approved=True, tasks=tasks or None)
        else:
            res = resume_research(res["thread_id"], approved=True)
    if a.trace:
        for e in res["events"]:
            print(f"[{e['seconds']:>5}s] {e['node']:<13} {e['detail']}")
        print()
    print(res["report"])
    print(f"(saved: {res['report_path']})")


def _cmd_eval(a: argparse.Namespace) -> None:
    from sentinel.config import ROOT

    sys.path.insert(0, str(ROOT))
    from eval.run_eval import main as eval_main

    eval_main(["--mode", a.mode, "--limit", str(a.limit)] + (["--ragas"] if a.ragas else []))


def _cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("sentinel.api.main:app", host=a.host, port=a.port, reload=False)


def _cmd_demo(_: argparse.Namespace) -> None:
    os.environ["SENTINEL_OFFLINE"] = "1"
    from sentinel.config import reload_settings

    reload_settings()
    from sentinel.rag.ingest import ensure_kb
    from sentinel.runner import run_research

    ensure_kb()
    q = "Why does IMU-only dead reckoning drift, and how many missions are in the missions database?"
    res = run_research(q, require_approval=False)
    for e in res["events"]:
        print(f"[{e['seconds']:>5}s] {e['node']:<13} {e['detail']}")
    print("\n" + res["report"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sentinel", description="Sentinel multi-agent research platform")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="ingest data/raw into the knowledge base")
    s.add_argument("--reset", action="store_true", help="drop and rebuild the KB collection")
    s.set_defaults(fn=_cmd_ingest)

    s = sub.add_parser("ask", help="run a research job")
    s.add_argument("query")
    s.add_argument("--approve", action="store_true", help="pause for plan approval (human-in-the-loop)")
    s.add_argument("--trace", action="store_true", help="print node-by-node trace")
    s.set_defaults(fn=_cmd_ask)

    s = sub.add_parser("eval", help="run the evaluation suite")
    s.add_argument("--mode", choices=["retrieval", "agent", "all"], default="retrieval")
    s.add_argument("--limit", type=int, default=0, help="only first N questions (0 = all)")
    s.add_argument("--ragas", action="store_true", help="also compute RAGAS metrics (needs extras + LLM)")
    s.set_defaults(fn=_cmd_eval)

    s = sub.add_parser("serve", help="start the FastAPI server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)

    s = sub.add_parser("demo", help="offline end-to-end demo (no API key needed)")
    s.set_defaults(fn=_cmd_demo)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
