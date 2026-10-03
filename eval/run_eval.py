"""Evaluation harness.

  retrieval  ablation on the 30 document questions: dense | bm25 | hybrid | hybrid+rerank
             metrics: source hit@k, answer-bearing-chunk recall@k, MRR@5 (needs no LLM)
  agent      end-to-end on all 50 questions: naive RAG vs hybrid RAG vs the full agentic graph
             metrics: answer correctness (all ground-truth keywords present), latency, LLM/tool calls,
             critic score. Uses whatever LLM is configured (offline fake => plumbing check only).

Usage:  python -m eval.run_eval --mode retrieval|agent|all [--limit N] [--ragas]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.config import get_settings  # noqa: E402
from sentinel.llm import get_llm, usage  # noqa: E402
from sentinel.rag.ingest import ensure_kb  # noqa: E402
from sentinel.rag.retriever import HybridRetriever  # noqa: E402

GOLDEN = ROOT / "data" / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5)


def load_golden(limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        docs = [r for r in rows if r["type"] == "doc"][: max(1, limit // 2)]
        sqls = [r for r in rows if r["type"] == "sql"][: max(1, limit - len(docs))]
        rows = docs + sqls
    return rows


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def has_all(text: str, kws: list[str]) -> bool:
    t = norm(text)
    return all(k.lower() in t for k in kws)


# ------------------------------------------------------------------ retrieval ablation
def eval_retrieval(rows: list[dict], style: str | None = None) -> dict:
    s = get_settings()
    retr = HybridRetriever(s)
    docs = [r for r in rows if r["type"] == "doc" and (style is None or r.get("style") == style)]
    configs = {
        "dense only": dict(mode="dense", rerank=False),
        "bm25 only": dict(mode="bm25", rerank=False),
        "hybrid (RRF)": dict(mode="hybrid", rerank=False),
        "hybrid + rerank": dict(mode="hybrid", rerank=True),
    }
    out = {}
    for name, kw in configs.items():
        hit = {k: 0 for k in KS}
        rec = {k: 0 for k in KS}
        rr = []
        for r in docs:
            hits = retr.search(r["question"], k=max(KS), **kw)
            src_rank = next((i for i, h in enumerate(hits, 1) if h.source == r["expected_source"]), None)
            kw_rank = next((i for i, h in enumerate(hits, 1) if has_all(h.text, r["keywords"])), None)
            for k in KS:
                hit[k] += int(src_rank is not None and src_rank <= k)
                rec[k] += int(kw_rank is not None and kw_rank <= k)
            rr.append(1.0 / kw_rank if kw_rank else 0.0)
        n = len(docs) or 1
        out[name] = {
            **{f"source_hit@{k}": round(hit[k] / n, 3) for k in KS},
            **{f"answer_chunk_recall@{k}": round(rec[k] / n, 3) for k in KS},
            "MRR@5 (answer chunk)": round(sum(rr) / n, 3),
        }
    return {"questions": len(docs), "embeddings": s.embeddings_provider, "reranker": s.reranker_provider, "results": out}


# ------------------------------------------------------------------ end-to-end systems
_NAIVE_SYS = (
    "Answer the question using ONLY the context passages. Be concise (1-3 sentences). "
    "If the context does not contain the answer, say so."
)


def _single_shot(retr: HybridRetriever, q: str, mode: str, rerank: bool) -> tuple[str, list[str]]:
    hits = retr.search(q, k=3, mode=mode, rerank=rerank)
    ctx = "\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1)) or "(no passages)"
    ans = get_llm().complete("naive_rag", _NAIVE_SYS, f"CONTEXT:\n{ctx}\n\nQUESTION: {q}")
    return ans, [h.text for h in hits]


def _main_sections(report: str) -> str:
    m = re.search(r"## Executive summary(.*?)## Limitations", report, re.DOTALL)
    return m.group(1) if m else report


def eval_agent(rows: list[dict], ragas: bool = False) -> dict:
    from sentinel.runner import run_research

    s = get_settings()
    ensure_kb(s)
    retr = HybridRetriever(s)
    systems = {
        "naive RAG (dense, 1 call)": lambda r: _single_shot(retr, r["question"], "dense", False),
        "hybrid RAG (RRF+rerank, 1 call)": lambda r: _single_shot(retr, r["question"], "hybrid", True),
    }
    results: dict[str, dict] = {}
    samples: list[dict] = []

    def record(name: str, r: dict, ok: bool, secs: float, extra: dict | None = None) -> None:
        d = results.setdefault(name, {"rows": []})
        snap = usage.snapshot()
        d["rows"].append({"id": r["id"], "type": r["type"], "ok": ok, "seconds": secs, "llm_calls": snap["llm_calls"], **(extra or {})})

    for name, fn in systems.items():
        for r in rows:
            usage.reset()
            t0 = time.time()
            ans, _ctx = fn(r)
            record(name, r, has_all(ans, r["keywords"]), time.time() - t0)

    name = "agentic (Sentinel)"
    for r in rows:
        usage.reset()
        t0 = time.time()
        res = run_research(r["question"], thread_id=f"eval-{uuid.uuid4().hex[:8]}", require_approval=False)
        body = _main_sections(res["report"])
        crit = res.get("critique", {})
        record(
            name, r, has_all(body, r["keywords"]), time.time() - t0,
            {"critic_score": crit.get("score", 0), "tool_calls": res.get("usage", {}).get("tool_calls", 0),
             "critic_passed": not res["report"].startswith("> **Verification warning")},
        )
        if ragas:
            samples.append({"user_input": r["question"], "response": body.strip(), "reference": r["ground_truth"]})

    summary = {}
    for sysname, d in results.items():
        rs = d["rows"]

        def acc(sub: list[dict]) -> float | None:
            return round(sum(x["ok"] for x in sub) / len(sub), 3) if sub else None

        summary[sysname] = {
            "accuracy_all": acc(rs),
            "accuracy_doc": acc([x for x in rs if x["type"] == "doc"]),
            "accuracy_sql": acc([x for x in rs if x["type"] == "sql"]),
            "avg_latency_s": round(statistics.mean(x["seconds"] for x in rs), 2),
            "avg_llm_calls": round(statistics.mean(x["llm_calls"] for x in rs), 1),
        }
        if sysname.startswith("agentic"):
            summary[sysname]["avg_critic_score"] = round(statistics.mean(x["critic_score"] for x in rs), 3)
            summary[sysname]["critic_pass_rate"] = round(sum(x["critic_passed"] for x in rs) / len(rs), 3)
            summary[sysname]["avg_tool_calls"] = round(statistics.mean(x["tool_calls"] for x in rs), 1)

    out = {"questions": len(rows), "llm": f"{s.llm_provider}:{s.llm_model}", "offline": s.offline, "summary": summary}
    if ragas:
        out["ragas"] = _run_ragas(samples)
    return out


def _run_ragas(samples: list[dict]) -> dict | str:
    """Experimental: RAGAS needs `pip install 'sentinel[eval]'`, a real LLM and an embedding model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except ImportError as e:
        return f"skipped (missing packages: {e})"
    s = get_settings()
    if s.llm_provider == "fake":
        return "skipped (RAGAS needs a real LLM; offline fake provider selected)"
    from sentinel.rag.embeddings import get_embedder  # noqa: F401  (ensures settings validated)

    llm = get_llm(s)
    if llm._model is None:
        llm._model = llm._build()
    # contexts: re-retrieve with the production retriever for each sample
    retr = HybridRetriever(s)
    for smp in samples:
        smp["retrieved_contexts"] = [h.text for h in retr.search(smp["user_input"], k=4)]
    ds = EvaluationDataset.from_list(samples)
    res = evaluate(
        ds,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=LangchainLLMWrapper(llm._model),
        embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=s.embeddings_model)),
    )
    return {k: round(float(v), 3) for k, v in res._repr_dict.items()}


# ------------------------------------------------------------------ reporting
def _md_table(title: str, data: dict[str, dict]) -> str:
    cols: list[str] = []
    for row in data.values():
        for c in row:
            if c not in cols:
                cols.append(c)
    lines = [f"### {title}", "", "| System | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, row in data.items():
        lines.append(f"| {name} | " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "agent", "all"], default="retrieval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ragas", action="store_true")
    a = ap.parse_args(argv)

    s = get_settings()
    ensure_kb(s)
    rows = load_golden(a.limit)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    md = [f"# Evaluation run {stamp}", "",
          f"- LLM: `{s.llm_provider}:{s.llm_model}`  | embeddings: `{s.embeddings_provider}`  | reranker: `{s.reranker_provider}`",
          f"- Offline fake LLM: **{s.offline}**" + (" (agent numbers validate the pipeline, not model quality)" if s.offline else ""), ""]
    payload: dict = {"stamp": stamp}

    if a.mode in ("retrieval", "all"):
        for label, style in (("all document questions", None), ("paraphrased (hard) questions only", "paraphrased")):
            r = eval_retrieval(rows, style)
            payload.setdefault("retrieval", {})[label] = r
            md.append(_md_table(f"Retrieval ablation: {label} (n={r['questions']})", r["results"]))
            print(md[-1])
    if a.mode in ("agent", "all"):
        r = eval_agent(rows, a.ragas)
        payload["agent"] = r
        md.append(_md_table(f"End-to-end ({r['questions']} questions)", r["summary"]))
        print(md[-1])
        if "ragas" in r:
            md.append(f"RAGAS: `{r['ragas']}`\n")

    (RESULTS / f"{stamp}_{a.mode}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS / "latest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved -> {RESULTS / 'latest.md'}")


if __name__ == "__main__":
    main()
