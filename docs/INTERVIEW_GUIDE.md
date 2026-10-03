# How to present Sentinel

## Resume entry
**Sentinel: Multi-Agent Deep Research Platform** | LangGraph, ChromaDB, FastAPI, Streamlit, Docker
- Built a supervisor-routed multi-agent system (planner, researcher, analyst, writer, critic) in LangGraph with a reflection loop, human-in-the-loop plan approval and persistent checkpoints.
- Implemented hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion, reranking, query expansion) over section-aware chunks, plus semantic and episodic long-term memory in ChromaDB.
- Added a claim-level citation verifier: every sentence cites evidence ids and a critic validates each claim against its sources, re-researching failures (max 3 loops).
- Secured tool use (read-only Text-to-SQL, restricted Python subprocess, injection filtering, run budgets); shipped FastAPI + SSE streaming, Streamlit UI, Docker Compose, CI and 49 offline tests.
- Evaluated on a 50-question golden set with a retrieval ablation (dense vs BM25 vs hybrid vs rerank) and an end-to-end naive-vs-hybrid-vs-agentic comparison.

**Only add numbers you measured yourself.** After you run `sentinel eval --mode all` with a real LLM and real embeddings, add the real figures (e.g. "hybrid + rerank raised answer-chunk recall@3 from X to Y on 15 paraphrased questions").

## 3-minute demo script
1. `sentinel ask "<question mixing a document fact and a database number>" --trace`: point at the trace: planner splits into a research task and an analysis task, researcher calls `kb_search`, analyst writes SQL, writer cites `[E#]`, critic passes.
2. Same with `--approve`: edit the plan, show human-in-the-loop.
3. Open the report: Sources and Verification sections. Show `pytest`'s critic-failure test: a fabricated citation is caught and the writer is re-invoked with feedback.
4. Show the retrieval ablation table and say what it does and does not prove.

## Likely questions
**Why a graph, not one agent loop?** Distinct responsibilities, explicit control flow, per-node tracing, resumable checkpoints and a place to insert human approval. The critic → supervisor edge is a real cycle, which a linear chain cannot express.

**Is the supervisor an LLM?** No, on purpose. Routing is a function of the task queue, so code is cheaper and cannot hallucinate a route. Autonomy is in the nodes: tool choice, planning, claim judging.

**How does the agent choose tools?** Each turn the model sees the task, tool specs and previous observations and returns a JSON action. It can chain tools (e.g. `kb_search` then `web_search`, or `sql_query` then `calculator`) and decides when to finish. Budgets and a forced-finish turn bound it.

**Why RRF?** Dense and BM25 scores are on different scales; RRF uses only ranks, needs no tuning, and is robust. On my paraphrased questions it lifted recall@3 for the weak dense retriever from 0.53 to 0.93.

**What does the reranker add?** Fusion is recall-oriented; reranking restores precision at the top (MRR@5 0.70 → 0.90 on the hard set in my offline run).

**How do you reduce hallucination?** Evidence ids + a critic. Hard guarantees: cited ids exist, factual sentences are cited, flagged content is excluded. Soft check: an LLM judges entailment. Remaining failures are surfaced as a warning banner instead of hidden.

**What are the limits of the critic?** It is an LLM, so it can miss subtle unsupported claims or be too strict. I would calibrate it by hand-labelling 50 claims and measuring its precision/recall.

**How do you handle prompt injection?** Retrieved text is untrusted: sentence-level heuristic filter, flagged evidence is not shown to the writer, prompts say observations are data. It is defence in depth, not a proof.

**Is the Python sandbox safe?** No. It is layered hardening (AST allow-list, restricted builtins, rlimits, timeout), and I say so. For untrusted users it needs container/VM isolation.

**How did you test an LLM app?** A deterministic fake LLM drives the entire graph in CI, scripted LLMs force failure paths (bad citations, infinite tool loops, garbage JSON), and a test wraps outputs in chatty markdown to mimic real models.

**What does the offline demo prove?** Plumbing, control flow, tools, retrieval, verification logic. Not language quality. Real quality needs a real model; that's why the eval harness is model-agnostic.

**What would you do next?** Parallel research subgraphs, a calibrated critic, a Postgres checkpointer, streaming tokens, and an MCP server exposing the tools.
