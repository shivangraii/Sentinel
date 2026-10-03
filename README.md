# 🛰️ Sentinel: Multi-Agent Deep Research Platform

Sentinel takes a research question, plans it, researches it with **hybrid RAG + web/arXiv + SQL/Python tools**,
writes a cited report, and then **checks its own claims against the evidence** before returning anything.
It is built with LangGraph, ChromaDB, FastAPI and Streamlit, and the whole pipeline runs **offline with no API key**
(deterministic fake LLM) so it can be tested, demoed and run in CI.

```
question ─► recall ─► planner ─► approve_plan (optional human-in-the-loop)
                                   │
                              supervisor ◄──────────────┐
                               │   │   │                │
                 researcher ◄──┘   │   └──► writer      │
                 (kb/web/arXiv)    │          │         │
                 analyst ◄─────────┘          ▼         │
                 (SQL/Python/calc)          critic ─ fail (feedback + follow-up tasks)
                                              │ pass / max loops
                                          finalize ─► report + sources + verification + episodic memory
```

## What makes it more than "an agent with tools"

| Capability | How it works |
|---|---|
| **Multi-agent graph** | Planner, Researcher, Analyst, Writer, Critic as LangGraph nodes; a deterministic Supervisor routes work (control flow in code, judgement in LLM nodes) |
| **Autonomous tool selection** | Researcher/Analyst run a JSON action loop: the model picks the next tool from observations, chains calls, and decides when to finish (works with any chat model, no native tool-calling needed) |
| **Hybrid retrieval** | Dense (Chroma) + BM25, fused with Reciprocal Rank Fusion, then reranked (cross-encoder or offline lexical); multi-query expansion and optional HyDE; heading-aware chunks with contextual headers |
| **Self-verification** | Every sentence must cite `[E#]` evidence ids; the Critic checks ids exist, sentences are cited, and an LLM judge checks each claim is entailed by its evidence. Failures produce feedback and follow-up research tasks (max 3 loops), and the final report carries a warning banner if claims stay unverified |
| **3-layer memory** | Short-term: LangGraph SQLite checkpointer (resume, interrupts). Semantic: good web/arXiv evidence is written back to the Chroma KB. Episodic: past reports are recalled for related questions |
| **Human-in-the-loop** | `require_approval` pauses after planning (LangGraph `interrupt`); approve, edit the plan, or reject via CLI, API or UI |
| **Guardrails** | Read-only SQL (statement allow-list + `mode=ro` + `query_only`), AST-restricted Python subprocess with rlimits, safe calculator, prompt-injection filter on all retrieved text (flagged evidence is never shown to the writer), tool-call and time budgets |
| **Observability** | Per-node timing events in the API response + JSONL traces; optional Langfuse callback |
| **Evaluation** | 50-question golden set (30 document + 20 SQL with computed ground truth), retrieval ablation, end-to-end comparison (naive RAG vs hybrid RAG vs agentic), optional RAGAS |
| **Serving** | FastAPI (sync + SSE streaming, approval endpoints, upload/ingest), Streamlit UI, Docker Compose, CI |

## Quickstart

### 1. Offline demo (no API key, ~1 minute)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
sentinel demo                  # full graph run with the offline fake LLM
pytest                         # 49 tests, all offline
```
`sentinel demo` runs the real graph, retrieval, SQL tool and critic. Only the *language model* is a deterministic
stand-in that copies retrieved sentences, so reports in this mode are extractive, not intelligent.

### 2. With a real LLM
```bash
pip install -e ".[gemini,embeddings,search,ui]"     # or groq / ollama / anthropic
cp .env.example .env                                 # add GOOGLE_API_KEY (Gemini has a free tier)
sentinel ingest --reset                              # embed data/raw into ChromaDB
sentinel ask "Compare CBBA and leader-follower control, and how do their mission success rates differ?" --trace
sentinel ask "..." --approve                         # pause to approve/edit the plan
```
Provider model names change over time: edit `SENTINEL_LLM_MODEL` in `.env` if a default stops working.

### 3. API + UI
```bash
sentinel serve                       # http://localhost:8000/docs
streamlit run ui/app.py              # http://localhost:8501
# or:  docker compose up --build     (set SENTINEL_OFFLINE=1 in .env for a keyless demo)
```

```bash
curl -s localhost:8000/research -H 'content-type: application/json' \
  -d '{"query": "Why does IMU-only dead reckoning drift?"}' | jq .report

# human-in-the-loop
curl -s localhost:8000/research -H 'content-type: application/json' \
  -d '{"query": "...", "require_approval": true}'          # -> status: awaiting_approval, plan, thread_id
curl -s localhost:8000/approve -H 'content-type: application/json' \
  -d '{"thread_id": "<id>", "approved": true, "tasks": ["optional edited task", "analysis: SQL task"]}'
```
Endpoints: `POST /research`, `POST /research/stream` (SSE), `POST /approve`, `POST /approve/stream`, `POST /ingest`,
`POST /upload`, `GET /episodes`, `GET /health`.

## Configuration
`configs/settings.yaml` holds defaults; environment variables (`.env.example`) override them.

| Variable | Meaning |
|---|---|
| `SENTINEL_LLM_PROVIDER` / `SENTINEL_LLM_MODEL` | `gemini`, `groq`, `ollama`, `anthropic`, or `fake` |
| `SENTINEL_EMBEDDINGS` | `sentence-transformers` (default, `BAAI/bge-small-en-v1.5`) or `hash` (offline) |
| `SENTINEL_RERANKER` | `cross-encoder` or `lexical` |
| `SENTINEL_SEARCH` | `duckduckgo`, `tavily`, or `none` |
| `SENTINEL_REQUIRE_APPROVAL` | default for plan approval |
| `SENTINEL_MAX_LOOPS` | critic retry cap (default 3) |
| `SENTINEL_OFFLINE=1` | forces fake LLM + hash embeddings + lexical reranker + no web search |

Prompts live in `configs/prompts/*.md`, so you can tune agent behaviour without touching code.

## Evaluation

The golden set (`data/eval/golden_set.jsonl`) has 30 document questions (half are paraphrased to share few words with the
source, as a harder retrieval test) and 20 database questions whose answers are computed by SQL from `data/sample.db`
(a **synthetic** missions database, reproducible via `scripts/make_sample_db.py`).

```bash
sentinel eval --mode retrieval       # no LLM needed
sentinel eval --mode all             # + naive vs hybrid vs agentic with your configured LLM
sentinel eval --mode all --ragas     # + RAGAS faithfulness/relevancy (experimental)
```

**Retrieval ablation** (measured here with the offline `hash` embedder and lexical reranker; answer-chunk recall = a retrieved chunk contains all ground-truth keywords):

| Paraphrased (hard) questions, n=15 | recall@1 | recall@3 | recall@5 | MRR@5 |
|---|---|---|---|---|
| dense only | 0.333 | 0.533 | 0.733 | 0.477 |
| BM25 only | 0.867 | 0.933 | 0.933 | 0.900 |
| hybrid (RRF) | 0.533 | 0.933 | 0.933 | 0.700 |
| hybrid + rerank | 0.867 | 0.933 | 0.933 | 0.900 |

How to read this honestly: the offline hash embedder is a weak, purely lexical dense model, so dense-only is poor and BM25
is strong on a tiny 21-chunk corpus. RRF fusion fixes dense's weakness (recall@3 0.53 → 0.93) and reranking restores top-1
precision. With real embeddings (`bge-small`) and a bigger corpus the ordering will differ, so **re-run
`sentinel eval --mode retrieval` after switching `SENTINEL_EMBEDDINGS`** and report those numbers instead.
Full tables: `eval/results/latest.md`.

**End-to-end comparison:** `eval/results/latest.md` also contains an end-to-end table, but it was produced with the offline
fake LLM, so it only proves the harness works (and that SQL questions are unanswerable without the analyst). It says nothing about model
quality. Run `sentinel eval --mode all` with a real LLM to get meaningful numbers, then cite those.

## Testing
`pytest` runs 49 offline tests (≈ 1 min): chunking, RRF math, hybrid retrieval, SQL/Python/calculator safety (including
attempted writes, imports and infinite loops), injection filtering, arXiv parsing, the full graph, the critic failure and
recovery loop, max-loop warning banner, tool budgets, HITL approve/edit/reject, API endpoints including SSE streaming and
approval round-trips, upload validation, and robustness to chatty/fenced JSON from LLMs.

## Project layout
```
configs/        settings.yaml, prompts/*.md
data/raw/       sample knowledge base (5 UAV/defence-analytics notes)   data/sample.db  synthetic missions DB
data/eval/      golden_set.jsonl
src/sentinel/
  graph.py state.py runner.py llm.py fake_llm.py config.py cli.py
  agents/       planner, supervisor, workers (researcher+analyst), tool_loop, writer, critic, memory_nodes
  tools/        kb_search, web_search, arxiv_search, sql_tool, python_sandbox, calculator, registry
  rag/          ingest (section-aware chunking), embeddings, store (Chroma), retriever (dense+BM25+RRF), reranker, query_rewrite
  memory/       checkpointer (short-term), semantic, episodic
  guardrails/   injection_filter, budget
  observability/tracing.py
  api/          FastAPI app + schemas
ui/app.py       Streamlit        eval/run_eval.py   evaluation harness        tests/   49 tests
```
See `docs/ARCHITECTURE.md` for the design and `docs/INTERVIEW_GUIDE.md` for how to present it.

## Known limitations (read before claiming anything)
- **Not exercised in the build environment** (no keys/network/GPU there): real Gemini/Groq/Ollama/Anthropic calls,
  `sentence-transformers` models, DuckDuckGo/Tavily/arXiv network requests, Langfuse, RAGAS, and the Docker build.
  Their code paths are small and isolated, but run them once yourself and fix any API drift before depending on them.
- The Python sandbox is **best-effort hardening, not a security boundary**; do not expose it to untrusted users without
  container isolation (the compose file drops privileges but does not remove network access).
- The critic's entailment judge is an LLM, so it can be wrong in both directions; the deterministic citation checks are the hard guarantee.
- The sample knowledge base is small and the missions database is synthetic. Replace `data/raw` and the DB for real use.
- The usage tracker and run budgets are process-wide, so concurrent runs share counters (fine for a demo, not multi-tenant).

## Ideas to extend
Per-claim confidence in the UI, a LangGraph subgraph per sub-question for parallel research, a Postgres checkpointer,
conversation follow-ups on the same `thread_id`, and an MCP server exposing `kb_search`/`sql_query`.
