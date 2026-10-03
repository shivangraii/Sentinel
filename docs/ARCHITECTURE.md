# Architecture

## State (`state.py`)
`AgentState` is a TypedDict. List fields with append reducers: `evidence`, `findings`, `events`. Everything else is replaced by the node that writes it.

| Field | Written by | Purpose |
|---|---|---|
| `query`, `run_id`, `require_approval` | runner | input and run identity (`run_id` = LangGraph `thread_id`) |
| `memory_context` | recall | related past reports (episodic memory) |
| `tasks[]` `{id, question, type, status}` | planner, workers, critic, approve_plan | the work queue; `type` is `research` or `analysis` |
| `evidence[]` `{id: "E#", text, source, kind, flagged, task_id}` | researcher, analyst | every citable fact; ids are globally unique per run |
| `findings[]` | researcher, analyst | per-task cited answer + tool steps |
| `draft`, `feedback` | writer, critic | current report text and critic feedback |
| `critique`, `passed`, `loops` | critic | verification result and retry counter |
| `report`, `report_path`, `usage` | finalize | final output |

## Nodes
1. **recall**: episodic memory lookup (cosine distance threshold) → `memory_context`.
2. **planner**: LLM returns 2-5 typed sub-tasks as JSON; prompt includes the DB schema so it only plans `analysis` tasks the data can answer. Falls back to one research task if parsing fails.
3. **approve_plan**: with `require_approval`, calls `interrupt()`; the SQLite checkpointer persists the paused graph. Resume with `Command(resume={"approved": bool, "tasks": [...]})`. The node is idempotent because LangGraph re-executes it on resume.
4. **supervisor**: deterministic router. First pending task decides researcher vs analyst; no pending tasks → writer. (Deliberately not an LLM: control flow in code is cheaper, testable and cannot loop.)
5. **researcher / analyst** (`agents/workers.py`, `agents/tool_loop.py`): each turn the model returns `{"action": tool, "args": {...}}` or `{"action": "finish", "answer": ...}`. Observations are rendered with evidence ids. Safety: per-run `Budget` (tool calls, wall time), max steps per task with a forced-finish turn, unknown-tool and garbage-output handling, de-duplication of evidence, injection-flagged evidence marked in the prompt.
6. **writer**: sees findings + non-flagged evidence only, must cite `[E#]` after each factual sentence; on retry also sees critic feedback and the previous draft.
7. **critic** (`agents/critic.py`):
   - extracts claim sentences from Summary/Findings (skips headings, short lines, Limitations);
   - deterministic: uncited sentences and unknown/flagged ids are unsupported;
   - LLM judge: batch verdicts `{i, supported, reason}` against the cited evidence text; a claim the judge fails to return is *not* silently passed;
   - `score = supported / checked`; pass if `score >= critic_threshold` (0.8) and no invalid ids;
   - on failure asks a second prompt for feedback and up to 2 follow-up research questions, appended as new pending tasks; `loops += 1`.
   - routing: pass or `loops >= max_loops` → finalize, otherwise → supervisor (so new tasks are researched before rewriting).
8. **finalize**: adds a warning banner if unverified, a Sources section (only ids actually cited), a Verification section (score, loops, tool/LLM calls), writes `data/reports/*.md`, and stores an episode.

## Retrieval pipeline (`rag/`)
- **Ingest**: markdown is split by headings; each section is sentence-chunked (180 words, 40 overlap). The stored document is the clean body; the embedding is computed on `title > section\nbody` (contextual headers). Chunk ids are `source::n`, so re-ingesting is an idempotent upsert. Collections are namespaced by embedding model, so vectors from different models are never mixed.
- **Search**: for each query variant, dense top-k (Chroma, cosine) and BM25 top-k (rank-bm25 over title+section+body) → RRF (`1/(60+rank)`) over all lists → top `3k` pool → rerank against the *original* query → top k. Optional `where` metadata filter.
- **Expansion** (`query_rewrite.py`): LLM-written alternative queries and an optional HyDE passage. Failure falls back to the original query.
- **Embedders/rerankers**: `bge-small` + `bge-reranker-base` by default; `hash` embedder and `lexical` reranker for offline use.

## Memory
| Layer | Implementation | Lifetime |
|---|---|---|
| Short-term | LangGraph `SqliteSaver` keyed by `thread_id` | one run / resumable |
| Semantic | `remember_evidence` upserts web/arXiv evidence (not flagged) into the KB collection | permanent, retrievable by `kb_search` |
| Episodic | Chroma `episodes_*` collection: query + summary + report path | permanent, recalled by `recall` |

## Guardrails
- SQL: must start with SELECT/WITH, single statement, forbidden-keyword scan (string literals stripped first), DB opened with `mode=ro`, `PRAGMA query_only=ON`, 50-row cap.
- Python: AST allow-list (imports, no dunder access, no `open/exec/eval`), restricted builtins and import hook inside `python -I`, CPU/memory rlimits, 5 s timeout, 4 KB output cap. Best-effort only.
- Calculator: AST evaluator with whitelisted functions, exponent cap.
- Injection filter: sentence-level removal of instruction-like text in retrieved content; `flagged` evidence is excluded from the writer and not citable.
- Budgets: tool calls per run, wall-clock per run, tool steps per task, critic loops.

## Observability
Every node is wrapped by `@traced`: it appends `{node, seconds, detail}` to `state.events` (returned by the API and shown in the UI) and a JSONL line to `traces/`. If `LANGFUSE_*` keys are set, a Langfuse callback handler is attached to the graph config (best effort; v2 and v3 import paths).

## Why these trade-offs
- JSON action loop instead of native function-calling: portable across providers and local models, trivially testable with a scripted LLM.
- Deterministic supervisor: reliability and cost over "agentic" showmanship; the autonomy lives where judgement is needed.
- Citations by evidence id: makes verification mechanical (ids exist? claim entailed?) instead of asking a model whether its own answer "looks right".
- Offline fake LLM: lets CI cover the whole graph, retries and HITL paths deterministically.
