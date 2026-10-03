# Evaluation run 20261003-045111

- LLM: `fake:gemini-2.5-flash`  | embeddings: `hash`  | reranker: `lexical`
- Offline fake LLM: **True** (agent numbers validate the pipeline, not model quality)

### Retrieval ablation: all document questions (n=30)

| System | source_hit@1 | source_hit@3 | source_hit@5 | answer_chunk_recall@1 | answer_chunk_recall@3 | answer_chunk_recall@5 | MRR@5 (answer chunk) |
|---|---|---|---|---|---|---|---|
| dense only | 0.667 | 0.867 | 0.967 | 0.6 | 0.767 | 0.867 | 0.705 |
| bm25 only | 0.967 | 1.0 | 1.0 | 0.933 | 0.967 | 0.967 | 0.95 |
| hybrid (RRF) | 0.867 | 1.0 | 1.0 | 0.767 | 0.967 | 0.967 | 0.85 |
| hybrid + rerank | 0.967 | 1.0 | 1.0 | 0.933 | 0.967 | 0.967 | 0.95 |

### Retrieval ablation: paraphrased (hard) questions only (n=15)

| System | source_hit@1 | source_hit@3 | source_hit@5 | answer_chunk_recall@1 | answer_chunk_recall@3 | answer_chunk_recall@5 | MRR@5 (answer chunk) |
|---|---|---|---|---|---|---|---|
| dense only | 0.467 | 0.733 | 0.933 | 0.333 | 0.533 | 0.733 | 0.477 |
| bm25 only | 0.933 | 1.0 | 1.0 | 0.867 | 0.933 | 0.933 | 0.9 |
| hybrid (RRF) | 0.733 | 1.0 | 1.0 | 0.533 | 0.933 | 0.933 | 0.7 |
| hybrid + rerank | 0.933 | 1.0 | 1.0 | 0.867 | 0.933 | 0.933 | 0.9 |

### End-to-end (50 questions)

| System | accuracy_all | accuracy_doc | accuracy_sql | avg_latency_s | avg_llm_calls | avg_critic_score | critic_pass_rate | avg_tool_calls |
|---|---|---|---|---|---|---|---|---|
| naive RAG (dense, 1 call) | 0.08 | 0.133 | 0.0 | 0.0 | 1 |  |  |  |
| hybrid RAG (RRF+rerank, 1 call) | 0.18 | 0.267 | 0.05 | 0.0 | 1 |  |  |  |
| agentic (Sentinel) | 0.26 | 0.267 | 0.25 | 0.03 | 6.9 | 1.0 | 1.0 | 1.4 |
