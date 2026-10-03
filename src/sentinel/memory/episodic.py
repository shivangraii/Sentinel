"""Episodic memory: one record per finished run (question + report summary) for later recall."""
from __future__ import annotations

import time

from sentinel.config import Settings, get_settings
from sentinel.rag import store
from sentinel.rag.embeddings import get_embedder

EPISODES = "episodes"


def save_episode(run_id: str, query: str, summary: str, report_path: str, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    doc = f"{query}\n{summary}"
    store.add_texts(
        EPISODES,
        [doc],
        [run_id],
        [{"query": query, "summary": summary[:800], "report_path": report_path, "ts": int(time.time())}],
        s,
    )


def recall(query: str, k: int = 2, max_distance: float = 0.55, settings: Settings | None = None) -> list[dict]:
    s = settings or get_settings()
    col = store.get_collection(EPISODES, s)
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[get_embedder(s).embed_query(query)],
        n_results=min(k, col.count()),
        include=["metadatas", "distances"],
    )
    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        if dist <= max_distance:
            out.append({**meta, "distance": round(float(dist), 3)})
    return out


def list_episodes(limit: int = 20, settings: Settings | None = None) -> list[dict]:
    s = settings or get_settings()
    data = store.get_collection(EPISODES, s).get(include=["metadatas"])
    rows = [{"run_id": i, **(m or {})} for i, m in zip(data["ids"], data["metadatas"])]
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return rows[:limit]
