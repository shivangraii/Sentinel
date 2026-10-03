"""Long-term semantic memory: good web/arXiv evidence is written back to the knowledge base,
so later questions can reuse it through the same hybrid retriever."""
from __future__ import annotations

import hashlib

from sentinel.config import Settings, get_settings
from sentinel.rag import store
from sentinel.rag.ingest import KB


def remember_evidence(items: list[dict], settings: Settings | None = None) -> int:
    s = settings or get_settings()
    texts, ids, metas = [], [], []
    for it in items:
        if it.get("kind") not in ("web", "arxiv") or it.get("flagged"):
            continue
        key = hashlib.sha1(f"{it['source']}|{it['text'][:200]}".encode()).hexdigest()[:16]
        texts.append(it["text"])
        ids.append(f"mem::{key}")
        metas.append(
            {
                "source": it["source"],
                "chunk": 0,
                "title": it.get("title", "") or it["source"],
                "source_type": it["kind"],
            }
        )
    return store.add_texts(KB, texts, ids, metas, s)
