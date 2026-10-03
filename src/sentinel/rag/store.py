"""ChromaDB access. We compute embeddings ourselves so the backend is swappable."""
from __future__ import annotations

import re
import threading

import chromadb

from sentinel.config import Settings, get_settings
from sentinel.rag.embeddings import get_embedder

_clients: dict[str, object] = {}
_lock = threading.Lock()
_versions: dict[str, int] = {}


def _client(settings: Settings):
    path = str(settings.chroma_dir)
    with _lock:
        if path not in _clients:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            _clients[path] = chromadb.PersistentClient(path=path)
        return _clients[path]


def collection_name(base: str, settings: Settings) -> str:
    """Collections are namespaced by embedding model so vectors are never mixed."""
    tag = "hash" if settings.embeddings_provider == "hash" else settings.embeddings_model.split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9_-]", "-", tag)[:40].strip("-_") or "emb"
    return f"{base}_{tag}"


def get_collection(base: str, settings: Settings | None = None):
    s = settings or get_settings()
    return _client(s).get_or_create_collection(
        name=collection_name(base, s),
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def _vkey(base: str, s: Settings) -> str:
    return f"{s.chroma_dir}:{collection_name(base, s)}"


def index_version(base: str, settings: Settings | None = None) -> int:
    s = settings or get_settings()
    return _versions.get(_vkey(base, s), 0)


def bump_version(base: str, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    k = _vkey(base, s)
    _versions[k] = _versions.get(k, 0) + 1


def add_texts(
    base: str,
    texts: list[str],
    ids: list[str],
    metadatas: list[dict],
    settings: Settings | None = None,
    embed_texts: list[str] | None = None,
) -> int:
    """Upsert texts into a collection (idempotent by id). `embed_texts` lets the embedding be
    computed on a contextualised version of the stored text."""
    s = settings or get_settings()
    if not texts:
        return 0
    emb = get_embedder(s).embed_documents(embed_texts or texts)
    get_collection(base, s).upsert(ids=ids, documents=texts, embeddings=emb, metadatas=metadatas)
    bump_version(base, s)
    return len(texts)


def reset_collection(base: str, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    try:
        _client(s).delete_collection(collection_name(base, s))
    except Exception:
        pass
    bump_version(base, s)
