"""Central configuration: configs/settings.yaml  <  .env / environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.getenv("SENTINEL_HOME") or Path(__file__).resolve().parents[2])


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no python-dotenv dependency). Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if val and key not in os.environ:
            os.environ[key] = val


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # llm
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1
    ollama_base_url: str = "http://localhost:11434"
    # embeddings / reranker
    embeddings_provider: str = "sentence-transformers"
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    reranker_provider: str = "cross-encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    # search
    search_provider: str = "duckduckgo"
    search_max_results: int = 5
    # retrieval
    chunk_size: int = 180
    chunk_overlap: int = 40
    top_k_dense: int = 12
    top_k_bm25: int = 12
    top_k_final: int = 5
    rrf_k: int = 60
    # agent
    max_loops: int = 3
    max_tool_steps: int = 4
    max_tool_calls_per_run: int = 14
    max_runtime_s: int = 300
    critic_threshold: float = 0.8
    require_approval: bool = False
    # paths
    storage_dir: Path = field(default_factory=lambda: ROOT / "storage")
    raw_dir: Path = field(default_factory=lambda: ROOT / "data/raw")
    db_path: Path = field(default_factory=lambda: ROOT / "data/sample.db")
    reports_dir: Path = field(default_factory=lambda: ROOT / "data/reports")
    traces_dir: Path = field(default_factory=lambda: ROOT / "traces")
    offline: bool = False

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def checkpoint_db(self) -> Path:
        return self.storage_dir / "checkpoints.sqlite"

    def with_overrides(self, **kw) -> Settings:
        return replace(self, **kw)


def _p(base: Path, val: str) -> Path:
    path = Path(val)
    return path if path.is_absolute() else base / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    cfg: dict = {}
    yml = ROOT / "configs/settings.yaml"
    if yml.exists():
        cfg = yaml.safe_load(yml.read_text()) or {}
    llm, emb, rr = cfg.get("llm", {}), cfg.get("embeddings", {}), cfg.get("reranker", {})
    se, rt, ag, pa = (cfg.get(k, {}) for k in ("search", "retrieval", "agent", "paths"))
    env = os.environ.get

    s = Settings(
        llm_provider=env("SENTINEL_LLM_PROVIDER") or llm.get("provider", "gemini"),
        llm_model=env("SENTINEL_LLM_MODEL") or llm.get("model", "gemini-2.5-flash"),
        llm_temperature=float(llm.get("temperature", 0.1)),
        ollama_base_url=env("OLLAMA_BASE_URL") or "http://localhost:11434",
        embeddings_provider=env("SENTINEL_EMBEDDINGS") or emb.get("provider", "sentence-transformers"),
        embeddings_model=env("SENTINEL_EMBEDDING_MODEL") or emb.get("model", "BAAI/bge-small-en-v1.5"),
        reranker_provider=env("SENTINEL_RERANKER") or rr.get("provider", "cross-encoder"),
        reranker_model=env("SENTINEL_RERANKER_MODEL") or rr.get("model", "BAAI/bge-reranker-base"),
        search_provider=env("SENTINEL_SEARCH") or se.get("provider", "duckduckgo"),
        search_max_results=int(se.get("max_results", 5)),
        chunk_size=int(rt.get("chunk_size", 180)),
        chunk_overlap=int(rt.get("chunk_overlap", 40)),
        top_k_dense=int(rt.get("top_k_dense", 12)),
        top_k_bm25=int(rt.get("top_k_bm25", 12)),
        top_k_final=int(rt.get("top_k_final", 5)),
        rrf_k=int(rt.get("rrf_k", 60)),
        max_loops=int(env("SENTINEL_MAX_LOOPS") or ag.get("max_loops", 3)),
        max_tool_steps=int(ag.get("max_tool_steps", 4)),
        max_tool_calls_per_run=int(ag.get("max_tool_calls_per_run", 14)),
        max_runtime_s=int(ag.get("max_runtime_s", 300)),
        critic_threshold=float(ag.get("critic_threshold", 0.8)),
        require_approval=_truthy(env("SENTINEL_REQUIRE_APPROVAL"), bool(ag.get("require_approval", False))),
        storage_dir=_p(ROOT, env("SENTINEL_STORAGE_DIR") or pa.get("storage_dir", "storage")),
        raw_dir=_p(ROOT, pa.get("raw_dir", "data/raw")),
        db_path=_p(ROOT, env("SENTINEL_DB_PATH") or pa.get("db_path", "data/sample.db")),
        reports_dir=_p(ROOT, env("SENTINEL_REPORTS_DIR") or pa.get("reports_dir", "data/reports")),
        traces_dir=_p(ROOT, env("SENTINEL_TRACES_DIR") or pa.get("traces_dir", "traces")),
    )
    if _truthy(env("SENTINEL_OFFLINE")):
        s = s.with_overrides(
            llm_provider="fake",
            embeddings_provider="hash",
            reranker_provider="lexical",
            search_provider="none",
            offline=True,
        )
    return s


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
