"""Test environment: fully offline (fake LLM, hash embeddings, lexical reranker) with throwaway storage."""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="sentinel-tests-"))
os.environ.update(
    SENTINEL_OFFLINE="1",
    SENTINEL_STORAGE_DIR=str(_TMP / "storage"),
    SENTINEL_REPORTS_DIR=str(_TMP / "reports"),
    SENTINEL_TRACES_DIR=str(_TMP / "traces"),
    SENTINEL_REQUIRE_APPROVAL="false",
)

import pytest  # noqa: E402

from sentinel.config import ROOT, reload_settings  # noqa: E402

reload_settings()


@pytest.fixture(scope="session")
def kb():
    """Knowledge base ingested once per test session."""
    from sentinel.rag.ingest import ingest_directory

    return ingest_directory(reset=True)


@pytest.fixture
def sample_db() -> Path:
    return ROOT / "data" / "sample.db"
