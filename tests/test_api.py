import json

import pytest
from fastapi.testclient import TestClient

from sentinel.api.main import app

Q = "Why does IMU-only dead reckoning drift?"


@pytest.fixture(scope="module")
def client(kb):
    with TestClient(app) as c:
        yield c


def _sse_events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["offline"] is True and r["llm"].startswith("fake")


def test_research_endpoint(client):
    r = client.post("/research", json={"query": Q})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done" and "## Sources" in body["report"]
    assert body["critique"]["score"] >= 0.8 and body["events"][-1]["node"] == "finalize"


def test_research_validation(client):
    assert client.post("/research", json={"query": "x"}).status_code == 422


def test_approval_flow(client):
    first = client.post("/research", json={"query": Q, "require_approval": True}).json()
    assert first["status"] == "awaiting_approval" and first["plan"]
    done = client.post("/approve", json={"thread_id": first["thread_id"], "approved": True}).json()
    assert done["status"] == "done" and "## Sources" in done["report"]


def test_stream_endpoint(client):
    r = client.post("/research/stream", json={"query": Q})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r)
    assert events[0]["type"] == "start" and events[-1]["type"] == "result"
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert "planner" in nodes and "critic" in nodes and nodes[-1] == "finalize"
    assert events[-1]["status"] == "done"


def test_stream_approval_roundtrip(client):
    events = _sse_events(client.post("/research/stream", json={"query": Q, "require_approval": True}))
    assert any(e["type"] == "interrupt" for e in events)
    final = events[-1]
    assert final["status"] == "awaiting_approval"
    resumed = _sse_events(client.post("/approve/stream", json={"thread_id": final["thread_id"], "approved": True}))
    assert resumed[-1]["type"] == "result" and resumed[-1]["status"] == "done"


def test_upload_and_ingest(client, tmp_path):
    bad = client.post("/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400
    ok = client.post("/upload", files={"file": ("note.md", b"# Test note\n\nThe zebra protocol uses seventeen flags.", "text/markdown")})
    assert ok.status_code == 200 and ok.json()["saved"] == "uploads/note.md"
    r = client.post("/research", json={"query": "How many flags does the zebra protocol use?"}).json()
    assert "seventeen" in r["report"]
    # cleanup uploaded file from the repo data dir
    from sentinel.config import get_settings

    (get_settings().raw_dir / "uploads" / "note.md").unlink()
    client.post("/ingest", json={"reset": True})


def test_episodes_recorded(client):
    eps = client.get("/episodes").json()
    assert eps and "query" in eps[0]


def test_public_demo_blocks_writes(client, monkeypatch):
    monkeypatch.setenv("SENTINEL_PUBLIC_DEMO", "1")
    assert client.get("/health").json()["public_demo"] is True
    assert client.post("/ingest", json={"reset": True}).status_code == 403
    assert client.post("/upload", files={"file": ("x.md", b"# hi")}).status_code == 403
    # reads / research still work
    assert client.get("/episodes").status_code == 200
