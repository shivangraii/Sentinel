"""FastAPI service.

POST /research          run a job to completion (or until it pauses for plan approval)
POST /research/stream   same, streaming node-by-node progress as Server-Sent Events
POST /approve           resume a paused job (approve / reject / edit plan)
POST /approve/stream    resume, streaming
POST /ingest            (re)ingest data/raw into the knowledge base
POST /upload            upload a PDF/MD/TXT and ingest it
GET  /episodes          recent finished runs (episodic memory)
GET  /health
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sentinel import __version__
from sentinel.api.schemas import ApproveRequest, IngestRequest, ResearchRequest
from sentinel.config import get_settings
from sentinel.memory import episodic
from sentinel.rag.ingest import SUPPORTED, ensure_kb, ingest_directory
from sentinel.runner import resume_research, run_research, stream_research


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_kb()
    except Exception as e:  # never block startup on the KB (e.g. missing embedding model)
        print(f"[sentinel] KB bootstrap skipped: {e}")
    yield


app = FastAPI(title="Sentinel", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for ev in events:
            yield f"data: {json.dumps(ev, default=str)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok", "version": __version__, "llm": f"{s.llm_provider}:{s.llm_model}",
        "embeddings": s.embeddings_provider, "reranker": s.reranker_provider,
        "search": s.search_provider, "offline": s.offline,
    }


@app.post("/research")
def research(req: ResearchRequest) -> dict:
    try:
        return run_research(req.query, req.thread_id, req.require_approval)
    except RuntimeError as e:  # e.g. missing provider package / API key
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/research/stream")
def research_stream(req: ResearchRequest) -> StreamingResponse:
    gen = stream_research(req.query, req.thread_id, req.require_approval)
    return StreamingResponse(_sse(gen), media_type="text/event-stream")


@app.post("/approve")
def approve(req: ApproveRequest) -> dict:
    try:
        return resume_research(req.thread_id, req.approved, req.tasks)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/approve/stream")
def approve_stream(req: ApproveRequest) -> StreamingResponse:
    decision: dict = {"approved": req.approved}
    if req.tasks:
        decision["tasks"] = req.tasks
    gen = stream_research(thread_id=req.thread_id, resume=decision)
    return StreamingResponse(_sse(gen), media_type="text/event-stream")


@app.post("/ingest")
def ingest(req: IngestRequest | None = None) -> dict:
    return ingest_directory(reset=bool(req and req.reset))


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    from pathlib import Path

    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED))}")
    dest_dir = get_settings().raw_dir / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB)")
    (dest_dir / name).write_bytes(data)
    return {"saved": f"uploads/{name}", **ingest_directory()}


@app.get("/episodes")
def episodes(limit: int = 20) -> list[dict]:
    return episodic.list_episodes(limit)
