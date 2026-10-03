"""Load -> chunk -> embed -> store. Idempotent: re-ingesting the same files upserts by chunk id."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sentinel.config import Settings, get_settings
from sentinel.rag import store
from sentinel.rag.text import first_heading

KB = "kb"
SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Document:
    source: str
    text: str


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(raw_dir: Path) -> list[Document]:
    docs = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            text = read_file(p).strip()
            if text:
                docs.append(Document(source=str(p.relative_to(raw_dir)), text=text))
    return docs


_SENT = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=[-*#] )")


def chunk_text(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Sentence-aware sliding window. `size`/`overlap` are in words (approximate tokens)."""
    sents = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur and cur_words + w > size:
            chunks.append(" ".join(cur))
            keep: list[str] = []
            kw = 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if kw + pw > overlap:
                    break
                keep.insert(0, prev)
                kw += pw
            cur, cur_words = keep, kw
        cur.append(s)
        cur_words += w
    if cur:
        tail = " ".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def split_sections(text: str, fallback_title: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into (title, [(section_heading, body), ...]). Plain text -> one section."""
    title = first_heading(text) or fallback_title
    sections: list[tuple[str, str]] = []
    section, buf = "", []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((section, body))

    for line in text.splitlines():
        m = _HEAD.match(line)
        if m:
            flush()
            buf = []
            if len(m.group(1)) == 1 and not sections and not section:
                section = ""  # document title line
            else:
                section = m.group(2)
        else:
            buf.append(line)
    flush()
    return title, sections


def ingest_documents(docs: list[Document], settings: Settings | None = None, source_type: str = "file") -> dict:
    """Each chunk is stored as clean body text; the embedding is computed on
    '<title> > <section>\n<body>' (contextual chunk headers) to improve dense retrieval."""
    s = settings or get_settings()
    texts, embed_texts, ids, metas = [], [], [], []
    for d in docs:
        title, sections = split_sections(d.text, Path(d.source).stem)
        i = 0
        for section, body in sections:
            header = f"{title} > {section}" if section else title
            for c in chunk_text(body, s.chunk_size, s.chunk_overlap):
                texts.append(c)
                embed_texts.append(f"{header}\n{c}")
                ids.append(f"{d.source}::{i}")
                metas.append(
                    {"source": d.source, "chunk": i, "title": title, "section": section, "source_type": source_type}
                )
                i += 1
    n = store.add_texts(KB, texts, ids, metas, s, embed_texts=embed_texts)
    return {"documents": len(docs), "chunks": n}


def ingest_directory(raw_dir: Path | None = None, settings: Settings | None = None, reset: bool = False) -> dict:
    s = settings or get_settings()
    if reset:
        store.reset_collection(KB, s)
    docs = load_documents(raw_dir or s.raw_dir)
    result = ingest_documents(docs, s)
    result["total_in_store"] = store.get_collection(KB, s).count()
    return result


def ensure_kb(settings: Settings | None = None) -> int:
    """Ingest data/raw on first run if the knowledge base is empty. Returns chunk count."""
    s = settings or get_settings()
    col = store.get_collection(KB, s)
    if col.count() == 0 and s.raw_dir.exists():
        ingest_directory(settings=s)
    return col.count()
