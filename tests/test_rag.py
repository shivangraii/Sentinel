from sentinel.rag.ingest import chunk_text, ingest_directory, split_sections
from sentinel.rag.reranker import LexicalReranker
from sentinel.rag.retriever import HybridRetriever, rrf_fuse


def test_chunk_text_respects_size_and_overlap():
    sents = [f"Sentence number {i} has exactly seven words." for i in range(40)]
    chunks = chunk_text(" ".join(sents), size=30, overlap=8)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 30 + 8 for c in chunks)
    # overlap: last sentence of a chunk is repeated at the start of the next
    assert chunks[0].split(". ")[-1].strip(". ") in chunks[1]


def test_chunk_text_short_input():
    assert chunk_text("Just one short sentence.") == ["Just one short sentence."]
    assert chunk_text("") == []


def test_split_sections_markdown():
    md = "# Title\n\nIntro text here.\n\n## Part A\n\nBody A.\n\n## Part B\n\nBody B."
    title, secs = split_sections(md, "fallback")
    assert title == "Title"
    assert secs == [("", "Intro text here."), ("Part A", "Body A."), ("Part B", "Body B.")]
    t2, s2 = split_sections("plain text only", "doc")
    assert t2 == "doc" and s2 == [("", "plain text only")]


def test_rrf_fuse_math():
    scores = rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60)
    assert scores["a"] == 1 / 61 + 1 / 62
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["c"] == 1 / 63
    assert list(scores)[0] in ("a", "b")


def test_ingest_is_idempotent(kb):
    assert kb["chunks"] >= 15 and kb["documents"] == 5
    again = ingest_directory()
    assert again["total_in_store"] == kb["total_in_store"]


def test_hybrid_retrieval_finds_right_source(kb):
    r = HybridRetriever()
    cases = {
        "Who created MAVLink?": "uav_communications.md",
        "Why does IMU-only dead reckoning drift?": "gps_denied_navigation.md",
        "How much smaller is an INT8 model than FP32?": "edge_ai_uav.md",
        "What are boids separation alignment cohesion rules?": "swarm_coordination.md",
        "Which flight log formats exist for PX4 and ArduPilot?": "mission_data_analytics.md",
    }
    for q, src in cases.items():
        hits = r.search(q, k=3)
        assert hits and hits[0].source == src, (q, [h.source for h in hits])


def test_retrieval_modes_and_multiquery(kb):
    r = HybridRetriever()
    for mode in ("dense", "bm25", "hybrid"):
        assert r.search("quantization", k=2, mode=mode, rerank=False)
    multi = r.search(["knowledge distillation", "student teacher model compression"], k=2)
    assert multi and multi[0].source == "edge_ai_uav.md"
    assert r.search("   ") == []


def test_metadata_filter(kb):
    r = HybridRetriever()
    hits = r.search("battery", k=5, where={"source": "edge_ai_uav.md"})
    assert hits and all(h.source == "edge_ai_uav.md" for h in hits)


def test_lexical_reranker_prefers_matching_passage():
    rr = LexicalReranker()
    s = rr.score("dead reckoning drift", ["cats are small animals", "dead reckoning drift grows over time"])
    assert s[1] > s[0]
