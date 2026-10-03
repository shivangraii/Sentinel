from sentinel.agents.critic import cited_ids, extract_claims


def test_cited_ids_variants():
    assert cited_ids("Fact one [E1][E2]. Fact two [E3, E4].") == ["E1", "E2", "E3", "E4"]
    assert cited_ids("No citations here, just [a link] and [12].") == []


def test_extract_claims_skips_headings_short_and_limitations():
    draft = """# Title

## Executive summary
Swarms coordinate using local rules and shared goals [E1]. [E2]
Too short [E3].

## Findings
### Sub heading
- Boids use three rules for stable flocking behaviour [E4].
1. This numbered claim has enough words to count as one [E5][E6].
Is this a question with enough words to be skipped?

## Limitations
This section is skipped entirely even though it has many many words.
"""
    claims = extract_claims(draft)
    texts = [c["text"] for c in claims]
    assert len(claims) == 3
    assert claims[0]["ids"] == ["E1", "E2"]  # citation-only fragment merged into previous sentence
    assert claims[1]["ids"] == ["E4"] and claims[2]["ids"] == ["E5", "E6"]
    assert not any("Limitations" in t or "skipped" in t or "question" in t for t in texts)
