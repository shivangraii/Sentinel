import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sentinel.config import get_settings
from sentinel.fake_llm import fake_respond
from sentinel.llm import LLMClient, extract_json, usage
from sentinel.runner import run_research


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here you go:\n{"action": "finish", "answer": "x"}\nHope that helps.', {"action": "finish", "answer": "x"}),
        ('Thought: ...\n["q1", "q2"]', ["q1", "q2"]),
        ('{"nested": {"b": [1, 2]}} trailing', {"nested": {"b": [1, 2]}}),
    ],
)
def test_extract_json_variants(text, expected):
    assert extract_json(text) == expected


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json at all")


def _client_with(responses):
    c = LLMClient(get_settings().with_overrides(llm_provider="ollama"))
    c._model = FakeListChatModel(responses=responses)  # bypasses provider packages
    return c


def test_real_path_complete_and_usage():
    usage.reset()
    c = _client_with(["hello world"])
    assert c.complete("writer", "sys", "user") == "hello world"
    assert usage.snapshot()["llm_calls"] == 1


def test_complete_json_repairs_once():
    c = _client_with(["this is not json", '{"ok": true}'])
    assert c.complete_json("planner", "sys", "user") == {"ok": True}
    c2 = _client_with(["nope", "still nope"])
    with pytest.raises(ValueError):
        c2.complete_json("planner", "sys", "user")


def test_graph_survives_chatty_markdown_wrapped_json(kb, monkeypatch):
    """Real models wrap JSON in prose and code fences; the whole graph must still work."""
    orig = fake_respond

    def chatty(role, system, user):
        out = orig(role, system, user)
        if role in ("planner", "researcher_step", "analyst_step", "critic_judge", "critic_followup", "rewriter"):
            return f"Sure, here is my answer:\n```json\n{out}\n```\nLet me know if you need more."
        return out

    monkeypatch.setattr("sentinel.fake_llm.fake_respond", chatty)
    res = run_research("Why does IMU-only dead reckoning drift?", require_approval=False)
    assert res["status"] == "done" and res["critique"]["score"] >= 0.8 and "## Sources" in res["report"]
    assert json.dumps(res["plan"])  # plan survived parsing
