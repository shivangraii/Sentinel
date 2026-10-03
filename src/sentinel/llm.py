"""Provider-agnostic LLM wrapper with JSON parsing, usage tracking and an offline fake."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from sentinel.config import Settings, get_settings


class UsageTracker:
    """Rough per-process usage counters (tokens approximated as chars/4)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.in_tokens = 0
            self.out_tokens = 0
            self.seconds = 0.0

    def add(self, prompt: str, output: str, seconds: float) -> None:
        with self._lock:
            self.calls += 1
            self.in_tokens += max(1, len(prompt) // 4)
            self.out_tokens += max(1, len(output) // 4)
            self.seconds += seconds

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "approx_input_tokens": self.in_tokens,
                "approx_output_tokens": self.out_tokens,
                "llm_seconds": round(self.seconds, 2),
            }


usage = UsageTracker()


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array found in an LLM reply (handles ``` fences and chatter)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in LLM output: {text[:200]!r}")


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
        return "".join(parts)
    return str(content)


_INSTALL_HINT = {
    "gemini": 'pip install "sentinel[gemini]"',
    "groq": 'pip install "sentinel[groq]"',
    "ollama": 'pip install "sentinel[ollama]"',
    "anthropic": 'pip install "sentinel[anthropic]"',
}


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._model = None

    # -- model construction (lazy so offline mode needs no provider packages) --
    def _build(self):
        p, m, t = self.s.llm_provider, self.s.llm_model, self.s.llm_temperature
        try:
            if p == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=m, temperature=t, google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            if p == "groq":
                from langchain_groq import ChatGroq

                return ChatGroq(model=m, temperature=t)
            if p == "ollama":
                from langchain_ollama import ChatOllama

                return ChatOllama(model=m, temperature=t, base_url=self.s.ollama_base_url)
            if p == "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(model=m, temperature=t)
        except ImportError as e:
            raise RuntimeError(
                f"Provider '{p}' needs extra packages: {_INSTALL_HINT.get(p, '')} ({e})"
            ) from e
        raise ValueError(f"Unknown LLM provider: {p!r}")

    def complete(self, role: str, system: str, user: str) -> str:
        """Single chat completion. `role` is a label (planner, writer, ...) used for tracing/fake."""
        start = time.time()
        if self.s.llm_provider == "fake":
            from sentinel.fake_llm import fake_respond

            out = fake_respond(role, system, user)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            if self._model is None:
                self._model = self._build()
            msg = self._model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            out = _content_to_str(msg.content)
        usage.add(system + user, out, time.time() - start)
        return out

    def complete_json(self, role: str, system: str, user: str, retries: int = 1) -> Any:
        """Completion that must return JSON; on parse failure asks the model to repair once."""
        raw = self.complete(role, system, user)
        for attempt in range(retries + 1):
            try:
                return extract_json(raw)
            except ValueError:
                if attempt == retries:
                    raise
                raw = self.complete(
                    role,
                    system,
                    user
                    + "\n\nYour previous reply was not valid JSON. Reply with ONLY a valid JSON value, "
                    "no prose, no code fences.\nPrevious reply:\n"
                    + raw[:1500],
                )


_clients: dict[tuple, LLMClient] = {}


def get_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    key = (s.llm_provider, s.llm_model, s.ollama_base_url)
    if key not in _clients:
        _clients[key] = LLMClient(s)
    return _clients[key]
