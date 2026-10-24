"""ZeroGPU helper: direct call without `spaces`, GPU dispatch when present, CPU fallback on any GPU failure."""
import types

import pytest

from sentinel import gpu


@pytest.fixture
def registered():
    calls = []

    def fn(payload, on_gpu):
        calls.append(on_gpu)
        return f"{payload}:{'gpu' if on_gpu else 'cpu'}"

    gpu.register("t", fn)
    yield calls
    gpu._REGISTRY.pop("t", None)


def test_direct_call_without_spaces(registered, monkeypatch):
    monkeypatch.setattr(gpu, "spaces", None)
    assert gpu.run("t", "x") == "x:cpu" and registered == [False]


def test_dispatch_and_fallback_with_spaces(registered, monkeypatch):
    monkeypatch.setattr(gpu, "spaces", types.SimpleNamespace())
    monkeypatch.setattr(gpu, "_dispatch_gpu", gpu._dispatch)
    assert gpu.run("t", "x") == "x:gpu"  # GPU path used

    def boom(key, payload):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(gpu, "_dispatch_gpu", boom)
    assert gpu.run("t", "y") == "y:cpu"  # any GPU failure falls back to CPU, run never fails
    with gpu.cpu_only():
        monkeypatch.setattr(gpu, "_dispatch_gpu", gpu._dispatch)
        assert gpu.run("t", "z") == "z:cpu"  # cpu_only never touches the GPU path
