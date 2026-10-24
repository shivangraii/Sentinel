"""Optional Hugging Face ZeroGPU support for the neural parts of retrieval (embedder + cross-encoder reranker).

Why this module exists
  * A ZeroGPU Space crashes at startup unless at least one `@spaces.GPU` function exists at import time, and a GPU
    is only attached while such a function runs. So there is ONE module-level decorated dispatcher (`_dispatch_gpu`);
    model code registers plain callables by key and the dispatcher looks them up.
  * Models live on CPU. Inside the GPU window they are moved to CUDA, used, and moved back, so CPU-only code paths
    (startup ingestion, tests, local runs, CPU Spaces) keep working unchanged.
  * If the GPU call fails for any reason (visitor out of quota, worker without the registry, ...) we fall back to CPU.
    A research run therefore never fails because of the GPU layer.

Outside Hugging Face (`spaces` not installed) everything here is a direct function call.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

try:  # preinstalled on Hugging Face Gradio Spaces; absent locally
    import spaces
except ImportError:  # pragma: no cover - exercised via tests with a stub
    spaces = None

GPU_SECONDS = 30  # short windows get higher queue priority on ZeroGPU

# key -> fn(payload, on_gpu) ; registered by SentenceTransformerEmbedder / CrossEncoderReranker
_REGISTRY: dict[str, Callable[[Any, bool], Any]] = {}
_cpu_only = False


def register(key: str, fn: Callable[[Any, bool], Any]) -> None:
    _REGISTRY[key] = fn


def enabled() -> bool:
    return spaces is not None and os.getenv("SENTINEL_USE_GPU", "1").strip().lower() not in {"0", "false", "no", "off"}


def _dispatch(key: str, payload: Any) -> Any:
    return _REGISTRY[key](payload, True)


# Must exist at import time (ZeroGPU scans for it during startup). Effect-free on non-ZeroGPU hardware.
_dispatch_gpu = spaces.GPU(duration=GPU_SECONDS)(_dispatch) if spaces is not None else _dispatch


@contextmanager
def cpu_only():
    """Force CPU for the duration of the block (e.g. startup ingestion, which has no visitor GPU quota)."""
    global _cpu_only
    prev, _cpu_only = _cpu_only, True
    try:
        yield
    finally:
        _cpu_only = prev


def run(key: str, payload: Any) -> Any:
    """Run a registered callable on the GPU when possible, otherwise on CPU."""
    fn = _REGISTRY[key]
    if _cpu_only or not enabled():
        return fn(payload, False)
    try:
        return _dispatch_gpu(key, payload)
    except Exception:  # quota exhausted, no GPU free, worker lacks registry, ...
        return fn(payload, False)


def with_device(model: Any, on_gpu: bool, fn: Callable[[], Any]) -> Any:
    """Call fn() with `model` temporarily on CUDA (only if a GPU is really available), then restore it to CPU."""
    if on_gpu:
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False
    else:
        use_cuda = False
    if not use_cuda:
        return fn()
    import torch

    prev_target = getattr(model, "_target_device", None)  # older sentence-transformers keep their own device field
    model.to("cuda")
    if prev_target is not None:
        model._target_device = torch.device("cuda")
    try:
        return fn()
    finally:
        model.to("cpu")
        if prev_target is not None:
            model._target_device = prev_target
