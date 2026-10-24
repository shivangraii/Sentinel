"""Tiny in-process sliding-window rate limiter: caps how many research runs a public demo starts per window,
so a shared API key cannot be drained by visitors. Process-wide on purpose (one demo, one key)."""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, window_s: float = 3600.0) -> None:
        self.max_calls = max_calls  # <= 0 disables the limit
        self.window_s = window_s
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, now: float | None = None) -> tuple[bool, float]:
        """Returns (allowed, seconds_until_next_slot)."""
        if self.max_calls <= 0:
            return True, 0.0
        now = time.time() if now is None else now
        with self._lock:
            while self._hits and now - self._hits[0] >= self.window_s:
                self._hits.popleft()
            if len(self._hits) >= self.max_calls:
                return False, self.window_s - (now - self._hits[0])
            self._hits.append(now)
            return True, 0.0
