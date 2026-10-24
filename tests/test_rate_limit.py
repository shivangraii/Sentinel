from sentinel.guardrails.rate_limit import RateLimiter


def test_blocks_after_limit_and_reports_wait():
    rl = RateLimiter(max_calls=2, window_s=100)
    assert rl.acquire(now=0)[0] and rl.acquire(now=10)[0]
    ok, wait = rl.acquire(now=20)
    assert not ok and wait == 80  # oldest hit (t=0) expires at t=100


def test_window_slides_and_zero_disables():
    rl = RateLimiter(max_calls=1, window_s=100)
    assert rl.acquire(now=0)[0] and not rl.acquire(now=50)[0]
    assert rl.acquire(now=100)[0]  # first hit aged out
    off = RateLimiter(max_calls=0)
    assert all(off.acquire(now=i)[0] for i in range(1000))
