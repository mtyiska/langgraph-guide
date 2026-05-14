import sys
import time
import concurrent.futures
sys.path.append("../..")

from retry import retry_with_backoff

print("=== Experiment 1 — Retry with Exponential Backoff ===\n")

# ── Test 1: Fails twice, succeeds on third attempt ─────────────────────────────

attempt_log = []

def flaky_tool_fn(call_id: str = "default") -> str:
    attempt_log.append((call_id, time.perf_counter()))
    count = sum(1 for c, _ in attempt_log if c == call_id)
    if count < 3:
        raise IOError(f"Simulated transient failure (attempt {count})")
    return f"Success on attempt {count}"

wrapped_flaky = retry_with_backoff(
    max_attempts=3,
    base_delay=0.1,
    exceptions=(IOError,),
)(flaky_tool_fn)

print("Test 1: Fails twice, succeeds on third attempt")
attempt_log.clear()
t0 = time.perf_counter()
result = wrapped_flaky("t1")
duration = time.perf_counter() - t0

attempts = [(c, t - t0) for c, t in attempt_log if c == "t1"]
for i, (_, ts) in enumerate(attempts):
    print(f"  Attempt {i + 1}: t={ts:.3f}s")
print(f"  Result: {result}")
print(f"  Total duration: {duration:.3f}s (expected ~0.3s for two 0.1s + 0.2s delays)\n")

assert "Success" in result

# ── Test 2: Always fails — verify exception is re-raised ──────────────────────

def always_fails() -> str:
    raise IOError("Permanent failure")

wrapped_failing = retry_with_backoff(
    max_attempts=3,
    base_delay=0.05,
    exceptions=(IOError,),
)(always_fails)

print("Test 2: Always fails — verify exception propagates and timing is correct")
t0 = time.perf_counter()
try:
    wrapped_failing()
    assert False, "Should have raised"
except IOError as e:
    duration = time.perf_counter() - t0
    print(f"  Correctly raised IOError: {e}")
    print(f"  Total duration: {duration:.3f}s (expected ~0.15s for 0.05 + 0.10)\n")

assert duration < 1.0, "Retry loop waited too long"

# ── Test 3: Non-retryable exception propagates immediately ────────────────────

def raises_value_error() -> str:
    raise ValueError("This should NOT be retried")

wrapped_value_error = retry_with_backoff(
    max_attempts=3,
    base_delay=0.1,
    exceptions=(IOError,),   # ValueError not in this tuple
)(raises_value_error)

print("Test 3: Non-retryable exception propagates immediately")
t0 = time.perf_counter()
try:
    wrapped_value_error()
    assert False, "Should have raised"
except ValueError as e:
    duration = time.perf_counter() - t0
    print(f"  Correctly propagated ValueError immediately: {e}")
    print(f"  Duration: {duration:.4f}s (should be near 0 — no retry delay)\n")

assert duration < 0.05, "Non-retryable exception should propagate instantly"

# ── Test 4: Jitter prevents thundering herd ───────────────────────────────────

print("Test 4: Jitter spreads retry timestamps across concurrent callers")

call_timestamps = []

def concurrent_flaky(worker_id: int) -> str:
    for attempt in range(3):
        if attempt < 2:
            time.sleep(0.01)
            raise IOError(f"Worker {worker_id} attempt {attempt} failed")
        call_timestamps.append((worker_id, time.perf_counter()))
        return f"Worker {worker_id} succeeded"

# Without jitter
no_jitter_fn = retry_with_backoff(
    max_attempts=3, base_delay=0.05, exceptions=(IOError,), jitter=False
)(concurrent_flaky)

with_jitter_fn = retry_with_backoff(
    max_attempts=3, base_delay=0.05, exceptions=(IOError,), jitter=True
)(concurrent_flaky)

for label, fn in [("No jitter", no_jitter_fn), ("With jitter", with_jitter_fn)]:
    call_timestamps.clear()
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(fn, i) for i in range(5)]
        concurrent.futures.wait(futures)

    if call_timestamps:
        times = sorted(t - t_start for _, t in call_timestamps)
        spread = max(times) - min(times)
        print(f"  {label}: success times spread = {spread:.4f}s | times = {[f'{t:.3f}' for t in times]}")
    else:
        print(f"  {label}: no timestamps collected")

print("\nAll retry tests passed.")