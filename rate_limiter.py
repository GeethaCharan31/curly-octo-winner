"""
rate_limiter.py — a single, process-wide sliding-window limiter shared by
every LLM call in the pipeline.

Why this is needed: Stages 2/3/4 (agents.py) and Stage 5 (blender_codegen.py
via asset_stage.py) all run their per-item work on ThreadPoolExecutors
(MAX_ASSET_WORKERS, MAX_SECTION_WORKERS in config.py). Without a shared gate,
those worker threads would collectively fire far more than 15 calls/minute
at the API even though each *stage* looks sequential from the outside.

Usage: don't call llm.invoke(...) directly anywhere. Call
llm_utils.invoke_llm(llm, messages) instead — it acquires this limiter first.
"""

import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, period_sec: float = 60.0):
        self.max_calls = max_calls
        self.period_sec = period_sec
        self._calls = deque()       # monotonic timestamps of recent calls
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Blocks the calling thread until it's safe to make one more call."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period_sec:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                # Window is full — figure out how long until the oldest call
                # ages out, and sleep *outside* the lock so other threads
                # aren't blocked from checking/queuing themselves.
                sleep_for = self.period_sec - (now - self._calls[0]) + 0.05

            time.sleep(max(sleep_for, 0.05))


# ---------------------------------------------------------------------------
# One shared instance for the whole pipeline. All LLM calls (prose + code,
# across every stage) draw from this single budget, since they're hitting
# the same per-project/per-key quota on the Gemini Flash model.
# ---------------------------------------------------------------------------
from config import LLM_CALLS_PER_MINUTE  # noqa: E402  (import after class def is fine)

llm_rate_limiter = SlidingWindowRateLimiter(max_calls=LLM_CALLS_PER_MINUTE, period_sec=60.0)