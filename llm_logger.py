"""
llm_logger.py — persistent, thread-safe logging of every LLM call the pipeline makes.

Design:
  - One LLMCallLogger instance per course run, created in state.new_course_state()
    and threaded through CourseState (like everything else).
  - Every call to invoke_llm() logs exactly one record, success or failure,
    including retried-then-succeeded calls (attempt_count > 1 tells you it was flaky).
  - Records are appended to media/courses/<course_id>/state/llm_calls.jsonl as they
    happen — JSONL, not one big JSON array, so a crash mid-run doesn't lose earlier
    calls and concurrent ThreadPoolExecutor workers (Stages 2-5) can all append
    safely without a read-modify-write race.
  - A rollup (media/courses/<course_id>/state/llm_calls_summary.json) is rewritten
    after every record so you always have an up-to-date total without re-parsing
    the JSONL file.

This intentionally does NOT import anything from llm_utils.py, to avoid a circular
import (llm_utils imports this module). Callers pass already-extracted plain
strings — see invoke_llm() in llm_utils.py for the call pattern.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional


class LLMCallLogger:
    def __init__(self, course_id: str, log_dir: Path):
        self.course_id = course_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "llm_calls.jsonl"
        self.summary_path = self.log_dir / "llm_calls_summary.json"

        self._lock = threading.Lock()
        self._total_calls = 0
        self._failed_calls = 0
        self._retried_calls = 0          # calls where attempt_count > 1
        self._total_latency_sec = 0.0
        self._by_stage: dict = {}        # stage -> {"calls": n, "failed": n, "latency_sec": f}

    def log(
        self,
        *,
        stage: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        started_at: float,
        attempt_count: int,
        success: bool,
        error: Optional[str] = None,
        section_id: Optional[str] = None,
        asset_key: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        finished_at = time.time()
        latency_sec = round(finished_at - started_at, 3)

        record = {
            "course_id": self.course_id,
            "stage": stage,
            "section_id": section_id,
            "asset_key": asset_key,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": response_text,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_sec": latency_sec,
            "attempt_count": attempt_count,
            "success": success,
            "error": error,
            "extra": extra or {},
        }

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            self._total_calls += 1
            self._total_latency_sec += latency_sec
            if not success:
                self._failed_calls += 1
            if attempt_count > 1:
                self._retried_calls += 1

            bucket = self._by_stage.setdefault(stage, {"calls": 0, "failed": 0, "latency_sec": 0.0})
            bucket["calls"] += 1
            bucket["latency_sec"] = round(bucket["latency_sec"] + latency_sec, 2)
            if not success:
                bucket["failed"] += 1

            self._write_summary_locked()

    def _write_summary_locked(self) -> None:
        summary = {
            "course_id": self.course_id,
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "retried_calls": self._retried_calls,
            "total_llm_latency_sec": round(self._total_latency_sec, 2),
            "by_stage": self._by_stage,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2))

    def summary(self) -> dict:
        with self._lock:
            return {
                "course_id": self.course_id,
                "total_calls": self._total_calls,
                "failed_calls": self._failed_calls,
                "retried_calls": self._retried_calls,
                "total_llm_latency_sec": round(self._total_latency_sec, 2),
                "by_stage": dict(self._by_stage),
            }