"""
llm_utils.py — small helpers reused by every agent.
"""

import json
import re
import time
from typing import Union, Any, Optional

from rate_limiter import llm_rate_limiter
from llm_logger import LLMCallLogger

# Errors worth retrying: rate-limit / transient-server errors. Checked by
# substring on str(exc) rather than importing google.api_core exception
# classes directly, so this doesn't hard-depend on a specific google-genai
# SDK version's exception hierarchy.
_RETRYABLE_MARKERS = ("429", "rate limit", "resource has been exhausted",
                      "resourceexhausted", "503", "unavailable", "deadline exceeded")


def _extract_prompts(messages) -> tuple[str, str]:
    """Split a langchain message list into (system_prompt, user_prompt) plain
    strings for logging. Multiple HumanMessages (shouldn't normally happen
    here) are joined with a blank line."""
    system_parts, user_parts = [], []
    for m in messages:
        text = extract_text(m.content)
        if m.__class__.__name__ == "SystemMessage":
            system_parts.append(text)
        else:
            user_parts.append(text)
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def invoke_llm(
    llm,
    messages,
    max_retries: int = 5,
    *,
    logger: Optional[LLMCallLogger] = None,
    stage: Optional[str] = None,
    model_name: Optional[str] = None,
    section_id: Optional[str] = None,
    asset_key: Optional[str] = None,
    extra: Optional[dict] = None,
):
    """Every LLM call in the pipeline should go through this — never call
    llm.invoke(...) directly. This is the single choke point that (a) blocks
    on the shared rate limiter before every call, (b) retries with
    exponential backoff on rate-limit/transient errors, and (c) — if a
    `logger` is passed — records exactly one entry per call (including
    retried-then-succeeded calls) to the course's llm_calls.jsonl.

    `logger` is optional and defaults to None so this stays usable outside
    the course pipeline (tests, one-off scripts) without dragging a logger
    along. In agents.py/blender_codegen.py/asset_stage.py we always pass
    state["llm_logger"] plus stage/section_id/asset_key so every record is
    attributable to the exact section or asset it was generating.
    """
    started = time.time()
    last_exc = None
    attempt_count = 0
    model_label = model_name or getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
    system_prompt, user_prompt = (_extract_prompts(messages) if logger is not None else ("", ""))

    for attempt in range(max_retries):
        attempt_count = attempt + 1
        llm_rate_limiter.acquire()
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if not any(marker in msg for marker in _RETRYABLE_MARKERS):
                if logger is not None:
                    logger.log(
                        stage=stage or "unknown", model=model_label,
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        response_text="", started_at=started, attempt_count=attempt_count,
                        success=False, error=str(exc), section_id=section_id,
                        asset_key=asset_key, extra=extra,
                    )
                raise  # not a rate-limit/transient error — fail fast
            backoff = min(2 ** attempt * 2, 60)  # 2s, 4s, 8s, 16s, 32s (capped at 60s)
            print(f"[invoke_llm] retryable error (attempt {attempt + 1}/{max_retries}), "
                  f"backing off {backoff}s: {exc}")
            time.sleep(backoff)
            continue
        else:
            if logger is not None:
                logger.log(
                    stage=stage or "unknown", model=model_label,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    response_text=extract_text(response.content), started_at=started,
                    attempt_count=attempt_count, success=True, error=None,
                    section_id=section_id, asset_key=asset_key, extra=extra,
                )
            return response

    if logger is not None:
        logger.log(
            stage=stage or "unknown", model=model_label,
            system_prompt=system_prompt, user_prompt=user_prompt,
            response_text="", started_at=started, attempt_count=attempt_count,
            success=False, error=str(last_exc), section_id=section_id,
            asset_key=asset_key, extra=extra,
        )
    raise last_exc


def extract_text(content: Union[str, list]) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()


def strip_code_fences(raw: str) -> str:
    """Remove ``` / ```json / ```python fences an LLM added despite instructions."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    # drop first fence line and a trailing fence line if present
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_response(raw: str) -> Optional[Any]:
    """Best-effort JSON parse: strip fences, try direct parse, then fall back
    to extracting the first {...}/[...] block."""
    cleaned = strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def dedup_key(asset: str, state: str) -> str:
    """Normalize (asset, state) into a stable, filesystem-safe dedup key."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")
    return f"{norm(asset)}__{norm(state)}"