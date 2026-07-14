"""
graph.py — wires the 6 stages into a LangGraph StateGraph, matching the
"sequential barrier through asset generation, fan-out only at render" shape
from the design doc:

    course_draft -> content -> narration -> asset_requirements
                 -> asset_generation -> video (parallel)  -> END

Each node also checkpoints its output JSON to
media/courses/<course_id>/state/<stage>.json, so a crashed run can be
resumed from the last completed stage (see cli.py --resume) without
re-paying for earlier LLM calls or renders.
"""

import json
from pathlib import Path
from typing import Optional

from langgraph.graph import StateGraph, END

from config import course_dir
from state import CourseState
from agents import course_draft_agent, content_stage, narration_stage, asset_requirements_stage
from asset_stage import asset_generation_stage
from video_stage import video_stage

STAGE_ORDER = [
    "course_draft",
    "content",
    "narration",
    "asset_requirements",
    "asset_generation",
    "video",
]

# Which CourseState keys each stage is responsible for persisting.
STAGE_OUTPUT_KEYS = {
    "course_draft":        ["sections"],
    "content":             ["content"],
    "narration":           ["narration"],
    "asset_requirements":  ["asset_requirements", "deduplicated_assets"],
    "asset_generation":    ["asset_library"],
    "video":               ["video_paths", "render_errors"],
}


def _checkpoint(course_id: str, stage: str, state: CourseState) -> None:
    path = course_dir(course_id) / "state" / f"{stage}.json"
    payload = {k: state[k] for k in STAGE_OUTPUT_KEYS[stage]}
    path.write_text(json.dumps(payload, indent=2))


def _load_checkpoint(course_id: str, stage: str) -> Optional[dict]:
    path = course_dir(course_id) / "state" / f"{stage}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _wrap(stage_name: str, fn):
    """Wrap a stage function so it (a) skips work if a checkpoint already
    exists for this course+stage, and (b) writes one when it finishes."""
    def node(state: CourseState) -> CourseState:
        cached = _load_checkpoint(state["course_id"], stage_name)
        if cached is not None:
            print(f"[graph] '{stage_name}' — resuming from checkpoint, skipping re-generation")
            return {**state, **cached}
        new_state = fn(state)
        _checkpoint(state["course_id"], stage_name, new_state)
        return new_state
    return node


def build_graph():
    graph = StateGraph(CourseState)
    graph.add_node("course_draft",       _wrap("course_draft", course_draft_agent))
    graph.add_node("content",            _wrap("content", content_stage))
    graph.add_node("narration",          _wrap("narration", narration_stage))
    graph.add_node("asset_requirements", _wrap("asset_requirements", asset_requirements_stage))
    graph.add_node("asset_generation",   _wrap("asset_generation", asset_generation_stage))
    graph.add_node("video",              _wrap("video", video_stage))

    graph.set_entry_point("course_draft")
    graph.add_edge("course_draft",       "content")
    graph.add_edge("content",            "narration")
    graph.add_edge("narration",          "asset_requirements")
    graph.add_edge("asset_requirements", "asset_generation")
    graph.add_edge("asset_generation",   "video")
    graph.add_edge("video",              END)

    return graph.compile()