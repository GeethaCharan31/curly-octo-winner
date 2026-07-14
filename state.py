"""
state.py — the single state object threaded through the LangGraph pipeline.

Matches the data contracts in the design doc: each stage reads what it needs
from previous stages and adds its own key. Nothing is mutated in place —
nodes return {**state, "new_key": value} so LangGraph's state merging stays
predictable.
"""

from typing import TypedDict, List, Dict, Any

from config import course_dir
from llm_logger import LLMCallLogger


class Section(TypedDict):
    section_id: str
    title: str
    summary: str


class NarrationSegment(TypedDict):
    text: str
    visual_cue: str       # short description of what should be happening on screen
    emphasis: str          # "normal" | "slow" | "highlight" — pacing hint for TTS + animation


class AssetRequirement(TypedDict):
    section_id: str
    asset: str
    state: str


class DedupedAsset(TypedDict):
    key: str               # f"{asset}__{state}", normalized — the dedup key
    asset: str
    state: str
    used_by: List[str]     # section_ids that reference this asset


class CourseState(TypedDict):
    # Inputs
    topic: str
    audience: str
    course_id: str

    # Stage 1
    sections: List[Section]

    # Stage 2
    content: Dict[str, str]                       # section_id -> full script/content

    # Stage 3
    narration: Dict[str, List[NarrationSegment]]   # section_id -> ordered segments

    # Stage 4
    asset_requirements: List[AssetRequirement]     # raw, pre-dedup, per section
    deduplicated_assets: List[DedupedAsset]        # union, deduplicated

    # Stage 5
    asset_library: Dict[str, str]                  # dedup key -> .glb filepath

    # Stage 6
    video_paths: Dict[str, str]                    # section_id -> final muxed mp4 path
    render_errors: Dict[str, str]                  # section_id -> error message (best-effort)

    # Cross-cutting: NOT persisted to any stage checkpoint (graph.py's
    # STAGE_OUTPUT_KEYS never lists this key), since it's a live object, not
    # data. It's threaded through state purely so every invoke_llm() call at
    # every stage/thread can log to the same course-scoped file without each
    # module having to import config/course_dir itself.
    llm_logger: Any


def new_course_state(topic: str, audience: str, course_id: str) -> CourseState:
    return CourseState(
        topic=topic,
        audience=audience,
        course_id=course_id,
        sections=[],
        content={},
        narration={},
        asset_requirements=[],
        deduplicated_assets=[],
        asset_library={},
        video_paths={},
        render_errors={},
        llm_logger=LLMCallLogger(course_id, course_dir(course_id) / "state"),
    )