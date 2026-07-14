"""
video_stage.py — Stage 6: the only fan-out stage, per the design doc. Each
section reads its own narration timeline plus the (already fully-built) shared
asset pool, so sections never block on each other or on further LLM planning
calls. This is why it can be a plain thread pool rather than another LangGraph
barrier node.

Per-section pipeline:
  1. Synthesize gTTS narration for each segment -> real per-segment durations
     (this is the ground truth timeline; we do NOT estimate from word count).
  2. Concatenate narration clips into one audio track.
  3. Resolve which of this section's required assets actually made it into
     asset_library (Stage 5 is best-effort; missing assets are dropped rather
     than failing the section).
  4. Generate the Blender scene script for this section against the real cue
     timeline + resolved asset manifest.
  5. Render silent video, then mux the narration track on top.
"""

import time
import concurrent.futures as cf
from pathlib import Path

from config import COURSES_DIR, MAX_SECTION_WORKERS
from state import CourseState
from blender_codegen import generate_section_scene_script, repair_blender_script, SCENE_SYSTEM_PROMPT
from blender_runner import (
    synthesize_narration_segments,
    concat_audio_clips,
    render_section_silent_video,
    mux_audio_into_video,
)
from llm_utils import dedup_key

# 1 initial generation + up to 2 error-feedback repair attempts before a
# section's render is given up on (mirrors asset_stage.py's MAX_ASSET_BUILD_ATTEMPTS).
MAX_SCENE_BUILD_ATTEMPTS = 3


def _section_asset_manifest(state: CourseState, section_id: str) -> dict:
    """dedup keys required by this section -> resolved glb path, dropping any
    asset that Stage 5 failed to build."""
    needed_keys = {
        dedup_key(r["asset"], r["state"])
        for r in state["asset_requirements"]
        if r["section_id"] == section_id
    }
    return {
        key: state["asset_library"][key]
        for key in needed_keys
        if key in state["asset_library"]
    }


def _render_one_section(state: CourseState, section: dict) -> str:
    sid = section["section_id"]
    section_dir = COURSES_DIR / state["course_id"]
    audio_dir = section_dir / "audio" / sid
    video_dir = section_dir / "video" / sid
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Narration audio + real timing
    segments = state["narration"][sid]
    timeline = synthesize_narration_segments(segments, audio_dir)
    narration_track = concat_audio_clips([seg["audio_path"] for seg in timeline], audio_dir / "narration.mp3")

    # 3. Resolve assets actually available for this section
    asset_manifest = _section_asset_manifest(state, sid)

    # 4-5. Scene script + render, generated against the *real* cue timeline.
    # Persisted right next to the video it renders. On a render failure, the
    # exact traceback is fed back to the code model for a repair attempt
    # rather than dropping the section outright (mirrors asset_stage.py).
    silent_path = video_dir / "silent.mp4"
    scene_script = None
    last_exc = None

    for attempt in range(1, MAX_SCENE_BUILD_ATTEMPTS + 1):
        script_save_path = video_dir / (
            "scene_script.py" if attempt == 1 else f"scene_script__attempt{attempt}.py"
        )
        if attempt == 1:
            scene_script = generate_section_scene_script(
                section_title=section["title"],
                section_summary=section["summary"],
                asset_names=list(asset_manifest.keys()),
                logger=state["llm_logger"],
                section_id=sid,
                script_save_path=script_save_path,
            )
        else:
            print(f"[video_stage] '{sid}' repairing scene script after attempt {attempt - 1} failure")
            scene_script = repair_blender_script(
                SCENE_SYSTEM_PROMPT, scene_script, str(last_exc),
                logger=state["llm_logger"], stage="video_scene_script_repair", section_id=sid,
                script_save_path=script_save_path, attempt=attempt,
            )
        try:
            render_section_silent_video(
                scene_script, silent_path, asset_manifest, timeline, script_save_path=script_save_path,
            )
            break
        except Exception as exc:
            last_exc = exc
            print(f"[video_stage] '{sid}' render attempt {attempt}/{MAX_SCENE_BUILD_ATTEMPTS} failed: {exc}")
            if attempt == MAX_SCENE_BUILD_ATTEMPTS:
                raise

    # 6. Mux narration onto the (now successfully rendered) silent video
    final_path = video_dir / "final.mp4"
    mux_audio_into_video(silent_path, narration_track, final_path)

    return str(final_path)


def video_stage(state: CourseState) -> CourseState:
    print(f"\n[video_stage] rendering {len(state['sections'])} section videos in parallel")
    start = time.time()

    video_paths: dict = {}
    render_errors: dict = {}

    with cf.ThreadPoolExecutor(max_workers=MAX_SECTION_WORKERS) as ex:
        futures = {
            ex.submit(_render_one_section, state, section): section["section_id"]
            for section in state["sections"]
        }
        for fut in cf.as_completed(futures):
            sid = futures[fut]
            try:
                video_paths[sid] = fut.result()
                print(f"[video_stage] '{sid}' rendered OK")
            except Exception as exc:
                render_errors[sid] = str(exc)
                print(f"[video_stage] '{sid}' FAILED: {exc}")

    print(f"[video_stage] done in {time.time()-start:.2f}s "
          f"({len(video_paths)} ok, {len(render_errors)} failed)")
    return {**state, "video_paths": video_paths, "render_errors": render_errors}