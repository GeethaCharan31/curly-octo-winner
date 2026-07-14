"""
blender_runner.py — everything that shells out: Blender headless, gTTS, ffmpeg.

Kept separate from blender_codegen.py so the "what code do we generate" logic
and the "how do we actually execute it" logic don't get tangled.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

from gtts import gTTS

from config import BLENDER_BIN, BLENDER_TIMEOUT_SEC, RENDER_TAIL_BUFFER_SEC


# ---------------------------------------------------------------------------
# Blender subprocess execution
# ---------------------------------------------------------------------------

def run_blender_script(script_code: str, extra_args: List[str], script_save_path: Path) -> subprocess.CompletedProcess:
    """Write `script_code` to `script_save_path` and run it headless with
    `extra_args` passed after the `--` separator (retrievable in-script via
    sys.argv[-N:]).

    The script is always persisted at `script_save_path` — this used to be an
    anonymous tempfile that got deleted the moment the run succeeded, which
    meant a working render couldn't be reproduced, tweaked, or diffed against
    a later regeneration, and a failing one only survived by accident (the
    old code skipped the delete on failure). Callers pass a deliberate,
    content-addressed path (asset key or course/section id) so these scripts
    live permanently alongside the media they produced — see asset_stage.py
    and video_stage.py for where those paths come from.

    IMPORTANT: --python-exit-code is required. Without it, Blender's default
    behavior is to print a Python traceback to stderr but still exit 0 when
    the --python script raises an unhandled exception — so a buggy
    LLM-generated script would look like a "success" that just happened not
    to produce output. Passing --python-exit-code 1 makes Blender actually
    return that code on an unhandled exception, so failures surface as
    failures instead of silently-missing files three stages later.
    """
    script_save_path = Path(script_save_path)
    script_save_path.parent.mkdir(parents=True, exist_ok=True)
    script_save_path.write_text(script_code)

    cmd = [
        BLENDER_BIN, "--background",
        "--python-exit-code", "1",
        "--python", str(script_save_path),
        "--", *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT_SEC)

    if result.returncode != 0:
        raise RuntimeError(
            f"Blender script failed (exit {result.returncode}). Generated script kept at: {script_save_path}\n"
            f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
        )

    return result


def _missing_output_error(output_path: Path, result: subprocess.CompletedProcess) -> RuntimeError:
    """Even with --python-exit-code, keep this belt-and-suspenders: if a
    script somehow exits 0 without producing its file (e.g. exported to the
    wrong path), dump full stdout/stderr instead of a bare one-liner so the
    actual cause is visible instead of guesswork."""
    return RuntimeError(
        f"Blender exited 0 but {output_path} was not created.\n"
        f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
    )


def generate_asset_glb(asset_script_code: str, output_path: Path, script_save_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_blender_script(asset_script_code, extra_args=[str(output_path)], script_save_path=script_save_path)
    if not output_path.exists():
        raise _missing_output_error(output_path, result)
    return output_path


def render_section_silent_video(
    scene_script_code: str,
    output_video_path: Path,
    asset_manifest: Dict[str, str],
    cue_timeline: List[dict],
    script_save_path: Path,
) -> Path:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "asset_manifest.json"
        timeline_path = Path(tmpdir) / "cue_timeline.json"
        manifest_path.write_text(json.dumps(asset_manifest))
        timeline_path.write_text(json.dumps(cue_timeline))

        result = run_blender_script(
            scene_script_code,
            extra_args=[str(output_video_path), str(manifest_path), str(timeline_path)],
            script_save_path=script_save_path,
        )

    if not output_video_path.exists():
        raise _missing_output_error(output_video_path, result)
    return output_video_path


# ---------------------------------------------------------------------------
# Narration audio (gTTS) + timing extraction
# ---------------------------------------------------------------------------

def _ffprobe_duration_sec(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    return float(result.stdout.strip())


def synthesize_narration_segments(segments: List[dict], out_dir: Path) -> List[dict]:
    """
    Generate one gTTS clip per narration segment, measure its real duration via
    ffprobe, and return the segments annotated with start_sec/end_sec — this
    timeline is what drives the Blender animation cues (Stage 6 scene script)
    so visuals stay in sync with actual spoken length rather than a guess.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline = []
    cursor = 0.0

    for i, seg in enumerate(segments):
        clip_path = out_dir / f"seg_{i:03d}.mp3"
        gTTS(text=seg["text"], lang="en", tld="co.in").save(str(clip_path))
        duration = _ffprobe_duration_sec(clip_path)

        timeline.append({
            **seg,
            "audio_path": str(clip_path),
            "start_sec": round(cursor, 3),
            "end_sec": round(cursor + duration, 3),
        })
        cursor += duration

    return timeline


def concat_audio_clips(clip_paths: List[Path], output_path: Path) -> Path:
    """Concatenate narration clips into one track using ffmpeg's concat demuxer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_path = f.name

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
           "-c", "copy", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-1500:]}")
    return output_path


def mux_audio_into_video(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Combine the silent Blender render with the concatenated narration track."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-1500:]}")
    return output_path