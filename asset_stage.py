"""
asset_stage.py — Stage 5: generate the deduplicated .glb asset library.

Caching is two-layered:
  1. In-process: within a single run, each dedup key is only ever built once
     (guaranteed by iterating `state["deduplicated_assets"]`, which is already
     unique per key).
  2. Cross-course: a manifest.json in ASSET_LIBRARY_DIR persists key -> glb
     path, so a second course that needs "motor_rotor__exploded" reuses the
     asset built for an earlier course instead of paying for another Blender
     job + LLM call. This is what makes the shared asset library actually pay
     off over time, not just within one course.

Generation of each (still-missing) asset is parallelized with a thread pool:
the LLM codegen call and the Blender subprocess are both mostly-idle-waiting
work, so a handful of assets can build concurrently.
"""

import json
import time
import concurrent.futures as cf
from pathlib import Path

from config import ASSET_LIBRARY_DIR, MAX_ASSET_WORKERS
from state import CourseState
from blender_codegen import generate_asset_script, repair_blender_script, ASSET_SYSTEM_PROMPT
from blender_runner import generate_asset_glb

MANIFEST_PATH = ASSET_LIBRARY_DIR / "manifest.json"

# 1 initial generation + up to 2 error-feedback repair attempts before an
# asset is given up on. Each attempt's script is kept (see script_save_path
# below) so a persistently-failing asset is still fully debuggable afterward.
MAX_ASSET_BUILD_ATTEMPTS = 3


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _build_one_asset(dedup_key: str, asset_name: str, asset_state: str, used_by: list, logger) -> Path:
    out_path = ASSET_LIBRARY_DIR / f"{dedup_key}.glb"
    context = f"used in course sections: {used_by}"
    scripts_dir = ASSET_LIBRARY_DIR / "scripts"

    script_code = None
    last_exc = None

    for attempt in range(1, MAX_ASSET_BUILD_ATTEMPTS + 1):
        script_save_path = scripts_dir / (
            f"{dedup_key}.py" if attempt == 1 else f"{dedup_key}__attempt{attempt}.py"
        )
        if attempt == 1:
            script_code = generate_asset_script(
                asset_name, asset_state, context, logger=logger, script_save_path=script_save_path,
            )
        else:
            print(f"[asset_generation_stage] '{dedup_key}' repairing after attempt {attempt - 1} failure")
            script_code = repair_blender_script(
                ASSET_SYSTEM_PROMPT.format(asset=asset_name, state=asset_state),
                script_code, str(last_exc),
                logger=logger, stage="asset_generation_repair", asset_key=dedup_key,
                script_save_path=script_save_path, attempt=attempt,
            )
        try:
            return generate_asset_glb(script_code, out_path, script_save_path=script_save_path)
        except Exception as exc:
            last_exc = exc
            print(f"[asset_generation_stage] '{dedup_key}' attempt {attempt}/{MAX_ASSET_BUILD_ATTEMPTS} failed: {exc}")

    raise last_exc


def asset_generation_stage(state: CourseState) -> CourseState:
    print(f"\n[asset_generation_stage] {len(state['deduplicated_assets'])} deduplicated assets to resolve")
    start = time.time()

    # manifest.json write/read is not thread-safe across processes; fine for a
    # single-process run. Swap for a real lock/db if you parallelize courses too.
    manifest = _load_manifest()
    asset_library: dict = {}
    to_build = []

    for asset in state["deduplicated_assets"]:
        key = asset["key"]
        cached_path = manifest.get(key)
        if cached_path and Path(cached_path).exists():
            asset_library[key] = cached_path
        else:
            to_build.append(asset)

    print(f"[asset_generation_stage] {len(asset_library)} cache hits, {len(to_build)} to build")

    if to_build:
        with cf.ThreadPoolExecutor(max_workers=MAX_ASSET_WORKERS) as ex:
            futures = {
                ex.submit(_build_one_asset, a["key"], a["asset"], a["state"], a["used_by"], state["llm_logger"]): a["key"]
                for a in to_build
            }
            for fut in cf.as_completed(futures):
                key = futures[fut]
                try:
                    path = fut.result()
                    asset_library[key] = str(path)
                    manifest[key] = str(path)
                except Exception as exc:
                    # Best-effort: a failed asset shouldn't sink the whole course.
                    # Sections referencing it will just skip that asset at render time.
                    print(f"[asset_generation_stage] FAILED building '{key}': {exc}")

        _save_manifest(manifest)

    print(f"[asset_generation_stage] done in {time.time()-start:.2f}s "
          f"({len(asset_library)}/{len(state['deduplicated_assets'])} assets available)")
    return {**state, "asset_library": asset_library}