# course_pipeline

Implements the pipeline from the design doc: `(topic, audience) -> course draft -> per-section content -> per-section narration strategy -> deduplicated asset requirements -> deduplicated asset generation -> parallel video render`.

## Files


| File                 | Role                                                                                                                                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`          | LLM clients (Gemini via `langchain-google-genai`), filesystem layout, Blender/render settings                                                                                                                   |
| `state.py`           | `CourseState` TypedDict — the single object threaded through LangGraph, including the per-course `LLMCallLogger`                                                                                                |
| `llm_utils.py`       | text extraction, tolerant JSON parsing, the `(asset, state) -> dedup key` normalizer, and `invoke_llm` — the single choke point every LLM call goes through (rate limiting, retries, logging)                   |
| `llm_logger.py`      | thread-safe, append-only logging of every LLM call (full prompt, full response, timing, stage/section/asset, success/failure) to a per-course JSONL file                                                        |
| `agents.py`          | Stages 1–4 (course draft, content, narration strategy, asset requirement extraction+dedup) — all cheap LLM/JSON, no rendering                                                                                   |
| `blender_codegen.py` | LLM prompts that generate Blender Python scripts — one for building a reusable asset (Stage 5), one for a section's scene+render (Stage 6), plus the shared self-repair prompt used by both when a script fails |
| `blender_runner.py`  | Everything that shells out: `blender --background --python`, gTTS narration synthesis, ffmpeg concat/mux — and persists every generated script permanently instead of discarding it                             |
| `asset_stage.py`     | Stage 5 — builds only the assets missing from the shared, cross-course `.glb` library (`media/assets/manifest.json`), with a generate → run → repair-on-failure loop                                            |
| `video_stage.py`     | Stage 6 — the only parallel/fan-out stage; per section: real gTTS timing → resolved asset manifest → scene script → render (with the same repair loop) → mux                                                    |
| `graph.py`           | Wires it all into a LangGraph `StateGraph`, with per-stage JSON checkpointing for resumability                                                                                                                  |
| `cli.py`             | Entrypoint; folds the run's LLM usage summary into `final_summary.json`                                                                                                                                         |


## Design choices worth knowing about

- **Assets are `.glb` files, not one shared `.blend`.** Each asset-generation
script only has to know how to build *its own* object and export it —
it never needs to reason about a shared file's existing object graph. That
keeps the codegen prompt small and the script self-contained/re-runnable.
Section scene scripts just `import_scene.gltf` whatever they need. This is
also what makes the manifest cache trivial: a `.glb` on disk plus a JSON
line is a complete, portable unit of reuse.
- **Dedup key is `f"{normalized_asset}__{normalized_state}"`.** Built in
`llm_utils.dedup_key`, used identically in Stage 4 (union), Stage 5
(what to build / cache lookup), and Stage 6 (what a section is allowed to
reference).
- **The narration timeline is real audio, not an estimate.** Stage 6 runs
gTTS *before* generating the scene script, measures each clip's actual
duration with `ffprobe`, and only then asks the LLM to write animation
keyframes against that real `start_sec`/`end_sec` timeline. This avoids the
usual failure mode of animation drifting out of sync with narration.
Audio is muxed onto the silent Blender render afterward with ffmpeg rather
than relying on Blender's own audio mixdown, which is inconsistent across
headless builds.
- **Stages 1–5 are sequential barriers; Stage 6 is the only fan-out**,
matching the doc exactly. Within a barrier stage, per-section work still
runs on a thread pool (`MAX_SECTION_WORKERS`) since the LLM calls are
independent and I/O-bound — the barrier is about *stage-to-stage* ordering,
not banning within-stage concurrency.
- **Two-layer caching on assets**: within a run, `deduplicated_assets` is
already unique per key; across runs/courses, `asset_stage.py` checks
`media/assets/manifest.json` first, so a second course needing
`motor_rotor__exploded` reuses the first course's asset instead of paying
for another LLM+Blender job.
- **Resumability**: `graph.py` writes `media/courses/<id>/state/<stage>.json`
after every stage and skips regenerating a stage if its checkpoint already
exists. Delete the relevant file(s) to force a stage to re-run.
- **Best-effort degradation**: an asset that still fails after every repair
attempt doesn't kill the course (that section just renders without it); a
section that still fails after every repair attempt doesn't kill the course
either — it's recorded in `render_errors` and the CLI reports it at the end.
- **Content/narration prompts know the asset stage's real constraints.**
Stage 2/3 are explicitly told that Stage 5 can only build procedural
primitive geometry — no textures, no imported meshes, no on-screen
text/diagrams/charts — so they don't ask for visuals three stages later
that have no way to actually get built.
- **Every generated Blender script is persisted, not a deleted tempfile.**
Asset scripts live at `media/assets/scripts/<dedup_key>.py`, next to the
`.glb` they built; scene scripts live at
`media/courses/<id>/video/<section_id>/scene_script.py`, next to the video
they rendered. If a script needed a repair attempt, every attempt is kept
(`..._attempt2.py`, `..._attempt3.py`, ...) so a persistently-failing asset
or section is fully debuggable after the fact.
- **Every LLM call is logged.** `invoke_llm` (the single choke point all
stages call through) writes one record per call — full system+user prompt,
full response text, model, stage, section_id/asset_key, latency, attempt
count, success/error — to
`media/courses/<id>/state/llm_calls.jsonl`, with a live-updated
`llm_calls_summary.json` alongside it. `cli.py` folds that summary into
`final_summary.json` at the end of a run.
- **Failed Blender scripts self-repair.** If a generated asset or scene
script fails to run, the exact traceback is fed back to the code model
(`blender_codegen.repair_blender_script`) for up to 2 repair attempts
before that asset/section is given up on, rather than dropping it on the
first hallucinated API call. This is what the persisted per-attempt scripts
above are tracking.

## Setup

```bash
pip install -r requirements.txt
add GOOGLE_API_KEY=your_key_here in keys.py
# blender, ffmpeg, ffprobe must be on PATH (you mentioned Blender headless is
# already installed)
```

## Run

```bash
python cli.py --topic "Electric Vehicle Fundamentals"               --audience "ITI graduates seeking EV technician roles, Class 10 pass, no prior EV exposure"               --course-id ev-fundamentals-iti
```

Output:

- `media/courses/ev-fundamentals-iti/state/*.json` — per-stage checkpoints, `llm_calls.jsonl` + `llm_calls_summary.json`, and `final_summary.json`
- `media/courses/ev-fundamentals-iti/video/<section_id>/final.mp4` — final narrated videos
- `media/courses/ev-fundamentals-iti/video/<section_id>/scene_script.py` (+ `scene_script__attemptN.py` if it needed repair) — the exact script that rendered each video
- `media/assets/*.glb` + `manifest.json` — the growing, shared asset library
- `media/assets/scripts/*.py` — the exact script that built each cached asset

## Debugging a failed or odd-looking asset/section

1. Check `render_errors` in `final_summary.json` (or the CLI's end-of-run
  printout) for which section(s) failed.
2. Open `llm_calls.jsonl` and filter by `section_id`/`asset_key` to see every
  prompt and response involved, including repair attempts (`stage` will be
   `asset_generation_repair` or `video_scene_script_repair`).
3. The scripts themselves (`scene_script*.py` / `media/assets/scripts/*.py`)
  are runnable directly: `blender --background --python-exit-code 1  --python <script>.py -- <the same args the pipeline passed>` — check the
   corresponding `llm_calls.jsonl` record's `extra.script_save_path` entry if
   you're not sure which record produced which file.

## Realistic expectations / where you'll want to iterate

- **Blender codegen is still the highest-risk step.** LLM-generated `bpy`
scripts for arbitrary topics will sometimes fail (bad API usage,
unrealistic proportions, degenerate meshes) — the self-repair loop resolves
a good share of these automatically (bad kwargs, removed/renamed
attributes) but won't catch everything, particularly failures that are
semantically wrong rather than an outright error (a script that runs fine
but produces a badly-proportioned or visually confusing asset).
- Rendering quality/detail is bounded by what an LLM can write in raw `bpy`
primitives — good for schematic/exploded-view instructional style, not
photorealism.
- `MAX_ASSET_WORKERS` / `MAX_SECTION_WORKERS` in `config.py` control
concurrency; tune down if Blender headless instances compete too hard for
CPU/RAM on your machine. `MAX_ASSET_BUILD_ATTEMPTS` (`asset_stage.py`) and
`MAX_SCENE_BUILD_ATTEMPTS` (`video_stage.py`) control how many repair
attempts a failing script gets before it's given up on.
- Content/narration/asset-requirement stages currently abort the *entire*
stage if any single section's LLM call raises (unlike `video_stage`, which
degrades per-section). Combined with stage-level checkpointing, one bad
section mid-stage means redoing the whole stage's LLM calls on retry. Worth
applying the same per-item try/except + caching pattern `asset_stage.py`
already uses for `.glb`s if this becomes a real cost at your `LLM_CALLS_PER_MINUTE`
budget.

