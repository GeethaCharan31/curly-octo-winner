# course_pipeline

Implements the pipeline from the design doc: `(topic, audience) -> course draft
-> per-section content -> per-section narration strategy -> deduplicated
asset requirements -> deduplicated asset generation -> parallel video render`.

## Files

| File | Role |
|---|---|
| `config.py` | LLM clients (Gemini via `langchain-google-genai`), filesystem layout, Blender/render settings |
| `state.py` | `CourseState` TypedDict — the single object threaded through LangGraph |
| `llm_utils.py` | text extraction, tolerant JSON parsing, the `(asset, state) -> dedup key` normalizer |
| `agents.py` | Stages 1–4 (course draft, content, narration strategy, asset requirement extraction+dedup) — all cheap LLM/JSON, no rendering |
| `blender_codegen.py` | LLM prompts that generate Blender Python scripts — one for building a reusable asset (Stage 5), one for a section's scene+render (Stage 6) |
| `blender_runner.py` | Everything that shells out: `blender --background --python`, gTTS narration synthesis, ffmpeg concat/mux |
| `asset_stage.py` | Stage 5 — builds only the assets missing from the shared, cross-course `.glb` library (`media/assets/manifest.json`) |
| `video_stage.py` | Stage 6 — the only parallel/fan-out stage; per section: real gTTS timing → resolved asset manifest → scene script → render → mux |
| `graph.py` | Wires it all into a LangGraph `StateGraph`, with per-stage JSON checkpointing for resumability |
| `cli.py` | Entrypoint |

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
- **Best-effort degradation**: a failed asset doesn't kill the course (that
  section just renders without it); a failed section render doesn't kill the
  course either — it's recorded in `render_errors` and the CLI reports it at
  the end, same spirit as the reference file's video-generation try/except.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here
# blender, ffmpeg, ffprobe must be on PATH (you mentioned Blender headless is
# already installed)
```

## Run

```bash
python -m course_pipeline.cli \
  --topic "Electric Vehicle Fundamentals" \
  --audience "ITI graduates seeking EV technician roles, Class 10 pass, no prior EV exposure" \
  --course-id ev-fundamentals-iti
```

Output:
- `media/courses/ev-fundamentals-iti/state/*.json` — per-stage checkpoints + `final_summary.json`
- `media/courses/ev-fundamentals-iti/video/<section_id>/final.mp4` — final narrated videos
- `media/assets/*.glb` + `manifest.json` — the growing, shared asset library

## Realistic expectations / where you'll want to iterate

- **Blender codegen is the highest-risk step**, same as Manim was in your
  reference file — LLM-generated `bpy` scripts for arbitrary topics will
  sometimes fail (bad API usage, unrealistic proportions, degenerate meshes).
  `blender_runner.run_blender_script` surfaces the real stderr on failure so
  you can see exactly what broke; you'll likely want a retry-with-error-fed-
  back-to-the-LLM loop here before this is production-solid — the design doc
  explicitly scopes that kind of critic/review loop out of the MVU, so it's
  left as a clean extension point rather than built in.
- Rendering quality/detail is bounded by what an LLM can write in raw `bpy`
  primitives — good for schematic/exploded-view instructional style, not
  photorealism.
- `MAX_ASSET_WORKERS` / `MAX_SECTION_WORKERS` in `config.py` control
  concurrency; tune down if Blender headless instances compete too hard for
  CPU/RAM on your machine.