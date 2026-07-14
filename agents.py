"""
agents.py — Stages 1-4 of the pipeline (all cheap LLM/JSON work, no rendering).

Each stage function takes the full CourseState and returns a new CourseState
with its key(s) filled in. Per-section work within a stage is parallelized
with a thread pool (LLM calls are I/O bound), but the stage as a whole is a
barrier: nothing in Stage N+1 starts until every section has finished Stage N.
This matches the "sequential barrier through asset generation" note in the
design doc.
"""

import time
import concurrent.futures as cf
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from config import llm_prose, MAX_SECTION_WORKERS, PROSE_MODEL
from state import CourseState, Section, NarrationSegment, AssetRequirement, DedupedAsset
from llm_utils import extract_text, parse_json_response, dedup_key, invoke_llm


# ---------------------------------------------------------------------------
# Shared constraint block — every downstream stage that describes a *visual*
# needs to know what Stage 5/6 can actually build: procedural bpy primitives
# only, no textures, no imported meshes, no on-screen text/diagrams (see
# ASSET_SYSTEM_PROMPT / SCENE_SYSTEM_PROMPT in blender_codegen.py). Without
# this, content/narration are free to describe things like "show the wiring
# diagram" or "display the torque curve on screen" that Stage 4/5 then has no
# way to actually build, and you only find out three stages later when the
# asset planner either hallucinates a workaround or the scene script silently
# has nothing to show. Centralized here so it can't drift out of sync between
# the prompts that use it.
# ---------------------------------------------------------------------------

BLENDER_VISUAL_CONSTRAINTS = """
VISUAL CONSTRAINTS (what the animation pipeline can actually build):
Every visual is a procedurally-built 3D object made of primitive shapes (cubes, cylinders,
spheres, cones) with realistic materials — there are no photographs, no imported/scanned
meshes, no textures, and no on-screen text, labels, diagrams, or charts. If something needs
to be shown, it must be shown as a physical 3D form: an assembled object, an exploded/cutaway
view, a comparison of two physical states side by side, or a camera move/animation over time
(e.g. a part rotating, a mechanism engaging, a cross-section revealing an interior).
Do NOT call for: screenshots, schematics, wiring/circuit diagrams, graphs/charts, text overlays,
UI mockups, or anything that is inherently 2D information rather than a 3D physical object or
process. If a concept is fundamentally non-visual (e.g. a policy, a number, a comparison of
costs), narrate it in words rather than inventing a visual for it — an on-screen abstract prop
or a steady establishing shot of the relevant physical equipment is a better fallback than
forcing a diagram.
"""


# ---------------------------------------------------------------------------
# Stage 1 — Course Draft
# ---------------------------------------------------------------------------

DRAFT_SYSTEM_PROMPT = """You are an instructional designer building a vocational course outline.

Given a TOPIC and a TARGET AUDIENCE, produce a course broken into 4-8 top-level sections that
build on each other in a logical teaching order (foundational -> applied).

Calibrate everything to the stated audience: their prior knowledge, vocabulary, and goal.
Do not assume knowledge the audience doesn't have.

Use consistent terminology and naming for recurring objects/components across sections (e.g.
always "battery pack", never switch to "battery module" or "cell block" for the same thing in a
later section) — later stages build a shared, reusable library of 3D assets keyed off these
names, and inconsistent naming causes the same physical object to get modeled twice.

Return ONLY valid JSON — no markdown fences, no preamble — in this exact schema:
[
  {"section_id": "s1", "title": "...", "summary": "2-3 sentences on what this section covers and why it matters", "learning_objective": "one sentence: what the learner should be able to DO after this section"},
  ...
]
section_id must be short, unique, and ordered (s1, s2, s3, ...).
"""

def course_draft_agent(state: CourseState) -> CourseState:
    print(f"\n[course_draft] topic='{state['topic']}' audience='{state['audience']}'")
    start = time.time()
    messages = [
        SystemMessage(content=DRAFT_SYSTEM_PROMPT),
        HumanMessage(content=f"TOPIC: {state['topic']}\nTARGET AUDIENCE: {state['audience']}"),
    ]
    response = invoke_llm(
        llm_prose, messages,
        logger=state["llm_logger"], stage="course_draft", model_name=PROSE_MODEL,
    )
    raw = extract_text(response.content)
    sections = parse_json_response(raw)

    if not isinstance(sections, list) or not sections:
        raise RuntimeError(f"course_draft_agent: could not parse a section list from LLM output: {raw[:300]}")
    for s in sections:
        assert "section_id" in s and "title" in s and "summary" in s, f"malformed section: {s}"
        s.setdefault("learning_objective", "")

    print(f"[course_draft] {len(sections)} sections in {time.time()-start:.2f}s")
    return {**state, "sections": sections}


# ---------------------------------------------------------------------------
# Stage 2 — Content generation (per section, parallel within the stage)
# ---------------------------------------------------------------------------

CONTENT_SYSTEM_PROMPT = """You are an expert vocational trainer writing the instructional script for
ONE section of a larger course. Calibrate strictly to the audience's stated background — do not
assume prior exposure they don't have, and use vocabulary they'd recognize from their trade.

Guidelines:
- Open by connecting this section to what the audience already knows.
- Explain the core concept step by step, concretely (this will become a narrated 3D animation,
  so favor content that can be shown, not just told).
- Call out anything that will need a visual explicitly in the text — but only visuals that are
  physically buildable per the constraints below.
- End with a one-line bridge to why the next topic matters.
- 150-350 words: enough to animate a 60-120s section, concise enough to stay focused.
""" + BLENDER_VISUAL_CONSTRAINTS

def _generate_section_content(topic: str, audience: str, section: Section, logger, model_name: str) -> str:
    messages = [
        SystemMessage(content=CONTENT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"COURSE TOPIC: {topic}\nAUDIENCE: {audience}\n\n"
            f"SECTION TITLE: {section['title']}\nSECTION SUMMARY: {section['summary']}\n\n"
            "Write the full instructional content for this section now."
        )),
    ]
    response = invoke_llm(
        llm_prose, messages,
        logger=logger, stage="content", model_name=model_name, section_id=section["section_id"],
    )
    return extract_text(response.content)


def content_stage(state: CourseState) -> CourseState:
    print(f"\n[content_stage] generating content for {len(state['sections'])} sections")
    start = time.time()
    content: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=MAX_SECTION_WORKERS) as ex:
        futures = {
            ex.submit(
                _generate_section_content, state["topic"], state["audience"], s,
                state["llm_logger"], PROSE_MODEL,
            ): s["section_id"]
            for s in state["sections"]
        }
        for fut in cf.as_completed(futures):
            sid = futures[fut]
            content[sid] = fut.result()
    print(f"[content_stage] done in {time.time()-start:.2f}s")
    return {**state, "content": content}


# ---------------------------------------------------------------------------
# Stage 3 — Narration strategy (per section, parallel within the stage)
# ---------------------------------------------------------------------------

NARRATION_SYSTEM_PROMPT = """You are a narration director for a 3D-animated instructional video.

Given the section's written content, break it into an ORDERED sequence of short narration
segments (roughly one segment per sentence-or-two of spoken narration — 5 to 12 segments).
For each segment, describe what should be visually happening on screen at that moment
(visual_cue) so an animator can synchronize the animation to the voiceover, and flag pacing
(emphasis).
""" + BLENDER_VISUAL_CONSTRAINTS + """
Return ONLY valid JSON — no markdown fences, no preamble:
[
  {"text": "...", "visual_cue": "short description of what's shown/animated during this line", "emphasis": "normal"},
  ...
]
emphasis must be one of: "normal", "slow", "highlight".
The concatenation of all "text" fields, in order, must read as natural, complete narration —
do not drop or paraphrase content from the section.
"""

def _generate_section_narration(section: Section, content: str, logger, model_name: str) -> List[NarrationSegment]:
    messages = [
        SystemMessage(content=NARRATION_SYSTEM_PROMPT),
        HumanMessage(content=f"SECTION TITLE: {section['title']}\n\nSECTION CONTENT:\n{content}"),
    ]
    response = invoke_llm(
        llm_prose, messages,
        logger=logger, stage="narration", model_name=model_name, section_id=section["section_id"],
    )
    raw = extract_text(response.content)
    segments = parse_json_response(raw)
    if not isinstance(segments, list) or not segments:
        raise RuntimeError(f"narration parse failed for {section['section_id']}: {raw[:300]}")
    for seg in segments:
        seg.setdefault("emphasis", "normal")
        assert "text" in seg and "visual_cue" in seg
    return segments


def narration_stage(state: CourseState) -> CourseState:
    print(f"\n[narration_stage] generating narration strategy for {len(state['sections'])} sections")
    start = time.time()
    narration: dict[str, list] = {}
    with cf.ThreadPoolExecutor(max_workers=MAX_SECTION_WORKERS) as ex:
        futures = {
            ex.submit(
                _generate_section_narration, s, state["content"][s["section_id"]],
                state["llm_logger"], PROSE_MODEL,
            ): s["section_id"]
            for s in state["sections"]
        }
        for fut in cf.as_completed(futures):
            sid = futures[fut]
            narration[sid] = fut.result()
    print(f"[narration_stage] done in {time.time()-start:.2f}s")
    return {**state, "narration": narration}


# ---------------------------------------------------------------------------
# Stage 4 — Asset requirement collection (union across ALL sections, then dedup)
# ---------------------------------------------------------------------------

ASSET_REQ_SYSTEM_PROMPT = """You are a 3D asset planner for an instructional animation.

Given a section's content and its narration/visual cues, list every distinct 3D asset needed
to render it, and what STATE/variant of that asset is needed (e.g. an "EV motor" might be
needed in both an "assembled" and an "exploded" state — those are different asset states).

Keep asset names generic and reusable across sections (e.g. "motor_rotor", not
"motor_rotor_for_section_3"), so identical needs across sections collapse naturally. Reuse the
exact same name and wording the course content used for a given component — do not introduce a
synonym — since the dedup step matches on exact normalized name+state, not meaning.
Prefer a small vocabulary of reusable assets over one-off named objects.

Every asset you list must be buildable as procedural geometry (primitives + bmesh) with a
Principled BSDF material — no diagrams, screens, text panels, or 2D props disguised as "assets".

Return ONLY valid JSON — no markdown fences, no preamble:
[
  {"asset": "motor_rotor", "state": "exploded"},
  {"asset": "battery_pack", "state": "cutaway"},
  ...
]
3-8 entries. Keep names lowercase, snake_case-friendly, generic nouns.
"""

def _extract_section_assets(section: Section, content: str, narration: List[NarrationSegment], logger, model_name: str) -> List[dict]:
    cues = "\n".join(f"- {seg['visual_cue']}" for seg in narration)
    messages = [
        SystemMessage(content=ASSET_REQ_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"SECTION TITLE: {section['title']}\n\nCONTENT:\n{content}\n\nVISUAL CUES:\n{cues}"
        )),
    ]
    response = invoke_llm(
        llm_prose, messages,
        logger=logger, stage="asset_requirements", model_name=model_name, section_id=section["section_id"],
    )
    raw = extract_text(response.content)
    reqs = parse_json_response(raw)
    if not isinstance(reqs, list):
        raise RuntimeError(f"asset requirement parse failed for {section['section_id']}: {raw[:300]}")
    for r in reqs:
        assert "asset" in r and "state" in r
    return reqs


def asset_requirements_stage(state: CourseState) -> CourseState:
    print(f"\n[asset_requirements_stage] extracting asset needs for {len(state['sections'])} sections")
    start = time.time()

    raw_requirements: List[AssetRequirement] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_SECTION_WORKERS) as ex:
        futures = {
            ex.submit(
                _extract_section_assets,
                s, state["content"][s["section_id"]], state["narration"][s["section_id"]],
                state["llm_logger"], PROSE_MODEL,
            ): s["section_id"]
            for s in state["sections"]
        }
        for fut in cf.as_completed(futures):
            sid = futures[fut]
            for r in fut.result():
                raw_requirements.append({"section_id": sid, "asset": r["asset"], "state": r["state"]})

    # --- Union + dedup by (asset, state) ---
    dedup_map: dict[str, DedupedAsset] = {}
    for req in raw_requirements:
        key = dedup_key(req["asset"], req["state"])
        if key not in dedup_map:
            dedup_map[key] = {
                "key": key, "asset": req["asset"], "state": req["state"], "used_by": [],
            }
        if req["section_id"] not in dedup_map[key]["used_by"]:
            dedup_map[key]["used_by"].append(req["section_id"])

    deduplicated_assets = list(dedup_map.values())
    print(
        f"[asset_requirements_stage] {len(raw_requirements)} raw requirements -> "
        f"{len(deduplicated_assets)} deduplicated assets in {time.time()-start:.2f}s"
    )
    return {**state, "asset_requirements": raw_requirements, "deduplicated_assets": deduplicated_assets}