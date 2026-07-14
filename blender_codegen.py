"""
blender_codegen.py — turns LLM output into runnable `blender --background --python <script>` files.

1. Asset script  (Stage 5, once per deduplicated asset)
   Procedurally builds ONE object/group at the world origin and exports it to a
   .glb.

2. Section scene script (Stage 6, once per section)
   Imports the section's required .glb assets from the shared library, arranges/
   animates them in sync with the narration cue timeline, and renders a SILENT
   video (audio is muxed on afterward with ffmpeg, since Blender's headless
   audio mixdown is unreliable across builds).
"""

import time
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import llm_code, RENDER_FPS, RENDER_RES_X, RENDER_RES_Y, RENDER_ENGINE, RENDER_SAMPLES, RENDER_VIEW_TRANSFORM, CODE_MODEL
from llm_utils import extract_text, strip_code_fences, invoke_llm


# ---------------------------------------------------------------------------
# Asset script generation
# ---------------------------------------------------------------------------

ASSET_SYSTEM_PROMPT = """You are an expert Blender Python (bpy) developer building a single reusable
3D asset for a REALISTIC instructional animation, headless via `blender --background --python`.
The goal is photorealism, not stylised/toy CGI — every material and shape decision should serve that.

You are generating the asset "{asset}" in its "{state}" state/variant.

CRITICAL OUTPUT CONTRACT:
1. The script receives its output path as the LAST command-line argument after `--`,
   i.e. read it with: `output_path = sys.argv[-1]`.
2. Build the asset procedurally using bpy primitives (cubes, cylinders, spheres, cones) and
   bmesh/modifiers — do NOT reference any external file (no image textures, no imported meshes).
3. Center the finished asset's origin at the world origin (0, 0, 0), with a sensible default
   scale (roughly 1-3 Blender units across its largest dimension) so it can be scaled/placed
   consistently when imported into a scene later.
4. Group every object belonging to this asset under a single empty/parent named exactly
   "{asset}" so downstream scripts can find and manipulate it as one unit.
5. Respect real-world proportions: reason about the actual object's dimensions and part
   relationships before placing primitives (e.g. a gripper's jaws must be sized to plausibly
   close around what it grips; a wheel's hub/spoke/rim thickness ratios should look manufactured,
   not arbitrary). Do not guess proportions carelessly.
6. Materials: use the Principled BSDF exclusively, with values appropriate to the REAL material
   being represented (brushed/machined metal: metallic=0.9, roughness=0.3-0.4; painted
   metal/plastic: metallic=0.0, roughness=0.25-0.35; rubber: metallic=0.0, roughness=0.85-0.95;
   unfinished wood: metallic=0.0, roughness=0.5-0.6). Build materials EXACTLY like this — do not
   deviate from this node/link API, it is version-critical for Blender 4.x:
       mat = bpy.data.materials.new(name="MatName")
       mat.use_nodes = True
       nodes = mat.node_tree.nodes
       links = mat.node_tree.links
       bsdf = nodes.get("Principled BSDF")
       bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
       bsdf.inputs["Metallic"].default_value = 0.0
       bsdf.inputs["Roughness"].default_value = 0.3
       noise = nodes.new("ShaderNodeTexNoise")
       noise.inputs["Scale"].default_value = 8.0
       map_range = nodes.new("ShaderNodeMapRange")
       map_range.inputs["To Min"].default_value = 0.25   # keep roughness variation subtle
       map_range.inputs["To Max"].default_value = 0.4
       links.new(noise.outputs["Fac"], map_range.inputs["Value"])
       links.new(map_range.outputs["Result"], bsdf.inputs["Roughness"])
       obj.data.materials.append(mat)
   NEVER call `nodes.link(...)` (nodes has no `link` method) — connections always go through
   `mat.node_tree.links.new(output_socket, input_socket)`.
   For a "transparent"/"glass"/"see-through" state variant: do NOT set `mat.shadow_method` or
   `mat.blend_method` — both attributes were REMOVED from Blender's Material API in 4.2+
   (EEVEE Next handles this differently and referencing either raises AttributeError). Use:
       mat.surface_render_method = 'BLENDED'
       bsdf.inputs["Alpha"].default_value = 0.25   # 0.15-0.35 reads as clearly see-through
   This alpha-blend approach is reliable across 4.x point releases. True refractive glass (via
   the "Transmission Weight" input plus raytraced refraction) looks better but its exact API
   has moved across recent Blender versions — prefer the alpha-blend approach above rather than
   guess at refraction attribute names.
7. Add a Bevel modifier to every hard-surface mesh object, with the bevel WIDTH scaled to that
   object's own size (~1-2% of its largest local dimension, computed from its own geometry —
   do NOT hardcode the same absolute width across differently-sized parts, e.g. a bolt head and
   a housing panel need different bevel widths even though both use 2-3 segments and
   harden_normals=True). Apply smooth shading EXACTLY like this:
       dims = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
       bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
       bevel.width = max(dims * 0.015, 0.002)   # scale to the object, with a sane floor
       bevel.segments = 3
       bevel.harden_normals = True
       bpy.context.view_layer.objects.active = obj
       obj.select_set(True)
       bpy.ops.object.shade_smooth()
       obj.select_set(False)
   NEVER use `obj.data.use_auto_smooth` (removed in Blender 4.1+) or call `obj.shade_smooth()` as
   a method — it is an operator: `bpy.ops.object.shade_smooth()`, and requires the object to be
   active and selected first, exactly as shown above. `harden_normals=True` on the Bevel modifier
   is sufficient shading correction — no separate auto-smooth step is needed.
8. If you use bmesh to build custom geometry, ONLY use these bmesh.ops calls, with exactly
   these keyword arguments (Blender 4.x verified signatures — do not invent others):
       bmesh.ops.create_circle(bm, cap_ends=True, radius=1.0, segments=32)   # NOT "diameter"
       bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=1.0, radius2=0.0, depth=1.0)
       bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
       bmesh.ops.translate(bm, verts=[...], vec=(0,0,0))
   There is NO `bmesh.ops.create_face`. To create a face from existing verts, call
   `bm.faces.new([v1, v2, v3, v4])` directly (a method on bm.faces, not a bmesh.ops function).
   Call `bm.faces.ensure_lookup_table()` (and `.verts`/`.edges` as needed) immediately after
   any operation that adds or removes geometry, and before indexing into bm.faces[]/verts[]/
   edges[] by number.
9. For ANY part with repeating teeth, cogs, splines, or ribs (gears, sprockets, heat-sink fins,
   knurling) — do NOT hand-build this with bmesh vertex/face bookkeeping, it is the single
   most common source of script failures. Build it this way instead:
       # 1. Build the toothed body as a cylinder.
       bpy.ops.mesh.primitive_cylinder_add(radius=outer_radius, depth=thickness, vertices=64)
       body = bpy.context.active_object

       # 2. Build ONE tooth as a small cube, positioned at the rim.
       bpy.ops.mesh.primitive_cube_add(size=1.0)
       tooth = bpy.context.active_object
       tooth.scale = (tooth_w, tooth_w, thickness / 2)
       tooth.location = (outer_radius * 0.98, 0, 0)
       bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

       # 3. Array the tooth around the axis using an Empty as the offset object.
       axis_empty = bpy.data.objects.new("GearToothAxis", None)
       bpy.context.collection.objects.link(axis_empty)
       axis_empty.rotation_euler = (0, 0, 2 * math.pi / num_teeth)
       arr = tooth.modifiers.new("ToothArray", 'ARRAY')
       arr.count = num_teeth
       arr.use_relative_offset = False
       arr.use_object_offset = True
       arr.offset_object = axis_empty
       bpy.context.view_layer.objects.active = tooth
       bpy.ops.object.modifier_apply(modifier="ToothArray")

       # 4. Boolean-union the teeth onto the body, then clean up the helper objects.
       boolean = body.modifiers.new("TeethUnion", 'BOOLEAN')
       boolean.operation = 'UNION'
       boolean.object = tooth
       bpy.context.view_layer.objects.active = body
       bpy.ops.object.modifier_apply(modifier="TeethUnion")
       bpy.data.objects.remove(tooth, do_unlink=True)
       bpy.data.objects.remove(axis_empty, do_unlink=True)
   This produces a correct gear/sprocket with zero manual vertex math and no bmesh calls at all.
10. At the very end, export ONLY the objects you created to glTF:
       bpy.ops.object.select_all(action='DESELECT')
       for obj in bpy.data.objects:
           obj.select_set(True)
       bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)

CRITICAL BPY RULES:
1. Modern Blender Python API (4.x): use bpy.ops.mesh.primitive_*_add(), obj.rotation_euler,
   obj.location, obj.scale — all as mathutils.Vector or plain tuples of 3 floats.
2. No LaTeX, no external addons beyond the built-in glTF exporter (io_scene_gltf2, already
   enabled in standard Blender builds), no image texture files of any kind.
3. Wrap the whole build in a function `def build():` and call it at module level, so errors
   surface with a clean traceback.
4. Import at least: `import bpy`, `import sys`, `import math`.

Return ONLY raw executable Python. No markdown fences, no preamble. Start immediately with imports.
"""

def generate_asset_script(
    asset: str, asset_state: str, used_by_context: str,
    *, logger=None, section_id: Optional[str] = None, script_save_path: Optional[Path] = None,
) -> str:
    system_prompt = ASSET_SYSTEM_PROMPT.format(asset=asset, state=asset_state)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Build the asset now. Context on how it's used in the course "
            f"(for level of detail/shape guidance only — do not add scene dressing): "
            f"{used_by_context}"
        )),
    ]
    response = invoke_llm(
        llm_code, messages,
        logger=logger, stage="asset_generation", model_name=CODE_MODEL,
        asset_key=f"{asset}__{asset_state}",
        extra={"script_save_path": str(script_save_path)} if script_save_path else None,
    )
    return strip_code_fences(extract_text(response.content))


# ---------------------------------------------------------------------------
# Section scene/render script generation
# ---------------------------------------------------------------------------

SCENE_SYSTEM_PROMPT = f"""You are an expert Blender Python (bpy) developer building ONE section of a
REALISTIC instructional 3D animation, headless via `blender --background --python`. Every lighting,
camera, and render decision should push toward photorealism, not a flat/stylised CG look.

CRITICAL OUTPUT CONTRACT:
1. The script receives, after `--`, these command-line arguments in order via sys.argv[-3:]:
   [0] output_video_path  (render destination, .mp4)
   [1] asset_manifest_path (a JSON file: {{"asset_name": "/abs/path/to/asset.glb", ...}})
   [2] cue_timeline_path   (a JSON file: a list of
       {{"text": "...", "visual_cue": "...", "emphasis": "...", "start_sec": 0.0, "end_sec": 4.2}})
   Read them with:
       import sys, json
       output_video_path, asset_manifest_path, cue_timeline_path = sys.argv[-3:]
       asset_manifest = json.load(open(asset_manifest_path))
       cue_timeline = json.load(open(cue_timeline_path))
2. Clear the default scene at the start (same as asset scripts).
3. Set up the world (a fresh/imported scene may have scene.world = None — never assume it exists):
       scene = bpy.context.scene
       if scene.world is None:
           scene.world = bpy.data.worlds.new("World")
       scene.world.use_nodes = True
       bg_node = scene.world.node_tree.nodes.get("Background")
       if bg_node is None:
           bg_node = scene.world.node_tree.nodes.new("ShaderNodeBackground")
       bg_node.inputs[0].default_value = (0.05, 0.05, 0.05, 1)   # low-energy ambient fill
       bg_node.inputs[1].default_value = 1.0   # strength
4. Set fps = {RENDER_FPS}. For each cue_timeline entry, convert start_sec/end_sec to frames
   (frame = round(seconds * fps)) and keyframe camera/object movement, visibility, or highlight
   (e.g. emission strength pulse) so the requested visual_cue is happening on screen during that
   exact frame range. "emphasis": "slow" cues should hold the camera/motion steady rather than
   rushing; "highlight" cues should visually emphasize the relevant object (e.g. a brief scale
   pulse or color flash), not move the camera erratically. Ease all keyframes (use 'BEZIER'
   interpolation with eased handles, not 'LINEAR') — linear motion reads as robotic/fake.
5. Lighting — use a 3-point setup, not a single light:
     - Key: Area light, higher energy, ~5000-5500K, 30-45 degrees off-axis from camera
     - Fill: Area light, ~25-35% of key's energy, cooler or neutral, opposite side of key
     - Rim/back: Area or Sun light at a grazing angle behind the subject, to separate it from
       the background
   Give every light a nonzero size/shadow_soft_size so shadows are soft, not razor-edged.
   Set world background to a subtle gradient or low-energy environment color (not pure black)
   so shadows have ambient fill and don't crush to solid black.
6. Camera: realistic focal length (scene.camera.data.lens between 35 and 50mm), positioned at a
   slight three-quarter angle by default (avoid dead-center/orthographic-looking framing unless
   a cue specifically calls for straight-on). Enable depth of field:
       cam.data.dof.use_dof = True
       cam.data.dof.aperture_fstop = 2.8   # 2.0-4.0 range; lower = more background blur
       cam.data.dof.focus_object = <the main subject's empty>
7. Render settings (set exactly):
       scene.render.engine = '{RENDER_ENGINE}'
       scene.eevee.taa_render_samples = {RENDER_SAMPLES}
       scene.eevee.use_raytracing = True
       scene.eevee.use_shadows = True
       scene.view_settings.view_transform = '{RENDER_VIEW_TRANSFORM}'
       scene.render.resolution_x = {RENDER_RES_X}
       scene.render.resolution_y = {RENDER_RES_Y}
       scene.render.fps = {RENDER_FPS}
       scene.frame_start = 1
       scene.frame_end = <last cue's end frame, plus a small buffer>
       scene.render.filepath = output_video_path
       scene.render.image_settings.file_format = 'FFMPEG'
       scene.render.ffmpeg.format = 'MPEG4'
       scene.render.ffmpeg.codec = 'H264'
       scene.render.ffmpeg.audio_codec = 'NONE'
8. Render with: bpy.ops.render.render(animation=True, write_still=False)
9. If an asset referenced by a visual_cue is missing from asset_manifest (Stage 5 can drop
   assets that failed to build), do not fail the script — silently skip staging that asset and
   keep going with whatever is available, so the section still renders something rather than
   erroring out entirely.

CRITICAL BPY RULES:
1. Modern Blender Python API (4.x) syntax only.
2. NO external textures, NO LaTeX/Tex objects, text on screen only via bpy.data.curves.new(type='FONT')
   if absolutely needed (prefer pure 3D staging over on-screen text — the narration carries the words).
3. ALL positions/vectors must be plain 3-tuples of floats or mathutils.Vector — never bare lists used
   in arithmetic without conversion.
4. Wrap the whole thing in `def build():` and call it at module level.
5. Do not reference any asset name that isn't a key in asset_manifest.
6. Assume imported assets already have realistic PBR materials and bevels from the asset stage —
   do not flatten or override their materials; only add lights, camera, and animation here.
7. Never set `mat.shadow_method` or `mat.blend_method` on any Material you create — both were
   removed from Blender's Material API in 4.2+ (EEVEE Next); use `mat.surface_render_method`
   ('OPAQUE'/'BLENDED'/'DITHERED') instead if you need to set render method at all.

Return ONLY raw executable Python. No markdown fences, no preamble. Start immediately with imports.
"""

def generate_section_scene_script(
    section_title: str, section_summary: str, asset_names: List[str],
    *, logger=None, section_id: Optional[str] = None, script_save_path: Optional[Path] = None,
) -> str:
    messages = [
        SystemMessage(content=SCENE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"SECTION TITLE: {section_title}\nSECTION SUMMARY: {section_summary}\n\n"
            f"Available asset names in asset_manifest for this section: {asset_names}\n\n"
            "Build the scene and render it now, following the cue_timeline exactly."
        )),
    ]
    response = invoke_llm(
        llm_code, messages,
        logger=logger, stage="video_scene_script", model_name=CODE_MODEL, section_id=section_id,
        extra={"script_save_path": str(script_save_path)} if script_save_path else None,
    )
    return strip_code_fences(extract_text(response.content))


# ---------------------------------------------------------------------------
# Self-repair: feed a failed script's exact traceback back to the code model.
#
# Shared by asset_stage.py and video_stage.py — both build in a loop of
# (generate -> run -> on failure, repair -> run again) up to a small attempt
# cap, rather than dropping the asset/section on the first failure. This
# turns systematic hallucinated-API errors (wrong bmesh kwargs, removed
# Material attributes, etc.) into a self-correcting retry instead of a
# permanently missing asset, while every attempt's script and LLM record
# still gets persisted (see script_save_path handling in the callers).
# ---------------------------------------------------------------------------

REPAIR_SYSTEM_PROMPT_SUFFIX = """

--- SELF-REPAIR MODE ---
You previously generated a script that failed when actually executed inside Blender. You will
be given that exact script and the exact error/traceback it produced. Fix it precisely:
- Keep the same overall design, proportions, materials, and animation intent — change only what
  is necessary to eliminate the error.
- Common causes: a bmesh.ops/bpy.ops call with an invalid or renamed keyword argument, or a
  Material/Object attribute that doesn't exist in Blender 4.x (verify against the API reference
  given above rather than guessing from an older/remembered Blender version).
- Keep the exact same output contract (how output_path / asset_manifest / cue_timeline are read,
  what gets exported/rendered, etc) — the failure is in implementation, not the contract.
- Return the COMPLETE corrected script, not a diff or just the changed function.

Return ONLY raw executable Python. No markdown fences, no preamble, no explanation.
"""


def repair_blender_script(
    original_system_prompt: str,
    previous_code: str,
    error_message: str,
    *,
    logger=None,
    stage: str,
    section_id: Optional[str] = None,
    asset_key: Optional[str] = None,
    script_save_path: Optional[Path] = None,
    attempt: int = 2,
) -> str:
    messages = [
        SystemMessage(content=original_system_prompt + REPAIR_SYSTEM_PROMPT_SUFFIX),
        HumanMessage(content=(
            f"PREVIOUS SCRIPT:\n{previous_code}\n\n"
            f"ERROR WHEN RUN IN BLENDER:\n{error_message}\n\n"
            "Fix it and return the complete corrected script."
        )),
    ]
    response = invoke_llm(
        llm_code, messages,
        logger=logger, stage=stage, model_name=CODE_MODEL,
        section_id=section_id, asset_key=asset_key,
        extra={"script_save_path": str(script_save_path), "repair_attempt": attempt} if script_save_path
        else {"repair_attempt": attempt},
    )
    return strip_code_fences(extract_text(response.content))