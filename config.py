# """
# config.py — shared settings for the course-generation pipeline.

# Everything that's environment-specific (paths, model name, render settings)
# lives here so the rest of the code doesn't hardcode it.
# """

# from pathlib import Path
# import os

# from langchain_google_genai import ChatGoogleGenerativeAI

# # ---------------------------------------------------------------------------
# # LLM
# # ---------------------------------------------------------------------------
# # Code-gen stages (assets, scene scripts) benefit from a stronger model than prose stages.

# from keys import GOOGLE_API_KEY, GEMMA26B

# def make_llm(model: str, temperature: float = 0.4) -> ChatGoogleGenerativeAI:
#     return ChatGoogleGenerativeAI(
#         model=model,
#         google_api_key=GOOGLE_API_KEY,
#         temperature=temperature,
#     )

# # Two clients: one tuned for prose/JSON planning, one tuned for code-gen
# # (lower temperature -> more syntactically reliable Blender scripts).
# PROSE_MODEL = GEMMA26B
# CODE_MODEL  = GEMMA26B

# llm_prose = make_llm(PROSE_MODEL, temperature=0.5)
# llm_code  = make_llm(CODE_MODEL, temperature=0.2)

# # ---------------------------------------------------------------------------
# # Filesystem layout
# # ---------------------------------------------------------------------------
# #
# # media/
# #   courses/<course_id>/
# #     state/               <- JSON checkpoint per stage (resumability)
# #     audio/<section_id>/  <- per-segment gTTS clips + concatenated track
# #     video/<section_id>/  <- silent render + final muxed mp4
# #   assets/                <- shared, deduplicated .glb library (cross-course)

# MEDIA_ROOT   = Path(os.environ.get("COURSE_MEDIA_ROOT", "./media")).resolve()
# ASSET_LIBRARY_DIR = MEDIA_ROOT / "assets"
# COURSES_DIR  = MEDIA_ROOT / "courses"

# for d in (MEDIA_ROOT, ASSET_LIBRARY_DIR, COURSES_DIR):
#     d.mkdir(parents=True, exist_ok=True)


# def course_dir(course_id: str) -> Path:
#     d = COURSES_DIR / course_id
#     (d / "state").mkdir(parents=True, exist_ok=True)
#     (d / "audio").mkdir(parents=True, exist_ok=True)
#     (d / "video").mkdir(parents=True, exist_ok=True)
#     return d


# # ---------------------------------------------------------------------------
# # Render / Blender settings
# # ---------------------------------------------------------------------------

# BLENDER_BIN   = "/data1/home/geethacharan/blender_exp/blender-4.5.11-linux-x64/blender"
# RENDER_FPS    = 30
# RENDER_RES_X  = 1280
# RENDER_RES_Y  = 720
# RENDER_ENGINE = "BLENDER_EEVEE_NEXT"   # fast, good enough for stylised instructional 3D
# # A trailing-silence buffer (seconds) appended after narration ends per section,
# # so the animation doesn't get cut off mid-motion.
# RENDER_TAIL_BUFFER_SEC = 1.5

# BLENDER_TIMEOUT_SEC = 900   # per subprocess call (asset gen or a single section render)
# MAX_ASSET_WORKERS   = 3
# MAX_SECTION_WORKERS = 3     # parallel section renders in Stage 6


"""
config.py — shared settings for the course-generation pipeline.

Everything that's environment-specific (paths, model name, render settings)
lives here so the rest of the code doesn't hardcode it.
"""

from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from keys import GOOGLE_API_KEY, GEMMA26B, GEMINI_3_1_FLASH_LITE

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Swap the model name for whatever Gemini model you have access to
# (e.g. "gemini-2.0-flash", "gemini-2.5-pro"). Code-gen stages (assets,
# scene scripts) benefit from a stronger model than prose stages.

def make_llm(model: str, temperature: float = 0.4) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )

# Two clients: one tuned for prose/JSON planning, one tuned for code-gen
# (lower temperature -> more syntactically reliable Blender scripts).
PROSE_MODEL = GEMINI_3_1_FLASH_LITE
CODE_MODEL  = GEMINI_3_1_FLASH_LITE

llm_prose = make_llm(PROSE_MODEL, temperature=0.5)
llm_code  = make_llm(CODE_MODEL, temperature=0.2)

# Shared quota across BOTH clients above — set to your actual per-minute cap.
# All LLM calls in the pipeline (prose + code, across every stage/thread) draw
# down this single budget via rate_limiter.py, since they hit the same
# per-key/per-project quota on the Gemini Flash model.
LLM_CALLS_PER_MINUTE = int(os.environ.get("COURSE_LLM_RPM", "10"))

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
#
# media/
#   courses/<course_id>/
#     state/               <- JSON checkpoint per stage (resumability)
#     audio/<section_id>/  <- per-segment gTTS clips + concatenated track
#     video/<section_id>/  <- silent render + final muxed mp4
#   assets/                <- shared, deduplicated .glb library (cross-course)

MEDIA_ROOT   = Path(os.environ.get("COURSE_MEDIA_ROOT", "./media")).resolve()
ASSET_LIBRARY_DIR = MEDIA_ROOT / "assets"
COURSES_DIR  = MEDIA_ROOT / "courses"

for d in (MEDIA_ROOT, ASSET_LIBRARY_DIR, COURSES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def course_dir(course_id: str) -> Path:
    d = COURSES_DIR / course_id
    (d / "state").mkdir(parents=True, exist_ok=True)
    (d / "audio").mkdir(parents=True, exist_ok=True)
    (d / "video").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Render / Blender settings
# ---------------------------------------------------------------------------

BLENDER_BIN   = "/data1/home/geethacharan/blender_exp/blender-4.5.11-linux-x64/blender"
RENDER_FPS    = 30
RENDER_RES_X  = 1280
RENDER_RES_Y  = 720
RENDER_ENGINE = "BLENDER_EEVEE_NEXT"
RENDER_SAMPLES = 64          # Eevee Next viewport/render samples; raise to 128-256 if noisy with raytracing on
RENDER_VIEW_TRANSFORM = "AgX"   # was implicitly 'Standard' before — this alone fixes the flat/washed-out look
RENDER_TAIL_BUFFER_SEC = 1.5

BLENDER_TIMEOUT_SEC = 900
MAX_ASSET_WORKERS   = 3
MAX_SECTION_WORKERS = 3