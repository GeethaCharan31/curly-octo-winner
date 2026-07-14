"""
cli.py — run the pipeline end to end.

Usage:
    python -m course_pipeline.cli \\
        --topic "Electric Vehicle Fundamentals" \\
        --audience "ITI graduates seeking EV technician roles, Class 10 pass, no prior EV exposure" \\
        --course-id ev-fundamentals-iti

If --course-id is omitted, one is slugified from --topic. Re-running with the
same --course-id resumes from whatever stages already have a checkpoint in
media/courses/<course-id>/state/ (see graph.py) instead of starting over.

Every LLM call made along the way (prompts + full responses + timing) is
logged to media/courses/<course-id>/state/llm_calls.jsonl as it happens (see
llm_logger.py). This run's summary of that log — call counts, failures,
retries, latency per stage — is folded into final_summary.json below so
there's one file, next to the rendered media, that tells you both what was
produced and what it cost to produce it.

Requires:
    GOOGLE_API_KEY   env var set
    blender          on PATH (headless, so any build works — no GPU/display needed)
    ffmpeg/ffprobe   on PATH
    pip install langgraph langchain-google-genai gtts
"""

import argparse
import json
import re
import sys

from config import course_dir
from state import new_course_state
from graph import build_graph


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an AI-authored, audience-targeted course.")
    parser.add_argument("--topic", required=True, help='e.g. "Electric Vehicle Fundamentals"')
    parser.add_argument("--audience", required=True,
                         help='e.g. "ITI Mechanic (MV) trade, Class 10 pass, no prior EV exposure"')
    parser.add_argument("--course-id", default=None, help="Defaults to a slug of --topic.")
    args = parser.parse_args()

    course_id = args.course_id or _slugify(args.topic)
    print(f"=== Generating course '{course_id}' ===")
    print(f"Topic:    {args.topic}")
    print(f"Audience: {args.audience}")

    app = build_graph()
    initial_state = new_course_state(topic=args.topic, audience=args.audience, course_id=course_id)

    final_state = app.invoke(initial_state)

    llm_summary = final_state["llm_logger"].summary()

    summary_path = course_dir(course_id) / "state" / "final_summary.json"
    summary = {
        "course_id": course_id,
        "topic": args.topic,
        "audience": args.audience,
        "sections": final_state["sections"],
        "video_paths": final_state["video_paths"],
        "render_errors": final_state["render_errors"],
        "deduplicated_assets": final_state["deduplicated_assets"],
        "llm_usage": llm_summary,
        "llm_call_log": str(course_dir(course_id) / "state" / "llm_calls.jsonl"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Done ===")
    print(f"Sections:        {len(final_state['sections'])}")
    print(f"Videos rendered: {len(final_state['video_paths'])}")
    print(f"Videos failed:   {len(final_state['render_errors'])}")
    print(f"LLM calls:       {llm_summary['total_calls']} "
          f"({llm_summary['failed_calls']} failed, {llm_summary['retried_calls']} retried, "
          f"{llm_summary['total_llm_latency_sec']}s total)")
    print(f"Summary written: {summary_path}")

    if final_state["render_errors"]:
        print("\nFailed sections:")
        for sid, err in final_state["render_errors"].items():
            print(f"  - {sid}: {err[:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()