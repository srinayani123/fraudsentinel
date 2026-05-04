"""One-time backfill: add `summary` and `reasoning` fields to every fraud pattern.

- summary    — 1-2 sentences. WHAT this pattern looks like in the data — the
                giveaway signals an analyst would observe.
- reasoning  — 1-2 sentences. WHY it's fraud — what the bad actor is trying
                to do, why these signals together indicate criminal intent
                rather than normal customer behavior.

Both fields are stored on the case JSON, indexed in ChromaDB metadata, and
displayed on every pattern card across Investigate / Pattern Library / Test.

Run after generating new patterns:
    python -m src.agentic.summarize_patterns

Idempotent — skips files that already have both fields. Pass --force to regenerate.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL, FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


SYSTEM_PROMPT = """You are a fraud analyst writing two short UI strings for each fraud pattern in a fraud co-pilot tool.

You'll receive one fraud pattern (title, pattern type, narrative, indicators).
Output STRICT JSON with exactly these two fields:

{
  "summary": "1-2 sentences. WHAT this pattern looks like in the data — the giveaway signals an analyst would observe. Plain English. No jargon, no markdown, no quotes inside the string.",
  "reasoning": "1-2 sentences. WHY this is fraud — what the bad actor is trying to do, why these signals together indicate criminal intent rather than normal customer behavior."
}

Rules:
- Each field maximum 2 sentences. Aim for 1 if possible.
- Don't repeat the title verbatim — assume the title is shown alongside.
- No prefixes like "this pattern describes" or "in this case".
- Plain prose only. No markdown, no quotes inside the strings, no lists.
- summary = WHAT (observable signals). reasoning = WHY (criminal intent).

Output JSON only, no preamble or trailing text."""


def summarize_case(client, model: str, case: dict) -> dict:
    """Generate {summary, reasoning} dict for one case."""
    indicators_str = "\n".join(f"- {ind}" for ind in case.get("indicators", []))
    user_msg = (
        f"Title: {case.get('title', '')}\n"
        f"Pattern type: {case.get('pattern', '')}\n\n"
        f"Narrative:\n{case.get('narrative', '')}\n\n"
        f"Behavioral indicators:\n{indicators_str}"
    )

    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    parsed = json.loads(text)
    return {
        "summary": str(parsed.get("summary", "")).strip(),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def process_one(
    client, model: str, path: Path, force: bool = False
) -> tuple[Path, str]:
    """Returns (path, status). status: 'updated' | 'skipped' | 'error: ...'"""
    try:
        with open(path) as f:
            case = json.load(f)

        if not force and case.get("summary") and case.get("reasoning"):
            return path, "skipped"

        result = summarize_case(client, model, case)
        case["summary"] = result["summary"]
        case["reasoning"] = result["reasoning"]

        with open(path, "w") as f:
            json.dump(case, f, indent=2)

        return path, "updated"
    except Exception as e:
        return path, f"error: {type(e).__name__}: {str(e)[:120]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if summary/reasoning already present",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent API calls (default 4)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if not FRAUD_CASES_DIR.exists():
        raise RuntimeError(f"FRAUD_CASES_DIR does not exist: {FRAUD_CASES_DIR}")

    case_paths = sorted(FRAUD_CASES_DIR.glob("case_*.json"))
    if not case_paths:
        raise RuntimeError("No case files found")

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    logger.info(f"Using model: {model}")
    logger.info(f"Found {len(case_paths)} case files")
    logger.info(f"Force mode: {args.force} | Workers: {args.workers}")

    client = anthropic.Anthropic(api_key=api_key)
    counts = {"updated": 0, "skipped": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, client, model, p, args.force): p for p in case_paths
        }
        completed = 0
        for fut in as_completed(futures):
            path, status = fut.result()
            completed += 1
            if status == "updated":
                counts["updated"] += 1
            elif status == "skipped":
                counts["skipped"] += 1
            else:
                counts["error"] += 1
                logger.warning(f"  {path.name}: {status}")

            if completed % 25 == 0 or completed == len(case_paths):
                logger.info(
                    f"  {completed}/{len(case_paths)} processed | "
                    f"updated={counts['updated']} skipped={counts['skipped']} "
                    f"errors={counts['error']}"
                )

    logger.info("=" * 60)
    logger.info(
        f"Done. Updated: {counts['updated']}, "
        f"Skipped: {counts['skipped']}, "
        f"Errors: {counts['error']}"
    )
    logger.info("")
    logger.info("Next: python -m src.agentic.build_knowledge_base")


if __name__ == "__main__":
    main()
    