"""Generate calibration queries from the pattern catalog.

For each fraud pattern in the catalog, asks Claude to produce 2 short analyst-style
query paraphrases — the kind of natural-language question an analyst would type
when looking for that pattern. Used as supplementary in-domain examples during
OOD detector calibration so analyst-style queries don't get rejected.

Idempotent — skips patterns already in the output file. Re-run after adding new
patterns to extend the calibration set without regenerating existing queries.

Output: models/calibration_queries.json — schema:
    {
      "_meta": { "n_patterns": int, "n_queries": int, "generated_at": str },
      "queries": [
        {"pattern_id": "case_velocity_attack_001", "pattern": "velocity_attack",
         "queries": ["short query 1", "short query 2"]},
        ...
      ]
    }

Cost: ~$1 for 377 patterns (one Claude call per pattern, batched concurrent).
Runtime: ~5 minutes.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import anthropic
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL, FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()

OUTPUT_PATH = Path("models/calibration_queries.json")

MAX_CONCURRENT = 4
MAX_RETRIES = 3
COST_PER_PATTERN = (500 * 3 / 1_000_000) + (180 * 15 / 1_000_000)  # ~$0.0042 each


SYSTEM_PROMPT = """You are a fraud-domain query generator. You receive a fraud pattern (title, narrative, indicators) and produce 2 short analyst-style queries — the kind of natural-language question a fraud analyst would type when looking for this pattern.

REQUIREMENTS:
- Each query must be 8-15 words
- Each must sound like something an analyst would actually type, not a textbook description
- Use natural language, not feature names like "card1_txn_count_1h"
- Capture the DISTINCTIVE behavioral or contextual signal of the pattern, not its full narrative
- The two queries must be paraphrases — same intent, different wording
- Avoid generic queries that could match any fraud pattern; be specific to THIS one

OUTPUT STRICT JSON ONLY:
{
  "queries": ["short analyst-style query 1", "short analyst-style query 2"]
}

EXAMPLES:

For a card-testing burst pattern with rapid small charges:
{
  "queries": [
    "small test charges to validate stolen card numbers in rapid succession",
    "burst of tiny micro charges before larger purchase on same card"
  ]
}

For a device takeover with clean behavior:
{
  "queries": [
    "first time device card pairing with normal velocity and matching emails",
    "transaction from new device fingerprint despite valid credentials"
  ]
}

Output JSON only, no preamble."""


def build_user_prompt(case: dict) -> str:
    title = case.get("title", "")
    narrative = case.get("narrative", "")[:600]  # truncate to keep cost down
    pattern = case.get("pattern", "")
    return f"""Pattern category: {pattern}
Title: {title}

Narrative: {narrative}

Generate 2 analyst-style queries. Output JSON only."""


def generate_queries_for_pattern(client, model, case: dict) -> list[str] | None:
    user_prompt = build_user_prompt(case)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            queries = parsed.get("queries", [])
            if not isinstance(queries, list) or len(queries) < 2:
                raise ValueError(f"Expected 2 queries, got {len(queries) if isinstance(queries, list) else 0}")
            queries = [q for q in queries if isinstance(q, str) and 4 <= len(q.split()) <= 25]
            if len(queries) < 2:
                raise ValueError("After length filter, fewer than 2 queries remain")
            return queries[:2]
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            logger.debug(f"  {case.get('id', '?')} attempt {attempt+1}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"  {case.get('id', '?')} attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return None


def load_existing_output() -> dict:
    """Load existing output file if present (for idempotent re-runs)."""
    if not OUTPUT_PATH.exists():
        return {"queries": []}
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not parse existing {OUTPUT_PATH}: {e}. Starting fresh.")
        return {"queries": []}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    logger.info(f"Using model: {model}")

    # Load patterns
    case_paths = sorted(FRAUD_CASES_DIR.glob("case_*.json"))
    if not case_paths:
        raise RuntimeError("No case files found")

    cases = []
    for path in case_paths:
        try:
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            cases.append(case)
        except Exception as e:
            logger.warning(f"Skipping {path.name}: {e}")
    logger.info(f"Loaded {len(cases)} pattern files")

    # Load existing output for idempotent re-run
    existing = load_existing_output()
    existing_ids = {entry["pattern_id"] for entry in existing.get("queries", [])}
    logger.info(f"  {len(existing_ids)} patterns already have queries (will skip)")

    todo = [c for c in cases if c.get("id") not in existing_ids]
    logger.info(f"  {len(todo)} patterns need queries")

    if not todo:
        logger.info("Nothing to do — all patterns already have queries.")
        return

    logger.info(f"  estimated cost: ~${len(todo) * COST_PER_PATTERN:.2f}")

    client = anthropic.Anthropic(api_key=api_key)

    new_entries = []
    progress_lock = Lock()
    completed = 0
    failed = 0
    start_time = time.time()

    def process(case: dict):
        nonlocal completed, failed
        queries = generate_queries_for_pattern(client, model, case)
        with progress_lock:
            completed += 1
            if queries is None:
                failed += 1
                logger.warning(f"  {case.get('id', '?')}: failed after {MAX_RETRIES} attempts")
            else:
                new_entries.append({
                    "pattern_id": case.get("id"),
                    "pattern": case.get("pattern", ""),
                    "queries": queries,
                })

            if completed % 20 == 0 or completed == len(todo):
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 1)
                eta = (len(todo) - completed) / max(rate, 0.1)
                logger.info(
                    f"  {completed}/{len(todo)} | "
                    f"generated={completed - failed} failed={failed} | "
                    f"~${(completed - failed) * COST_PER_PATTERN:.2f} | "
                    f"ETA: {eta/60:.1f} min"
                )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = [pool.submit(process, c) for c in todo]
        for _ in as_completed(futures):
            pass

    # Combine with existing entries and write
    all_entries = existing.get("queries", []) + new_entries
    all_entries.sort(key=lambda e: e["pattern_id"])

    n_queries = sum(len(e["queries"]) for e in all_entries)
    output = {
        "_meta": {
            "n_patterns": len(all_entries),
            "n_queries": n_queries,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
        },
        "queries": all_entries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Generated queries for {completed - failed}/{len(todo)} new patterns in {elapsed/60:.1f} min")
    logger.info(f"Total entries in {OUTPUT_PATH.name}: {len(all_entries)} patterns, {n_queries} queries")
    logger.info(f"Failed: {failed}")
    logger.info("")
    logger.info("Next: python -m src.agentic.calibrate_similarity")


if __name__ == "__main__":
    main()
    