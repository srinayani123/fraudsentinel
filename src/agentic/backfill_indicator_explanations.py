"""Backfill plain-English indicator_explanations on existing pattern files.

For each case_*.json without an indicator_explanations field, sends the
indicators list to Claude and gets back a parallel list of human-readable
versions. The indicators field stays unchanged (machine-checkable thresholds);
the new field is added for UI display.

Idempotent — skips files that already have indicator_explanations.

Cost: ~$1.00 for 290 existing patterns.
Runtime: ~5 minutes (concurrent, 4 workers).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import anthropic
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL, FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


MAX_CONCURRENT = 4
MAX_RETRIES = 3
COST_PER_PATTERN = (700 * 3 / 1_000_000) + (350 * 15 / 1_000_000)  # ~$0.0073 each


SYSTEM_PROMPT = """You are a fraud-pattern documentation expert. You receive a list of technical fraud-detection indicators (with feature names, thresholds, and quantitative conditions) and produce a parallel list of plain-English explanations — one explanation per indicator, in the same order.

Each plain-English explanation must:
- Describe what the indicator MEANS for fraud detection in human language
- Be readable by a fraud analyst with NO ML background
- Be 8-18 words — sharp, not verbose
- Lead with what the analyst would observe or verify, not the feature name
- Not repeat the threshold or feature name (those stay in the technical indicator)

EXAMPLES:

Technical: "card1_txn_count_5min >= 12 (12-15 transactions in 5-minute window)"
Plain-English: "Card is being hammered with rapid-fire transactions in a tight window"

Technical: "P_emaildomain switches between disposable providers across consecutive transactions"
Plain-English: "Email domain rotates between throwaway services between purchases"

Technical: "card1_amt_zscore exceeds 8.5 indicating extreme deviation from cardholder spending baseline"
Plain-English: "Amount is wildly larger than this card's normal spending range"

Technical: "dist1 values show >500km geographic distance between transactions <90 seconds apart"
Plain-English: "Transactions appear in geographically impossible locations within seconds"

Technical: "Transactions span 3-4 different MCC codes within burst window (5411, 5812, 5999)"
Plain-English: "Card jumps between unrelated merchant types during the attack burst"

Technical: "D1 = 0 indicating first-time device-card pairing"
Plain-English: "The device has never been seen with this card before"

Technical: "C1 > 60 (p90 of top-XGBoost transactions), elevated card-counter aggregation"
Plain-English: "Card-level activity counter is far above the typical cardholder baseline"

OUTPUT STRICT JSON ONLY:
{
  "indicator_explanations": [
    "<plain-English explanation for indicator 1>",
    "<plain-English explanation for indicator 2>",
    ...
  ]
}

The list MUST have the same length as the input indicators list, in the same order. Output JSON only, no preamble."""


def build_user_prompt(indicators: list[str], pattern: str, title: str) -> str:
    indicators_block = "\n".join(f"  {i+1}. {ind}" for i, ind in enumerate(indicators))
    return f"""Pattern: {pattern}
Title: {title}

Technical indicators:
{indicators_block}

Generate the parallel indicator_explanations list. Same length, same order. Output JSON only."""


def enrich_one_file(client, model, path) -> tuple[bool, str]:
    """Returns (success, message)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"read failed: {e}"

    # Skip if already has the field with non-empty list
    existing = data.get("indicator_explanations")
    if isinstance(existing, list) and len(existing) > 0:
        return True, "already has indicator_explanations"

    indicators = data.get("indicators")
    if not isinstance(indicators, list) or not indicators:
        return False, "no indicators to enrich"

    pattern = data.get("pattern", "unknown")
    title = data.get("title", "")
    user_prompt = build_user_prompt(indicators, pattern, title)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=600,
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
            explanations = parsed.get("indicator_explanations", [])

            if not isinstance(explanations, list):
                raise ValueError("indicator_explanations is not a list")
            if len(explanations) != len(indicators):
                raise ValueError(
                    f"length mismatch: {len(explanations)} explanations vs "
                    f"{len(indicators)} indicators"
                )
            if not all(isinstance(x, str) and len(x) > 0 for x in explanations):
                raise ValueError("non-string or empty explanation")

            data["indicator_explanations"] = explanations
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True, "enriched"

        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            logger.debug(f"  {path.name} attempt {attempt+1}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.debug(f"  {path.name} attempt {attempt+1}: unexpected: {e}")
            time.sleep(2 ** attempt)

    return False, f"failed after {MAX_RETRIES} attempts"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    logger.info(f"Using model: {model}")

    paths = sorted(FRAUD_CASES_DIR.glob("case_*.json"))
    if not paths:
        logger.error(f"No case files in {FRAUD_CASES_DIR}")
        return

    logger.info(f"Found {len(paths)} pattern files")

    # Quick scan: how many already have the field?
    have_field = 0
    need_field = 0
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("indicator_explanations"), list) and d["indicator_explanations"]:
                have_field += 1
            else:
                need_field += 1
        except Exception:
            need_field += 1

    logger.info(f"  {have_field} already have indicator_explanations")
    logger.info(f"  {need_field} need backfill")
    logger.info(f"  estimated cost: ~${need_field * COST_PER_PATTERN:.2f}")

    if need_field == 0:
        logger.info("Nothing to do.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    succeeded = 0
    skipped = 0
    failed = 0
    progress_lock = Lock()
    completed = 0
    start_time = time.time()

    def process(path):
        nonlocal completed, succeeded, skipped, failed
        ok, msg = enrich_one_file(client, model, path)
        with progress_lock:
            completed += 1
            if ok and msg == "already has indicator_explanations":
                skipped += 1
            elif ok:
                succeeded += 1
            else:
                failed += 1
                logger.warning(f"  {path.name}: {msg}")

            if completed % 20 == 0 or completed == len(paths):
                elapsed = time.time() - start_time
                logger.info(
                    f"  {completed}/{len(paths)} | "
                    f"enriched={succeeded} skipped={skipped} failed={failed} | "
                    f"~${succeeded * COST_PER_PATTERN:.2f} | "
                    f"{elapsed:.0f}s elapsed"
                )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = [pool.submit(process, p) for p in paths]
        for _ in as_completed(futures):
            pass

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed/60:.1f} min")
    logger.info(f"  Enriched: {succeeded}")
    logger.info(f"  Skipped (already had field): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info("")
    logger.info("Next: python -m src.agentic.generate_new_patterns")


if __name__ == "__main__":
    main()
    