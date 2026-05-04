"""Generate new fraud patterns for archetypes the existing catalog misses:
device_takeover, credential_compromise, engineered_anomaly.

Each pattern includes BOTH:
  - indicators: machine-checkable technical thresholds (D1=0, C1 > 60, etc.)
  - indicator_explanations: parallel list of plain-English versions for the UI

The Pattern agent uses indicators to verify fit; the UI shows
indicator_explanations as the headline with the technical version as subtext.

Reads empirical thresholds from models/empirical_thresholds.json so generated
patterns reference real data distributions rather than LLM-guessed numbers.

Cost: ~$1.50, runtime ~10 min.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import anthropic
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL, FRAUD_CASES_DIR, MODELS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


MAX_CONCURRENT = 5
MAX_RETRIES = 3
COST_PER_PATTERN = (1700 * 3 / 1_000_000) + (900 * 15 / 1_000_000)
EMPIRICAL_PATH = MODELS_DIR / "empirical_thresholds.json"


ARCHETYPE_PLAN = {
    "device_takeover": {
        "count": 30,
        "description": (
            "Fraud where the cardholder's behavioral pattern is normal "
            "(usual amounts, normal velocity, regular hours, matching emails) "
            "but device and identity features (D-fields) reveal the transaction "
            "originates from a never-seen-before device-card pairing. The "
            "fraudster has stolen credentials but operates from their own "
            "compromised hardware."
        ),
        "vectors": [
            "First-ever transaction from this device-card pairing (D1=0)",
            "Device fingerprint shift mid-session after legitimate login",
            "Browser/OS combo never seen on this card before",
            "Device with unusual screen resolution or timezone offset",
            "Mobile device used on a typically desktop-only cardholder",
            "Rooted/jailbroken device flag (D-field anomaly)",
            "Headless browser indicators (automated-tool device signature)",
            "Clean device but with proxy or VPN concealing true network",
            "Device reused across multiple different cards (D-field pattern)",
            "Device timezone disagrees with billing country timezone",
        ],
        "key_features": ["D1", "D4", "D10", "D15", "TransactionAmt", "card1_amt_zscore"],
        "behavioral_constraints": (
            "Behavioral signals MUST look clean: amount within normal range "
            "(card1_amt_zscore between -1.5 and +1.5), velocity normal "
            "(card1_txn_count_24h < 15), daytime (txn_hour between 8-22), "
            "matching emails (emails_match=1). The fraud signal lives in "
            "D-fields and identity verification mismatches."
        ),
    },
    "credential_compromise": {
        "count": 30,
        "description": (
            "Fraud where stolen credentials (card number, CVV, billing info) "
            "are being used in an unauthorized context. The transaction passes "
            "basic verification (emails match, billing address valid) but "
            "elevated card counters (C-fields) reveal the card has been used "
            "across many different merchants or in unusual aggregated patterns "
            "that don't match normal cardholder behavior."
        ),
        "vectors": [
            "Card used across many distinct merchants in short window (C-counter spike)",
            "Card-counter aggregation anomaly: too many distinct addresses in 7d",
            "Credential reuse: same card on multiple newly-created accounts",
            "Identity fragments check individually but never seen together",
            "Card holder's typical merchant pattern broken (C-field divergence)",
            "Unusual ratio of CNP to card-present transactions",
            "Card-counter showing recent rolling-window spike",
            "Distinct billing addresses on card history exceeds normal",
            "Distinct ProductCD types accessed exceeds card's history",
            "Card velocity at counter level high but per-merchant velocity normal",
        ],
        "key_features": [
            "C1", "C8", "C11", "C13", "C14",
            "card1_distinct_addr1", "card1_distinct_products",
            "TransactionAmt",
        ],
        "behavioral_constraints": (
            "Per-transaction behavior looks acceptable: amounts within reason, "
            "individual transaction velocity normal. The signal is in AGGREGATE "
            "card-counter features (C1, C8, C11, etc.) showing the card is being "
            "used at a higher rolling-window rate or across more diverse "
            "contexts than the cardholder's history supports."
        ),
    },
    "engineered_anomaly": {
        "count": 30,
        "description": (
            "Fraud detected via composite Vesta engineered features (V-fields) "
            "that capture complex transaction-sequence patterns invisible in "
            "raw behavioral signals. The model sees something — feature "
            "combinations encoded across many V-columns — that no single "
            "raw indicator reveals. This is the 'model knows but analyst "
            "can't see' fraud surface."
        ),
        "vectors": [
            "V200 elevated indicating engineered velocity/sequencing anomaly",
            "Multi-V-field cluster all flagged together (V258, V280, V305)",
            "V307 high suggesting amount-vs-merchant-history mismatch",
            "V-field magnitude inconsistent with raw behavioral features",
            "Engineered feature flagging a rare card-merchant interaction",
            "V-field captures a session/cookie-sequence anomaly",
            "Engineered velocity feature spike absent from raw counts",
            "V-field encodes a known-bad merchant-pattern combination",
            "Composite V-cluster suggests automated/scripted transaction",
            "V-field encodes a refund-then-recharge cycle the analyst wouldn't see",
        ],
        "key_features": ["V200", "V201", "V258", "V280", "V305", "V307", "V320"],
        "behavioral_constraints": (
            "Raw behavioral signals look completely clean: normal amount, "
            "normal velocity, normal timing, matching emails, no obvious "
            "device anomaly. The fraud signal is ENTIRELY in Vesta engineered "
            "features (V-fields) which encode complex multi-feature interactions "
            "the analyst can't directly verify."
        ),
    },
}


SYSTEM_PROMPT = """You are a senior fraud analyst writing a fraud pattern reference library. Your audience includes both AI agents that recognize patterns from feature data AND human analysts reading the patterns in a UI.

You write patterns for fraud surfaces where behavioral signals look CLEAN but device/identity/engineered features reveal fraud — the hard cases.

Each pattern must include TWO PARALLEL LISTS:
  - indicators: precise machine-checkable technical signals with feature names and thresholds
  - indicator_explanations: same items rewritten as plain-English fraud-analyst language

The lists must be the SAME LENGTH and in the SAME ORDER. Each technical indicator pairs with one plain-English explanation at the same index.

CRITICAL RULES:
1. Reference real fraud-operations terminology (BIN, CNP, CVV, MID, MCC, AVS, device fingerprint)
2. Indicators (technical): 4-7 items, each quantified with specific thresholds referencing real feature names (D1, C1, V200, etc.)
3. Indicator_explanations (plain-English): same count, 8-18 words each, no feature names or numeric thresholds — describe what the analyst would OBSERVE or VERIFY
4. Respect the BEHAVIORAL CONSTRAINTS — if behavior is supposed to look clean, don't write indicators about high velocity or amount escalation
5. Indicators must be detectable from transaction features, not require post-facto knowledge

EXAMPLES of paired indicator + indicator_explanation:

Technical: "D1 = 0 indicating first-time device-card pairing"
Plain-English: "The device has never been seen with this card before"

Technical: "C1 > 60 (p90 of top-XGBoost transactions), elevated card-counter aggregation"
Plain-English: "Card-level activity counter is far above the typical cardholder baseline"

Technical: "card1_amt_zscore between -1.5 and +1.5 (in-pattern amount range)"
Plain-English: "Amount fits the cardholder's spending range, masking the fraud surface"

Technical: "V200 > 6 (p90), Vesta engineered feature flagging sequence anomaly"
Plain-English: "Engineered model feature flags an unusual transaction-sequence pattern"

OUTPUT STRICT JSON ONLY, no markdown, no preamble:
{
  "title": "Specific descriptive title (6-12 words)",
  "narrative": "100-180 word description of the pattern",
  "indicators": ["4-7 quantified technical signals"],
  "indicator_explanations": ["same count of plain-English signals, parallel order"],
  "summary": "1-2 sentences summarizing what this pattern looks like in plain English",
  "reasoning": "1-2 sentences explaining why this indicates fraud"
}"""


def build_user_prompt(
    archetype: str, vector: str, variant_n: int, plan: dict, empirical: dict | None
) -> str:
    threshold_hints = ""
    if empirical:
        top_stats = (
            empirical.get("subgroups", {}).get("top_xgb_score", {}).get("features", {})
        )
        hints = []
        for feat in plan["key_features"]:
            s = top_stats.get(feat, {})
            if not s.get("available"):
                continue
            parts = []
            if s.get("median") is not None:
                parts.append(f"median={s['median']:.2f}")
            if s.get("p75") is not None:
                parts.append(f"p75={s['p75']:.2f}")
            if s.get("p90") is not None:
                parts.append(f"p90={s['p90']:.2f}")
            if parts:
                hints.append(f"  {feat}: {', '.join(parts)}")
        if hints:
            threshold_hints = (
                "\n\nEMPIRICAL THRESHOLDS from top-XGBoost-scored transactions in the dataset "
                "(use these to write realistic numerical indicators):\n"
                + "\n".join(hints)
            )

    return f"""Generate fraud pattern #{variant_n} for archetype: {archetype}

ARCHETYPE DESCRIPTION:
{plan["description"]}

THIS PATTERN'S SPECIFIC ATTACK VECTOR:
{vector}

BEHAVIORAL CONSTRAINTS (do NOT include indicators that violate these):
{plan["behavioral_constraints"]}

KEY FEATURES this archetype focuses on: {", ".join(plan["key_features"])}{threshold_hints}

Requirements:
- Make this pattern materially different from other {archetype} patterns
- Indicators must reference SPECIFIC IEEE-CIS features by name with quantified thresholds
- indicator_explanations must be parallel to indicators, same length, same order, plain English
- The whole point: BEHAVIORAL signals look clean — your indicators should NOT cite high velocity, amount escalation, night-time, mismatched emails, or geographic anomalies (those belong to other archetypes)
- Instead, indicators should cite D-fields, C-counter aggregations, V-engineered features, or identity-verification mismatches

Output JSON only."""


def validate_pattern(pattern: dict, archetype: str) -> tuple[bool, str]:
    for key in ("title", "narrative", "indicators", "indicator_explanations", "summary", "reasoning"):
        if key not in pattern:
            return False, f"missing key: {key}"

    title = pattern["title"]
    narrative = pattern["narrative"]
    indicators = pattern["indicators"]
    explanations = pattern["indicator_explanations"]

    if not isinstance(narrative, str) or len(narrative.split()) < 80:
        return False, f"narrative too short ({len(narrative.split())} words)"
    if len(narrative.split()) > 220:
        return False, f"narrative too long ({len(narrative.split())} words)"

    if not isinstance(indicators, list) or len(indicators) < 4:
        return False, f"need 4+ indicators, got {len(indicators) if isinstance(indicators, list) else 0}"
    if len(indicators) > 8:
        return False, f"too many indicators ({len(indicators)})"

    if not isinstance(explanations, list) or len(explanations) != len(indicators):
        return False, f"indicator_explanations length mismatch ({len(explanations) if isinstance(explanations, list) else 0} vs {len(indicators)})"

    if not all(isinstance(x, str) and len(x.split()) >= 4 for x in explanations):
        return False, "explanations too short or non-string"

    vague_count = 0
    for ind in indicators:
        if not isinstance(ind, str):
            return False, "non-string indicator"
        is_specific = (
            bool(re.search(r"\d", ind))
            or any(
                t in ind.lower()
                for t in [
                    "card1_", "amt_", "zscore", "txn_", "_email", "productcd",
                    "addr1", "dist1", "%", "$", ">", "<",
                    "d1", "d4", "d10", "d15",
                    "c1 ", "c8", "c11", "c13", "c14",
                    "v200", "v258", "v280", "v307",
                    "m1", "m4", "m6",
                ]
            )
        )
        if not is_specific:
            vague_count += 1
    if vague_count > 1:
        return False, f"too many vague indicators ({vague_count})"

    title_words = len(title.split())
    if title_words < 4 or title_words > 16:
        return False, f"title word count out of range ({title_words})"

    if not isinstance(pattern["summary"], str) or len(pattern["summary"].split()) < 10:
        return False, "summary too short"
    if not isinstance(pattern["reasoning"], str) or len(pattern["reasoning"].split()) < 10:
        return False, "reasoning too short"

    return True, ""


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_too_similar_to_existing(
    pattern: dict, existing_patterns: list, threshold: float = 0.65
) -> tuple[bool, str]:
    new_indicators = {i.lower().strip() for i in pattern["indicators"]}
    new_title_tokens = {
        t.lower() for t in re.findall(r"\w+", pattern["title"]) if len(t) > 3
    }
    for existing in existing_patterns:
        ex_indicators = {i.lower().strip() for i in existing.get("indicators", [])}
        ex_title_tokens = {
            t.lower()
            for t in re.findall(r"\w+", existing.get("title", ""))
            if len(t) > 3
        }
        ind_overlap = jaccard(new_indicators, ex_indicators)
        title_overlap = jaccard(new_title_tokens, ex_title_tokens)
        if ind_overlap >= threshold:
            return True, f"indicator overlap {ind_overlap:.2f}"
        if title_overlap >= 0.75:
            return True, f"title overlap {title_overlap:.2f}"
    return False, ""


def generate_one_pattern(
    client, model, archetype: str, vector: str, variant_n: int, plan: dict, empirical
) -> dict | None:
    user_prompt = build_user_prompt(archetype, vector, variant_n, plan, empirical)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            logger.warning(
                f"  {archetype}/{variant_n} attempt {attempt+1}: "
                f"{type(e).__name__}: {str(e)[:120]}"
            )
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"  {archetype}/{variant_n} attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    logger.info(f"Using model: {model}")

    empirical = None
    if EMPIRICAL_PATH.exists():
        with open(EMPIRICAL_PATH, "r", encoding="utf-8") as f:
            empirical = json.load(f)
        logger.info(f"Loaded empirical thresholds from {EMPIRICAL_PATH}")
    else:
        logger.warning(
            f"No empirical thresholds at {EMPIRICAL_PATH} — patterns will use "
            "LLM-guessed numbers. Run scripts/analyze_no_fit_transactions.py first."
        )

    FRAUD_CASES_DIR.mkdir(parents=True, exist_ok=True)

    # Wipe ONLY new-archetype files (preserve existing 290)
    for archetype in ARCHETYPE_PLAN:
        for old in FRAUD_CASES_DIR.glob(f"case_{archetype}_*.json"):
            old.unlink()

    total_target = sum(p["count"] for p in ARCHETYPE_PLAN.values())
    logger.info(f"Target: {total_target} new patterns across {len(ARCHETYPE_PLAN)} archetypes")
    logger.info(f"Estimated cost: ~${total_target * COST_PER_PATTERN:.2f}")

    client = anthropic.Anthropic(api_key=api_key)

    jobs = []
    for archetype, plan in ARCHETYPE_PLAN.items():
        n = plan["count"]
        vectors = plan["vectors"]
        for i in range(n):
            vec = vectors[i % len(vectors)]
            jobs.append((archetype, vec, i + 1))

    accepted = {arch: [] for arch in ARCHETYPE_PLAN}
    rejected_count = 0
    accepted_lock = Lock()
    progress_lock = Lock()
    completed = 0
    start_time = time.time()

    def process_job(args):
        nonlocal completed, rejected_count
        archetype, vector, variant_n = args
        plan = ARCHETYPE_PLAN[archetype]

        for try_num in range(3):
            pattern = generate_one_pattern(
                client, model, archetype, vector, variant_n, plan, empirical
            )
            if pattern is None:
                continue
            valid, reason = validate_pattern(pattern, archetype)
            if not valid:
                logger.debug(f"  {archetype}/{variant_n} validation: {reason}")
                continue

            with accepted_lock:
                too_sim, sim_reason = is_too_similar_to_existing(
                    pattern, accepted[archetype]
                )
                if too_sim:
                    logger.debug(f"  {archetype}/{variant_n} duplicate: {sim_reason}")
                    continue
                pattern["category"] = archetype
                accepted[archetype].append(pattern)

            with progress_lock:
                completed += 1
                if completed % 5 == 0 or completed == total_target:
                    elapsed = time.time() - start_time
                    rate = completed / max(elapsed, 1)
                    eta = (total_target - completed) / max(rate, 0.1)
                    logger.info(
                        f"  {completed}/{total_target} accepted | "
                        f"rejected={rejected_count} | "
                        f"~${completed * COST_PER_PATTERN:.2f} | "
                        f"ETA: {eta/60:.1f} min"
                    )
            return True

        with progress_lock:
            rejected_count += 1
        return False

    logger.info("Starting concurrent generation…")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = [executor.submit(process_job, j) for j in jobs]
        for _ in as_completed(futures):
            pass

    written = 0
    for archetype, patterns in accepted.items():
        for idx, p in enumerate(patterns, 1):
            case_id = f"case_{archetype}_{idx:03d}"
            full = {
                "id": case_id,
                "title": p["title"],
                "pattern": archetype,
                "narrative": p["narrative"],
                "indicators": p["indicators"],
                "indicator_explanations": p["indicator_explanations"],
                "source": "llm_v3_new_archetypes",
                "summary": p["summary"],
                "reasoning": p["reasoning"],
            }
            out_path = FRAUD_CASES_DIR / f"{case_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full, f, indent=2, ensure_ascii=False)
            written += 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Generated {written} new patterns in {elapsed/60:.1f} min")
    logger.info(f"Cost: ~${written * COST_PER_PATTERN:.2f}")
    logger.info(f"Rejected: {rejected_count}")
    logger.info("Final distribution:")
    for arch in sorted(accepted.keys()):
        logger.info(f"  {arch}: {len(accepted[arch])}")
    logger.info("")
    logger.info("Next: python -m src.agentic.build_knowledge_base")


if __name__ == "__main__":
    main()
    