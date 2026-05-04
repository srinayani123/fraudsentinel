"""
Improved fraud pattern library generator.

Enumerate-and-elaborate approach: for each of 10 fraud categories, generate
N distinct scenarios with explicit diversity constraints.

Replaces the old generate_cases_llm.py + generate_cases_template.py pipeline.

Cost: ~$1.50 in Anthropic credits.
Time: ~12-15 minutes (concurrent, retry-resilient).
Output: ~300 high-quality patterns balanced across categories.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import anthropic
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL, FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


# ============================================================
# Configuration
# ============================================================
MAX_CONCURRENT = 5
MAX_RETRIES = 3
COST_PER_PATTERN = (1200 * 3 / 1_000_000) + (450 * 15 / 1_000_000)  # ~$0.005 each

# Category → (count, attack vectors to vary across patterns)
CATEGORY_PLAN = {
    "card_testing": {
        "count": 32,
        "vectors": [
            "Single tiny charge ($0.50-$2) at digital good merchant",
            "Multiple test charges across different micro-merchants",
            "Test charge then immediate large purchase same merchant",
            "Test on subscription service to validate without charging",
            "Refund manipulation to test card without losing money",
            "Charity donation as a test (low scrutiny)",
            "Mobile app store $0.99 purchase as validator",
            "Foreign exchange micro-charge (currency conversion fee only)",
            "Failed CVV attempts followed by success",
            "Test on a merchant with deferred capture",
        ],
    },
    "geo_anomaly": {
        "count": 28,
        "vectors": [
            "Impossible travel: domestic txn then international within 1 hour",
            "VPN/proxy usage masking true origin",
            "Card present transaction in unusual country",
            "Shipping address country differs from billing country",
            "Card used in country flagged for high CNP fraud",
            "Roaming card pattern that matches no airline/hotel record",
            "Suddenly active in country with currency the user never used",
            "IP geolocation contradicts billing country",
            "Tor exit node origin",
            "Cardholder's primary country shows no recent activity",
        ],
    },
    "account_takeover": {
        "count": 32,
        "vectors": [
            "Email change followed by password reset followed by purchase",
            "Shipping address added then large purchase to new address",
            "Phone number changed before transaction",
            "Login from never-seen device + password reset + purchase",
            "Credential stuffing match: account dormant then sudden activity",
            "Recovery email itself was compromised (chain takeover)",
            "Session hijack: legitimate login then cookie reuse from new IP",
            "SIM swap: phone-based 2FA bypassed",
            "OAuth token theft from third-party app",
            "Social engineering: customer service tricked into adding device",
        ],
    },
    "velocity_attack": {
        "count": 30,
        "vectors": [
            "Rapid burst: 10+ transactions in 5 minutes on same card",
            "Sustained drain: 50+ transactions across 24 hours",
            "Multi-merchant burst: same card, 8+ different merchants in 1 hour",
            "Refund-then-recharge cycling to inflate apparent volume",
            "Subscription cascade: signing up for many subscriptions in succession",
            "Gift card purchases in rapid succession (liquidation)",
            "P2P transfers in rapid succession (laundering)",
            "Multiple tickets/event purchases for resale",
            "Same merchant, small amounts, dozens of transactions",
            "Velocity spike correlated with merchant batch settlement window",
        ],
    },
    "synthetic_identity": {
        "count": 28,
        "vectors": [
            "Bust-out: account aged 3-6 months then sudden max-out",
            "Real SSN + fake name combo (Frankenstein identity)",
            "Authorized user fraud: added to thin-file cardholder",
            "Recently established credit profile with thin file",
            "Identity built across multiple merchants using same fragments",
            "Address tied to mail-drop or virtual mailbox service",
            "Phone number assigned to VOIP service with no history",
            "Email domain registered within last 90 days",
            "All identity fragments check individually but never seen together",
            "Pattern of small responsible payments then sudden default behavior",
        ],
    },
    "bin_attack": {
        "count": 28,
        "vectors": [
            "Sequential card numbers from same BIN tested in succession",
            "Algorithmic generation: Luhn-valid numbers tried in batch",
            "Targeted small-issuer BIN with weaker fraud controls",
            "Prepaid BIN attack: high approval, low cardholder accountability",
            "BIN range tested across many merchants simultaneously",
            "Recently breached BIN range showing unusual activity",
            "BIN lookup before each attempt to confirm card type",
            "BIN attack disguised as legitimate retailer signup flow",
            "Co-branded card BIN abuse",
            "Corporate card BIN attack at B2B merchants",
        ],
    },
    "friendly_fraud": {
        "count": 28,
        "vectors": [
            "Item received then disputed as 'not received'",
            "Digital good purchased then disputed as unauthorized",
            "Family member made purchase then cardholder disputes",
            "Buyer's remorse rebranded as fraud claim",
            "Subscription used for full period then disputed",
            "Service consumed (event ticket used) then disputed",
            "Repeat-customer pattern: history of disputes after delivery",
            "High-resale item purchased and resold before dispute",
            "Dispute filed exactly within chargeback window",
            "Multiple dispute attempts with refined story",
        ],
    },
    "temporal_anomaly": {
        "count": 28,
        "vectors": [
            "3-5am transactions on a strictly daytime user",
            "Activity surge during cardholder's known travel/sleep period",
            "First-ever weekend activity on a strictly weekday card",
            "Transaction during a holiday cardholder typically ignores",
            "Late-night activity matching attacker's timezone, not cardholder's",
            "Transaction at exactly the cardholder's known commute time but wrong location",
            "Activity in a normally dead hour for the merchant",
            "Burst pattern timed to fraud team shift change",
            "End-of-month surge inconsistent with cardholder's typical spending",
            "Activity during cardholder's confirmed offline period",
        ],
    },
    "email_risk": {
        "count": 28,
        "vectors": [
            "Disposable email service (mailinator, 10minutemail) on transaction",
            "Newly registered email domain (<30 days old)",
            "Privacy-focused mail provider with no other history",
            "Recipient email differs from purchaser email",
            "Email pattern: random characters + numbers (machine-generated)",
            "Free webmail account with no recovery info",
            "Catch-all email domain (any prefix accepted)",
            "Email previously associated with confirmed fraud",
            "Plus-addressing abuse (gmail.com+random for many accounts)",
            "Email domain on threat intelligence blocklist",
        ],
    },
    "subscription_probe": {
        "count": 28,
        "vectors": [
            "$0 trial signup to validate card without charge",
            "$0.99 micro-charge subscription as card validator",
            "Multiple streaming service signups in rapid succession",
            "VPN/privacy service signup using stolen card",
            "Adult content subscription as low-scrutiny validator",
            "Cloud storage trial signup pattern",
            "Free-trial-then-cancel cycle abuse",
            "Subscription bundling abuse (one card, many account creations)",
            "Recurring micro-payment to validate card stays alive",
            "Subscription chargeback after content consumption",
        ],
    },
}


SYSTEM_PROMPT = """You are a senior fraud analyst writing a fraud pattern reference library. Your audience is junior fraud analysts and AI agents that need to recognize fraud patterns.

Each pattern you write must be:
- A concrete, named scenario with specific behavioral details
- 100-180 words, dense with realistic specifics (amounts, time windows, feature signatures)
- Written as if describing a real case, not a textbook definition
- Distinct from any other pattern in the same category — no near-duplicates

CRITICAL RULES:
1. Reference real fraud-operations terminology naturally (BIN, CNP, velocity, bust-out, chargeback, MCC, AVS, CVV, MID, IIN)
2. Include 4-7 SPECIFIC behavioral indicators (not generic ones like "suspicious activity")
3. Specific = quantified. "5+ transactions in 10 minutes" is specific. "High velocity" is not.
4. Indicators should be detectable from transaction features, not require post-facto knowledge

Output STRICT JSON only, no markdown, no preamble:
{
  "title": "Specific descriptive title (6-12 words)",
  "narrative": "100-180 word description of the pattern",
  "indicators": ["4-7 specific quantified detection signals"]
}"""


def build_user_prompt(category: str, vector: str, variant_n: int) -> str:
    return f"""Generate fraud pattern #{variant_n} for category: {category}

This pattern's specific attack vector: {vector}

Requirements:
- Make this pattern materially different from other {category} patterns (not just renumbering)
- Use realistic transaction-level details (specific dollar amounts, time windows, feature names)
- Indicators must be 4-7 quantified signals an analyst could verify from transaction data
- Reference IEEE-CIS-style features where natural: TransactionAmt, card1_txn_count_*, card1_amt_zscore, P_emaildomain, R_emaildomain, ProductCD, addr1, dist1, is_night, txn_hour, etc.

Output JSON only."""


# ============================================================
# Validation
# ============================================================
def validate_pattern(pattern: dict, category: str) -> tuple[bool, str]:
    """Returns (is_valid, reason_if_not)."""
    # Required keys
    for key in ("title", "narrative", "indicators"):
        if key not in pattern:
            return False, f"missing key: {key}"

    title = pattern["title"]
    narrative = pattern["narrative"]
    indicators = pattern["indicators"]

    # Length checks
    if not isinstance(narrative, str) or len(narrative.split()) < 80:
        return False, f"narrative too short ({len(narrative.split())} words, need 80+)"
    if len(narrative.split()) > 220:
        return False, f"narrative too long ({len(narrative.split())} words, max 220)"

    # Indicators
    if not isinstance(indicators, list) or len(indicators) < 4:
        return False, f"need 4+ indicators, got {len(indicators) if isinstance(indicators, list) else 0}"
    if len(indicators) > 8:
        return False, f"too many indicators ({len(indicators)}, max 8)"

    # Indicators must be specific (contain a digit or common qualifier)
    vague_count = 0
    for ind in indicators:
        if not isinstance(ind, str):
            return False, "non-string indicator"
        # Heuristic: vague indicators have no number, no specific feature name, no quantifier
        is_specific = (
            bool(re.search(r"\d", ind)) or
            any(t in ind.lower() for t in [
                "card1_", "amt_", "zscore", "txn_", "_email", "productcd",
                "addr1", "dist1", "%", "$", ">"
            ])
        )
        if not is_specific:
            vague_count += 1
    if vague_count > 1:
        return False, f"too many vague indicators ({vague_count} of {len(indicators)})"

    # Title length
    title_words = len(title.split())
    if title_words < 4 or title_words > 16:
        return False, f"title word count out of range ({title_words})"

    return True, ""


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_too_similar_to_existing(
    pattern: dict, existing_patterns: list, threshold: float = 0.65
) -> tuple[bool, str]:
    """Reject if indicators or narrative tokens overlap >threshold with an existing pattern."""
    new_indicators = {i.lower().strip() for i in pattern["indicators"]}
    new_title_tokens = {
        t.lower() for t in re.findall(r"\w+", pattern["title"]) if len(t) > 3
    }

    for existing in existing_patterns:
        ex_indicators = {i.lower().strip() for i in existing.get("indicators", [])}
        ex_title_tokens = {
            t.lower() for t in re.findall(r"\w+", existing.get("title", "")) if len(t) > 3
        }

        ind_overlap = jaccard(new_indicators, ex_indicators)
        title_overlap = jaccard(new_title_tokens, ex_title_tokens)

        if ind_overlap >= threshold:
            return True, f"indicator overlap {ind_overlap:.2f} with {existing.get('id', '?')}"
        if title_overlap >= 0.75:
            return True, f"title overlap {title_overlap:.2f} with {existing.get('id', '?')}"

    return False, ""


# ============================================================
# Generation
# ============================================================
def generate_one_pattern(client, model, category: str, vector: str, variant_n: int) -> dict | None:
    """Returns parsed pattern dict or None on failure."""
    user_prompt = build_user_prompt(category, vector, variant_n)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=900,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            return parsed
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            logger.warning(f"  {category}/{variant_n} attempt {attempt+1}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"  {category}/{variant_n} attempt {attempt+1}: unexpected: {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add to .env.")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    logger.info(f"Using model: {model}")

    # Wipe existing fraud cases (we're rebuilding from scratch)
    FRAUD_CASES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAUD_CASES_DIR.glob("case_*.json"):
        old.unlink()
    for old in FRAUD_CASES_DIR.glob("llm_*.json"):
        old.unlink()
    for old in FRAUD_CASES_DIR.glob("tmpl_*.json"):
        old.unlink()
    logger.info("Wiped existing case files")

    total_target = sum(p["count"] for p in CATEGORY_PLAN.values())
    logger.info(f"Target: {total_target} patterns across {len(CATEGORY_PLAN)} categories")
    logger.info(f"Estimated cost: ~${total_target * COST_PER_PATTERN:.2f}")

    client = anthropic.Anthropic(api_key=api_key)

    # Build job list — round-robin across vectors so we get variety quickly
    jobs = []  # list of (category, vector, variant_n)
    for category, plan in CATEGORY_PLAN.items():
        n = plan["count"]
        vectors = plan["vectors"]
        for i in range(n):
            vec = vectors[i % len(vectors)]
            jobs.append((category, vec, i + 1))

    accepted: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_PLAN}
    rejected_count = 0
    accepted_lock = Lock()
    progress_lock = Lock()
    completed = 0
    start_time = time.time()

    def process_job(args):
        nonlocal completed, rejected_count
        category, vector, variant_n = args

        for try_num in range(3):  # up to 3 generation attempts per slot
            pattern = generate_one_pattern(client, model, category, vector, variant_n)
            if pattern is None:
                continue

            valid, reason = validate_pattern(pattern, category)
            if not valid:
                logger.debug(f"  {category}/{variant_n} validation: {reason}")
                continue

            with accepted_lock:
                too_sim, sim_reason = is_too_similar_to_existing(pattern, accepted[category])
                if too_sim:
                    logger.debug(f"  {category}/{variant_n} duplicate: {sim_reason}")
                    continue

                # Accept it
                pattern["category"] = category
                accepted[category].append(pattern)

            with progress_lock:
                completed += 1
                if completed % 10 == 0 or completed == total_target:
                    elapsed = time.time() - start_time
                    rate = completed / max(elapsed, 1)
                    eta = (total_target - completed) / max(rate, 0.1)
                    logger.info(
                        f"  {completed}/{total_target} accepted | rejected={rejected_count} | "
                        f"~${completed * COST_PER_PATTERN:.2f} | ETA: {eta/60:.1f} min"
                    )
            return True

        # All 3 tries failed
        with progress_lock:
            rejected_count += 1
        return False

    logger.info("Starting concurrent generation...")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = [executor.submit(process_job, j) for j in jobs]
        for _ in as_completed(futures):
            pass

    # Write all accepted patterns to disk
    written = 0
    for category, patterns in accepted.items():
        for idx, p in enumerate(patterns, 1):
            case_id = f"case_{category}_{idx:03d}"
            full = {
                "id": case_id,
                "title": p["title"],
                "pattern": category,
                "narrative": p["narrative"],
                "indicators": p["indicators"],
                "source": "llm_v2_curated",
            }
            out_path = FRAUD_CASES_DIR / f"{case_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full, f, indent=2, ensure_ascii=False)
            written += 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Generated {written} patterns in {elapsed/60:.1f} min")
    logger.info(f"Cost: ~${written * COST_PER_PATTERN:.2f}")
    logger.info(f"Rejected (failed validation/dedup): {rejected_count}")
    logger.info("Final distribution:")
    for cat in sorted(accepted.keys()):
        logger.info(f"  {cat}: {len(accepted[cat])}")
    logger.info("")
    logger.info("Next: python -m src.agentic.build_knowledge_base")


if __name__ == "__main__":
    main()
    