"""SHAP-grounded investigation coach.

When no catalogued fraud pattern fits a flagged transaction, this module
generates a verification checklist from the XGBoost SHAP attribution itself.

Hybrid streaming architecture:
  1. Rule-based baseline (instant, ~50ms) — one entry per qualifying SHAP
     driver. NO cross-feature dedup; multiple drivers in the same family each
     get their own placeholder, ready for the LLM to differentiate.
  2. LLM enrichment (~3-5s) — rewrites EVERY model-driven check with prose
     that LEADS WITH THE ACTION (not the feature name), names the fraud
     surface confidently, and differentiates same-family features by what
     each one likely captures.

Negative-SHAP drivers (those that decreased fraud probability) are filtered
out — investigators look for what made the score high, not what tempered it.

Baseline anti-fraud checks (step-up auth, disputes, hot-list) are always
appended because they apply to every fraud investigation regardless of the
model's signal.
"""

from __future__ import annotations

import json
import os

from src.utils.config import DEFAULT_ANTHROPIC_MODEL
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Filter out drivers below this magnitude (likely noise) or strongly negative
# (legitimate-signaling, not what we want analysts to verify when investigating
# fraud)
MIN_POSITIVE_SHAP = 0.05
MAX_NEGATIVE_SHAP = -0.3


# ============================================================================
# Auto-check helpers — used by rule entries to mark a check present/absent
# from the transaction data without LLM involvement
# ============================================================================
def _check_velocity_present(row: dict, driver: dict) -> str:
    feature = driver.get("feature", "")
    raw_val = driver.get("value_raw")

    if "txn_count_1h" in feature:
        try:
            v = int(raw_val) if raw_val is not None else 0
            return "present" if v >= 5 else "absent"
        except (TypeError, ValueError):
            return "manual"
    if "txn_count_24h" in feature:
        try:
            v = int(raw_val) if raw_val is not None else 0
            return "present" if v >= 20 else "absent"
        except (TypeError, ValueError):
            return "manual"
    if "seconds_since_last" in feature:
        try:
            v = float(raw_val) if raw_val is not None else 99999
            return "present" if v < 60 else "absent"
        except (TypeError, ValueError):
            return "manual"
    return "manual"


def _check_amount_present(row: dict, driver: dict) -> str:
    feature = driver.get("feature", "")
    raw_val = driver.get("value_raw")

    if "amt_zscore" in feature:
        try:
            z = abs(float(raw_val)) if raw_val is not None else 0
            return "present" if z >= 3.0 else "absent"
        except (TypeError, ValueError):
            return "manual"
    if feature == "TransactionAmt":
        try:
            amt = float(raw_val) if raw_val is not None else 0
            return "present" if amt >= 500 else "absent"
        except (TypeError, ValueError):
            return "manual"
    return "manual"


def _check_email_risk(row: dict, driver: dict) -> str:
    feature = driver.get("feature", "")
    raw_val = driver.get("value_raw")
    if "is_highrisk" in feature:
        try:
            return "present" if int(raw_val or 0) == 1 else "absent"
        except (TypeError, ValueError):
            return "manual"
    if "emails_match" in feature:
        try:
            return "absent" if int(raw_val or 1) == 1 else "present"
        except (TypeError, ValueError):
            return "manual"
    return "manual"


def _check_timing(row: dict, driver: dict) -> str:
    feature = driver.get("feature", "")
    raw_val = driver.get("value_raw")
    if feature == "is_night":
        try:
            return "present" if int(raw_val or 0) == 1 else "absent"
        except (TypeError, ValueError):
            return "manual"
    if feature == "txn_hour":
        try:
            h = int(raw_val) if raw_val is not None else -1
            return "present" if 0 <= h <= 5 else "absent"
        except (TypeError, ValueError):
            return "manual"
    return "manual"


# ============================================================================
# Rule mapping (baseline templates — LLM enrichment replaces these per-txn)
# ============================================================================
_FEATURE_RULES: dict[str, dict] = {
    "TransactionAmt": {
        "question": "Verify the transaction amount is consistent with the cardholder's typical spending",
        "rationale": "Large amounts well above the card's baseline are a classic fraud indicator.",
        "auto_check": _check_amount_present,
    },
    "card1_amt_zscore": {
        "question": "Confirm the amount fits this card's typical spending range",
        "rationale": "An amount many standard deviations from the card's average suggests compromise.",
        "auto_check": _check_amount_present,
    },
    "card1_txn_count_1h": {
        "question": "Check whether the cardholder is making rapid back-to-back purchases",
        "rationale": "High 1-hour velocity is a hallmark of card-testing attacks.",
        "auto_check": _check_velocity_present,
    },
    "card1_txn_count_24h": {
        "question": "Review the card's 24-hour transaction count for unusual volume",
        "rationale": "Sudden 24-hour bursts often indicate active fraud campaigns.",
        "auto_check": _check_velocity_present,
    },
    "card1_seconds_since_last": {
        "question": "Check the gap since the card's previous transaction",
        "rationale": "Transactions seconds apart often indicate automated card-testing.",
        "auto_check": _check_velocity_present,
    },
    "is_night": {
        "question": "Verify whether the cardholder typically transacts at night",
        "rationale": "Off-hours activity on a daytime user is a strong takeover signal.",
        "auto_check": _check_timing,
    },
    "txn_hour": {
        "question": "Check whether the transaction time fits the cardholder's pattern",
        "rationale": "Unusual transaction hours warrant review.",
        "auto_check": _check_timing,
    },
    "P_emaildomain_is_highrisk": {
        "question": "Verify the purchaser's email domain reputation",
        "rationale": "High-risk email domains correlate strongly with fraud.",
        "auto_check": _check_email_risk,
    },
    "R_emaildomain_is_highrisk": {
        "question": "Verify the recipient's email domain reputation",
        "rationale": "Recipient email risk is especially predictive in CNP fraud.",
        "auto_check": _check_email_risk,
    },
    "emails_match": {
        "question": "Check whether purchaser and recipient emails match",
        "rationale": "Mismatched purchaser/recipient emails are common in fraudulent gift purchases.",
        "auto_check": _check_email_risk,
    },
    "dist1": {
        "question": "Review the geographic distance for this transaction",
        "rationale": "Implausible distance from the cardholder's usual location indicates potential takeover.",
    },
    "dist2": {
        "question": "Verify shipping vs billing geographic consistency",
        "rationale": "Large bill-to/ship-to gaps are common fraud signals.",
    },
}


_FAMILY_RULES: dict[str, dict] = {
    "device_identity": {
        "question": "Verify whether this card has been used on this device before",
        "rationale": "Device fingerprinting features carry strong fraud signal — an unrecognized device-card pairing is a top takeover indicator.",
    },
    "card_counter": {
        "question": "Review the card's recent usage counters for unusual activity",
        "rationale": "Card-related counters track aggregated card behavior. Sudden shifts here often indicate compromised credentials being tested.",
    },
    "engineered": {
        "question": "Cross-check this transaction against the card's full transaction history",
        "rationale": "Vesta engineered features encode complex behavioral patterns. When they drive the score, manual review of recent history is warranted.",
    },
    "verification": {
        "question": "Confirm address/identity match flags through manual review",
        "rationale": "Match/verification flags compare submitted data against records. When these drive the score, identity may be partially valid but suspect.",
    },
    "amount": {
        "question": "Verify the amount is consistent with the cardholder's pattern",
        "rationale": "Amount-related anomalies often reveal compromise.",
        "auto_check": _check_amount_present,
    },
    "behavioral_velocity_timing": {
        "question": "Review velocity and timing for this card's recent activity",
        "rationale": "Velocity/timing anomalies are leading indicators of card abuse.",
    },
    "geographic": {
        "question": "Verify geographic consistency between billing, shipping, and IP",
        "rationale": "Geographic inconsistency is among the strongest fraud signals.",
    },
    "card_metadata": {
        "question": "Cross-check card metadata against expected patterns",
        "rationale": "BIN-level anomalies can indicate test cards or stolen ranges.",
    },
    "email": {
        "question": "Verify email reputation for both purchaser and recipient",
        "rationale": "Email risk signals fraudulent intent.",
        "auto_check": _check_email_risk,
    },
}


# Baseline anti-fraud checks — always included, intentionally fixed because
# they apply to every fraud investigation regardless of model signal
_BASELINE_CHECKS = [
    {
        "feature": "_baseline_step_up_auth",
        "label": "Confirm cardholder is reachable for verification",
        "rationale": "Step-up authentication should be triggered for any high-risk transaction before settlement.",
        "status": "manual",
        "actual_value": None,
        "category": "Baseline check",
    },
    {
        "feature": "_baseline_recent_disputes",
        "label": "Check for recent disputes or chargebacks on this card",
        "rationale": "Recent dispute history dramatically raises the prior probability of fraud.",
        "status": "manual",
        "actual_value": None,
        "category": "Baseline check",
    },
    {
        "feature": "_baseline_card_status",
        "label": "Verify the card is not on the issuer's hot-list or watch list",
        "rationale": "Cards may have been flagged through external channels not visible to the model.",
        "status": "manual",
        "actual_value": None,
        "category": "Baseline check",
    },
]


# ============================================================================
# Rule-based baseline (instant, ~50ms)
# ============================================================================
def _filter_drivers(drivers: list[dict]) -> list[dict]:
    """Skip noise drivers and strongly-negative ones.

    We don't want analysts to verify features that the model thinks are
    REASSURING — when investigating fraud, focus on positive-SHAP features.
    """
    filtered = []
    for d in drivers:
        shap_val = d.get("shap", 0)
        if shap_val is None:
            continue
        if abs(shap_val) < MIN_POSITIVE_SHAP:
            continue
        if shap_val < MAX_NEGATIVE_SHAP:
            continue
        filtered.append(d)
    return filtered


def _categorize_check(check: dict, driver: dict) -> dict:
    if "category" not in check:
        check["category"] = "Model-driven check"
    check["shap_value"] = driver.get("shap")
    check["feature"] = driver.get("feature")
    check["feature_label"] = driver.get("label")
    check["feature_value"] = driver.get("value")
    check["family"] = driver.get("family")
    return check


def build_rule_based_checklist(
    xgb_attribution: dict,
    row_dict: dict,
    top_k: int = 5,
) -> list[dict]:
    """Generate the baseline rule-based checklist (instant).

    IMPORTANT — no cross-feature dedup. Multiple drivers from the same family
    each get their OWN placeholder entry. This lets the LLM enrichment stage
    produce differentiated, feature-specific questions for each one. Without
    this, two C-fields would collapse to one generic "review card counters"
    entry and lose information.
    """
    if not xgb_attribution or xgb_attribution.get("error"):
        return list(_BASELINE_CHECKS)

    drivers_raw = xgb_attribution.get("top_drivers", [])
    drivers = _filter_drivers(drivers_raw)[:top_k]
    if not drivers:
        return list(_BASELINE_CHECKS)

    checks = []

    for driver in drivers:
        feature = driver.get("feature", "")
        family = driver.get("family", "other")

        rule = _FEATURE_RULES.get(feature) or _FAMILY_RULES.get(family)

        if rule is None:
            checks.append(
                _categorize_check(
                    {
                        "label": f"Investigate the influence of {driver.get('label', feature)}",
                        "rationale": "AI is generating a tailored question for this feature…",
                        "status": "manual",
                        "actual_value": driver.get("value"),
                        "category": "Pending AI",
                    },
                    driver,
                )
            )
            continue

        auto_check_fn = rule.get("auto_check")
        if auto_check_fn is not None:
            try:
                status = auto_check_fn(row_dict, driver)
            except Exception as e:
                logger.debug(f"auto-check failed for {feature}: {e}")
                status = "manual"
        else:
            status = "manual"

        check = {
            "label": rule["question"],
            "rationale": rule["rationale"],
            "status": status,
            "actual_value": driver.get("value"),
            "category": "Model-driven check",
        }
        checks.append(_categorize_check(check, driver))

    checks.extend(_BASELINE_CHECKS)
    return checks


# ============================================================================
# LLM enrichment — rewrites ALL model-driven checks per-transaction
# ============================================================================
SHAP_COACH_PROMPT = """You are a fraud-investigation coach writing verification checklist items for a senior fraud analyst. You receive the SHAP attribution for a flagged transaction plus a baseline checklist.

YOUR JOB: Rewrite every entry whose category is "Model-driven check" or "Pending AI" into a sharp, action-led verification question. DO NOT modify "Baseline check" entries.

OUTPUT — STRICT JSON ONLY:
{
  "enriched_checks": [
    {
      "feature": "<feature_name from the original entry, exactly as given>",
      "label": "<the verification action — see rules below>",
      "rationale": "<one sentence explaining what this feature likely captures and why it drove the score>"
    },
    ...
  ]
}

THE LABEL (verification action) — HARD RULES:

1. LEAD WITH THE ACTION VERB. The first word is what the analyst should DO.
   ✗ WRONG: "Investigate V200=8 (SHAP +3.02): this is..."
   ✓ RIGHT: "Pull the card's recent transaction sequence and check for…"
   ✓ RIGHT: "Cross-check the device fingerprint against the card's last 5 transactions"
   ✓ RIGHT: "Confirm the cardholder authorized purchases at this merchant before"

2. NO FEATURE NAMES IN THE LABEL. Don't write "V200=8" or "SHAP +3.02" in the label.
   The UI already shows feature name and SHAP value as separate pills. Repeating them is clutter.

3. DIFFERENTIATE same-family features. If three card-counter features (C1, C8, C11) appear, each should get a DIFFERENT question. Use the SHAP magnitude and feature value to infer what each likely captures:
   - High count + high SHAP → count-based aggregation (transaction volume)
   - Low count + high SHAP → recency-based aggregation
   - Same family but different scales → different roll-up windows
   Don't write three identical "review the counters" questions.

4. NAME THE FRAUD SURFACE confidently in the rationale, not the question. The rationale handles the "why this matters" framing.

5. KEEP IT UNDER 22 WORDS. Sharp, not verbose.

THE RATIONALE — RULES:

- One sentence, max 25 words.
- Start with what the feature LIKELY CAPTURES (you can hypothesize based on family).
- For opaque V-fields: "A Vesta engineered feature whose magnitude here suggests [your inference based on SHAP magnitude]."
- For C-fields: name the likely aggregation (count / sum / distinct merchants / window length).
- For D-fields: name the device/identity aspect (fingerprint stability / first-seen / mismatch).
- Don't hedge with "may" / "could" / "might" — be confident, the SHAP value is the evidence.

EXAMPLES:

For a C-field with high count + high SHAP:
  feature: "C1", value: 181, shap: +2.37
  label: "Pull the card's full transaction count history — 181 is well above typical for normal cardholders"
  rationale: "C1 captures aggregated card-level transaction volume; high values with positive SHAP indicate this card has been used far more than peer cards."

For a different C-field, similar family:
  feature: "C8", value: 22, shap: +1.4
  label: "Verify how many distinct merchants this card has touched recently"
  rationale: "C8 captures merchant-diversity counts; elevated values indicate the card hopping between unrelated merchants — a card-testing signal."

For an opaque V-field:
  feature: "V200", value: 8, shap: +3.02
  label: "Run a Vesta deep-dive: this engineered feature is the dominant fraud driver"
  rationale: "V200 is a Vesta-engineered feature whose magnitude here suggests sequencing or timing anomalies invisible in raw card metrics."

For a D-field:
  feature: "D1", value: 0, shap: +0.42
  label: "Cross-check whether this card has been seen on the current device fingerprint before"
  rationale: "D1 captures device-card pairing stability; a value of 0 indicates first-time device pairing, a top takeover predictor."

OUTPUT JSON ONLY, no preamble, no commentary."""


def enrich_checklist_with_llm(
    rule_based_checklist: list[dict],
    xgb_attribution: dict,
    row_dict: dict,
    api_key: str | None = None,
) -> tuple[list[dict], dict]:
    """Send all model-driven checks to the LLM for transaction-specific rewriting.

    Replaces "Model-driven check" and "Pending AI" entries with "AI-generated check"
    entries with sharp action-led questions tailored to each driver.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_based_checklist, {"error": "No ANTHROPIC_API_KEY available"}

    enrichable = [
        c for c in rule_based_checklist
        if c.get("category") in ("Model-driven check", "Pending AI")
    ]
    if not enrichable:
        return rule_based_checklist, {"enriched": 0, "skipped": "no enrichable checks"}

    try:
        import anthropic
    except ImportError:
        return rule_based_checklist, {"error": "anthropic package not available"}

    client = anthropic.Anthropic(api_key=api_key)

    drivers_payload = xgb_attribution.get("top_drivers", []) if xgb_attribution else []
    checklist_payload = [
        {
            "feature": c.get("feature"),
            "label": c.get("label"),
            "category": c.get("category"),
            "shap_value": c.get("shap_value"),
            "feature_value": c.get("feature_value"),
            "family": c.get("family"),
        }
        for c in rule_based_checklist
    ]

    user_msg = f"""SHAP ATTRIBUTION (top drivers, ranked by |SHAP|):
{json.dumps(drivers_payload, indent=2, default=str)}

CURRENT BASELINE CHECKLIST:
{json.dumps(checklist_payload, indent=2, default=str)}

Generate enriched_checks. Rewrite EVERY entry whose category is "Model-driven check" or "Pending AI". Skip "Baseline check" entries entirely. Each rewrite must lead with an action verb, omit feature names from the label, and differentiate same-family features by what they likely capture.

Output JSON only."""

    try:
        resp = client.messages.create(
            model=DEFAULT_ANTHROPIC_MODEL,
            max_tokens=1500,
            system=SHAP_COACH_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e}")
        return rule_based_checklist, {"error": f"{type(e).__name__}: {e}"}

    enrichments = {
        e["feature"]: e
        for e in parsed.get("enriched_checks", [])
        if "feature" in e
    }
    if not enrichments:
        return rule_based_checklist, {"enriched": 0}

    out = []
    enriched_count = 0
    for c in rule_based_checklist:
        if c.get("category") in ("Model-driven check", "Pending AI"):
            f = c.get("feature")
            if f in enrichments:
                e = enrichments[f]
                out.append({
                    **c,
                    "label": e.get("label", c["label"]),
                    "rationale": e.get("rationale", c.get("rationale", "")),
                    "category": "AI-generated check",
                })
                enriched_count += 1
                continue
        out.append(c)

    return out, {"enriched": enriched_count, "total_enrichable": len(enrichable)}


# ============================================================================
# Top-level entry point
# ============================================================================
def build_shap_grounded_checklist(
    xgb_attribution: dict,
    row_dict: dict,
    top_k: int = 5,
) -> dict:
    """Build the rule-based baseline (instant). The page calls this on tab open
    for immediate render, then triggers enrich_checklist_with_llm for
    transaction-specific tailoring.
    """
    checks = build_rule_based_checklist(xgb_attribution, row_dict, top_k=top_k)

    summary = {"present": 0, "absent": 0, "manual": 0}
    for c in checks:
        status = c.get("status", "manual")
        if status in summary:
            summary[status] += 1
        else:
            summary["manual"] += 1

    drivers = xgb_attribution.get("top_drivers", []) if xgb_attribution else []
    filtered_drivers = _filter_drivers(drivers)[:top_k]

    return {
        "checks": checks,
        "summary": summary,
        "source": "shap_grounded",
        "drivers_used": [d.get("feature") for d in filtered_drivers],
    }
