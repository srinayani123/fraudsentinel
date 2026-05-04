"""Pattern Coach — converts a matched fraud pattern's indicators into a verifiable
checklist for the current transaction.

The coach receives BOTH the technical indicators AND their plain-English
explanations from the matched pattern. The LLM uses the explanations as
human-readable labels while keeping the technical indicators as the
machine-checkable thresholds (mapped to feature names + operators + values).

Result: each check has a plain-English headline (analyst sees first) and a
technical threshold receipt (shown as monospace subtext) — same paired
rendering as the SHAP-grounded checklist and the Pattern Library page.

Robust against both list and JSON-stringified inputs for `indicators` and
`indicator_explanations`, since both shapes can appear depending on whether
the pattern came from a fresh search or a cached session value.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


SYSTEM_PROMPT = """You are a fraud investigation tooling agent. Your job: convert a matched fraud pattern's indicators (technical thresholds + plain-English explanations) into structured, machine-verifiable checks against a single transaction.

For each indicator from the matched pattern, output ONE check object describing what to verify. The "label" should be the plain-English explanation (what the analyst reads first). The "technical" field should be the precise threshold (shown as monospace receipt below the label). The check type and parameters drive automated verification.

Each check must have one of these types:

1. "numeric" — feature compared to a threshold
   {
     "type": "numeric",
     "label": "<plain-English explanation as the user-facing headline>",
     "technical": "<original technical indicator string>",
     "feature": "<feature_name>",
     "operator": ">=" | ">" | "<=" | "<" | "==" | "!=",
     "value": <number>
   }

2. "categorical" — feature value is in a set
   {
     "type": "categorical",
     "label": "<plain-English explanation>",
     "technical": "<original technical indicator>",
     "feature": "<feature_name>",
     "values": ["...", "..."]
   }

3. "manual" — requires human judgment, can't be verified from features alone
   {
     "type": "manual",
     "label": "<plain-English explanation as the user-facing headline>",
     "technical": "<original technical indicator string>"
   }

AVAILABLE FEATURES (use these exact names for numeric/categorical checks):

Transaction-level:
- TransactionAmt (float, dollars)
- ProductCD (string: W, C, R, H, S)
- txn_hour (0-23)
- is_night (0/1)
- txn_dayofweek (0-6, Monday=0)
- emails_match (0/1)
- addr1 (float)
- dist1 (float, miles)

Card-aggregated counters and ratios:
- card1_txn_count_1h, card1_txn_count_24h, card1_txn_count_7d (int)
- card1_amt_zscore (float, z-score of current amount vs card mean)
- card1_amt_max_24h, card1_amt_sum_1h, card1_amt_sum_24h, card1_amt_sum_7d (float)
- card1_amt_mean, card1_amt_std (float)
- card1_total_txns (int, lifetime count)
- card1_distinct_products (int, distinct ProductCD values)
- card1_distinct_addr1 (int, distinct shipping addresses)
- card1_seconds_since_last (int)

Card-counter aggregations (from raw IEEE-CIS):
- C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14 (float)

Device fingerprint counters (from raw IEEE-CIS):
- D1, D2, D3, D4, D10, D15 (float)

Vesta engineered features (from raw IEEE-CIS, sparse):
- V1, V12, V14, V20, V30, V40, V50, V70, V90, V100, V130, V160, V200, V250, V300 (float)

Email/risk:
- P_emaildomain (string), P_emaildomain_is_highrisk (0/1), P_emaildomain_isnull (0/1)
- R_emaildomain (string), R_emaildomain_is_highrisk (0/1), R_emaildomain_isnull (0/1)

PATTERN-MATCHING TIPS:
- Indicators like "C1 > 60 (p90)" → numeric check on C1, operator >, value 60
- Indicators like "D1 = 0 indicating first-time pairing" → numeric check on D1, operator ==, value 0
- Indicators like "V200 > 6.0 (p90 threshold)" → numeric check on V200, operator >, value 6.0
- Indicators like "card1_amt_zscore between -1.5 and +1.5" → use TWO checks isn't supported, just use ONE check with a reasonable operator; OR mark "manual" if the indicator describes a range
- Indicators like "TransactionAmt_card1_mean_ratio between 0.7 and 1.3" → "manual" (this exact feature isn't in the list above)
- Indicators like "addr1_P_emaildomain match = 1" → "manual" (this is a derived field not in the dataset)
- Indicators that mention features NOT in the list above → "manual"

CRITICAL RULES:
- The "label" MUST be the plain-English explanation provided for that indicator — verbatim or near-verbatim. Do NOT paraphrase technical jargon as the label.
- The "technical" field MUST preserve the original technical indicator string verbatim.
- Each indicator from the input becomes exactly ONE check.
- Generate one check per indicator provided (typically 4-7 checks).
- For range indicators (e.g. "between 0.7 and 1.3"), pick the more decisive single threshold for the numeric check, or mark "manual" if neither bound is more decisive than the other.

Output STRICT JSON only:
{
  "checks": [
    { ... check 1 ... },
    { ... check 2 ... }
  ]
}"""


def _coerce_to_list(raw: Any) -> list[str]:
    """Defensive parser: accept list, JSON-stringified list, or anything else.

    pattern_coach can be called with input from multiple sources (fresh ChromaDB
    query, session_state cache, agent retrieval). One of those paths previously
    leaked JSON strings. This guard means the function never iterates a string
    character-by-character regardless of upstream shape.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        # Try JSON parse first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except (json.JSONDecodeError, ValueError):
            pass
        # If it's a plain string and not JSON, treat as a single indicator
        return [s]
    return []


def _build_prompt(matched_pattern: dict) -> str:
    pattern_title = matched_pattern.get("title", "")
    pattern_label = matched_pattern.get("pattern", "")
    indicators = _coerce_to_list(matched_pattern.get("indicators"))
    explanations = _coerce_to_list(matched_pattern.get("indicator_explanations"))
    snippet = matched_pattern.get("snippet", matched_pattern.get("narrative", ""))[:400]

    paired_lines = []
    for i, ind in enumerate(indicators):
        explanation = explanations[i] if i < len(explanations) else None
        if explanation:
            paired_lines.append(
                f"{i+1}. INDICATOR: {ind}\n   EXPLANATION: {explanation}"
            )
        else:
            paired_lines.append(f"{i+1}. INDICATOR: {ind}")

    paired_block = "\n\n".join(paired_lines) if paired_lines else "(no indicators listed)"

    return f"""Matched fraud pattern: {pattern_title} (category: {pattern_label})

Pattern description:
{snippet}

Indicators to convert into checks (each has a TECHNICAL form and a plain-English EXPLANATION — use the EXPLANATION as the user-facing label, preserve the INDICATOR as the technical receipt):

{paired_block}

For each indicator above, output exactly ONE check. The "label" field is the plain-English EXPLANATION. The "technical" field is the original INDICATOR string. Use "numeric" type for indicators referencing a feature in the available list with a clear threshold. Use "manual" type for indicators referencing features outside the available list, or for ambiguous range descriptions.

Output JSON only."""


def _evaluate_check(check: dict, txn_row: Any) -> dict:
    check_type = check.get("type", "manual")

    if check_type == "manual":
        return {**check, "status": "manual", "actual_value": None}

    feature = check.get("feature", "")
    if feature not in txn_row.index:
        return {
            **check,
            "status": "unknown",
            "actual_value": None,
            "note": f"Feature '{feature}' not available",
        }

    actual = txn_row[feature]
    # NaN check (covers float('nan') which is not equal to itself)
    if actual is None or (isinstance(actual, float) and actual != actual):
        return {
            **check,
            "status": "unknown",
            "actual_value": None,
            "note": "Feature value missing",
        }

    if check_type == "numeric":
        try:
            actual_num = float(actual)
            target = float(check.get("value", 0))
            op = check.get("operator", "==")
            ops = {
                ">=": actual_num >= target,
                ">": actual_num > target,
                "<=": actual_num <= target,
                "<": actual_num < target,
                "==": actual_num == target,
                "!=": actual_num != target,
            }
            present = ops.get(op, False)
            return {
                **check,
                "status": "present" if present else "absent",
                "actual_value": actual_num,
            }
        except (TypeError, ValueError):
            return {
                **check,
                "status": "unknown",
                "actual_value": str(actual),
                "note": "Could not parse as number",
            }

    if check_type == "categorical":
        try:
            actual_str = str(actual)
            values = [str(v) for v in check.get("values", [])]
            present = actual_str in values
            return {
                **check,
                "status": "present" if present else "absent",
                "actual_value": actual_str,
            }
        except Exception:
            return {**check, "status": "unknown", "actual_value": str(actual)}

    return {**check, "status": "unknown", "actual_value": None}


def build_checklist(
    matched_pattern: dict,
    txn_row: Any,
    api_key: str | None = None,
) -> dict:
    """Generate a structured investigation checklist for a transaction.

    Args:
        matched_pattern: pattern dict from ChromaDB retrieval
        txn_row: transaction row (pandas Series or dict)
        api_key: Anthropic API key. If None, falls back to ANTHROPIC_API_KEY env var.
                 Pass byok.get_api_key() from Streamlit context to support BYOK.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "error": "Anthropic API key required — paste it in Settings",
            "checks": [],
        }

    # Defensive: if indicators is a JSON string (legacy path), it'll still parse
    indicators_check = _coerce_to_list(matched_pattern.get("indicators"))
    if not indicators_check:
        return {"error": "Matched pattern has no indicators to verify", "checks": []}

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(matched_pattern)}],
        )
        text = resp.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        raw_checks = parsed.get("checks", [])

    except json.JSONDecodeError as e:
        return {"error": f"Could not parse coach response: {e}", "checks": []}
    except anthropic.APIError as e:
        return {"error": f"API error: {str(e)[:200]}", "checks": []}
    except Exception as e:
        logger.error(f"Pattern coach failed: {e}")
        return {"error": f"Coach failed: {type(e).__name__}", "checks": []}

    evaluated = [_evaluate_check(c, txn_row) for c in raw_checks]

    summary = {
        "present": sum(1 for c in evaluated if c["status"] == "present"),
        "absent": sum(1 for c in evaluated if c["status"] == "absent"),
        "manual": sum(1 for c in evaluated if c["status"] == "manual"),
        "unknown": sum(1 for c in evaluated if c["status"] == "unknown"),
    }

    return {
        "checks": evaluated,
        "matched_pattern_title": matched_pattern.get("title", ""),
        "matched_pattern_id": matched_pattern.get("id", ""),
        "matched_pattern_category": matched_pattern.get("pattern", ""),
        "summary": summary,
    }
