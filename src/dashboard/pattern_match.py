"""Shared pattern-matching helper used by Investigate, Monitor, Test, and Insights.

Calls into agentic.tools.search_fraud_cases with skip_ood=True because the queries
built here come from already-flagged transactions — they are fraud-by-construction
and should bypass the OOD gate (which exists for user-typed queries on Pattern
Library "Test the matcher").
"""

from __future__ import annotations

import logging
from typing import Optional

from src.dashboard.agent_log import log_pattern_match

logger = logging.getLogger(__name__)


def _build_query_text(row) -> str:
    """Turn a transaction row into a natural-language query for semantic search."""
    parts = []

    amt = row.get("TransactionAmt")
    if amt is not None:
        try:
            parts.append(f"transaction amount ${float(amt):.2f}")
        except (TypeError, ValueError):
            pass

    product = row.get("ProductCD")
    if product:
        parts.append(f"on channel {product}")

    vel_24h = row.get("card1_txn_count_24h")
    if vel_24h is not None:
        try:
            v = int(vel_24h)
            if v > 1:
                parts.append(f"with {v} transactions on this card in last 24 hours")
        except (TypeError, ValueError):
            pass

    vel_1h = row.get("card1_txn_count_1h")
    if vel_1h is not None:
        try:
            v = int(vel_1h)
            if v > 0:
                parts.append(f"and {v} in the last hour")
        except (TypeError, ValueError):
            pass

    z = row.get("card1_amt_zscore")
    if z is not None:
        try:
            z_val = float(z)
            if abs(z_val) >= 1.0:
                direction = "above" if z_val > 0 else "below"
                parts.append(f"amount {direction} card's typical pattern (z-score {z_val:+.1f})")
        except (TypeError, ValueError):
            pass

    is_night = row.get("is_night")
    if is_night is not None:
        try:
            if int(is_night) == 1:
                parts.append("at night-time")
        except (TypeError, ValueError):
            pass

    score = row.get("xgb_score")
    if score is not None:
        try:
            if float(score) >= 0.6:
                parts.append("flagged as high risk")
        except (TypeError, ValueError):
            pass

    if not parts:
        return "high risk fraud transaction with anomalous behavior"
    return " ".join(parts)


def find_similar_patterns(row, top_k: int = 3, log_for_txn_id: Optional[str] = None) -> list:
    """Return top_k most similar fraud patterns for a transaction row.

    Bypasses the OOD gate because the query is built from a flagged transaction
    that we already know is fraud-relevant. The OOD gate is reserved for
    user-typed queries on Pattern Library "Test the matcher".

    Returns:
      - list of matches normally
      - empty list on error or empty result (logged)
    """
    query = _build_query_text(row)

    try:
        from src.agentic.tools import search_fraud_cases
        # skip_ood=True because transaction-built queries are fraud by construction.
        results = search_fraud_cases(query, top_k=top_k, skip_ood=True)
    except Exception as e:
        logger.warning(f"Pattern search failed: {e}")
        return []

    # Even with skip_ood=True we keep the OOD-rejected check defensively, in
    # case the function signature ever changes — better safe than silently broken.
    if results and isinstance(results[0], dict) and results[0].get("ood_rejected"):
        logger.info(f"Query rejected as OOD (unexpected with skip_ood=True): {results[0].get('reason')}")
        return []

    # Filter out error responses
    if results and isinstance(results[0], dict) and "error" in results[0]:
        logger.warning(f"Pattern search returned error: {results[0]['error']}")
        return []

    # Log if requested
    if log_for_txn_id and results:
        try:
            top_match = results[0]
            log_pattern_match(
                txn_id=str(log_for_txn_id),
                pattern=top_match.get("pattern", "unknown"),
                case_id=top_match.get("id", ""),
                similarity=float(top_match.get("similarity") or 0),
            )
        except Exception:
            pass

    return results


def top_pattern_label(row) -> Optional[str]:
    """Quick lookup: just the top pattern name for a transaction.

    Used by Monitor table badges. Returns None if no match found.
    """
    matches = find_similar_patterns(row, top_k=1)
    if not matches:
        return None
    pattern = matches[0].get("pattern", "")
    if not pattern:
        return None
    return pattern.replace("_", " ").title()
