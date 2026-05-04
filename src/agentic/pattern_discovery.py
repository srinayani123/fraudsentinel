"""Pattern Discovery Agent — finds coordinated attack patterns in recent flagged transactions.

Unlike per-transaction scoring (rules + ML), this looks ACROSS transactions for
patterns that suggest coordination: same BIN range, time clustering, channel
clustering, amount clustering, etc.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.utils.config import DEFAULT_ANTHROPIC_MODEL
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


SYSTEM_PROMPT = """You are a senior fraud analyst examining a batch of recent high-risk transactions to identify coordinated attacks — multiple transactions that, taken together, suggest organized activity rather than independent isolated fraud.

You receive a structured summary of recent flagged transactions plus aggregated statistics (BIN clusters, time clusters, channel clusters, etc.).

Your job: identify groups of transactions that appear coordinated, and explain the suspected attack pattern.

A "coordinated cluster" requires AT LEAST 3 transactions sharing meaningful similarity:
- Same or adjacent BIN (card1 prefix)
- Same channel (ProductCD) within a tight time window
- Similar amounts (e.g. all $0.50-$2 — card testing) or stair-stepped amounts
- Same hour-of-day across different cards (suggests automation)
- Similar velocity profiles

Do NOT flag a cluster just because transactions are all "high risk" — they're flagged because they ARE high risk. Look for SHARED structure beyond risk score.

If nothing meaningful clusters, say so honestly. Do not fabricate.

Output STRICT JSON only, no preamble:
{
  "clusters": [
    {
      "label": "Short name for this attack pattern (4-8 words)",
      "pattern_type": "one of: card_testing_burst, bin_attack, velocity_drain, coordinated_geo, time_burst, channel_targeting, other",
      "transaction_ids": ["list of TransactionID values that belong to this cluster"],
      "confidence": "high | medium | low",
      "reasoning": "2-3 sentence analyst-grade explanation of why these transactions appear coordinated and what attack this suggests",
      "recommended_action": "1 sentence: what should the fraud team do about this?"
    }
  ],
  "summary": "1-2 sentence top-level finding across all clusters. If no clusters, say so.",
  "transactions_analyzed": <integer count>
}"""


def _bucket(values, bucket_size):
    """Group values into buckets of given size. Returns dict of bucket → count."""
    buckets = defaultdict(int)
    for v in values:
        b = (v // bucket_size) * bucket_size
        buckets[int(b)] += 1
    return dict(buckets)


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Pre-compute aggregates so the LLM doesn't need to do statistics — just reasoning."""
    if df.empty:
        return {"transaction_count": 0}

    # Basic counts
    summary: dict[str, Any] = {
        "transaction_count": int(len(df)),
        "total_amount": float(df["TransactionAmt"].sum()),
        "score_range": [float(df["xgb_score"].min()), float(df["xgb_score"].max())],
        "score_avg": float(df["xgb_score"].mean()),
    }

    # BIN clusters (card1 prefix — first 4 digits typically)
    if "card1" in df.columns:
        # Group by BIN range (first 4 digits of card1)
        df_b = df.copy()
        df_b["bin_prefix"] = (df_b["card1"] // 1000).astype(int)  # rough BIN bucketing
        bin_counts = df_b["bin_prefix"].value_counts().head(10).to_dict()
        # Only report BIN groups with 2+ txns (otherwise it's noise)
        summary["bin_clusters"] = {
            f"{k}xxx": int(v) for k, v in bin_counts.items() if v >= 2
        }

    # Time clusters (hour-of-day)
    if "TransactionDT" in df.columns:
        hours = ((df["TransactionDT"] % 86400) // 3600).astype(int)
        hour_counts = Counter(hours)
        # Only report hours with 3+ transactions (potential burst)
        summary["hour_clusters"] = {
            f"{h:02d}:00": int(c) for h, c in hour_counts.most_common() if c >= 3
        }

    # Channel clusters
    if "ProductCD" in df.columns:
        channel_counts = df["ProductCD"].value_counts().to_dict()
        summary["channel_distribution"] = {str(k): int(v) for k, v in channel_counts.items()}

    # Amount clusters (small/medium/large)
    amounts = df["TransactionAmt"]
    summary["amount_distribution"] = {
        "tiny_under_5": int((amounts < 5).sum()),
        "small_5_to_50": int(((amounts >= 5) & (amounts < 50)).sum()),
        "medium_50_to_500": int(((amounts >= 50) & (amounts < 500)).sum()),
        "large_500_plus": int((amounts >= 500).sum()),
    }

    # Email risk concentration
    if "P_emaildomain" in df.columns:
        email_counts = df["P_emaildomain"].value_counts().head(5).to_dict()
        summary["top_email_domains"] = {str(k): int(v) for k, v in email_counts.items() if pd.notna(k)}

    return summary


def build_transaction_list(df: pd.DataFrame, max_n: int = 30) -> list[dict[str, Any]]:
    """Build a compact list of transactions for the LLM to reference."""
    df = df.head(max_n).copy()
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "TransactionID": str(row.get("TransactionID", "")),
            "amount": round(float(row.get("TransactionAmt", 0)), 2),
            "card_bin": int(row.get("card1", 0)) // 1000 if pd.notna(row.get("card1")) else None,
            "channel": str(row.get("ProductCD", "?")),
            "hour": int((row.get("TransactionDT", 0) % 86400) // 3600) if pd.notna(row.get("TransactionDT")) else None,
            "score": round(float(row.get("xgb_score", 0)), 3),
            "velocity_24h": int(row.get("card1_txn_count_24h", 0)) if pd.notna(row.get("card1_txn_count_24h")) else 0,
        })
    return rows


def discover_patterns(df: pd.DataFrame, min_score: float = 0.6, max_txns: int = 30) -> dict[str, Any]:
    """Run pattern discovery on a dataframe of transactions.

    Args:
        df: DataFrame with transaction data and xgb_score column
        min_score: Only analyze transactions with score >= this threshold
        max_txns: Cap on transactions sent to LLM (cost control)

    Returns dict with 'clusters', 'summary', 'transactions_analyzed' keys, or error.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}

    # Filter to high-risk
    df_risk = df[df["xgb_score"] >= min_score].copy()
    if len(df_risk) < 3:
        return {
            "clusters": [],
            "summary": f"Only {len(df_risk)} high-risk transactions in this window — not enough for coordinated-attack analysis.",
            "transactions_analyzed": int(len(df_risk)),
        }

    # Cap to max_txns most recent
    df_risk = df_risk.tail(max_txns)

    # Build the prompt content
    summary = build_summary(df_risk)
    txn_list = build_transaction_list(df_risk, max_n=max_txns)

    user_prompt = f"""Analyze these {len(txn_list)} recent flagged transactions for coordinated attack patterns.

AGGREGATED STATISTICS:
{json.dumps(summary, indent=2)}

INDIVIDUAL TRANSACTIONS:
{json.dumps(txn_list, indent=2)}

Identify any clusters that suggest coordinated activity. Output JSON only."""

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = resp.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        # Defensive: ensure required keys
        result.setdefault("clusters", [])
        result.setdefault("summary", "")
        result["transactions_analyzed"] = int(len(df_risk))

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Pattern discovery returned invalid JSON: {e}")
        return {"error": f"Could not parse agent response: {e}"}
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return {"error": f"API error: {str(e)[:200]}"}
    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        return {"error": f"Discovery failed: {type(e).__name__}: {str(e)[:200]}"}
    