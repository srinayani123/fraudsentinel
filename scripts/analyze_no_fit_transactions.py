"""Analyze the high-XGBoost transactions in the demo sample to derive empirical
thresholds for the new pattern archetypes (device_takeover, credential_compromise,
engineered_anomaly).

These archetypes target the fraud surfaces XGBoost actually flags but the
existing 290-pattern catalog doesn't cover: device fingerprint anomalies,
elevated card counters, abnormal Vesta engineered features.

Output:
    models/empirical_thresholds.json — keyed by feature, with median/p75/p90/p99
    statistics computed over the top-300 highest-XGBoost transactions plus
    the same statistics computed over confirmed isFraud=1 transactions for
    comparison.

The pattern generator (src/agentic/generate_new_patterns.py) reads this file
and uses the thresholds in its prompts so generated patterns reference real
data distributions, not LLM guesses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR, SAMPLES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = MODELS_DIR / "empirical_thresholds.json"
TOP_N_BY_SCORE = 300


# Feature families we care about for the new archetypes
DEVICE_FEATURES = [f"D{i}" for i in range(1, 16)]
CARD_COUNTER_FEATURES = [f"C{i}" for i in range(1, 15)]
ENGINEERED_FEATURES = [f"V{i}" for i in [200, 201, 202, 220, 240, 258, 280, 305, 307, 320]]
VERIFICATION_FEATURES = [f"M{i}" for i in range(1, 10)]
BEHAVIORAL_FEATURES = [
    "TransactionAmt",
    "card1_txn_count_1h",
    "card1_txn_count_24h",
    "card1_amt_zscore",
    "card1_seconds_since_last",
    "is_night",
    "txn_hour",
    "emails_match",
]


def _safe_quantile(series: pd.Series, q: float) -> float | None:
    """Compute a quantile, handling missing data."""
    try:
        s = series.dropna()
        if len(s) == 0:
            return None
        return float(np.quantile(s, q))
    except Exception as e:
        logger.debug(f"quantile failed: {e}")
        return None


def _safe_mean(series: pd.Series) -> float | None:
    try:
        s = series.dropna()
        if len(s) == 0:
            return None
        return float(s.mean())
    except Exception:
        return None


def _safe_count_nonzero(series: pd.Series) -> int:
    try:
        return int((series.dropna() != 0).sum())
    except Exception:
        return 0


def compute_feature_stats(df: pd.DataFrame, feature: str) -> dict:
    """Compute median/p75/p90/p99/mean for a feature."""
    if feature not in df.columns:
        return {"available": False}

    return {
        "available": True,
        "median": _safe_quantile(df[feature], 0.50),
        "p75": _safe_quantile(df[feature], 0.75),
        "p90": _safe_quantile(df[feature], 0.90),
        "p99": _safe_quantile(df[feature], 0.99),
        "mean": _safe_mean(df[feature]),
        "count_nonzero": _safe_count_nonzero(df[feature]),
        "n": int(df[feature].dropna().shape[0]),
    }


def analyze_subgroup(df: pd.DataFrame, label: str) -> dict:
    """Compute stats for one subgroup (high-XGBoost or confirmed-fraud)."""
    logger.info(f"Analyzing subgroup: {label} ({len(df):,} rows)")

    out = {"label": label, "n_rows": int(len(df)), "features": {}}
    all_features = (
        DEVICE_FEATURES
        + CARD_COUNTER_FEATURES
        + ENGINEERED_FEATURES
        + VERIFICATION_FEATURES
        + BEHAVIORAL_FEATURES
    )
    for feat in all_features:
        out["features"][feat] = compute_feature_stats(df, feat)

    return out


def main():
    sample_path = SAMPLES_DIR / "demo_transactions.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(f"Demo sample not found at {sample_path}")

    logger.info(f"Loading {sample_path}")
    df = pd.read_parquet(sample_path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Score with XGBoost
    logger.info("Scoring with XGBoost…")
    from src.ml_models.inference import score_dataframe
    df["xgb_score"] = score_dataframe(df)

    # Subgroups for comparison
    top_score_df = df.nlargest(TOP_N_BY_SCORE, "xgb_score")
    fraud_df = df[df.get("isFraud", 0) == 1] if "isFraud" in df.columns else pd.DataFrame()
    legit_df = df[df.get("isFraud", 0) == 0] if "isFraud" in df.columns else pd.DataFrame()

    output = {
        "_meta": {
            "source_file": str(sample_path),
            "total_rows": int(len(df)),
            "top_n_by_score": TOP_N_BY_SCORE,
            "xgb_score_threshold_top_n": float(top_score_df["xgb_score"].min()),
            "n_confirmed_fraud": int(len(fraud_df)),
            "n_confirmed_legit": int(len(legit_df)),
        },
        "subgroups": {},
    }

    output["subgroups"]["top_xgb_score"] = analyze_subgroup(
        top_score_df, "top_xgb_score"
    )
    if not fraud_df.empty:
        output["subgroups"]["confirmed_fraud"] = analyze_subgroup(
            fraud_df, "confirmed_fraud"
        )
    if not legit_df.empty:
        output["subgroups"]["confirmed_legit"] = analyze_subgroup(
            legit_df, "confirmed_legit"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Wrote thresholds to {OUTPUT_PATH}")

    # Quick text summary for the user
    logger.info("")
    logger.info("KEY FINDINGS (top-XGBoost subgroup):")
    top_stats = output["subgroups"]["top_xgb_score"]["features"]
    interesting = ["C1", "C8", "C11", "C13", "V200", "V258", "V307", "D1", "D4", "D10"]
    for feat in interesting:
        s = top_stats.get(feat, {})
        if not s.get("available"):
            continue
        logger.info(
            f"  {feat:<10} median={s.get('median', '—'):<10} "
            f"p75={s.get('p75', '—'):<10} p90={s.get('p90', '—'):<10} "
            f"p99={s.get('p99', '—')}"
        )

    logger.info("")
    logger.info("Next: python -m src.agentic.generate_new_patterns")


if __name__ == "__main__":
    main()
    