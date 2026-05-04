"""Pre-compute aggregate statistics from the analyst's transaction selection.

Workers consume these aggregates rather than raw rows. This is critical:
  - Sending 50+ raw rows to each Worker would burn tokens and add noise
  - Pre-computed stats give Workers exactly the patterns they need to spot
  - Aggregates also let us compare fraud-vs-legit distributions cheaply

The compute is pure-Python pandas, runs in <100ms, no LLM calls.

Column names match the IEEE-CIS demo dataset confirmed in the user's environment:
  C1-C14, D1-D4/D10/D15, V1-V300 sparse Vesta features, P_emaildomain*,
  R_emaildomain*, card1_txn_count_*, card1_amt_*, card1_distinct_*, is_night,
  txn_hour, txn_dayofweek, ProductCD, isFraud, TransactionDT, etc.
"""

from __future__ import annotations

import pandas as pd

from src.agentic.rule_generator.types import AggregateInput


# ----------------------------------------------------------------------
# Helpers — defensive accessors for series/quantiles/rates
# ----------------------------------------------------------------------
def _safe_p(series: pd.Series, q: float, default: float = 0.0) -> float:
    """Compute a quantile, returning default if series is empty/NaN."""
    if series is None or len(series) == 0:
        return default
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return default
    return float(s.quantile(q))


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return default
    return float(s.mean())


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _format_iso_date(ts) -> str:
    """Convert timestamp-like value to ISO date string. Handles NaT."""
    if ts is None or pd.isna(ts):
        return "unknown"
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


# ----------------------------------------------------------------------
# Filter helper — analyst-facing input set
# ----------------------------------------------------------------------
def filter_transactions(
    df: pd.DataFrame,
    risk_band: str = "high_critical",
    date_range_days: int = 30,
    score_col: str = "xgb_score",
    legit_sample_multiplier: int = 5,
) -> pd.DataFrame:
    """Filter the analyst's full dataset down to the rule-generation input set.

    The output combines:
      - All transactions in the chosen risk_band (where the model thinks fraud
        is happening)
      - A random sample of legitimate (unflagged) transactions for baseline
        comparison, sized to legit_sample_multiplier x flagged_count

    This baseline is critical: without it, every fraud-rate calculation
    degenerates to 100% and Workers can't propose threshold-based rules.

    Args:
        df: full dataset with xgb_score column
        risk_band: "all" | "high" (>= 0.6 and < 0.85) | "critical" (>= 0.85) |
                   "high_critical" (>= 0.6)
        date_range_days: only include transactions from the last N days
                         (uses TransactionDT column if available)
        score_col: column to filter on for risk_band
        legit_sample_multiplier: how many legit (unflagged) transactions to
                                 include per flagged transaction (default 5x)

    Returns:
        filtered DataFrame combining flagged + legit baseline
    """
    out = df.copy()

    # ---- Apply date filter first (to the whole dataset) ----
    # IEEE-CIS TransactionDT is seconds-since-arbitrary-epoch.
    # We approximate by taking the last N days from the max date.
    if "TransactionDT" in out.columns and len(out) > 0:
        try:
            max_dt = out["TransactionDT"].max()
            seconds_in_day = 86400
            min_dt = max_dt - (date_range_days * seconds_in_day)
            out = out[out["TransactionDT"] >= min_dt]
        except Exception:
            pass  # fail open

    if score_col not in out.columns:
        return out

    # ---- Pick the flagged set per risk_band ----
    if risk_band == "critical":
        flagged_mask = out[score_col] >= 0.85
    elif risk_band == "high":
        flagged_mask = (out[score_col] >= 0.6) & (out[score_col] < 0.85)
    elif risk_band == "high_critical":
        flagged_mask = out[score_col] >= 0.6
    else:  # "all"
        return out

    flagged = out[flagged_mask]

    # ---- Add legit baseline ----
    # Sample unflagged transactions (model-clean) for distributional contrast.
    # This is what makes "fraud cards average X transactions vs Y for legit"
    # statements possible in the aggregates.
    legit_pool = out[~flagged_mask]
    if len(flagged) > 0 and len(legit_pool) > 0:
        sample_size = min(len(legit_pool), len(flagged) * legit_sample_multiplier)
        # Use a fixed seed for reproducibility within a single Streamlit session
        legit_sample = legit_pool.sample(n=sample_size, random_state=42)
        return pd.concat([flagged, legit_sample], ignore_index=True)

    return flagged



# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def compute_aggregates(
    df: pd.DataFrame,
    risk_band: str = "high_critical",
) -> AggregateInput:
    """Reduce a DataFrame of selected transactions to aggregate stats.

    Args:
        df: pandas DataFrame with the IEEE-CIS feature schema
        risk_band: "all" | "high" | "critical" | "high_critical" (informational only)

    Returns:
        AggregateInput populated from the dataframe
    """
    n = len(df)
    has_fraud_label = "isFraud" in df.columns

    if has_fraud_label:
        fraud_mask = df["isFraud"].astype(int) == 1
        legit_mask = ~fraud_mask
        n_fraud = int(fraud_mask.sum())
    else:
        # If no labels available, treat all as fraud-suspect for rule generation
        fraud_mask = pd.Series([True] * n, index=df.index)
        legit_mask = pd.Series([False] * n, index=df.index)
        n_fraud = n

    fraud_df = df[fraud_mask] if has_fraud_label else df
    legit_df = df[legit_mask] if has_fraud_label else df.head(0)

    # ---- Date range ----
    if "timestamp" in df.columns:
        date_min = _format_iso_date(df["timestamp"].min())
        date_max = _format_iso_date(df["timestamp"].max())
    elif "TransactionDT" in df.columns:
        date_min = _format_iso_date(df["TransactionDT"].min())
        date_max = _format_iso_date(df["TransactionDT"].max())
    else:
        date_min = date_max = "unknown"

    # ---- Velocity ----
    velocity_1h_p50 = _safe_p(df.get("card1_txn_count_1h"), 0.50)
    velocity_1h_p95 = _safe_p(df.get("card1_txn_count_1h"), 0.95)
    velocity_24h_p50 = _safe_p(df.get("card1_txn_count_24h"), 0.50)
    velocity_24h_p95 = _safe_p(df.get("card1_txn_count_24h"), 0.95)
    velocity_7d_p50 = _safe_p(df.get("card1_txn_count_7d"), 0.50)
    velocity_7d_p95 = _safe_p(df.get("card1_txn_count_7d"), 0.95)
    velocity_24h_fraud_mean = _safe_mean(fraud_df.get("card1_txn_count_24h"))
    velocity_24h_legit_mean = _safe_mean(legit_df.get("card1_txn_count_24h"))

    # ---- Email risk ----
    if "P_emaildomain_is_highrisk" in df.columns and has_fraud_label:
        high_risk = df[df["P_emaildomain_is_highrisk"] == 1]
        low_risk = df[df["P_emaildomain_is_highrisk"] != 1]
        email_high_risk_fraud_rate = _safe_rate(
            int(high_risk["isFraud"].sum()) if len(high_risk) > 0 else 0,
            len(high_risk),
        )
        email_low_risk_fraud_rate = _safe_rate(
            int(low_risk["isFraud"].sum()) if len(low_risk) > 0 else 0,
            len(low_risk),
        )
    else:
        email_high_risk_fraud_rate = 0.0
        email_low_risk_fraud_rate = 0.0

    if "emails_match" in df.columns and has_fraud_label:
        mismatch = df[df["emails_match"] == 0]
        email_mismatch_fraud_rate = _safe_rate(
            int(mismatch["isFraud"].sum()) if len(mismatch) > 0 else 0,
            len(mismatch),
        )
    else:
        email_mismatch_fraud_rate = 0.0

    if "P_emaildomain_isnull" in df.columns and has_fraud_label:
        null_email = df[df["P_emaildomain_isnull"] == 1]
        email_null_fraud_rate = _safe_rate(
            int(null_email["isFraud"].sum()) if len(null_email) > 0 else 0,
            len(null_email),
        )
    else:
        email_null_fraud_rate = 0.0

    # Top fraud email domains (only domains with at least 3 transactions to avoid noise)
    top_fraud_email_domains: list[dict] = []
    if "P_emaildomain" in df.columns and has_fraud_label and n_fraud > 0:
        domain_stats = (
            df.groupby("P_emaildomain")
            .agg(n=("isFraud", "size"), fraud_count=("isFraud", "sum"))
            .reset_index()
        )
        domain_stats = domain_stats[domain_stats["n"] >= 3]
        if len(domain_stats) > 0:
            domain_stats["fraud_rate"] = domain_stats["fraud_count"] / domain_stats["n"]
            domain_stats = domain_stats.sort_values("fraud_rate", ascending=False).head(5)
            top_fraud_email_domains = [
                {
                    "domain": str(r["P_emaildomain"]),
                    "fraud_rate": float(r["fraud_rate"]),
                    "n": int(r["n"]),
                }
                for _, r in domain_stats.iterrows()
            ]

    # ---- Device / identity hopping ----
    fraud_card_distinct_addr1_p95 = _safe_p(fraud_df.get("card1_distinct_addr1"), 0.95)
    legit_card_distinct_addr1_p95 = _safe_p(legit_df.get("card1_distinct_addr1"), 0.95)
    fraud_card_distinct_products_p95 = _safe_p(fraud_df.get("card1_distinct_products"), 0.95)
    legit_card_distinct_products_p95 = _safe_p(legit_df.get("card1_distinct_products"), 0.95)

    # D1 == 0 indicates first-time device pairing, a fraud signal in IEEE-CIS
    if "D1" in df.columns and has_fraud_label:
        d1_zero = df[df["D1"] == 0]
        d1_nonzero = df[(df["D1"].notna()) & (df["D1"] != 0)]
        d1_zero_fraud_rate = _safe_rate(
            int(d1_zero["isFraud"].sum()) if len(d1_zero) > 0 else 0,
            len(d1_zero),
        )
        d1_nonzero_fraud_rate = _safe_rate(
            int(d1_nonzero["isFraud"].sum()) if len(d1_nonzero) > 0 else 0,
            len(d1_nonzero),
        )
    else:
        d1_zero_fraud_rate = 0.0
        d1_nonzero_fraud_rate = 0.0

    # ---- Amount ----
    amount_p50 = _safe_p(df.get("TransactionAmt"), 0.50)
    amount_p95 = _safe_p(df.get("TransactionAmt"), 0.95)
    amount_p99 = _safe_p(df.get("TransactionAmt"), 0.99)
    amount_zscore_fraud_p95 = _safe_p(fraud_df.get("card1_amt_zscore"), 0.95)
    amount_zscore_legit_p95 = _safe_p(legit_df.get("card1_amt_zscore"), 0.95)
    amount_sum_24h_fraud_p95 = _safe_p(fraud_df.get("card1_amt_sum_24h"), 0.95)
    amount_sum_24h_legit_p95 = _safe_p(legit_df.get("card1_amt_sum_24h"), 0.95)

    # ---- Channel breakdown ----
    products_with_high_fraud: list[dict] = []
    if "ProductCD" in df.columns and has_fraud_label and n_fraud > 0:
        prod_stats = (
            df.groupby("ProductCD")
            .agg(n=("isFraud", "size"), fraud_count=("isFraud", "sum"))
            .reset_index()
        )
        prod_stats["fraud_rate"] = prod_stats["fraud_count"] / prod_stats["n"]
        prod_stats = prod_stats.sort_values("fraud_rate", ascending=False).head(5)
        products_with_high_fraud = [
            {
                "product": str(r["ProductCD"]),
                "fraud_rate": float(r["fraud_rate"]),
                "n": int(r["n"]),
            }
            for _, r in prod_stats.iterrows()
        ]

    # ---- Time-of-day ----
    night_fraud_rate = 0.0
    day_fraud_rate = 0.0
    hour_with_highest_fraud = 0
    hour_with_highest_fraud_rate = 0.0

    if "is_night" in df.columns and has_fraud_label:
        night = df[df["is_night"] == 1]
        day = df[df["is_night"] != 1]
        night_fraud_rate = _safe_rate(
            int(night["isFraud"].sum()) if len(night) > 0 else 0,
            len(night),
        )
        day_fraud_rate = _safe_rate(
            int(day["isFraud"].sum()) if len(day) > 0 else 0,
            len(day),
        )

    if "txn_hour" in df.columns and has_fraud_label and n_fraud > 0:
        hour_stats = (
            df.groupby("txn_hour")
            .agg(n=("isFraud", "size"), fraud_count=("isFraud", "sum"))
            .reset_index()
        )
        hour_stats = hour_stats[hour_stats["n"] >= 5]  # min sample
        if len(hour_stats) > 0:
            hour_stats["fraud_rate"] = hour_stats["fraud_count"] / hour_stats["n"]
            top_hour = hour_stats.sort_values("fraud_rate", ascending=False).iloc[0]
            hour_with_highest_fraud = int(top_hour["txn_hour"])
            hour_with_highest_fraud_rate = float(top_hour["fraud_rate"])

    return AggregateInput(
        n_transactions=n,
        n_fraud=n_fraud,
        fraud_rate=_safe_rate(n_fraud, n),
        date_range=(date_min, date_max),
        risk_band=risk_band,
        velocity_1h_p50=velocity_1h_p50,
        velocity_1h_p95=velocity_1h_p95,
        velocity_24h_p50=velocity_24h_p50,
        velocity_24h_p95=velocity_24h_p95,
        velocity_7d_p50=velocity_7d_p50,
        velocity_7d_p95=velocity_7d_p95,
        velocity_24h_fraud_mean=velocity_24h_fraud_mean,
        velocity_24h_legit_mean=velocity_24h_legit_mean,
        email_high_risk_fraud_rate=email_high_risk_fraud_rate,
        email_low_risk_fraud_rate=email_low_risk_fraud_rate,
        email_mismatch_fraud_rate=email_mismatch_fraud_rate,
        email_null_fraud_rate=email_null_fraud_rate,
        top_fraud_email_domains=top_fraud_email_domains,
        fraud_card_distinct_addr1_p95=fraud_card_distinct_addr1_p95,
        legit_card_distinct_addr1_p95=legit_card_distinct_addr1_p95,
        fraud_card_distinct_products_p95=fraud_card_distinct_products_p95,
        legit_card_distinct_products_p95=legit_card_distinct_products_p95,
        d1_zero_fraud_rate=d1_zero_fraud_rate,
        d1_nonzero_fraud_rate=d1_nonzero_fraud_rate,
        amount_p50=amount_p50,
        amount_p95=amount_p95,
        amount_p99=amount_p99,
        amount_zscore_fraud_p95=amount_zscore_fraud_p95,
        amount_zscore_legit_p95=amount_zscore_legit_p95,
        amount_sum_24h_fraud_p95=amount_sum_24h_fraud_p95,
        amount_sum_24h_legit_p95=amount_sum_24h_legit_p95,
        products_with_high_fraud=products_with_high_fraud,
        night_fraud_rate=night_fraud_rate,
        day_fraud_rate=day_fraud_rate,
        hour_with_highest_fraud=hour_with_highest_fraud,
        hour_with_highest_fraud_rate=hour_with_highest_fraud_rate,
    )
