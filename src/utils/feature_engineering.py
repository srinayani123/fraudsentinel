"""
Pure-Python equivalents of the Spark feature pipeline, for live single-transaction scoring.

When a transaction comes in via the API or playground, we don't run Spark — we compute
the same features in process from the transaction + the recent card history.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

HIGH_RISK_DOMAINS = {"protonmail.com", "mail.com", "outlook.com"}


def add_time_features(row: dict) -> dict:
    """Derive hour/day/is_night from TransactionDT seconds, if not already provided."""
    if "txn_hour" not in row and "TransactionDT" in row:
        dt = int(row["TransactionDT"])
        row["txn_hour"] = (dt // 3600) % 24
        row["txn_day"] = dt // 86400
        row["txn_dayofweek"] = (dt // 86400) % 7
    if "is_night" not in row and "txn_hour" in row:
        h = row["txn_hour"]
        row["is_night"] = int(h >= 22 or h <= 5)
    return row


def add_email_features(row: dict) -> dict:
    """Email risk + match indicators."""
    p = row.get("P_emaildomain")
    r = row.get("R_emaildomain")
    row["P_emaildomain_isnull"] = int(p is None or p == "")
    row["R_emaildomain_isnull"] = int(r is None or r == "")
    row["P_emaildomain_is_highrisk"] = int((p or "") in HIGH_RISK_DOMAINS)
    row["R_emaildomain_is_highrisk"] = int((r or "") in HIGH_RISK_DOMAINS)
    row["emails_match"] = int(p is not None and p == r)
    return row


def compute_velocity_features(
    row: dict,
    history: Iterable[dict] | pd.DataFrame | None,
) -> dict:
    """Compute card1_txn_count_{1h,24h,7d} and amount aggregates from recent history.

    `history` is the list of prior transactions for the same card1, each with
    at minimum 'TransactionDT' and 'TransactionAmt'. The current transaction
    itself should NOT be in history.
    """
    if history is None:
        # Defaults
        for win in ("1h", "24h", "7d"):
            row.setdefault(f"card1_txn_count_{win}", 0)
            row.setdefault(f"card1_amt_sum_{win}", 0.0)
        row.setdefault("card1_amt_max_24h", 0.0)
        row.setdefault("card1_seconds_since_last", -1)
        return row

    if isinstance(history, pd.DataFrame):
        hist_df = history
    else:
        hist_df = pd.DataFrame(list(history))

    if hist_df.empty:
        for win in ("1h", "24h", "7d"):
            row[f"card1_txn_count_{win}"] = 0
            row[f"card1_amt_sum_{win}"] = 0.0
        row["card1_amt_max_24h"] = 0.0
        row["card1_seconds_since_last"] = -1
        return row

    current_dt = int(row.get("TransactionDT", hist_df["TransactionDT"].max() + 1))

    for hours, label in [(1, "1h"), (24, "24h"), (168, "7d")]:
        seconds = hours * 3600
        window = hist_df[
            (hist_df["TransactionDT"] >= current_dt - seconds)
            & (hist_df["TransactionDT"] < current_dt)
        ]
        row[f"card1_txn_count_{label}"] = int(len(window))
        row[f"card1_amt_sum_{label}"] = float(window["TransactionAmt"].sum())
        if label == "24h":
            row["card1_amt_max_24h"] = float(
                window["TransactionAmt"].max() if len(window) else 0.0
            )

    last_dt = int(hist_df["TransactionDT"].max())
    row["card1_seconds_since_last"] = max(current_dt - last_dt, 0)
    return row


def compute_behavioral_features(
    row: dict,
    history: Iterable[dict] | pd.DataFrame | None,
) -> dict:
    """Per-card aggregates: mean, std, distinct counts, z-score."""
    if history is None or (isinstance(history, pd.DataFrame) and history.empty):
        row.setdefault("card1_amt_mean", row.get("TransactionAmt", 0.0))
        row.setdefault("card1_amt_std", 0.0)
        row.setdefault("card1_amt_zscore", 0.0)
        row.setdefault("card1_total_txns", 0)
        row.setdefault("card1_distinct_products", 0)
        row.setdefault("card1_distinct_addr1", 0)
        return row

    hist_df = pd.DataFrame(list(history)) if not isinstance(history, pd.DataFrame) else history

    mean = float(hist_df["TransactionAmt"].mean())
    std = float(hist_df["TransactionAmt"].std()) if len(hist_df) > 1 else 0.0
    row["card1_amt_mean"] = mean
    row["card1_amt_std"] = std
    if std > 0:
        row["card1_amt_zscore"] = (row.get("TransactionAmt", 0.0) - mean) / std
    else:
        row["card1_amt_zscore"] = 0.0
    row["card1_total_txns"] = int(len(hist_df))
    if "ProductCD" in hist_df.columns:
        row["card1_distinct_products"] = int(hist_df["ProductCD"].nunique())
    if "addr1" in hist_df.columns:
        row["card1_distinct_addr1"] = int(hist_df["addr1"].nunique())
    return row


def engineer_features(
    transaction: dict,
    card_history: pd.DataFrame | None = None,
) -> dict:
    """One-stop feature engineering for live scoring.

    Returns a dict with all features the trained models expect.
    Missing features default to safe sentinels (-999 or 0).
    """
    row = dict(transaction)
    row = add_time_features(row)
    row = add_email_features(row)
    row = compute_velocity_features(row, card_history)
    row = compute_behavioral_features(row, card_history)
    return row
