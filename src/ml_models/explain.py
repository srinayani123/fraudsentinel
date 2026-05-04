"""Per-prediction explainability for the fraud-detection models.

Two complementary techniques:

1. XGBoost SHAP attribution — for the fraud probability, computes which
   features actually contributed and by how much. Uses TreeExplainer (exact,
   ~50ms per prediction). Handles categorical features by reading the model's
   expected feature_types and casting the input DataFrame to match.
   Reference: Lundberg & Lee (2017), "A Unified Approach to Interpreting
   Model Predictions" (NeurIPS).

2. LSTM timestep attribution — for the autoencoder anomaly score, computes
   which transaction in the recent sequence had the highest reconstruction
   error. Tells the analyst WHEN in the timeline the anomaly started, not
   just that it exists.

Both grounding signals feed into the Investigator agent so its prose cites
real model attributions instead of speculation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 8
LSTM_SEQ_TOP_K = 5

_SHAP_EXPLAINER_CACHE: dict[int, Any] = {}


# ============================================================================
# Helpers
# ============================================================================
def _humanize_feature_name(feature: str) -> str:
    NICE_NAMES = {
        "TransactionAmt": "Transaction amount",
        "ProductCD": "Channel",
        "card1": "Card BIN",
        "card2": "Card issuing bank",
        "card3": "Card country",
        "card4": "Card brand",
        "card5": "Card subtype",
        "card6": "Card type (credit/debit)",
        "addr1": "Billing region",
        "addr2": "Billing country",
        "dist1": "Distance metric 1",
        "dist2": "Distance metric 2",
        "P_emaildomain": "Purchaser email domain",
        "R_emaildomain": "Recipient email domain",
        "card1_txn_count_1h": "Transactions on this card (1h)",
        "card1_txn_count_24h": "Transactions on this card (24h)",
        "card1_txn_count_7d": "Transactions on this card (7d)",
        "card1_amt_zscore": "Amount vs card's typical (z-score)",
        "card1_amt_sum_24h": "Total spend on card (24h)",
        "card1_seconds_since_last": "Seconds since last txn",
        "card1_total_txns": "Total card lifetime txns",
        "card1_distinct_addr1": "Distinct shipping regions on card",
        "card1_distinct_products": "Distinct product types on card",
        "is_night": "Night-time transaction flag",
        "txn_hour": "Transaction hour of day",
        "txn_dayofweek": "Day of week",
        "emails_match": "Purchaser/recipient email match",
        "P_emaildomain_is_highrisk": "Purchaser email is high-risk domain",
        "R_emaildomain_is_highrisk": "Recipient email is high-risk domain",
    }

    if feature in NICE_NAMES:
        return NICE_NAMES[feature]

    if feature.startswith("D") and feature[1:].isdigit():
        return f"{feature} (device/identity field)"
    if feature.startswith("C") and feature[1:].isdigit():
        return f"{feature} (card-related counter)"
    if feature.startswith("V") and feature[1:].isdigit():
        return f"{feature} (Vesta engineered feature)"
    if feature.startswith("M") and feature[1:].isdigit():
        return f"{feature} (match/verification flag)"

    return feature.replace("_", " ")


def _categorize_feature_family(feature: str) -> str:
    if feature.startswith("D") and feature[1:].isdigit():
        return "device_identity"
    if feature.startswith("C") and feature[1:].isdigit():
        return "card_counter"
    if feature.startswith("V") and feature[1:].isdigit():
        return "engineered"
    if feature.startswith("M") and feature[1:].isdigit():
        return "verification"
    if "amt" in feature.lower() or feature == "TransactionAmt":
        return "amount"
    if (
        "txn_count" in feature
        or "txn_hour" in feature
        or "is_night" in feature
        or "seconds_since" in feature
    ):
        return "behavioral_velocity_timing"
    if feature.startswith("addr") or feature.startswith("dist"):
        return "geographic"
    if feature.startswith("card"):
        return "card_metadata"
    if "email" in feature:
        return "email"
    return "other"


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, float):
        if abs(value) < 1e-9:
            return "0"
        return f"{value:.4g}"
    return str(value)


def _coerce_X_to_model_dtypes(X: pd.DataFrame, model) -> pd.DataFrame:
    """Cast DataFrame columns to match the XGBoost model's expected dtypes.

    XGBoost was trained with object/string columns as 'category' dtype; SHAP
    rejects 'object' dtype with the error "DataFrame.dtypes for data must be
    int, float, bool or category". This reads the booster's feature_types
    list (one of 'int', 'float', 'q', 'i', 'c') and coerces each column.

    Falls back to converting any remaining object column to category if the
    model's feature_types are unavailable.
    """
    try:
        booster = model.get_booster()
        expected_types = booster.feature_types
        if expected_types and len(expected_types) == len(X.columns):
            for col, t in zip(X.columns, expected_types):
                if t == "c":
                    X[col] = X[col].astype("category")
                elif t in ("int", "i"):
                    X[col] = pd.to_numeric(X[col], errors="coerce").astype("Int64")
                elif t in ("float", "q", "float32"):
                    X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
            return X
    except Exception as e:
        logger.debug(f"Could not read booster feature_types: {e}")

    # Fallback: any object column becomes category
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")
    return X


# ============================================================================
# XGBoost SHAP attribution
# ============================================================================
def _build_explainer(model):
    import shap

    key = id(model)
    if key in _SHAP_EXPLAINER_CACHE:
        return _SHAP_EXPLAINER_CACHE[key]

    explainer = shap.TreeExplainer(model)
    _SHAP_EXPLAINER_CACHE[key] = explainer
    logger.info("Built SHAP TreeExplainer for XGBoost model")
    return explainer


def explain_xgboost_prediction(
    model,
    feature_names: list[str],
    row_dict: dict,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Compute SHAP attributions for a single XGBoost prediction.

    Returns:
        {
          "predicted_proba": float,
          "base_value": float,
          "top_drivers": [{feature, label, value, shap, direction, family}, ...],
          "by_family": {family_name: total_shap_contribution},
          "signal_balance": "fraud_signals_dominate" | "legit_signals_dominate" | "mixed",
          "total_positive_contribution": float,
          "total_negative_contribution": float,
        }
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        return {
            "error": "SHAP not installed. Run: pip install shap --break-system-packages",
            "top_drivers": [],
        }

    feature_values = []
    for f in feature_names:
        v = row_dict.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            feature_values.append(np.nan)
        else:
            feature_values.append(v)

    X = pd.DataFrame([feature_values], columns=feature_names)

    # Cast columns to match the model's expected dtypes (categorical / int / float)
    X = _coerce_X_to_model_dtypes(X, model)

    try:
        proba_arr = model.predict_proba(X)
        predicted_proba = (
            float(proba_arr[0, 1]) if proba_arr.shape[1] >= 2 else float(proba_arr[0, 0])
        )
    except Exception as e:
        logger.warning(f"predict_proba failed: {e}")
        predicted_proba = None

    try:
        explainer = _build_explainer(model)
        # check_additivity=False because categorical encoding can introduce
        # tiny floating-point drift that fails SHAP's strict additivity check
        shap_values = explainer.shap_values(X, check_additivity=False)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        sv = np.asarray(sv).reshape(-1)

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).reshape(-1)[-1])
        else:
            base_value = float(base_value)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {
            "error": f"SHAP computation failed: {type(e).__name__}: {e}",
            "predicted_proba": predicted_proba,
            "top_drivers": [],
        }

    pairs = []
    for i, fname in enumerate(feature_names):
        try:
            shap_val = float(sv[i])
            raw_val = feature_values[i]
            pairs.append({"feature": fname, "value": raw_val, "shap": shap_val})
        except (IndexError, TypeError, ValueError):
            continue

    pairs.sort(key=lambda p: abs(p["shap"]), reverse=True)
    top = pairs[:top_k]

    top_drivers = []
    for p in top:
        direction = (
            "increased fraud probability"
            if p["shap"] > 0
            else "decreased fraud probability"
        )
        top_drivers.append(
            {
                "feature": p["feature"],
                "label": _humanize_feature_name(p["feature"]),
                "value": _format_value(p["value"]),
                "value_raw": p["value"]
                if not (isinstance(p["value"], float) and np.isnan(p["value"]))
                else None,
                "shap": round(p["shap"], 4),
                "direction": direction,
                "family": _categorize_feature_family(p["feature"]),
            }
        )

    by_family: dict[str, float] = {}
    for p in pairs:
        fam = _categorize_feature_family(p["feature"])
        by_family[fam] = by_family.get(fam, 0.0) + p["shap"]
    by_family_sorted = {
        k: round(v, 4)
        for k, v in sorted(by_family.items(), key=lambda kv: abs(kv[1]), reverse=True)
    }

    pos_contrib = sum(p["shap"] for p in pairs if p["shap"] > 0)
    neg_contrib = sum(p["shap"] for p in pairs if p["shap"] < 0)
    if abs(pos_contrib) > 2 * abs(neg_contrib):
        signal_balance = "fraud_signals_dominate"
    elif abs(neg_contrib) > 2 * abs(pos_contrib):
        signal_balance = "legit_signals_dominate"
    else:
        signal_balance = "mixed"

    return {
        "error": None,
        "predicted_proba": predicted_proba,
        "base_value": round(base_value, 4),
        "top_drivers": top_drivers,
        "by_family": by_family_sorted,
        "signal_balance": signal_balance,
        "total_positive_contribution": round(pos_contrib, 4),
        "total_negative_contribution": round(neg_contrib, 4),
    }


def format_xgboost_attribution_for_llm(attribution: dict) -> str:
    if attribution.get("error"):
        return f"SHAP attribution unavailable: {attribution['error']}"

    drivers = attribution.get("top_drivers", [])
    if not drivers:
        return "No SHAP attribution available."

    lines = []
    lines.append(
        f"XGBoost predicted fraud probability: {attribution.get('predicted_proba', 0):.4f}"
    )
    lines.append(f"Signal balance: {attribution.get('signal_balance', 'unknown')}")
    lines.append("")
    lines.append("TOP FEATURE CONTRIBUTORS (ranked by |SHAP|):")
    for i, d in enumerate(drivers, 1):
        sign = "+" if d["shap"] > 0 else "-"
        lines.append(
            f"  {i}. {d['label']} = {d['value']}  "
            f"-> {sign}{abs(d['shap']):.3f} ({d['direction']})  [family: {d['family']}]"
        )
    lines.append("")
    lines.append("CONTRIBUTION BY FEATURE FAMILY:")
    for family, total in attribution.get("by_family", {}).items():
        sign = "+" if total > 0 else "-"
        lines.append(f"  {family}: {sign}{abs(total):.3f}")
    return "\n".join(lines)


# ============================================================================
# LSTM timestep attribution
# ============================================================================
def explain_lstm_sequence(
    transactions: pd.DataFrame,
    top_k: int = LSTM_SEQ_TOP_K,
) -> dict:
    """Identify which transactions in the recent sequence drove the LSTM anomaly.

    Args:
        transactions: card's recent transactions, time-ordered (oldest first).
        top_k: number of top-anomalous timesteps to surface.

    Returns:
        {
          "anomaly_score": float,       # raw reconstruction error (sum)
          "normalized_score": float,    # 0-1
          "is_anomaly": bool,
          "threshold": float,
          "sequence_length": int,
          "current_step_error": float,  # error attributed to LATEST txn
          "current_step_rank": int,     # 1 = most anomalous in window
          "top_steps": [...],
          "verdict": "anomaly_at_current" | "anomaly_earlier_in_window" | "no_anomaly",
        }
    """
    try:
        import torch

        from src.dl_models.inference import load_model as _load_lstm
    except ImportError as e:
        return {
            "error": f"LSTM inference module unavailable: {e}",
            "top_steps": [],
        }

    try:
        model, scaler, feature_columns, seq_len, threshold = _load_lstm()
    except Exception as e:
        logger.warning(f"LSTM model not loaded: {e}")
        return {"error": f"LSTM model unavailable: {e}", "top_steps": []}

    if transactions is None or len(transactions) == 0:
        return {"error": "No transactions provided", "top_steps": []}

    df = transactions.copy()
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT")

    # Pad if shorter than seq_len (same logic as score_sequence)
    if len(df) < seq_len:
        padding = pd.concat([df.iloc[[0]]] * (seq_len - len(df)))
        df = pd.concat([padding, df], ignore_index=True)

    df_window = df.iloc[-seq_len:].reset_index(drop=True)

    seq = df_window[feature_columns].fillna(0).values.astype(np.float32)
    seq_scaled = scaler.transform(seq)
    x = torch.from_numpy(seq_scaled).unsqueeze(0)  # (1, seq_len, n_features)

    try:
        with torch.no_grad():
            reconstructed = model(x)
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]
        reconstructed_np = reconstructed.squeeze(0).cpu().numpy()
    except Exception as e:
        logger.warning(f"LSTM forward pass failed: {e}")
        return {
            "error": f"LSTM forward pass failed: {type(e).__name__}: {e}",
            "top_steps": [],
        }

    # Per-timestep MSE
    diff = seq_scaled - reconstructed_np
    per_step_error = np.mean(diff ** 2, axis=1)

    total_error = float(per_step_error.sum())
    normalized = float(
        1.0 / (1.0 + np.exp(-(total_error / max(threshold, 1e-6) - 1.0) * 3.0))
    )
    is_anomaly = bool(total_error > threshold)

    order = np.argsort(per_step_error)[::-1]
    top = []
    for rank, pos in enumerate(order[:top_k], start=1):
        is_current = pos == seq_len - 1
        row = df_window.iloc[int(pos)]
        amt = row.get("TransactionAmt")
        try:
            amt = float(amt) if amt is not None else None
        except (TypeError, ValueError):
            amt = None
        hour = row.get("txn_hour")
        try:
            hour = int(hour) if hour is not None else None
        except (TypeError, ValueError):
            hour = None
        top.append(
            {
                "position": int(pos),
                "rank": int(rank),
                "error": round(float(per_step_error[pos]), 6),
                "is_current": bool(is_current),
                "amount": amt,
                "hour": hour,
            }
        )

    current_step_error = float(per_step_error[-1])
    current_rank = int(np.where(order == seq_len - 1)[0][0]) + 1

    if not is_anomaly:
        verdict = "no_anomaly"
    elif current_rank == 1:
        verdict = "anomaly_at_current"
    else:
        verdict = "anomaly_earlier_in_window"

    return {
        "error": None,
        "anomaly_score": round(total_error, 6),
        "normalized_score": round(normalized, 4),
        "is_anomaly": is_anomaly,
        "threshold": float(threshold),
        "sequence_length": int(seq_len),
        "current_step_error": round(current_step_error, 6),
        "current_step_rank": current_rank,
        "top_steps": top,
        "verdict": verdict,
    }


def format_lstm_attribution_for_llm(attribution: dict) -> str:
    if attribution.get("error"):
        return f"LSTM timestep analysis unavailable: {attribution['error']}"

    if not attribution.get("top_steps"):
        return "No LSTM timestep analysis available."

    lines = []
    lines.append(
        f"LSTM autoencoder anomaly score: {attribution.get('anomaly_score', 0):.4f} "
        f"(threshold {attribution.get('threshold', 0):.4f}; "
        f"{'ABOVE' if attribution.get('is_anomaly') else 'BELOW'} threshold)"
    )
    lines.append(
        f"Window: {attribution.get('sequence_length', 0)} most recent transactions on this card"
    )
    lines.append(f"Verdict: {attribution.get('verdict', 'unknown')}")
    lines.append("")
    lines.append(
        f"Current transaction's reconstruction error: "
        f"{attribution.get('current_step_error', 0):.4f} "
        f"(rank {attribution.get('current_step_rank', '?')} of "
        f"{attribution.get('sequence_length', 0)} steps)"
    )
    lines.append("")
    lines.append("TOP-ANOMALOUS TIMESTEPS in the window (rank by reconstruction error):")
    for s in attribution["top_steps"]:
        marker = " <- CURRENT" if s["is_current"] else ""
        amt_str = f"${s['amount']:.2f}" if s["amount"] is not None else "?"
        hour_str = f"{s['hour']:02d}h" if s["hour"] is not None else "?"
        lines.append(
            f"  rank {s['rank']}: position {s['position']} "
            f"(error {s['error']:.4f}, amt {amt_str}, hour {hour_str}){marker}"
        )

    return "\n".join(lines)


def format_combined_attribution_for_llm(
    xgb_attribution: dict | None,
    lstm_attribution: dict | None,
) -> str:
    sections = []
    if xgb_attribution:
        sections.append("=== XGBOOST SHAP ATTRIBUTION (per-feature) ===")
        sections.append(format_xgboost_attribution_for_llm(xgb_attribution))
    if lstm_attribution:
        sections.append("")
        sections.append("=== LSTM TIMESTEP ATTRIBUTION (per-timestep) ===")
        sections.append(format_lstm_attribution_for_llm(lstm_attribution))
    return "\n".join(sections) if sections else "No attribution available."
