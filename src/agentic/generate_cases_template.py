
"""
Template-based fraud case generator (clusters 200-399 of k=400).

Run AFTER generate_cases_llm.py. Uses identical k-means parameters so cluster
assignments match exactly — this generator handles the second half.

Free, fast, deterministic.
"""

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.dl_models.train_lstm_ae import LSTM_FEATURES
from src.utils.config import FEATURES_DIR, FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

N_CLUSTERS = 400
TEMPLATE_CLUSTER_RANGE = (200, 400)


def categorize_fraud_pattern(row) -> tuple[str, list[str]]:
    indicators = []
    amt = float(row.get("TransactionAmt", 0))
    z = float(row.get("card1_amt_zscore", 0))
    vel_1h = int(row.get("card1_txn_count_1h", 0))
    vel_24h = int(row.get("card1_txn_count_24h", 0))
    vel_7d = int(row.get("card1_txn_count_7d", 0))
    is_night = int(row.get("is_night", 0))
    p_email_risk = int(row.get("P_emaildomain_is_highrisk", 0))
    emails_match = int(row.get("emails_match", 1))
    distinct_products = int(row.get("card1_distinct_products", 0))
    total_txns = int(row.get("card1_total_txns", 0))

    scores = {}
    if amt < 10 and vel_1h >= 2:
        scores["card_testing"] = 3 + vel_1h
        indicators += ["small_test_charge", "rapid_followup"]
    if vel_24h >= 8 or vel_1h >= 4:
        scores["velocity_attack"] = vel_24h + vel_1h * 2
        indicators.append("high_velocity")
        if distinct_products >= 3:
            indicators.append("diverse_merchants")
    if emails_match == 0 and z > 2.5:
        scores["account_takeover"] = 5 + z
        indicators += ["email_mismatch", "post_change_large_purchase"]
    if 10 <= total_txns <= 60 and z > 3:
        scores["synthetic_identity"] = z + (total_txns / 10)
        indicators += ["moderate_history_then_bustout", "extreme_zscore"]
    if is_night == 1 and z > 1.5:
        scores["temporal_anomaly"] = z + 2
        indicators += ["off_hours", "elevated_amount"]
    if p_email_risk == 1:
        scores["email_risk"] = 2 + (z if z > 0 else 0)
        indicators.append("high_risk_email_domain")
    if amt < 5 and vel_1h <= 1 and total_txns < 10:
        scores["bin_attack"] = 4
        indicators.append("low_amount_test")
    if 0.99 <= amt <= 14.99 and vel_24h <= 3:
        scores["subscription_probe"] = 3
        indicators.append("subscription_amount")
    dist1 = row.get("dist1")
    if dist1 is not None and not pd.isna(dist1) and float(dist1) > 100:
        scores["geo_anomaly"] = float(dist1) / 50
        indicators.append("distance_anomaly")
    if not scores:
        scores["friendly_fraud"] = 1
        indicators.append("normal_looking_transaction")

    pattern = max(scores, key=scores.get)
    if not indicators:
        indicators = ["fraud_signature"]
    return pattern, indicators[:5]


def render_narrative(row, pattern: str) -> str:
    amt = float(row.get("TransactionAmt", 0))
    z = float(row.get("card1_amt_zscore", 0))
    vel_1h = int(row.get("card1_txn_count_1h", 0))
    vel_24h = int(row.get("card1_txn_count_24h", 0))
    vel_7d = int(row.get("card1_txn_count_7d", 0))
    hour = int(row.get("txn_hour", -1))
    is_night = int(row.get("is_night", 0))
    product = str(row.get("ProductCD", "?"))
    email_risk = int(row.get("P_emaildomain_is_highrisk", 0))
    emails_match = int(row.get("emails_match", 1))
    total_txns = int(row.get("card1_total_txns", 0))
    distinct_products = int(row.get("card1_distinct_products", 0))
    seconds_since = int(row.get("card1_seconds_since_last", -1))

    parts = [f"Confirmed fraud transaction matching {pattern.replace('_', ' ')} pattern."]

    if z > 4:
        parts.append(f"Transaction amount of ${amt:.2f} is an extreme outlier (z-score {z:.2f}).")
    elif z > 2:
        parts.append(f"Transaction amount of ${amt:.2f} is significantly above the card's typical pattern (z-score {z:.2f}).")
    elif z < -1:
        parts.append(f"Small charge of ${amt:.2f}, well below typical (z-score {z:.2f}) - consistent with card validation testing.")
    else:
        parts.append(f"Transaction amount: ${amt:.2f}.")

    if vel_1h >= 5:
        parts.append(f"Card velocity is critically elevated: {vel_1h} transactions in the last hour, {vel_24h} in 24 hours.")
    elif vel_24h >= 8:
        parts.append(f"High 24-hour velocity ({vel_24h} transactions), with {vel_1h} in the last hour.")
    elif vel_7d >= 20:
        parts.append(f"Sustained activity: {vel_7d} transactions in the last 7 days.")

    if is_night == 1:
        parts.append(f"Transaction occurred at {hour:02d}:00, outside the user's typical daytime activity window.")

    if distinct_products >= 4:
        parts.append(f"Card touched {distinct_products} distinct product categories - unusual diversity suggests liquidation rather than normal shopping.")

    if email_risk == 1:
        parts.append("Associated with a high-risk email domain.")
    if emails_match == 0:
        parts.append("Purchaser email does not match recipient email - possible account takeover or address swap.")

    if total_txns < 10:
        parts.append(f"Card has minimal history ({total_txns} prior transactions).")
    elif 10 <= total_txns <= 60:
        parts.append(f"Card has moderate history ({total_txns} transactions) - consistent with synthetic identity or seasoned account.")

    if seconds_since >= 0 and seconds_since < 300:
        parts.append(f"Only {seconds_since}s since the previous transaction on this card.")

    parts.append(f"Product code: {product}.")
    return " ".join(parts)


def main():
    path = FEATURES_DIR / "engineered.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run Spark feature pipeline first. Missing: {path}")

    logger.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    fraud = df[df["isFraud"] == 1].copy()
    logger.info(f"Loaded {len(fraud):,} fraud transactions")

    avail = [c for c in LSTM_FEATURES if c in fraud.columns]
    X = fraud[avail].fillna(0).values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(N_CLUSTERS, len(fraud) // 5)
    logger.info(f"Running k-means with k={n_clusters} (must match LLM generator's k-means)...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    logger.info(f"This generator handles clusters {TEMPLATE_CLUSTER_RANGE[0]}-{TEMPLATE_CLUSTER_RANGE[1]-1}...")
    selected_rows = []
    for cluster_id in range(TEMPLATE_CLUSTER_RANGE[0], TEMPLATE_CLUSTER_RANGE[1]):
        cluster_mask = labels == cluster_id
        if not cluster_mask.any():
            continue
        cluster_X = X_scaled[cluster_mask]
        centroid = km.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_X - centroid, axis=1)
        nearest = np.argmin(distances)
        cluster_indices = np.where(cluster_mask)[0]
        selected_rows.append((cluster_id, fraud.iloc[cluster_indices[nearest]]))

    logger.info(f"Generating templated narratives for {len(selected_rows)} clusters...")
    FRAUD_CASES_DIR.mkdir(parents=True, exist_ok=True)
    pattern_counts = {}

    for cluster_id, row in selected_rows:
        case_id = f"tmpl_{cluster_id:03d}"
        pattern, indicators = categorize_fraud_pattern(row)
        narrative = render_narrative(row, pattern)
        title = f"Auto-{case_id}: {pattern.replace('_', ' ').title()}"
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        case = {
            "id": case_id,
            "title": title,
            "pattern": pattern,
            "narrative": narrative,
            "indicators": indicators,
            "source": "template_generated_from_real_fraud",
            "txn_id": int(row.get("TransactionID", -1)),
            "cluster_id": int(cluster_id),
            "feature_summary": {
                "amount": float(row.get("TransactionAmt", 0)),
                "z_score": float(row.get("card1_amt_zscore", 0)),
                "velocity_1h": int(row.get("card1_txn_count_1h", 0)),
                "velocity_24h": int(row.get("card1_txn_count_24h", 0)),
                "is_night": int(row.get("is_night", 0)),
                "email_risk": int(row.get("P_emaildomain_is_highrisk", 0)),
            },
        }
        path = FRAUD_CASES_DIR / f"{case_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2)

    logger.info(f"Wrote {len(selected_rows)} template cases")
    logger.info("Pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {pattern}: {count}")
    logger.info("Next: python -m src.agentic.build_knowledge_base")


if __name__ == "__main__":
    main()
