
"""
LLM-powered fraud case generator (clusters 0-199 of k=400).

Run BEFORE generate_cases_template.py. Both generators use k=400 with the same
random seed, so clusters are deterministic across runs.

Cost: ~$1.20 in Anthropic credits.
Time: 30-45 minutes (concurrent, retry-resilient).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.dl_models.train_lstm_ae import LSTM_FEATURES
from src.utils.config import DEFAULT_ANTHROPIC_MODEL, FEATURES_DIR, FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

load_dotenv()

N_CLUSTERS = 400
LLM_CLUSTER_RANGE = (0, 200)
MAX_CONCURRENT = 5
MAX_RETRIES = 3
COST_PER_CALL = (600 * 3 / 1_000_000) + (250 * 15 / 1_000_000)


SYSTEM_PROMPT = """You are a senior fraud analyst with deep expertise in payment card fraud patterns.

Your job: given a single fraud transaction's feature signature, write a precise, analyst-grade narrative describing why it is fraud and what attack pattern it likely represents.

Style requirements:
- 4-6 sentences, dense and informative
- Reference the actual feature values (don't be vague)
- Hypothesize intent and likely attacker behavior
- Use payment-fraud terminology naturally (BIN, CNP, velocity, bust-out, etc.) where it fits
- No filler ("It is interesting to note that...", "This is a classic example of...")
- Don't repeat the feature values back as a list - weave them into the prose

Output requirements (STRICT JSON, nothing else):
{
  "title": "Short descriptive title, 5-10 words",
  "pattern": "one of: card_testing, geo_anomaly, account_takeover, velocity_attack, synthetic_identity, bin_attack, friendly_fraud, temporal_anomaly, email_risk, subscription_probe",
  "narrative": "The 4-6 sentence narrative",
  "indicators": ["list", "of", "3-5", "fraud", "indicators"]
}"""


def build_user_prompt(row: pd.Series) -> str:
    amt = float(row.get("TransactionAmt", 0))
    z = float(row.get("card1_amt_zscore", 0))
    vel_1h = int(row.get("card1_txn_count_1h", 0))
    vel_24h = int(row.get("card1_txn_count_24h", 0))
    vel_7d = int(row.get("card1_txn_count_7d", 0))
    hour = int(row.get("txn_hour", -1))
    is_night = int(row.get("is_night", 0))
    product = str(row.get("ProductCD", "?"))
    p_email_risk = int(row.get("P_emaildomain_is_highrisk", 0))
    emails_match = int(row.get("emails_match", 1))
    distinct_products = int(row.get("card1_distinct_products", 0))
    total_txns = int(row.get("card1_total_txns", 0))
    seconds_since = int(row.get("card1_seconds_since_last", -1))
    distinct_addr = int(row.get("card1_distinct_addr1", 0))

    return f"""This transaction was confirmed fraud. Analyze its feature signature:

TRANSACTION
- Amount: ${amt:.2f}
- Product code: {product}
- Hour of day: {hour:02d}:00 ({"NIGHT" if is_night else "day"})

CARD HISTORY CONTEXT
- Total prior transactions for this card: {total_txns}
- Distinct product categories used: {distinct_products}
- Distinct billing addresses: {distinct_addr}
- Z-score of this amount vs card's history: {z:.2f}

VELOCITY (recent activity on this card)
- Transactions in last 1 hour: {vel_1h}
- Transactions in last 24 hours: {vel_24h}
- Transactions in last 7 days: {vel_7d}
- Seconds since previous transaction: {seconds_since if seconds_since >= 0 else "N/A"}

EMAIL SIGNALS
- Purchaser email is high-risk domain: {"YES" if p_email_risk else "no"}
- Purchaser and recipient emails match: {"YES" if emails_match else "NO"}

Write the case narrative as JSON only."""


def template_fallback(row: pd.Series) -> dict:
    amt = float(row.get("TransactionAmt", 0))
    z = float(row.get("card1_amt_zscore", 0))
    vel_24h = int(row.get("card1_txn_count_24h", 0))
    is_night = int(row.get("is_night", 0))
    if vel_24h >= 8:
        pattern = "velocity_attack"
    elif z > 3:
        pattern = "synthetic_identity"
    elif is_night:
        pattern = "temporal_anomaly"
    elif amt < 5:
        pattern = "card_testing"
    else:
        pattern = "friendly_fraud"
    return {
        "title": f"Auto fraud case ({pattern.replace('_', ' ')})",
        "pattern": pattern,
        "narrative": (
            f"Confirmed fraud transaction with amount ${amt:.2f}, z-score {z:.2f} "
            f"vs card history, and velocity of {vel_24h} transactions in 24 hours. "
            f"Pattern signature consistent with {pattern.replace('_', ' ')}."
        ),
        "indicators": [pattern, "fraud_signature"],
    }


def generate_one_case(row, case_id, client, model):
    user_prompt = build_user_prompt(row)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            for key in ("title", "pattern", "narrative", "indicators"):
                if key not in parsed:
                    raise ValueError(f"Missing key: {key}")
            return parsed, True
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            logger.warning(f"  {case_id} attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"  {case_id} attempt {attempt+1} unexpected: {e}")
            time.sleep(2 ** attempt)
    logger.warning(f"  {case_id}: all retries failed, using template fallback")
    return template_fallback(row), False


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add to .env.")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    logger.info(f"Using model: {model}")

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
    logger.info(f"Running k-means with k={n_clusters} (deterministic, will match template generator)...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    logger.info(f"Selecting centroid-nearest fraud row per cluster (this generator handles clusters {LLM_CLUSTER_RANGE[0]}-{LLM_CLUSTER_RANGE[1]-1})...")
    selected_rows = []
    for cluster_id in range(LLM_CLUSTER_RANGE[0], LLM_CLUSTER_RANGE[1]):
        cluster_mask = labels == cluster_id
        if not cluster_mask.any():
            continue
        cluster_X = X_scaled[cluster_mask]
        centroid = km.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_X - centroid, axis=1)
        nearest = np.argmin(distances)
        cluster_indices = np.where(cluster_mask)[0]
        selected_rows.append((cluster_id, fraud.iloc[cluster_indices[nearest]]))

    n_total = len(selected_rows)
    estimated_cost = n_total * COST_PER_CALL
    logger.info(f"Will generate {n_total} cases via {model}")
    logger.info(f"Estimated cost: ~${estimated_cost:.2f}  (max concurrent: {MAX_CONCURRENT})")

    client = anthropic.Anthropic(api_key=api_key)
    FRAUD_CASES_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    llm_count = 0
    fallback_count = 0
    pattern_counts = {}
    progress_lock = Lock()
    start_time = time.time()

    def process_case(args):
        nonlocal completed, llm_count, fallback_count
        cluster_id, row = args
        case_id = f"llm_{cluster_id:03d}"
        case_dict, used_llm = generate_one_case(row, case_id, client, model)

        full_case = {
            "id": case_id,
            "title": case_dict["title"],
            "pattern": case_dict["pattern"],
            "narrative": case_dict["narrative"],
            "indicators": case_dict["indicators"],
            "source": "llm_generated_from_real_fraud" if used_llm else "template_fallback",
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
        out_path = FRAUD_CASES_DIR / f"{case_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_case, f, indent=2)

        with progress_lock:
            completed += 1
            if used_llm:
                llm_count += 1
            else:
                fallback_count += 1
            pattern_counts[full_case["pattern"]] = pattern_counts.get(full_case["pattern"], 0) + 1
            if completed % 10 == 0 or completed == n_total:
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 1)
                eta = (n_total - completed) / max(rate, 0.1)
                logger.info(
                    f"  {completed}/{n_total} ({llm_count} LLM, {fallback_count} fallback) | "
                    f"~${completed * COST_PER_CALL:.2f} | ETA: {eta/60:.1f} min"
                )
        return case_id

    logger.info("Starting concurrent generation...")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = [executor.submit(process_case, args) for args in selected_rows]
        for _ in as_completed(futures):
            pass

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Generated {n_total} LLM cases in {elapsed/60:.1f} min")
    logger.info(f"  LLM: {llm_count}  Fallback: {fallback_count}")
    logger.info(f"  Cost: ~${llm_count * COST_PER_CALL:.2f}")
    logger.info("Pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {pattern}: {count}")
    logger.info("Next: python -m src.agentic.generate_cases_template")


if __name__ == "__main__":
    main()
