"""
Create the 10k-transaction sample that ships with the live demo.

Strategy:
- Keep all fraud transactions in a stratified slice (oversample fraud for demo richness)
- Mix in legit transactions to maintain ~5% fraud rate (vs ~3.5% in original)
- Include feature-engineered version + raw original for the dashboard

Output: data/samples/demo_transactions.parquet
"""

import numpy as np
import pandas as pd

from src.utils.config import FEATURES_DIR, SAMPLE_SIZE, SAMPLES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    features_path = FEATURES_DIR / "engineered.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found. Run the Spark feature pipeline first:\n"
            f"  python -m src.spark_pipeline.build_features"
        )

    logger.info(f"Loading engineered features from {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Original fraud rate: {df['isFraud'].mean():.4%}")

    # Stratified sample: target ~5% fraud rate for richer demo
    target_fraud_rate = 0.05
    n_fraud = int(SAMPLE_SIZE * target_fraud_rate)
    n_legit = SAMPLE_SIZE - n_fraud

    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]

    n_fraud = min(n_fraud, len(fraud_df))
    n_legit = min(n_legit, len(legit_df))

    fraud_sample = fraud_df.sample(n=n_fraud, random_state=42)
    legit_sample = legit_df.sample(n=n_legit, random_state=42)

    sample = pd.concat([fraud_sample, legit_sample], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(f"Sample: {len(sample):,} rows, {sample['isFraud'].mean():.4%} fraud")

    # Add a synthetic timestamp for the live feed visualization
    # Original IEEE-CIS uses TransactionDT (seconds from a reference)
    # We'll convert to datetime for prettier display
    if "TransactionDT" in sample.columns:
        ref = pd.Timestamp("2017-12-01")
        sample["timestamp"] = ref + pd.to_timedelta(sample["TransactionDT"], unit="s")
    else:
        sample["timestamp"] = pd.Timestamp.now() - pd.to_timedelta(
            np.random.randint(0, 86400 * 30, len(sample)), unit="s"
        )

    # Sort by timestamp so the live feed plays back chronologically
    sample = sample.sort_values("timestamp").reset_index(drop=True)

    out_path = SAMPLES_DIR / "demo_transactions.parquet"
    sample.to_parquet(out_path, index=False)
    logger.info(f"✅ Wrote demo sample to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Also write a tiny CSV preview for quick inspection
    preview = sample.head(100)
    preview.to_csv(SAMPLES_DIR / "demo_preview.csv", index=False)
    logger.info(f"   Preview: {SAMPLES_DIR / 'demo_preview.csv'}")


if __name__ == "__main__":
    main()
