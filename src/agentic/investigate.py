"""
Run an agentic investigation on the top-N highest-risk transactions.

Useful for batch testing without firing up the dashboard. Requires:
- ANTHROPIC_API_KEY in env
- Trained models + ChromaDB built
- Demo sample built
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv

from src.agentic.orchestrator import FraudInvestigator
from src.ml_models.inference import score_dataframe
from src.utils.config import SAMPLES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(top_n: int = 3):
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")

    sample_path = SAMPLES_DIR / "demo_transactions.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    df = pd.read_parquet(sample_path)
    logger.info(f"Loaded {len(df):,} transactions")

    logger.info("Scoring with XGBoost...")
    df["xgb_score"] = score_dataframe(df)

    top = df.nlargest(top_n, "xgb_score")
    logger.info(f"Top {top_n} highest-risk transactions selected.")

    investigator = FraudInvestigator(api_key=api_key)

    for i, (_, row) in enumerate(top.iterrows()):
        logger.info(f"\n{'='*70}\nInvestigating transaction {i+1}/{top_n}: {row.get('TransactionID', '?')}\n{'='*70}")
        txn = row.to_dict()
        # Drop heavy fields for the LLM
        txn.pop("isFraud", None)

        # We'd compute LSTM score here if model is loaded; for the CLI, skip
        ctx = investigator.investigate(
            transaction=txn,
            xgb_score=float(row["xgb_score"]),
            lstm_score=0.5,  # placeholder
            lstm_anomaly=False,
        )
        print("\n--- TRIAGE ---")
        print(json.dumps(ctx.triage_result, indent=2))
        print("\n--- INVESTIGATOR ---")
        print(ctx.investigator_findings)
        print("\n--- PATTERN ---")
        print(ctx.pattern_analysis)
        print("\n--- REPORT ---")
        print(ctx.final_report)


if __name__ == "__main__":
    main()
