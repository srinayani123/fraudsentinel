
"""
Train LSTM Autoencoder on legitimate-only transaction sequences.

Output: a model that produces an "anomaly score" per transaction.
Logs to MLflow.

Usage:
    python -m src.dl_models.train_lstm_ae                     # default config
    python -m src.dl_models.train_lstm_ae --use-best-params   # use Optuna-tuned config
"""

import argparse
import json

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.dl_models.model import LSTMAutoencoder
from src.utils.config import (
    FEATURES_DIR,
    LSTM_CONFIG,
    LSTM_MODEL_PATH,
    LSTM_THRESHOLD_PATH,
    MLFLOW_EXPERIMENT_LSTM,
    MODELS_DIR,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Behavioral features used for the LSTM. Must be numeric.
LSTM_FEATURES = [
    "TransactionAmt",
    "txn_hour",
    "is_night",
    "card1_txn_count_1h",
    "card1_txn_count_24h",
    "card1_txn_count_7d",
    "card1_amt_sum_1h",
    "card1_amt_sum_24h",
    "card1_amt_max_24h",
    "card1_seconds_since_last",
    "card1_amt_mean",
    "card1_amt_zscore",
    "card1_distinct_products",
    "card1_distinct_addr1",
    "P_emaildomain_isnull",
    "emails_match",
]
SEQ_LEN = LSTM_CONFIG["sequence_length"]


class SequenceDataset(Dataset):
    """Builds rolling windows of length SEQ_LEN per card1."""

    def __init__(self, df: pd.DataFrame, scaler: StandardScaler):
        self.sequences = []
        df = df.sort_values(["card1", "TransactionDT"])
        for card_id, group in df.groupby("card1"):
            if len(group) < SEQ_LEN:
                continue
            arr = group[LSTM_FEATURES].fillna(0).values.astype(np.float32)
            arr = scaler.transform(arr)
            for i in range(len(arr) - SEQ_LEN + 1):
                self.sequences.append(arr[i : i + SEQ_LEN])
        self.sequences = np.array(self.sequences, dtype=np.float32)
        logger.info(f"Built {len(self.sequences):,} sequences of length {SEQ_LEN}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])


def main():
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder on engineered features.")
    parser.add_argument(
        "--use-best-params",
        action="store_true",
        help="Load best hyperparameters from Optuna study (models/best_lstm_params.json)",
    )
    args, _ = parser.parse_known_args()

    path = FEATURES_DIR / "engineered.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run Spark feature pipeline first. Missing: {path}")

    logger.info("Loading features...")
    df = pd.read_parquet(path)

    missing = [c for c in LSTM_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Required features missing: {missing}")

    # Train only on legitimate transactions
    df_legit = df[df["isFraud"] == 0].copy()
    df_fraud = df[df["isFraud"] == 1].copy()
    logger.info(f"Legit: {len(df_legit):,} | Fraud: {len(df_fraud):,}")

    # Cap memory: subsample if huge
    if len(df_legit) > 200_000:
        logger.info("Subsampling legit to 200k for training tractability.")
        df_legit = df_legit.sample(n=200_000, random_state=42)

    # Load config — defaults or Optuna-tuned
    config = dict(LSTM_CONFIG)
    run_label = "lstm_autoencoder"
    if args.use_best_params:
        bp_path = MODELS_DIR / "best_lstm_params.json"
        if not bp_path.exists():
            logger.warning(
                f"--use-best-params requested but {bp_path} not found. "
                f"Falling back to defaults. Run 'python -m src.dl_models.tune_lstm' first."
            )
        else:
            with open(bp_path) as f:
                bp = json.load(f)
            logger.info(
                f"Loading Optuna-tuned hyperparameters from {bp_path.name} "
                f"(study best fraud recall: {bp['best_fraud_recall']:.4f})"
            )
            config.update(bp["best_params"])
            run_label = "lstm_optuna_tuned"

    # Fit scaler on legit data
    scaler = StandardScaler()
    scaler.fit(df_legit[LSTM_FEATURES].fillna(0).values)

    train_ds = SequenceDataset(df_legit, scaler)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = LSTMAutoencoder(
        input_dim=len(LSTM_FEATURES),
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    epochs = config.get("epochs", 30)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_LSTM)
    with mlflow.start_run(run_name=run_label) as run:
        mlflow.log_params(config)
        mlflow.log_param("n_features", len(LSTM_FEATURES))
        mlflow.log_param("n_train_sequences", len(train_ds))
        mlflow.log_param("used_best_params", args.use_best_params)

        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * batch.size(0)

            avg_loss = total_loss / len(train_ds)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            logger.info(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        # Compute anomaly threshold on a held-out legit sample (95th percentile)
        model.eval()
        sample_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
        errors = []
        with torch.no_grad():
            for batch in sample_loader:
                batch = batch.to(device)
                err = model.reconstruction_error(batch).cpu().numpy()
                errors.extend(err.tolist())
                if len(errors) >= 20_000:
                    break
        errors = np.array(errors)
        threshold = float(np.percentile(errors, 95))
        logger.info(f"Anomaly threshold (95th pct of legit error): {threshold:.6f}")

        # If we have fraud data, evaluate
        if len(df_fraud) > 0:
            fraud_ds = SequenceDataset(df_fraud, scaler)
            if len(fraud_ds) > 0:
                fraud_loader = DataLoader(fraud_ds, batch_size=512, shuffle=False)
                fraud_errors = []
                with torch.no_grad():
                    for batch in fraud_loader:
                        batch = batch.to(device)
                        err = model.reconstruction_error(batch).cpu().numpy()
                        fraud_errors.extend(err.tolist())
                if fraud_errors:
                    fraud_errors = np.array(fraud_errors)
                    fraud_recall = float((fraud_errors > threshold).mean())
                    legit_fpr = float((errors > threshold).mean())
                    logger.info(f"Fraud recall @ threshold: {fraud_recall:.4f}")
                    logger.info(f"Legit FPR    @ threshold: {legit_fpr:.4f}")
                    mlflow.log_metric("fraud_recall_at_threshold", fraud_recall)
                    mlflow.log_metric("legit_fpr_at_threshold", legit_fpr)

        # Save artifacts
        torch.save(model.state_dict(), LSTM_MODEL_PATH)
        joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
        with open(LSTM_THRESHOLD_PATH, "w") as f:
            json.dump(
                {
                    "threshold": threshold,
                    "feature_columns": LSTM_FEATURES,
                    "sequence_length": SEQ_LEN,
                    "config": config,
                },
                f,
                indent=2,
            )

        mlflow.pytorch.log_model(model, "lstm_autoencoder")
        mlflow.log_artifact(str(LSTM_THRESHOLD_PATH))

        logger.info(f"✅ LSTM saved → {LSTM_MODEL_PATH}")
        logger.info(f"   Threshold/config → {LSTM_THRESHOLD_PATH}")
        logger.info(f"   Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
