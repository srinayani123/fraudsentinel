
"""
Optuna hyperparameter optimization for LSTM Autoencoder.

Runs 8 Bayesian trials over architecture + training hyperparameters.
Optimizes fraud recall at 5% legit FPR.

Each trial logged to MLflow. Best config saved to models/best_lstm_params.json.
After optimization, run:
    python -m src.dl_models.train_lstm_ae --use-best-params
to produce the final tuned model artifact.
"""

import json

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.dl_models.model import LSTMAutoencoder
from src.dl_models.train_lstm_ae import LSTM_FEATURES, SequenceDataset
from src.utils.config import (
    FEATURES_DIR,
    LSTM_CONFIG,
    MODELS_DIR,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

OPTUNA_EXPERIMENT = "fraudsentinel_lstm_optuna"
N_TRIALS = 8

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    path = FEATURES_DIR / "engineered.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run Spark pipeline first. Missing: {path}")
    return pd.read_parquet(path)


def build_loaders_and_eval_data(df, scaler, seq_len, batch_size):
    df_legit = df[df["isFraud"] == 0].copy()
    df_fraud = df[df["isFraud"] == 1].copy()
    if len(df_legit) > 200_000:
        df_legit = df_legit.sample(n=200_000, random_state=42)

    train_ds = SequenceDataset(df_legit, scaler)
    fraud_ds = SequenceDataset(df_fraud, scaler) if len(df_fraud) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, train_ds, fraud_ds


def evaluate(model, train_ds, fraud_ds, device):
    """Compute fraud recall at 5% legit FPR."""
    model.eval()
    sample_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
    legit_errors = []
    with torch.no_grad():
        for batch in sample_loader:
            batch = batch.to(device)
            err = model.reconstruction_error(batch).cpu().numpy()
            legit_errors.extend(err.tolist())
            if len(legit_errors) >= 20_000:
                break
    legit_errors = np.array(legit_errors)
    threshold = float(np.percentile(legit_errors, 95))

    if fraud_ds is None or len(fraud_ds) == 0:
        return 0.0, threshold

    fraud_loader = DataLoader(fraud_ds, batch_size=512, shuffle=False)
    fraud_errors = []
    with torch.no_grad():
        for batch in fraud_loader:
            batch = batch.to(device)
            err = model.reconstruction_error(batch).cpu().numpy()
            fraud_errors.extend(err.tolist())
    fraud_errors = np.array(fraud_errors)
    fraud_recall = float((fraud_errors > threshold).mean())
    return fraud_recall, threshold


def objective(trial, df):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 2e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = 20

    seq_len = LSTM_CONFIG["sequence_length"]
    df_legit_for_scaler = df[df["isFraud"] == 0].sample(n=min(200_000, (df["isFraud"]==0).sum()), random_state=42)
    scaler = StandardScaler()
    scaler.fit(df_legit_for_scaler[LSTM_FEATURES].fillna(0).values)

    train_loader, train_ds, fraud_ds = build_loaders_and_eval_data(df, scaler, seq_len, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        input_dim=len(LSTM_FEATURES),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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
        if avg_loss < best_loss:
            best_loss = avg_loss

    fraud_recall, threshold = evaluate(model, train_ds, fraud_ds, device)

    with mlflow.start_run(run_name=f"lstm_trial_{trial.number}"):
        mlflow.log_params(
            {
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )
        mlflow.log_metric("fraud_recall_at_5pct_fpr", fraud_recall)
        mlflow.log_metric("final_train_loss", best_loss)
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metric("trial_number", trial.number)

    return fraud_recall


def main():
    df = load_data()
    logger.info(f"Loaded {len(df):,} rows for LSTM tuning")
    logger.info(f"Will run {N_TRIALS} trials. Expected: 4-5 hours on CPU.")

    mlflow.set_experiment(OPTUNA_EXPERIMENT)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lstm_tuning")

    def callback(study, trial):
        logger.info(
            f"Trial {trial.number+1}/{N_TRIALS} | Fraud recall: {trial.value:.4f} | "
            f"Best so far: {study.best_value:.4f}"
        )

    study.optimize(
        lambda t: objective(t, df),
        n_trials=N_TRIALS,
        callbacks=[callback],
        show_progress_bar=False,
    )

    logger.info("=" * 60)
    logger.info("Optimization complete")
    logger.info(f"Best fraud recall (at 5% legit FPR): {study.best_value:.4f}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    out_path = MODELS_DIR / "best_lstm_params.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "best_fraud_recall": study.best_value,
                "best_params": study.best_params,
                "n_trials": N_TRIALS,
            },
            f,
            indent=2,
        )
    logger.info(f"Best params saved to {out_path}")
    logger.info("Next: python -m src.dl_models.train_lstm_ae --use-best-params")


if __name__ == "__main__":
    main()
