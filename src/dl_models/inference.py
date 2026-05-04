"""LSTM Autoencoder inference for live anomaly scoring."""

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch

from src.dl_models.model import LSTMAutoencoder
from src.utils.config import LSTM_MODEL_PATH, LSTM_THRESHOLD_PATH, MODELS_DIR


@lru_cache(maxsize=1)
def load_model():
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model not found at {LSTM_MODEL_PATH}. "
            "Train it first: python -m src.dl_models.train_lstm_ae"
        )

    with open(LSTM_THRESHOLD_PATH) as f:
        meta = json.load(f)

    config = meta["config"]
    feature_columns = meta["feature_columns"]
    seq_len = meta["sequence_length"]
    threshold = meta["threshold"]

    model = LSTMAutoencoder(
        input_dim=len(feature_columns),
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location="cpu"))
    model.eval()

    scaler = joblib.load(MODELS_DIR / "lstm_scaler.pkl")
    return model, scaler, feature_columns, seq_len, threshold


def score_sequence(transactions: pd.DataFrame) -> dict:
    """Score the most recent transaction in a card's history.

    Args:
        transactions: DataFrame of transactions for a single card, ordered by time,
                      length >= sequence_length.

    Returns:
        dict with 'anomaly_score' (raw error), 'normalized_score' (0-1), 'is_anomaly' bool
    """
    model, scaler, feature_columns, seq_len, threshold = load_model()

    if len(transactions) < seq_len:
        # Pad with the first transaction repeated
        padding = pd.concat([transactions.iloc[[0]]] * (seq_len - len(transactions)))
        transactions = pd.concat([padding, transactions], ignore_index=True)

    seq = transactions[feature_columns].fillna(0).values.astype(np.float32)
    seq = scaler.transform(seq[-seq_len:])
    x = torch.from_numpy(seq).unsqueeze(0)  # (1, seq_len, n_features)

    with torch.no_grad():
        err = float(model.reconstruction_error(x).item())

    # Normalize to 0-1 using the threshold as a soft pivot
    # score = 1 / (1 + exp(-(err/threshold - 1) * 5))
    normalized = float(1.0 / (1.0 + np.exp(-(err / max(threshold, 1e-6) - 1.0) * 3.0)))

    return {
        "anomaly_score": err,
        "normalized_score": normalized,
        "is_anomaly": err > threshold,
        "threshold": threshold,
    }
