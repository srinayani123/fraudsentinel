"""Load a trained XGBoost model and score transactions in real time."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.config import FEATURE_COLUMNS_PATH, XGB_MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {XGB_MODEL_PATH}. "
            "Train it first: python -m src.ml_models.train_xgboost"
        )
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)["feature_columns"]
    return model, feature_columns


def score_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Return fraud probability for each row in df. Aligns columns automatically."""
    model, feature_columns = load_model()

    X = df.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = -999
    X = X[feature_columns]

    # Encode categoricals (same way as training)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes.astype("int32")

    X = X.fillna(-999)
    return model.predict_proba(X)[:, 1]


def score_one(transaction: dict) -> float:
    """Convenience wrapper for scoring a single transaction (dict)."""
    df = pd.DataFrame([transaction])
    return float(score_dataframe(df)[0])
