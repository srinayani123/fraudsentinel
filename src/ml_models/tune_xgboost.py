
"""
Optuna hyperparameter optimization for XGBoost on IEEE-CIS fraud detection.

Runs 50 Bayesian trials over key hyperparameters, optimizing PR-AUC on a
validation split. Each trial logged to MLflow.

After optimization, the best params are printed. Then re-run train_xgboost.py
with the best params to produce the final model artifact.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.utils.config import (
    FEATURES_DIR,
    MODELS_DIR,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

OPTUNA_EXPERIMENT = "fraudsentinel_xgboost_optuna"
N_TRIALS = 50

# Reduce optuna log noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_features():
    path = FEATURES_DIR / "engineered.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run Spark pipeline first. Missing: {path}")
    logger.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    return df


def prepare_xy(df):
    target = "isFraud"
    drop_cols = ["TransactionID", "TransactionDT", target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes.astype("int32")
    X = X.fillna(-999)
    y = df[target].astype(int)
    return X, y


def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        # Tunable
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

    model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_proba)

    # Log to MLflow
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=False):
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("trial_number", trial.number)

    return pr_auc


def main():
    df = load_features()
    X, y = prepare_xy(df)
    logger.info(f"Total: {len(X):,} rows, {len(X.columns)} features, fraud rate {y.mean():.4%}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    logger.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

    mlflow.set_experiment(OPTUNA_EXPERIMENT)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_tuning")

    logger.info(f"Starting Optuna search: {N_TRIALS} trials. This will take 60-120 min.")
    logger.info(f"Track progress: mlflow ui  →  http://localhost:5000  → experiment '{OPTUNA_EXPERIMENT}'")

    def callback(study, trial):
        logger.info(
            f"Trial {trial.number+1}/{N_TRIALS} | PR-AUC: {trial.value:.4f} | "
            f"Best so far: {study.best_value:.4f}"
        )

    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=N_TRIALS,
        callbacks=[callback],
        show_progress_bar=False,
    )

    logger.info("=" * 60)
    logger.info(f"✅ Optimization complete!")
    logger.info(f"Best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Save best params to a JSON for the retrain step
    out_path = MODELS_DIR / "best_xgb_params.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "best_pr_auc": study.best_value,
                "best_params": study.best_params,
                "n_trials": N_TRIALS,
            },
            f,
            indent=2,
        )
    logger.info(f"Best params saved to {out_path}")
    logger.info("Next: run 'python -m src.ml_models.train_xgboost --use-best-params'")


if __name__ == "__main__":
    main()

