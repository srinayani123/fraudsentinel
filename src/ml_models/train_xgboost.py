
"""
Train XGBoost classifier on the engineered features.

- Logs everything to MLflow (params, metrics, feature importances, model artifact)
- Handles class imbalance via scale_pos_weight
- Reports PR-AUC, ROC-AUC, and recall @ fixed FPR (the metric that matters in fraud)

Usage:
    python -m src.ml_models.train_xgboost                     # train with default params
    python -m src.ml_models.train_xgboost --use-best-params   # train with Optuna-tuned params
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    FEATURE_COLUMNS_PATH,
    FEATURES_DIR,
    MLFLOW_EXPERIMENT_XGB,
    MODELS_DIR,
    SCALER_PATH,
    XGB_MODEL_PATH,
    XGB_PARAMS,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_features():
    path = FEATURES_DIR / "engineered.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run Spark feature pipeline first. Missing: {path}")
    logger.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Shape: {df.shape}, fraud rate: {df['isFraud'].mean():.4%}")
    return df


def prepare_xy(df):
    """Split into features/target. Cast categoricals to numeric codes."""
    target = "isFraud"
    drop_cols = ["TransactionID", "TransactionDT", target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals as numeric codes (XGBoost can handle them but pandas won't pass strings)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes.astype("int32")

    # Fill NaN with -999 (XGBoost handles this as a sentinel value)
    X = X.fillna(-999)

    y = df[target].astype(int)
    return X, y


def recall_at_fpr(y_true, y_scores, target_fpr=0.001):
    """Recall when we fix the false-positive rate. The fraud-detection metric that matters."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        return tpr[-1]
    return tpr[idx]


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on engineered IEEE-CIS features.")
    parser.add_argument(
        "--use-best-params",
        action="store_true",
        help="Load best hyperparameters from Optuna study (models/best_xgb_params.json)",
    )
    args, _ = parser.parse_known_args()

    df = load_features()
    X, y = prepare_xy(df)
    feature_cols = X.columns.tolist()
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Recompute scale_pos_weight from actual data
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    logger.info(f"scale_pos_weight = {scale_pos_weight:.2f} ({neg:,} neg / {pos:,} pos)")

    # Override with Optuna best params if requested
    run_label = "xgboost_baseline"
    if args.use_best_params:
        bp_path = MODELS_DIR / "best_xgb_params.json"
        if not bp_path.exists():
            logger.warning(
                f"--use-best-params flag set but {bp_path} not found. "
                f"Falling back to default config. Run 'python -m src.ml_models.tune_xgboost' first."
            )
        else:
            with open(bp_path) as f:
                bp = json.load(f)
            logger.info(
                f"Loading Optuna-tuned hyperparameters from {bp_path.name} "
                f"(study best PR-AUC: {bp['best_pr_auc']:.4f})"
            )
            params.update(bp["best_params"])
            run_label = "xgboost_optuna_tuned"

    # Scaler — XGBoost doesn't need it, but we save it for the LSTM that consumes the same features
    scaler = StandardScaler()
    scaler.fit(X_train.values)

    # MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_XGB)
    with mlflow.start_run(run_name=run_label) as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("train_fraud_rate", y_train.mean())
        mlflow.log_param("used_best_params", args.use_best_params)

        logger.info(f"Training XGBoost ({run_label})...")
        model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "pr_auc": average_precision_score(y_test, y_proba),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "recall_at_fpr_0.001": recall_at_fpr(y_test, y_proba, 0.001),
            "recall_at_fpr_0.01": recall_at_fpr(y_test, y_proba, 0.01),
        }
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Confusion matrix at 0.5 threshold
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion matrix @ 0.5:\n{cm}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['legit','fraud'])}")

        # Save artifacts
        XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(XGB_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        with open(FEATURE_COLUMNS_PATH, "w") as f:
            json.dump({"feature_columns": feature_cols}, f, indent=2)

        mlflow.xgboost.log_model(model, "xgboost_model")
        mlflow.log_artifact(str(FEATURE_COLUMNS_PATH))
        mlflow.log_artifact(str(SCALER_PATH))

        # Log feature importances
        importances = model.feature_importances_
        imp_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)
        imp_path = XGB_MODEL_PATH.parent / "feature_importances.csv"
        imp_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))

        logger.info(f"✅ Model saved → {XGB_MODEL_PATH}")
        logger.info(f"   Scaler  → {SCALER_PATH}")
        logger.info(f"   Run ID: {run.info.run_id}")
        logger.info("   View in MLflow UI:  mlflow ui  → http://localhost:5000")


if __name__ == "__main__":
    main()

