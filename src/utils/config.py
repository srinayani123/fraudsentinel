"""Central configuration for the FraudSentinel project."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# -------------------- Paths --------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SAMPLES_DIR = DATA_DIR / "samples"
FRAUD_CASES_DIR = DATA_DIR / "fraud_cases"

MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = MODELS_DIR / "chroma_db"

# Ensure dirs exist
for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, SAMPLES_DIR, FRAUD_CASES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Data --------------------
KAGGLE_COMPETITION = "ieee-fraud-detection"
SAMPLE_SIZE = 10_000  # Number of transactions to ship with the demo

# -------------------- Models --------------------
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
LSTM_THRESHOLD_PATH = MODELS_DIR / "lstm_threshold.json"

# -------------------- Agentic — model selection per stage --------------------
# Strategy:
#   - Triage and Pattern do bounded structured-output tasks. Haiku 4.5
#     matches Sonnet quality at much lower latency for these.
#   - Investigator does multi-turn tool use + reasoning. Stays on Sonnet.
#   - Report writes the analyst-facing synthesis where word choice matters.
#     Stays on Sonnet.
#
# All models can be overridden via .env. The DEFAULT_ANTHROPIC_MODEL is kept
# for backward-compat with code that doesn't yet read per-stage models.
#
# Model strings (as of November 2025):
#   - claude-sonnet-4-5           (~3-8x slower than Haiku, higher reasoning)
#   - claude-haiku-4-5-20251001   (fast, cheap, JSON-output friendly)
#   - claude-opus-4-7             (most capable, slowest)
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5-20251001")
PATTERN_MODEL = os.environ.get("PATTERN_MODEL", "claude-haiku-4-5-20251001")
INVESTIGATOR_MODEL = os.environ.get("INVESTIGATOR_MODEL", DEFAULT_ANTHROPIC_MODEL)
REPORT_MODEL = os.environ.get("REPORT_MODEL", DEFAULT_ANTHROPIC_MODEL)

# Rule Generator uses the same Sonnet/Haiku split — Planner and Workers
# benefit from Sonnet's reasoning when proposing thresholds, Synthesizer
# stays on Sonnet for the ranking judgment.
RULE_GEN_PLANNER_MODEL = os.environ.get("RULE_GEN_PLANNER_MODEL", DEFAULT_ANTHROPIC_MODEL)
RULE_GEN_WORKER_MODEL = os.environ.get("RULE_GEN_WORKER_MODEL", DEFAULT_ANTHROPIC_MODEL)
RULE_GEN_SYNTHESIZER_MODEL = os.environ.get("RULE_GEN_SYNTHESIZER_MODEL", DEFAULT_ANTHROPIC_MODEL)

CHROMA_COLLECTION_NAME = "fraud_cases"

# -------------------- ML hyperparams --------------------
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 25,  # ~1/0.035 fraud rate, will be recalculated from data
    "tree_method": "hist",
    "random_state": 42,
}

LSTM_CONFIG = {
    "sequence_length": 10,
    "input_dim": 16,
    "hidden_dim": 32,
    "latent_dim": 16,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 256,
    "epochs": 30,
    "learning_rate": 1e-3,
}

# -------------------- Risk scoring --------------------
RISK_WEIGHTS = {
    "xgboost": 0.6,
    "lstm_anomaly": 0.4,
}

RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.85,
}

# -------------------- MLflow --------------------
MLFLOW_EXPERIMENT_XGB = "fraudsentinel_xgboost"
MLFLOW_EXPERIMENT_LSTM = "fraudsentinel_lstm_ae"
