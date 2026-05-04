# Training Pipeline Walkthrough

End-to-end steps to train all models from scratch on IEEE-CIS, then bundle artifacts for the live demo.

> **Time:** ~30–60 minutes on a modern laptop with 16GB RAM.
> **Disk:** ~5–7 GB for raw + processed data + MLflow runs.

## Prerequisites

- Python 3.10+
- A Kaggle account with API token at `~/.kaggle/kaggle.json` (`chmod 600`)
- You've accepted the [IEEE-CIS competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules)
- (Optional) `ANTHROPIC_API_KEY` in `.env` for testing the agentic layer

## Step 0 — Setup

```bash
git clone <repo>
cd fraudsentinel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in keys
```

## Step 1 — Download IEEE-CIS

```bash
python -m src.data_ingestion.download_ieee_cis
```

Downloads ~1.3 GB into `data/raw/`. Idempotent — won't re-download if files already exist.

## Step 2 — Spark feature engineering

```bash
python -m src.spark_pipeline.build_features
```

What it does:
- Joins `train_transaction` + `train_identity`
- Time features (hour, day-of-week, is_night)
- Velocity features (count + sum amount over rolling 1h / 24h / 7d windows per card)
- Behavioral aggregates (mean, std, distinct merchants per card)
- Email-domain risk indicators
- Selects ~80 curated columns for downstream training

Output: `data/features/engineered.parquet/` (Spark writes a directory; pandas reads it fine)

## Step 3 — Train XGBoost

```bash
python -m src.ml_models.train_xgboost
```

What it does:
- Loads the engineered Parquet
- Stratified train/test split (80/20)
- Trains XGBoost with class-weighting (`scale_pos_weight` from data)
- Logs everything to MLflow (params, metrics, model artifact, feature importances)
- Saves: `models/xgboost_model.json`, `models/feature_columns.json`, `models/feature_scaler.pkl`

Open the MLflow UI to inspect:
```bash
mlflow ui
# → http://localhost:5000
```

## Step 4 — Train LSTM Autoencoder

```bash
python -m src.dl_models.train_lstm_ae
```

What it does:
- Subsamples 200k legitimate transactions
- Builds rolling 10-step sequences per card
- Trains an LSTM encoder/decoder reconstruction model
- Computes the 95th-percentile reconstruction error of legit data → anomaly threshold
- Evaluates fraud recall at that threshold
- Saves: `models/lstm_autoencoder.pt`, `models/lstm_threshold.json`, `models/lstm_scaler.pkl`

CPU training takes ~15 min on a laptop. GPU is much faster.

## Step 5 — Build the ChromaDB knowledge base

```bash
python -m src.agentic.build_knowledge_base
```

What it does:
- Defines 10 synthesized historical fraud-case narratives (in `src/agentic/build_knowledge_base.py`)
- Persists each as JSON in `data/fraud_cases/`
- Indexes them in ChromaDB at `models/chroma_db/` using `sentence-transformers/all-MiniLM-L6-v2`

To add new cases, edit the `FRAUD_CASES` list in that file and re-run.

## Step 6 — Build the demo sample

```bash
python -m src.data_ingestion.sample_for_demo
```

What it does:
- Loads engineered features
- Stratified samples 10,000 transactions (~5% fraud rate)
- Adds synthetic timestamps (derived from `TransactionDT`)
- Saves `data/samples/demo_transactions.parquet` — the file the live demo loads

## Step 7 — Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

Open http://localhost:8501. Paste your Anthropic key. Investigate.

## Step 8 — (Optional) Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

## Step 9 — (Optional) Run as Docker

```bash
docker compose -f docker/docker-compose.yml up --build
# Dashboard: http://localhost:8501
# API:       http://localhost:8000/docs
```

## Step 10 — (Optional) Deploy to Hugging Face Spaces

See [`DEPLOY_HF_SPACES.md`](DEPLOY_HF_SPACES.md).

---

## Re-running parts of the pipeline

The pipeline is idempotent — each step writes its own outputs. To force a re-run, delete the output and re-execute:

```bash
# Re-run just XGBoost
rm models/xgboost_model.json
python -m src.ml_models.train_xgboost

# Re-run the demo sample with different size
# (edit SAMPLE_SIZE in src/utils/config.py first)
rm data/samples/demo_transactions.parquet
python -m src.data_ingestion.sample_for_demo
```

## Tracking experiments

Every training run creates an MLflow run. Useful comparisons:
- Tune XGBoost hyperparams in `src/utils/config.py:XGB_PARAMS` and re-train
- Adjust LSTM `hidden_dim` / `latent_dim` and compare reconstruction errors
- Compare feature sets by editing `select_final_columns` in the Spark script
