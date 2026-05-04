# Deploying FraudSentinel to Hugging Face Spaces

This guide covers deploying the Streamlit dashboard as a public live demo.

## Why Streamlit SDK (not Docker SDK)?

We use the **Streamlit SDK** on Hugging Face Spaces:

- ✅ Native to HF — they handle Python env + Streamlit config
- ✅ Faster cold starts than Docker
- ✅ Simpler — just push the code, HF builds it

If you'd prefer Docker on Spaces (e.g., to bundle the FastAPI service alongside), HF supports that too — but you'd need to combine both services into one image since each Space runs one container. For most demos, Streamlit SDK is the right choice.

## Prerequisites

1. A Hugging Face account (free tier works)
2. The `huggingface_hub` CLI (`pip install huggingface_hub`)
3. Pre-trained model artifacts in `models/` and demo sample in `data/samples/`
   - These are committed to the repo; they're already there if you cloned a release
   - Or run the offline pipeline (see [`PIPELINE.md`](PIPELINE.md))

## What gets deployed

The Space loads:
- `src/dashboard/app.py` (entrypoint, declared in the README header)
- All `src/dashboard/pages/*.py`
- `src/agentic/`, `src/ml_models/`, `src/dl_models/`, `src/utils/`
- `models/` (pre-trained artifacts)
- `data/samples/demo_transactions.parquet`
- `data/fraud_cases/case_*.json`
- `requirements-hf.txt` (renamed to `requirements.txt` on the Space)

## Step-by-step deploy

### 1. Create a new Space

Go to https://huggingface.co/new-space, choose:
- **Space SDK:** Streamlit
- **Space hardware:** CPU basic (free) — 16GB RAM, 2 vCPU
- **Visibility:** Public (or Private if you prefer)
- **Name:** e.g., `fraudsentinel`

### 2. Clone the empty Space locally

```bash
huggingface-cli login  # paste your HF token
git clone https://huggingface.co/spaces/<your-username>/fraudsentinel hf-space
cd hf-space
```

### 3. Copy this project into the Space

```bash
# From inside hf-space/
cp -r ../fraudsentinel/src .
cp -r ../fraudsentinel/models .
cp -r ../fraudsentinel/data .   # only data/samples and data/fraud_cases
cp -r ../fraudsentinel/.streamlit .

# IMPORTANT: rename the HF-specific files
cp ../fraudsentinel/requirements-hf.txt requirements.txt
cp ../fraudsentinel/README_HF.md README.md
```

The `README_HF.md` has the YAML frontmatter that HF Spaces parses to configure the Space (title, emoji, SDK, app_file path, etc.).

### 4. Verify required files

```bash
ls
# Should see:
#   README.md          ← has YAML frontmatter
#   requirements.txt   ← lightweight deps (no Spark/MLflow)
#   src/               ← code
#   models/            ← trained artifacts
#   data/samples/      ← demo_transactions.parquet
#   data/fraud_cases/  ← case JSONs
#   .streamlit/        ← theme config
```

### 5. Check the model file sizes

Spaces has a 50GB limit but Git LFS is recommended for files > 10MB.

```bash
du -sh models/*
# If any single file > 10MB, use git LFS:
git lfs install
git lfs track "models/*.pt" "models/*.json" "models/chroma_db/*"
```

### 6. Push

```bash
git add -A
git commit -m "Initial deploy of FraudSentinel"
git push
```

The Space will start building. Watch the **Build logs** tab on the HF Space page.

First build takes 5–10 minutes (PyTorch install dominates).

### 7. Test

Once the Space is "Running":
1. Open the Space URL
2. Paste your Anthropic API key in the sidebar
3. Navigate to the **Investigation** page
4. Pick a high-risk transaction
5. Click **Run Investigation**

If anything errors, check the **Logs** tab.

## Troubleshooting

### "Application error" on launch
Usually a missing import or model file. Check Logs. Common fixes:
- `requirements.txt` missing a dependency → add and push
- `models/` directory missing → re-copy and push
- Path issue → ensure `app_file: src/dashboard/app.py` matches your structure

### "Out of memory"
Free Spaces have 16GB. PyTorch + ChromaDB + sentence-transformers + XGBoost + Streamlit fit comfortably, but:
- If you bundled raw IEEE-CIS data, remove it (only `data/samples/` should be deployed)
- Switch to a paid Space if you need the headroom

### "ChromaDB persistence error"
Free Spaces have ephemeral storage by default. The ChromaDB index is rebuilt at startup if missing. Make sure `models/chroma_db/` is in the deploy. For persistent state across restarts, attach a paid Persistent Storage add-on.

### "Module not found" for `src.something`
Make sure you copied `src/` to the Space root. The dashboard imports use absolute paths like `from src.utils.config import ...`.

### Slow first request
First-time loads of XGBoost + LSTM + sentence-transformers can take 30s. Subsequent requests are fast (cached models).

## Updating the Space

```bash
# Make changes locally, then:
cd hf-space
git pull  # in case anyone else pushed
# copy updated files
cp -r ../fraudsentinel/src .
git add -A
git commit -m "Update X"
git push
```

The Space rebuilds automatically.

## What if I need MLflow UI in the live demo?

You don't. MLflow is for offline training. The Space loads pre-trained models directly. If you want MLflow tracking visible to viewers, you can:
- Take screenshots of the local MLflow UI and embed them in `Analytics` page
- Spin up an MLflow tracking server on a separate VM and link to it
- Skip it — the Space just shows the trained model performance, which is what users care about

## Costs

- **Free tier:** $0/month, 16GB RAM, 2 vCPU, sleeps after inactivity
- **CPU upgrade:** $9/month for always-on
- **GPU:** unnecessary for this demo (CPU is plenty for inference)
- **Anthropic inference:** $0 for you (BYOK — users pay with their own keys)
