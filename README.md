# FraudSentinel

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)](https://streamlit.io)
[![Anthropic](https://img.shields.io/badge/Claude-Sonnet_4.5-orange)](https://www.anthropic.com)

Agentic fraud detection platform. Compresses a 15-minute analyst investigation into a 30-second multi-agent pipeline — flags a transaction, attributes the score, retrieves similar historical patterns, and synthesizes an analyst-facing decision.

> **🚀 Live Demo**: _coming soon — deployment in progress_
> **📓 IEEE-CIS Fraud Detection** dataset · **377 fraud patterns** indexed · **30s** investigation latency

---

<!-- SCREENSHOT: Main Investigate page showing the 5-tab decision package -->
<!-- ![Investigate page](docs/screenshots/investigate.png) -->

---

## What it does

- **Investigate** flagged transactions through a 4-agent pipeline (Triage → Investigator with tool use → Pattern matching → Report) with SHAP attribution woven between stages
- **Verify** retrieved patterns through a two-tier check — semantic retrieval finds candidates, a Pattern agent verifies indicator fit and falls back to SHAP-grounded analysis when no pattern cleanly matches
- **Generate** production fraud rules from a filtered transaction set using a Planner + 4 parallel Workers + Synthesizer pipeline. Output: ranked rules with SQL + plain English + estimated catch/FPR rates
- **Browse** 377 catalogued fraud archetypes across 13 categories with calibrated similarity matching
- **Bring your own Anthropic key** — visitors paste their key in Settings, lives only in the browser session

---

## How it works

The investigation pipeline runs four Claude agents in sequence with two deterministic attribution stages woven between them. **Haiku 4.5** handles bounded structured outputs (Triage routing, Pattern fit verdicts) and **Sonnet 4.5** handles multi-turn tool use (Investigator) and analyst-facing synthesis (Report). Each agent receives the previous stage's output plus ground-truth model attributions (SHAP for XGBoost, per-timestep reconstruction error for the LSTM autoencoder) so it cites real numbers instead of speculating.

The rule generator is the inverse pipeline — many transactions in, ranked production rules out. A Planner reads aggregate stats, dispatches 4 specialized Workers (Velocity / Email / Device / Amount) in parallel via `ThreadPoolExecutor`, and a Synthesizer ranks proposals across workers. ~60 seconds end-to-end via parallelism.

Pattern retrieval uses **bge-base-en-v1.5** embeddings indexed in **ChromaDB**, with a hybrid PCA + k-NN out-of-distribution detector calibrated against 752 auto-generated queries to ensure unrelated questions don't return spurious matches.

---

## Results

### Models

| Model | Metric | Score |
|---|---|---|
| XGBoost | PR-AUC | **0.8553** |
| XGBoost | ROC-AUC | **0.9698** |
| XGBoost | Recall @ 1% FPR | **81.7%** |
| LSTM Autoencoder | Recall @ 5% FPR | 14.9% |

### Retrieval

| Metric | Value |
|---|---|
| Patterns indexed | **377** |
| Archetype categories | 13 |
| OOD true-reject rate | **98%** |
| OOD false-reject rate | **0%** |
| Calibration queries | 752 |

### Pipeline latency

| Pipeline | Latency |
|---|---|
| Investigation (cold) | ~30s |
| Investigation (cached) | <500ms |
| Rule generation | ~60s |

---

## Tech stack

**ML**: XGBoost · PyTorch · SHAP · scikit-learn
**Agents**: Anthropic Claude (Sonnet 4.5 + Haiku 4.5) · direct SDK, no framework
**Retrieval**: ChromaDB · bge-base-en-v1.5 embeddings · PCA + k-NN OOD detector
**Frontend**: Streamlit · Plotly
**Auth**: Supabase · Google OAuth
**Backend**: FastAPI · Pydantic
**Container**: Docker
**Tracking**: MLflow

---

## Quick start

### Prerequisites

- Python 3.11
- An [Anthropic API key](https://console.anthropic.com)
- _(Windows only)_ Java + Hadoop binaries for PySpark (see `docs/SETUP.md` if needed)

### Install

```bash
git clone https://github.com/yourusername/fraudsentinel.git
cd fraudsentinel

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows PowerShell

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — minimum: ANTHROPIC_API_KEY (or skip and use BYOK in the UI)
```

### Run

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) and sign in with Google.

---

## Project structure

```
fraudsentinel/
├── src/
│   ├── agentic/               # Multi-agent pipelines
│   │   ├── orchestrator.py    # 4-agent investigation pipeline
│   │   ├── pattern_coach.py   # Indicator → checklist conversion
│   │   ├── shap_coach.py      # SHAP-grounded fallback checklist
│   │   ├── ood_detector.py    # PCA + k-NN OOD rejection
│   │   ├── tools.py           # Tool schemas for the Investigator
│   │   └── rule_generator/    # Planner + parallel workers + synthesizer
│   ├── dashboard/             # Streamlit UI
│   │   └── pages/             # Monitor, Investigate, Insights, Pattern Library, Settings
│   ├── ml_models/             # XGBoost training + SHAP attribution
│   ├── dl_models/             # LSTM autoencoder
│   ├── api/                   # FastAPI service
│   └── utils/                 # Config, logging
├── data/
│   ├── samples/               # Demo transactions (10K rows)
│   └── fraud_cases/           # 377-pattern archetype library
├── models/                    # Trained artifacts + ChromaDB
├── notebooks/                 # Training + evaluation notebooks
├── scripts/                   # Build + calibration utilities
├── docker/                    # Dockerfile + entrypoint
└── docs/                      # Architecture, setup, deployment
```

---

## Screenshots

<!-- ADD SCREENSHOTS HERE -->

### Multi-agent investigation

<!-- ![Investigate](docs/screenshots/investigate.png) -->

_Five-tab decision package: Summary · Score Attribution · Evidence · Patterns · Checklist_

### Two-tier pattern verification

<!-- ![Patterns](docs/screenshots/patterns.png) -->

_Semantic retrieval surfaces candidates; the Pattern agent verifies indicator fit and reports Strong / Partial / No fit_

### Rule generator

<!-- ![Rule Generator](docs/screenshots/rule_generator.png) -->

_Planner dispatches 4 parallel workers, synthesizer ranks rules with SQL + estimated catch/FPR_

### Insights dashboard

<!-- ![Insights](docs/screenshots/insights.png) -->

_Threshold tradeoff slider, hour-of-day fraud patterns, model confidence distribution_

---

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — local development setup
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Hugging Face Spaces deployment guide

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — 590,540 transactions provided by Vesta Corporation, with engineered fraud features across card aggregations (C1-C14), device fingerprints (D1-D15), and Vesta-derived velocity encodings (V1-V300+).

The pattern library was hand-curated to span 13 distinct fraud archetypes — from velocity attacks and card testing to engineered-feature anomalies and sophisticated device takeover patterns.

---

## License

MIT — see [LICENSE](LICENSE).

---
