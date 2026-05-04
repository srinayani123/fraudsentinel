# Architecture Deep-Dive

This document explains the design decisions behind FraudSentinel — the **why** behind each layer.

## Layered detection: defense in depth

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Multi-Agent Investigation (LLM, expensive, slow)   │  ← top 0.1% of transactions
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Ensemble (XGBoost 60% + LSTM 40%)                  │  ← every transaction
├─────────────────────────────────────────────────────────────┤
│ Layer 2: LSTM Autoencoder (unsupervised behavioral)         │  ← every transaction
├─────────────────────────────────────────────────────────────┤
│ Layer 1: XGBoost (supervised classifier)                    │  ← every transaction
├─────────────────────────────────────────────────────────────┤
│ Layer 0: PySpark feature engineering (offline batch)        │  ← every transaction
└─────────────────────────────────────────────────────────────┘
```

Each layer adds capability and cost. The agentic layer is gated behind the ML layers — only flagged transactions get the LLM treatment, which keeps inference cost bounded.

## Why these specific models?

### XGBoost for the supervised baseline

- **Tabular data:** XGBoost is still the SOTA for tabular fraud detection. Beats deep learning unless you have huge data + deep feature interactions.
- **Class imbalance handling:** `scale_pos_weight` is built in. No need to up-sample or use SMOTE.
- **Fast inference:** ~1ms per transaction. Production-friendly.
- **Interpretable:** Feature importances + SHAP work out of the box.

### LSTM Autoencoder for behavioral anomaly

- **Unsupervised:** Trains on legit-only data. Catches fraud patterns the labeled data didn't have.
- **Sequence-aware:** Models the user's *behavior over time*, not just point-in-time features. A $500 purchase by someone who normally spends $20 is a sequence anomaly even if the amount alone isn't.
- **Cold-start friendly:** Once you have a baseline of any user's history, you can detect deviations. Doesn't need fraud examples for that user.
- **Complementary to XGBoost:** XGBoost asks "does this look like known fraud?"; LSTM asks "does this look like this user?".

### Multi-agent GenAI for investigation

- **Triage Agent (cheap, fast):** Filters the haystack. Most flagged transactions are not worth deep investigation.
- **Investigator Agent (with tools):** Builds an evidence file. Tools are what make this an *agent* and not just a fancy prompt — it can `get_card_history`, `check_velocity`, etc., and decide what to look at next based on what it finds.
- **Pattern Agent (RAG):** Looks for analogues in the case library. Solves "is this like anything we've seen before?" without requiring an analyst to hand-label patterns.
- **Report Agent (composer):** Decouples investigation from communication. The Investigator produces dense facts; the Report Agent makes them readable.

This is also more **auditable** than a monolithic prompt. Each agent's output can be logged, re-run, or replaced independently.

## Why ChromaDB for the case library?

- **Local-first:** No external service needed. ChromaDB runs in-process with persistent storage on disk.
- **Cheap:** Free, embedded.
- **Good enough at scale:** For 10,000+ cases, performance is fine. For millions, you'd switch to Pinecone or Weaviate.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — small, fast, good for semantic similarity. ~80MB, runs on CPU.

For a production fraud system, you'd build the case library from real investigated cases (with privacy redaction). Here we synthesize representative narratives.

## Why MLflow?

- **Reproducibility:** Every model has a logged run with params, metrics, and artifacts. Can rebuild any prior model.
- **Comparison:** Side-by-side metric comparison across runs.
- **Model registry:** Versioned model tracking — "production" vs "staging" tags.
- **Standard in industry:** Most ML engineering job descriptions list MLflow or equivalent (Weights & Biases, DVC).

In production, you'd run a remote MLflow tracking server (Postgres backend, S3 artifact store) and have CI/CD pipelines log to it. For this project, local MLflow is sufficient.

## Why FastAPI + Docker if the demo is Streamlit?

Two different stories:

- **Streamlit dashboard** — for analysts and demos. Interactive. Conversational. Visual.
- **FastAPI service** — for *integration*. Other services in the org want a REST endpoint they can POST a transaction to and get a score back.

In a real deployment at Visa, the FastAPI service is what processes streaming transactions. The Streamlit dashboard is what the fraud ops team uses to investigate flagged ones.

Docker bundles both for portability.

## Why ship pre-trained models in the repo?

The alternative is to have users run the full pipeline (Spark, MLflow, training) before they can see the demo. That's fine for a code review but bad for a portfolio link you send to a recruiter.

Pre-shipped artifacts (~30MB of XGBoost + ~5MB LSTM + ~80MB sentence-transformer) make the live demo work zero-config.

## Why BYOK?

Three problems with not-BYOK on a public demo:

1. **Cost:** A few thousand viewers each running 4 LLM calls = real money.
2. **Abuse:** Open keys get scraped and abused.
3. **Rate limits:** A shared key throttles the whole demo.

BYOK solves all three. Users provide their own key, hold their own rate limits, pay their own cost. The key lives in `st.session_state` only — never persisted, never logged.

## What would change at Visa scale?

| Component | Demo | Production |
|---|---|---|
| Feature pipeline | Spark batch on laptop | Spark Streaming or Flink on a cluster |
| Feature store | Parquet files | Feast or a dedicated feature store |
| XGBoost serving | Loaded in-process | Triton Inference Server or NVIDIA Morpheus |
| LSTM serving | PyTorch in-process | TorchServe or ONNX Runtime |
| MLflow tracking | Local SQLite | Remote tracking server, Postgres + S3 |
| Agentic layer | Anthropic API direct | Through internal LLM gateway with rate limits and budget guardrails |
| ChromaDB | Embedded | Pinecone / Weaviate / managed vector DB |
| FastAPI | Single container | Auto-scaling fleet behind a load balancer |
| Monitoring | Streamlit Analytics page | Datadog / Grafana + PagerDuty |

The architecture stays the same. The implementations swap to managed/scaled equivalents.

## What's intentionally out of scope

- **Real-time streaming** — feature engineering is offline batch. Adding Kafka + Spark Streaming is straightforward but a lot of code for a portfolio piece.
- **Online learning** — XGBoost is retrained periodically, not continuously. Adding online learning would add complexity without proportional ML insight.
- **GraphQL / gRPC** — REST-only API.
- **Authentication / authorization** — would need to add for a real product. The BYOK pattern partially covers it for the demo.
- **Detailed cost tracking** — the agentic layer should track $/transaction in production. Skipped here.

## What's most interview-worthy

- The **multi-agent + RAG investigation flow** — concrete demonstration of agentic AI applied to fraud, which is exactly the Visa PFI ask.
- The **layered design** with cost-gated escalation — production-aware.
- **End-to-end MLOps:** Spark → MLflow → trained artifacts → FastAPI → Docker → live demo.
- **BYOK** — shows you've thought about how AI features actually get deployed publicly.
