# ============================================================================
# FraudSentinel — Hugging Face Spaces Dockerfile
# ============================================================================
# Target platform: Hugging Face Spaces (Docker SDK)
# Required port: 7860 (HF Spaces convention; auto-mapped to public HTTPS)
# Base: Python 3.11 slim — small, has libstdc++ for xgboost/torch
# Build time: ~10-15 min on HF servers (torch + xgboost + chromadb are heavy)
# ============================================================================

FROM python:3.11-slim

# ----------------------------------------------------------------------------
# System dependencies
# ----------------------------------------------------------------------------
# build-essential: needed for any pip wheels that compile from source
# git: needed if any pip install pulls from a git URL
# curl: useful for healthcheck and debugging
# libgomp1: required by xgboost (OpenMP runtime)
# ca-certificates: required for HTTPS calls to Anthropic / Supabase / etc.
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Set up a non-root user (HF Spaces requirement)
# HF Spaces runs containers as user 1000 by default, NOT root.
# Files we want the app to write to (cache, ChromaDB) must be writable by 1000.
# ----------------------------------------------------------------------------
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1

WORKDIR $HOME/app

# ----------------------------------------------------------------------------
# Python dependencies
# ----------------------------------------------------------------------------
# Copy requirements first to leverage Docker layer caching — if requirements
# don't change, this layer is cached and rebuilds are much faster
# ----------------------------------------------------------------------------
COPY --chown=user:user requirements-hf.txt ./
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements-hf.txt

# ----------------------------------------------------------------------------
# Application code
# ----------------------------------------------------------------------------
# Copy the rest of the app. .dockerignore excludes .venv, notebooks, mlruns,
# raw data, tests, etc. — see .dockerignore for the full list.
# ----------------------------------------------------------------------------
COPY --chown=user:user . ./

# ----------------------------------------------------------------------------
# Pre-warm: download embedding model so first user request isn't slow
# bge-base-en-v1.5 is ~440MB; downloading it at build time means cold-start
# users don't wait for the download.
# ----------------------------------------------------------------------------
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')" || echo "Pre-warm skipped (will download on first use)"

# ----------------------------------------------------------------------------
# Create writable dirs the app expects
# ----------------------------------------------------------------------------
RUN mkdir -p $HOME/app/data/cache && \
    mkdir -p $HOME/app/models/chroma_db

# ----------------------------------------------------------------------------
# Streamlit config — disable telemetry and CORS for embedded use
# ----------------------------------------------------------------------------
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    PYTHONPATH=/home/user/app

# ----------------------------------------------------------------------------
# Healthcheck — HF Spaces uses this to know when the app is ready
# ----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
EXPOSE 7860

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
     