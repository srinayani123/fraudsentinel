# ============================================================================
# FraudSentinel — Hugging Face Spaces Dockerfile
# ============================================================================
# Target: Hugging Face Spaces (Docker SDK)
# Required port: 7860 (HF Spaces convention; auto-mapped to public HTTPS)
# Base: Python 3.11 slim
# Build time: ~10-15 min on HF servers
# ============================================================================

FROM python:3.11-slim

# ----------------------------------------------------------------------------
# System dependencies
# ----------------------------------------------------------------------------
# build-essential: needed for pip wheels that compile from source
# git: needed if any pip install pulls from a git URL
# curl: useful for healthcheck and debugging
# libgomp1: required by xgboost (OpenMP runtime)
# ca-certificates: required for HTTPS to Anthropic / Supabase / etc.
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install Python deps as ROOT into system site-packages.
# This avoids the --user / sys.path mismatch where the streamlit CLI shim is
# on PATH but the streamlit package isn't on Python's import path.
# ----------------------------------------------------------------------------
WORKDIR /app

COPY requirements-hf.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-hf.txt

# ----------------------------------------------------------------------------
# Now create the non-root user that HF Spaces requires (UID 1000).
# /app is owned by user so the app can read/write runtime files.
# ----------------------------------------------------------------------------
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# ----------------------------------------------------------------------------
# Copy application code (after pip install for layer caching).
# ----------------------------------------------------------------------------
COPY --chown=user:user . /app/

# ----------------------------------------------------------------------------
# Pre-warm: download embedding model so first user request isn't slow.
# bge-base-en-v1.5 is ~440MB — downloading at build time means cold-start
# users don't wait. Failure here is non-fatal (will download on first use).
# ----------------------------------------------------------------------------
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')" || echo "Pre-warm skipped (will download on first use)"

# ----------------------------------------------------------------------------
# Create writable runtime dirs.
# ----------------------------------------------------------------------------
RUN mkdir -p /app/data/cache && \
    mkdir -p /app/models/chroma_db

# ----------------------------------------------------------------------------
# Streamlit runtime config.
# ----------------------------------------------------------------------------
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# ----------------------------------------------------------------------------
# Healthcheck — HF uses this to detect when the app is ready.
# ----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# ----------------------------------------------------------------------------
# Run.
# Use `python -m streamlit` (not bare `streamlit`) to guarantee the right
# Python interpreter and import path.
# ----------------------------------------------------------------------------
EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "src/dashboard/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
     