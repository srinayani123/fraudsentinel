"""Lightweight agent run logger. Records timing of each agent stage, pattern matches,
and pattern-discovery runs to parquet files so Insights can show real metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "agent_runs.parquet"
PATTERN_LOG_FILE = LOG_DIR / "pattern_matches.parquet"
DISCOVERY_LOG_FILE = LOG_DIR / "pattern_discovery_runs.parquet"


def _ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Agent stage timing
# ============================================================
def log_agent_run(
    txn_id: str,
    stage: str,
    duration_ms: float,
    status: str = "success",
    model: Optional[str] = None,
):
    """Append a single agent-stage timing record."""
    _ensure_dir()
    row = pd.DataFrame([{
        "ts": pd.Timestamp.utcnow(),
        "txn_id": str(txn_id),
        "stage": stage,
        "duration_ms": float(duration_ms),
        "status": status,
        "model": model or "",
    }])
    if LOG_FILE.exists():
        existing = pd.read_parquet(LOG_FILE)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_parquet(LOG_FILE, index=False)


@contextmanager
def time_stage(txn_id: str, stage: str, model: Optional[str] = None):
    """Context manager: `with time_stage('T123', 'triage'): do_work()`"""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        try:
            log_agent_run(txn_id, stage, duration_ms, status=status, model=model)
        except Exception:
            pass  # never let logging crash the actual flow


def load_recent_runs(limit_hours: int = 24 * 7) -> pd.DataFrame:
    """Load recent agent runs."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=["ts", "txn_id", "stage", "duration_ms", "status", "model"])
    df = pd.read_parquet(LOG_FILE)
    if "ts" in df.columns and len(df) > 0:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=limit_hours)
        df = df[df["ts"] >= cutoff]
    return df


# ============================================================
# Pattern match logging (powers Insights "Top patterns" chart)
# ============================================================
def log_pattern_match(txn_id: str, pattern: str, case_id: str = "", similarity: float = 0.0):
    """Log which pattern was matched for a given investigation."""
    _ensure_dir()
    row = pd.DataFrame([{
        "ts": pd.Timestamp.utcnow(),
        "txn_id": str(txn_id),
        "pattern": str(pattern),
        "case_id": str(case_id),
        "similarity": float(similarity),
    }])
    if PATTERN_LOG_FILE.exists():
        existing = pd.read_parquet(PATTERN_LOG_FILE)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_parquet(PATTERN_LOG_FILE, index=False)


def load_pattern_matches(limit_hours: int = 24 * 7) -> pd.DataFrame:
    """Load recent pattern match log."""
    if not PATTERN_LOG_FILE.exists():
        return pd.DataFrame(columns=["ts", "txn_id", "pattern", "case_id", "similarity"])
    df = pd.read_parquet(PATTERN_LOG_FILE)
    if "ts" in df.columns and len(df) > 0:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=limit_hours)
        df = df[df["ts"] >= cutoff]
    return df


# ============================================================
# Pattern Discovery logging (powers Insights "Discovery activity" view)
# ============================================================
def log_discovery_run(transactions_analyzed: int, clusters_found: int, duration_ms: float):
    """Log a pattern-discovery run.

    Each row represents one click of 'Analyze recent activity' on the Monitor page.
    """
    _ensure_dir()
    row = pd.DataFrame([{
        "ts": pd.Timestamp.utcnow(),
        "transactions_analyzed": int(transactions_analyzed),
        "clusters_found": int(clusters_found),
        "duration_ms": float(duration_ms),
    }])
    if DISCOVERY_LOG_FILE.exists():
        existing = pd.read_parquet(DISCOVERY_LOG_FILE)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_parquet(DISCOVERY_LOG_FILE, index=False)


def load_discovery_runs(limit_hours: int = 24 * 7) -> pd.DataFrame:
    """Load recent pattern-discovery runs."""
    if not DISCOVERY_LOG_FILE.exists():
        return pd.DataFrame(columns=["ts", "transactions_analyzed", "clusters_found", "duration_ms"])
    df = pd.read_parquet(DISCOVERY_LOG_FILE)
    if "ts" in df.columns and len(df) > 0:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=limit_hours)
        df = df[df["ts"] >= cutoff]
    return df
