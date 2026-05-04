"""Investigation cache — keyed by transaction_id + feature hash + model version.

Two-tier cache:
  1. Session-state (in-memory, per Streamlit session) — instant
  2. SQLite-backed (persistent across sessions) — ~5ms read

A cached investigation IS the previous output, byte-for-byte. Quality is
preserved exactly. The only failure mode is staleness:

  - If the model is retrained → model_version changes → old cache entries miss
  - If the transaction's features change → feature_hash changes → cache miss
  - If a different model is selected per stage → cache miss

Cache hits return immediately, so re-viewing the same transaction is instant
instead of paying ~45 seconds again.

Default location: data/cache/investigations.db
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

import streamlit as st

from src.utils.config import (
    DEFAULT_ANTHROPIC_MODEL,
    SAMPLES_DIR,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Cache version key — bump this when investigation OUTPUT format changes
# (e.g. new field added to the result dict). Bumping invalidates all
# existing cache entries safely.
# ----------------------------------------------------------------------
CACHE_FORMAT_VERSION = "v3"


# ----------------------------------------------------------------------
# Storage path
# ----------------------------------------------------------------------
CACHE_DB_PATH = SAMPLES_DIR.parent / "cache" / "investigations.db"


def _ensure_cache_dir() -> None:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS investigations (
            cache_key TEXT PRIMARY KEY,
            transaction_id TEXT,
            feature_hash TEXT,
            model_version TEXT,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_txn_id ON investigations(transaction_id)
    """)
    conn.commit()


# ----------------------------------------------------------------------
# Cache key generation
# ----------------------------------------------------------------------
# Features that, if changed, invalidate the cache. We do NOT hash every
# feature — just the ones that drive investigation outputs. This keeps the
# hash stable across cosmetic changes (e.g. new derived features added to
# the dataframe but not used by the agents).
# ----------------------------------------------------------------------
_CACHE_RELEVANT_FEATURES = (
    "TransactionAmt", "ProductCD", "card1",
    "card1_amt_zscore", "card1_txn_count_1h", "card1_txn_count_24h", "card1_txn_count_7d",
    "card1_distinct_addr1", "card1_distinct_products",
    "P_emaildomain", "P_emaildomain_is_highrisk", "emails_match",
    "is_night", "txn_hour",
    "C1", "C8", "C11", "C13",
    "D1", "D4", "D15",
    "V200",
)


def _feature_hash(transaction: dict) -> str:
    """Stable hash of the cache-relevant feature subset."""
    features = {}
    for k in _CACHE_RELEVANT_FEATURES:
        v = transaction.get(k)
        if v is not None:
            # Stringify floats with limited precision so 0.5 == 0.500000001
            if isinstance(v, float):
                features[k] = round(v, 4)
            else:
                features[k] = str(v) if not isinstance(v, (int, bool)) else v
    payload = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_version_string(triage_model: str, pattern_model: str,
                          investigator_model: str, report_model: str) -> str:
    """Combined model-version key — any model swap busts the cache."""
    return f"t={triage_model}|p={pattern_model}|i={investigator_model}|r={report_model}"


def _make_cache_key(
    transaction: dict,
    triage_model: str,
    pattern_model: str,
    investigator_model: str,
    report_model: str,
) -> str:
    txn_id = str(transaction.get("TransactionID", "unknown"))
    feature_hash = _feature_hash(transaction)
    model_ver = _model_version_string(
        triage_model, pattern_model, investigator_model, report_model
    )
    model_hash = hashlib.sha256(model_ver.encode()).hexdigest()[:8]
    return f"{CACHE_FORMAT_VERSION}:{txn_id}:{feature_hash}:{model_hash}"


# ----------------------------------------------------------------------
# Session-state layer (instant, per-session)
# ----------------------------------------------------------------------
def _session_get(cache_key: str) -> Optional[dict]:
    if "_investigation_cache_mem" not in st.session_state:
        st.session_state["_investigation_cache_mem"] = {}
    return st.session_state["_investigation_cache_mem"].get(cache_key)


def _session_set(cache_key: str, result: dict) -> None:
    if "_investigation_cache_mem" not in st.session_state:
        st.session_state["_investigation_cache_mem"] = {}
    st.session_state["_investigation_cache_mem"][cache_key] = result


# ----------------------------------------------------------------------
# SQLite layer (persistent, cross-session)
# ----------------------------------------------------------------------
def _db_get(cache_key: str) -> Optional[dict]:
    try:
        _ensure_cache_dir()
        with closing(sqlite3.connect(str(CACHE_DB_PATH))) as conn:
            _init_db(conn)
            cur = conn.execute(
                "SELECT result_json FROM investigations WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return json.loads(row[0])
    except Exception as e:
        logger.warning(f"Cache DB read failed: {e}")
        return None


def _db_set(cache_key: str, txn_id: str, feature_hash: str,
            model_ver: str, result: dict) -> None:
    try:
        _ensure_cache_dir()
        with closing(sqlite3.connect(str(CACHE_DB_PATH))) as conn:
            _init_db(conn)
            conn.execute("""
                INSERT OR REPLACE INTO investigations
                (cache_key, transaction_id, feature_hash, model_version, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"{CACHE_FORMAT_VERSION}:{txn_id}:{feature_hash}",
                txn_id, feature_hash, model_ver,
                json.dumps(result, default=str),
                time.time(),
            ))
            conn.commit()
    except Exception as e:
        logger.warning(f"Cache DB write failed: {e}")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def get_cached_investigation(
    transaction: dict,
    triage_model: str = DEFAULT_ANTHROPIC_MODEL,
    pattern_model: str = DEFAULT_ANTHROPIC_MODEL,
    investigator_model: str = DEFAULT_ANTHROPIC_MODEL,
    report_model: str = DEFAULT_ANTHROPIC_MODEL,
) -> Optional[dict]:
    """Look up a cached investigation. Returns None on miss.

    Two-tier lookup: session memory first (instant), then SQLite (~5ms).
    On SQLite hit, also populates session memory for subsequent reads.
    """
    cache_key = _make_cache_key(
        transaction, triage_model, pattern_model, investigator_model, report_model
    )

    # Tier 1: session memory
    mem_hit = _session_get(cache_key)
    if mem_hit is not None:
        logger.info(f"Investigation cache HIT (session) for {transaction.get('TransactionID')}")
        return mem_hit

    # Tier 2: SQLite
    db_hit = _db_get(cache_key)
    if db_hit is not None:
        logger.info(f"Investigation cache HIT (db) for {transaction.get('TransactionID')}")
        _session_set(cache_key, db_hit)  # promote to session for next read
        return db_hit

    return None


def store_investigation(
    transaction: dict,
    result: dict,
    triage_model: str = DEFAULT_ANTHROPIC_MODEL,
    pattern_model: str = DEFAULT_ANTHROPIC_MODEL,
    investigator_model: str = DEFAULT_ANTHROPIC_MODEL,
    report_model: str = DEFAULT_ANTHROPIC_MODEL,
) -> None:
    """Store a completed investigation in both cache tiers."""
    if result is None:
        return
    if result.get("error") and not result.get("report"):
        return  # don't cache failures

    cache_key = _make_cache_key(
        transaction, triage_model, pattern_model, investigator_model, report_model
    )

    _session_set(cache_key, result)

    txn_id = str(transaction.get("TransactionID", "unknown"))
    feature_hash = _feature_hash(transaction)
    model_ver = _model_version_string(
        triage_model, pattern_model, investigator_model, report_model
    )
    _db_set(cache_key, txn_id, feature_hash, model_ver, result)


def clear_session_cache() -> None:
    """Clear the in-session cache (does NOT clear SQLite)."""
    if "_investigation_cache_mem" in st.session_state:
        st.session_state["_investigation_cache_mem"] = {}


def cache_stats() -> dict:
    """Return basic cache statistics."""
    session_count = len(st.session_state.get("_investigation_cache_mem", {}))

    db_count = 0
    try:
        _ensure_cache_dir()
        with closing(sqlite3.connect(str(CACHE_DB_PATH))) as conn:
            _init_db(conn)
            cur = conn.execute("SELECT COUNT(*) FROM investigations")
            db_count = cur.fetchone()[0]
    except Exception:
        pass

    return {"session_entries": session_count, "db_entries": db_count}
