"""Tools available to the agents.

Each tool is a pure Python function. Agents invoke them via the Anthropic
tool-use API (or, when running standalone, by direct function call).

The semantic search tool (search_fraud_cases) includes a two-stage gate
for user-typed queries:
  1. Hybrid PCA + k-NN OOD detection (fast, deterministic) — rejects clearly
     out-of-domain queries when both signals agree
  2. LLM judge fallback — invoked only on borderline cases (cheap fallback)

Internal callers (e.g. transaction-built queries from Investigate) can pass
skip_ood=True to bypass the gate, since those queries are fraud-by-construction.

Pattern results carry both `summary` (plain-English description of WHAT the
pattern looks like) and `reasoning` (WHY it indicates fraud), plus the paired
`indicators` (technical thresholds) and `indicator_explanations` (plain-English).
ChromaDB stores list-typed metadata as JSON strings (only scalars are allowed),
so this module deserializes them back into lists when returning results to
callers — without this, downstream code (e.g. pattern_coach) iterates the JSON
string character-by-character, producing nonsense output.

References:
  - Triantafyllopoulos et al. (2026), "Knowing When Not to Answer: Lightweight
    KB-Aligned OOD Detection for Safe RAG" (arXiv:2508.02296)
  - Sun et al. (2022), "Out-of-distribution Detection with Deep Nearest Neighbors"
    (ICML 2022)
"""

from __future__ import annotations

import json as _json
from functools import lru_cache
from pathlib import Path as _Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions

from src.agentic.ood_detector import PCAOODDetector, llm_judge_is_fraud_query
from src.utils.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    SAMPLES_DIR,
)

# Must match build_knowledge_base.py — embeddings from different models live in
# different vector spaces and cannot be compared.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


# ----------------------------------------------------------------------
# Similarity calibration
#
# Display ceiling override: calibrated SIMILARITY_HIGH from MS MARCO is
# typically ~0.72, which means anything above that flattens to 100% on screen.
# This makes a top-5 result list show "100% / 100% / 100% / 100% / 100%" with
# no visible ranking difference. We override the display ceiling to 0.85 so
# the real cosine spread (e.g. 0.78 / 0.75 / 0.73 / 0.72 / 0.70) shows as
# distinct percentages (100% / 91% / 82% / 76% / 73%).
#
# This is a DISPLAY ceiling — the OOD thresholds and quality cutoff still
# come from the calibrated file. Truth is preserved; only the rescale is
# spread out for readability.
# ----------------------------------------------------------------------
_CALIBRATION_FILE = _Path("models/similarity_calibration.json")
_DISPLAY_CEILING_OVERRIDE = 0.85

if _CALIBRATION_FILE.exists():
    try:
        with open(_CALIBRATION_FILE) as _f:
            _cal = _json.load(_f)
        SIMILARITY_LOW = _cal["calibrated_low"]
        SIMILARITY_HIGH = max(_cal["calibrated_high"], _DISPLAY_CEILING_OVERRIDE)
        MIN_THRESHOLD = _cal["calibrated_threshold"]
    except Exception:
        SIMILARITY_LOW = 0.55
        SIMILARITY_HIGH = _DISPLAY_CEILING_OVERRIDE
        MIN_THRESHOLD = 0.50
else:
    SIMILARITY_LOW = 0.55
    SIMILARITY_HIGH = _DISPLAY_CEILING_OVERRIDE
    MIN_THRESHOLD = 0.50


def _rescale_similarity(raw_cosine: float) -> float:
    if raw_cosine is None:
        return 0.0
    span = SIMILARITY_HIGH - SIMILARITY_LOW
    if span <= 0:
        return 0.5
    rescaled = (raw_cosine - SIMILARITY_LOW) / span
    return max(0.0, min(1.0, rescaled))


def similarity_band(rescaled: float) -> str:
    if rescaled >= 0.75:
        return "Strongest match"
    elif rescaled >= 0.50:
        return "Strong match"
    elif rescaled >= 0.25:
        return "Moderate match"
    else:
        return "Weak match"


def _parse_list_metadata(raw_value: Any) -> list:
    """Parse a list-typed metadata field that ChromaDB stored as a JSON string.

    Returns a list. Handles four cases robustly:
      - already a list → return it
      - JSON-stringified list → parse and return
      - empty/None → return []
      - garbage (unparseable) → return [] (don't crash)
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return []
        try:
            parsed = _json.loads(raw_value)
            if isinstance(parsed, list):
                return parsed
            return []
        except (_json.JSONDecodeError, ValueError):
            return [raw_value]
    return []


# -------------------- Data access --------------------
@lru_cache(maxsize=1)
def _load_samples() -> pd.DataFrame:
    """Load the demo transaction sample once."""
    path = SAMPLES_DIR / "demo_transactions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _get_chroma_collection():
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedder)


@lru_cache(maxsize=1)
def _get_ood_detector():
    return PCAOODDetector.load()


@lru_cache(maxsize=1)
def _get_query_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


# -------------------- Tool implementations --------------------
def get_card_history(card_id: int, limit: int = 20) -> dict[str, Any]:
    df = _load_samples()
    if df.empty:
        return {"error": "Sample data not loaded", "card_id": card_id}

    history = df[df["card1"] == card_id].sort_values("TransactionDT").tail(limit)
    if len(history) == 0:
        return {"card_id": card_id, "transaction_count": 0, "message": "No history available"}

    summary = {
        "card_id": int(card_id),
        "transaction_count": int(len(history)),
        "amount_stats": {
            "mean": float(history["TransactionAmt"].mean()),
            "std": float(history["TransactionAmt"].std()) if len(history) > 1 else 0.0,
            "min": float(history["TransactionAmt"].min()),
            "max": float(history["TransactionAmt"].max()),
        },
        "fraud_count_in_history": int(history["isFraud"].sum())
        if "isFraud" in history.columns
        else 0,
        "recent_transactions": [
            {
                "amount": float(row["TransactionAmt"]),
                "product": str(row.get("ProductCD", "?")),
                "hour": int(row.get("txn_hour", -1)),
                "is_night": bool(row.get("is_night", 0)),
                "amt_zscore": float(row.get("card1_amt_zscore", 0.0)),
            }
            for _, row in history.tail(5).iterrows()
        ],
    }
    return summary


def get_merchant_profile(product_cd: str) -> dict[str, Any]:
    df = _load_samples()
    if df.empty or "ProductCD" not in df.columns:
        return {"error": "Sample data not loaded"}

    subset = df[df["ProductCD"] == product_cd]
    if len(subset) == 0:
        return {"product_cd": product_cd, "transaction_count": 0}

    fraud_rate = (
        float(subset["isFraud"].mean()) if "isFraud" in subset.columns else None
    )
    return {
        "product_cd": product_cd,
        "transaction_count": int(len(subset)),
        "fraud_rate": fraud_rate,
        "amount_stats": {
            "mean": float(subset["TransactionAmt"].mean()),
            "median": float(subset["TransactionAmt"].median()),
            "p95": float(subset["TransactionAmt"].quantile(0.95)),
        },
    }


def check_velocity(card_id: int) -> dict[str, Any]:
    df = _load_samples()
    if df.empty:
        return {"error": "Sample data not loaded"}

    history = df[df["card1"] == card_id]
    if len(history) == 0:
        return {"card_id": card_id, "message": "No data"}

    latest = history.sort_values("TransactionDT").iloc[-1]
    return {
        "card_id": int(card_id),
        "txn_count_last_1h": int(latest.get("card1_txn_count_1h", 0)),
        "txn_count_last_24h": int(latest.get("card1_txn_count_24h", 0)),
        "txn_count_last_7d": int(latest.get("card1_txn_count_7d", 0)),
        "amt_sum_last_24h": float(latest.get("card1_amt_sum_24h", 0.0)),
        "seconds_since_last_txn": int(latest.get("card1_seconds_since_last", -1)),
    }


def compare_to_user_pattern(card_id: int, current_amount: float) -> dict[str, Any]:
    df = _load_samples()
    if df.empty:
        return {"error": "Sample data not loaded"}

    history = df[df["card1"] == card_id]
    if len(history) < 2:
        return {"card_id": card_id, "message": "Insufficient history"}

    mean = float(history["TransactionAmt"].mean())
    std = float(history["TransactionAmt"].std())
    zscore = (current_amount - mean) / max(std, 1e-6)
    return {
        "card_id": int(card_id),
        "current_amount": current_amount,
        "historical_mean": mean,
        "historical_std": std,
        "z_score": zscore,
        "verdict": (
            "extreme_outlier" if abs(zscore) > 4
            else "outlier" if abs(zscore) > 2
            else "in_pattern"
        ),
    }


def search_fraud_cases(
    query: str,
    top_k: int = 3,
    skip_ood: bool = False,
) -> list[dict[str, Any]]:
    """Vector search the fraud knowledge base for similar patterns.

    Returns each match as a dict with:
      - id, title, pattern, snippet, reasoning, similarity, similarity_raw, match_band
      - indicators: list[str]              ← parsed from JSON-stringified metadata
      - indicator_explanations: list[str]  ← parsed from JSON-stringified metadata

    Two-stage gate before retrieval (skipped when skip_ood=True):
      Stage 1: Hybrid PCA + k-NN OOD check
      Stage 2: LLM judge for borderline cases
    """
    # ---- Stage 1: Hybrid PCA + k-NN OOD check ----
    if not skip_ood:
        detector = _get_ood_detector()
        if detector is not None:
            try:
                embedder = _get_query_embedder()
                query_emb = embedder.encode([query], normalize_embeddings=True)[0]
                ood_result = detector.classify(np.array(query_emb))
                verdict = ood_result["verdict"]

                if verdict == "out_of_domain":
                    return [{
                        "ood_rejected": True,
                        "reason": "Query is outside the fraud-domain subspace and far from all known patterns.",
                        "ood_method": "hybrid_pca_knn",
                        "relative_score": ood_result["relative_score"],
                        "pca_error": ood_result.get("pca_error"),
                        "knn_distance": ood_result.get("knn_distance"),
                    }]

                if verdict == "borderline":
                    judge = llm_judge_is_fraud_query(query)
                    if not judge.get("is_fraud_related", True):
                        return [{
                            "ood_rejected": True,
                            "reason": judge.get("reasoning", "LLM judge marked query as off-topic"),
                            "ood_method": "llm_judge",
                            "relative_score": ood_result["relative_score"],
                            "pca_error": ood_result.get("pca_error"),
                            "knn_distance": ood_result.get("knn_distance"),
                        }]
            except Exception:
                pass

    # ---- Stage 3: actual retrieval ----
    try:
        collection = _get_chroma_collection()
    except Exception as e:
        return [{"error": f"ChromaDB not available: {e}"}]

    try:
        results = collection.query(query_texts=[query], n_results=max(top_k * 2, 6))
    except Exception as e:
        return [{"error": f"ChromaDB query failed: {e}"}]

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    out = []
    for i in range(len(results["ids"][0])):
        raw_cosine = (
            1.0 - results["distances"][0][i] if results.get("distances") else None
        )

        if raw_cosine is not None and raw_cosine < MIN_THRESHOLD:
            continue

        rescaled = _rescale_similarity(raw_cosine) if raw_cosine is not None else None
        band = similarity_band(rescaled) if rescaled is not None else None

        md = results["metadatas"][0][i]
        summary = md.get("summary", "") or results["documents"][0][i][:280]

        # Parse list-typed metadata fields back from JSON strings.
        # ChromaDB only allows scalar metadata, so build_knowledge_base.py
        # stored these as json.dumps(...). We reverse that here so callers
        # receive real lists, not JSON-stringified blobs.
        indicators_list = _parse_list_metadata(md.get("indicators"))
        explanations_list = _parse_list_metadata(md.get("indicator_explanations"))

        out.append({
            "id": results["ids"][0][i],
            "title": md.get("title", ""),
            "pattern": md.get("pattern", ""),
            "indicators": indicators_list,
            "indicator_explanations": explanations_list,
            "snippet": summary,
            "reasoning": md.get("reasoning", ""),
            "similarity": rescaled,
            "similarity_raw": raw_cosine,
            "match_band": band,
        })

        if len(out) >= top_k:
            break

    return out


# -------------------- Tool schema (for Anthropic API) --------------------
TOOL_SCHEMAS = [
    {
        "name": "get_card_history",
        "description": "Fetch recent transaction history for a specific card.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {"type": "integer"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "get_merchant_profile",
        "description": "Get the risk profile and stats for a merchant/product code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_cd": {"type": "string"},
            },
            "required": ["product_cd"],
        },
    },
    {
        "name": "check_velocity",
        "description": "Check transaction velocity for a card.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {"type": "integer"},
            },
            "required": ["card_id"],
        },
    },
    {
        "name": "compare_to_user_pattern",
        "description": "Compare an amount to the card's historical pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_id": {"type": "integer"},
                "current_amount": {"type": "number"},
            },
            "required": ["card_id", "current_amount"],
        },
    },
    {
        "name": "search_fraud_cases",
        "description": "Search the historical fraud case knowledge base via semantic similarity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
                "skip_ood": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip OOD gate (for internal calls)",
                },
            },
            "required": ["query"],
        },
    },
]


TOOL_DISPATCH = {
    "get_card_history": get_card_history,
    "get_merchant_profile": get_merchant_profile,
    "check_velocity": check_velocity,
    "compare_to_user_pattern": compare_to_user_pattern,
    "search_fraud_cases": search_fraud_cases,
}


def execute_tool(name: str, params: dict) -> Any:
    if name not in TOOL_DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_DISPATCH[name](**params)
    except Exception as e:
        return {"error": f"Tool {name} failed: {type(e).__name__}: {e}"}
    