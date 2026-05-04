"""
FastAPI service for FraudSentinel.

Endpoints:
- GET  /health                Liveness check
- POST /score                 ML scoring only (no LLM, no key needed)
- POST /investigate           Full agentic investigation (BYOK)
- GET  /                      Redirects to /docs

Run:
    uvicorn src.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.agentic.orchestrator import FraudInvestigator
from src.api.schemas import (
    InvestigateRequest,
    InvestigateResponse,
    ScoreResponse,
    TransactionInput,
)
from src.ml_models.inference import score_dataframe
from src.utils.config import RISK_THRESHOLDS
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to load LSTM at startup (optional)
_lstm_available = False
try:
    from src.dl_models.inference import load_model as load_lstm

    load_lstm()
    _lstm_available = True
    logger.info("LSTM model loaded.")
except Exception as e:
    logger.warning(f"LSTM model not available: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up XGBoost model
    try:
        from src.ml_models.inference import load_model

        load_model()
        logger.info("XGBoost model loaded.")
    except Exception as e:
        logger.error(f"Failed to load XGBoost: {e}")
    yield


app = FastAPI(
    title="FraudSentinel API",
    description="Multi-layer fraud detection with optional agentic investigation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def classify_risk(score: float) -> tuple[str, str]:
    if score >= RISK_THRESHOLDS["high"]:
        return "CRITICAL", "BLOCK"
    if score >= RISK_THRESHOLDS["medium"]:
        return "HIGH", "REVIEW"
    if score >= RISK_THRESHOLDS["low"]:
        return "MEDIUM", "REVIEW"
    return "LOW", "APPROVE"


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "lstm_available": _lstm_available}


@app.post("/score", response_model=ScoreResponse)
def score(transaction: TransactionInput) -> ScoreResponse:
    """ML-only scoring. Fast path, no LLM."""
    try:
        # Build a one-row DataFrame from the input + extras
        row = transaction.model_dump()
        extras = row.pop("extra_features", {}) or {}
        row.update(extras)
        df = pd.DataFrame([row])

        xgb_score = float(score_dataframe(df)[0])

        lstm_score = None
        lstm_anomaly = None
        if _lstm_available:
            # For single-transaction scoring without history, we use a degenerate
            # sequence (just this row). In production, the API would also accept
            # a sequence parameter or look up recent history from a DB.
            try:
                from src.dl_models.inference import score_sequence

                lstm_result = score_sequence(df)
                lstm_score = lstm_result["normalized_score"]
                lstm_anomaly = lstm_result["is_anomaly"]
            except Exception as e:
                logger.warning(f"LSTM scoring failed: {e}")

        # Ensemble
        if lstm_score is not None:
            combined = 0.6 * xgb_score + 0.4 * lstm_score
        else:
            combined = xgb_score

        band, decision = classify_risk(combined)
        return ScoreResponse(
            xgboost_score=xgb_score,
            lstm_anomaly_score=lstm_score,
            lstm_is_anomaly=lstm_anomaly,
            combined_risk=combined,
            risk_band=band,
            decision=decision,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not loaded: {e}")
    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(500, f"Scoring failed: {e}")


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest) -> InvestigateResponse:
    """Full agentic investigation. Requires user-provided Anthropic key."""
    # Score first
    try:
        score_resp = score(req.transaction)
    except HTTPException:
        raise

    # Then investigate
    try:
        investigator = FraudInvestigator(api_key=req.anthropic_api_key)
        ctx = investigator.investigate(
            transaction=req.transaction.model_dump(),
            xgb_score=score_resp.xgboost_score,
            lstm_score=score_resp.lstm_anomaly_score or 0.5,
            lstm_anomaly=score_resp.lstm_is_anomaly or False,
        )
        return InvestigateResponse(
            triage=ctx.triage_result or {},
            investigator_findings=ctx.investigator_findings,
            pattern_matches=ctx.pattern_matches,
            pattern_analysis=ctx.pattern_analysis,
            final_report=ctx.final_report,
            tool_calls=ctx.investigator_tool_calls,
        )
    except Exception as e:
        logger.exception("Investigation failed")
        raise HTTPException(500, f"Investigation failed: {e}")
