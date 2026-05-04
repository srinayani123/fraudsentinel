"""Pydantic schemas for the FastAPI scoring service."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """A transaction to be scored. All fields optional except amount + card."""

    TransactionAmt: float = Field(..., description="Transaction amount")
    card1: int = Field(..., description="Card identifier (BIN-like)")
    ProductCD: Optional[str] = Field(None, description="Product code")
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    txn_hour: Optional[int] = None
    is_night: Optional[int] = None
    card1_txn_count_1h: Optional[int] = 0
    card1_txn_count_24h: Optional[int] = 0
    card1_amt_zscore: Optional[float] = 0.0
    extra_features: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Any additional features to merge in",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 245.99,
                "card1": 13926,
                "ProductCD": "W",
                "P_emaildomain": "gmail.com",
                "txn_hour": 14,
                "is_night": 0,
                "card1_txn_count_24h": 3,
                "card1_amt_zscore": 1.2,
            }
        }
    }


class ScoreResponse(BaseModel):
    xgboost_score: float = Field(..., description="XGBoost fraud probability [0,1]")
    lstm_anomaly_score: Optional[float] = Field(None, description="Normalized LSTM anomaly [0,1]")
    lstm_is_anomaly: Optional[bool] = None
    combined_risk: float = Field(..., description="Weighted ensemble risk [0,1]")
    risk_band: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    decision: str = Field(..., description="APPROVE / REVIEW / BLOCK")


class InvestigateRequest(BaseModel):
    """Request body for /investigate. Requires user-provided API key (BYOK)."""

    transaction: TransactionInput
    anthropic_api_key: str = Field(..., description="User's Anthropic API key (BYOK)")


class InvestigateResponse(BaseModel):
    triage: dict
    investigator_findings: Optional[str] = None
    pattern_matches: list[dict] = []
    pattern_analysis: Optional[str] = None
    final_report: Optional[str] = None
    tool_calls: list[dict] = []
