"""Basic smoke tests. Run with: pytest tests/"""

import pytest


def test_imports():
    """Every module should import cleanly."""
    from src.utils import config, logging, feature_engineering
    from src.agentic import prompts, tools
    from src.api import schemas
    from src.dl_models import model
    assert config.PROJECT_ROOT.exists()


def test_feature_engineering_no_history():
    """Feature engineering should work with no history (defaults sensibly)."""
    from src.utils.feature_engineering import engineer_features

    txn = {
        "TransactionAmt": 100.0,
        "card1": 12345,
        "P_emaildomain": "gmail.com",
        "TransactionDT": 86400 * 30,  # 30 days in
    }
    result = engineer_features(txn, card_history=None)
    assert "txn_hour" in result
    assert "is_night" in result
    assert "card1_txn_count_1h" in result
    assert result["card1_txn_count_1h"] == 0
    assert result["emails_match"] == 0  # no R_emaildomain


def test_feature_engineering_high_risk_email():
    from src.utils.feature_engineering import engineer_features

    txn = {
        "TransactionAmt": 100.0,
        "card1": 12345,
        "P_emaildomain": "protonmail.com",
        "TransactionDT": 0,
    }
    result = engineer_features(txn, card_history=None)
    assert result["P_emaildomain_is_highrisk"] == 1


def test_pydantic_schemas():
    """API schemas should validate."""
    from src.api.schemas import TransactionInput

    t = TransactionInput(TransactionAmt=100.0, card1=123)
    assert t.TransactionAmt == 100.0
    assert t.card1 == 123


def test_chat_agent_requires_key():
    """ChatAgent should refuse to construct without an API key."""
    from src.agentic.chat import ChatAgent

    with pytest.raises(ValueError):
        ChatAgent(api_key="")


def test_orchestrator_requires_key():
    from src.agentic.orchestrator import FraudInvestigator

    with pytest.raises(ValueError):
        FraudInvestigator(api_key="")


def test_risk_band():
    from src.dashboard.components import risk_band

    assert risk_band(0.0)[0] == "LOW"
    assert risk_band(0.5)[0] == "MEDIUM"
    assert risk_band(0.7)[0] == "HIGH"
    assert risk_band(0.95)[0] == "CRITICAL"


def test_lstm_model_constructs():
    """LSTM autoencoder should construct without errors."""
    import torch
    from src.dl_models.model import LSTMAutoencoder

    model = LSTMAutoencoder(input_dim=16)
    x = torch.randn(2, 10, 16)  # batch=2, seq=10, features=16
    out = model(x)
    assert out.shape == (2, 10, 16)

    err = model.reconstruction_error(x)
    assert err.shape == (2,)
