"""Shared dataclasses for the Rule Generator multi-agent system.

All types are JSON-serializable so we can stream stages over Streamlit's
session_state and cache results between runs (mirrors the InvestigationContext
pattern in src.agentic.orchestrator).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
@dataclass
class AggregateInput:
    """Pre-computed aggregate stats from the analyst's selected transaction set.

    Workers consume this — NOT individual transactions. Sending raw rows to
    Claude per-worker would be too expensive and noisy. The aggregates capture
    the distributional patterns Workers need to spot.
    """
    n_transactions: int
    n_fraud: int
    fraud_rate: float
    date_range: tuple[str, str]  # (min_date_iso, max_date_iso) or ("unknown", "unknown")
    risk_band: str  # "all" | "high" | "critical" | "high_critical"

    # Velocity-related stats
    velocity_1h_p50: float
    velocity_1h_p95: float
    velocity_24h_p50: float
    velocity_24h_p95: float
    velocity_7d_p50: float
    velocity_7d_p95: float
    velocity_24h_fraud_mean: float
    velocity_24h_legit_mean: float

    # Email-related stats
    email_high_risk_fraud_rate: float
    email_low_risk_fraud_rate: float
    email_mismatch_fraud_rate: float
    email_null_fraud_rate: float
    top_fraud_email_domains: list[dict[str, Any]]  # [{"domain": str, "fraud_rate": float, "n": int}]

    # Device/identity stats (using D1/D4/D15 — the IEEE-CIS device fingerprint fields)
    fraud_card_distinct_addr1_p95: float
    legit_card_distinct_addr1_p95: float
    fraud_card_distinct_products_p95: float
    legit_card_distinct_products_p95: float
    d1_zero_fraud_rate: float  # Fraud rate when D1 == 0 (first-time device)
    d1_nonzero_fraud_rate: float

    # Amount stats
    amount_p50: float
    amount_p95: float
    amount_p99: float
    amount_zscore_fraud_p95: float
    amount_zscore_legit_p95: float
    amount_sum_24h_fraud_p95: float
    amount_sum_24h_legit_p95: float

    # Channel breakdown
    products_with_high_fraud: list[dict[str, Any]]  # [{"product": str, "fraud_rate": float, "n": int}]

    # Time-of-day breakdown
    night_fraud_rate: float
    day_fraud_rate: float
    hour_with_highest_fraud: int  # 0-23
    hour_with_highest_fraud_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------
@dataclass
class RuleProposal:
    """A single proposed fraud rule with thresholds and estimated impact."""
    rule_name: str  # short, kebab-case identifier (e.g. "high-velocity-night")
    plain_english: str  # 1-2 sentence analyst-readable description
    rule_code_sql: str  # SQL WHERE clause ready for production
    rule_code_pseudo: str  # pseudo-code version for non-SQL rules engines
    feature_family: str  # "velocity" | "email" | "device" | "amount" | "composite"
    severity: str  # "block" | "review" | "monitor"
    estimated_catch_rate: str  # e.g. "~15% of fraud in this set"
    estimated_false_positive_rate: str  # e.g. "<2% of legitimate"
    rationale: str  # why this rule, what it catches
    evidence: list[str] = field(default_factory=list)  # specific numbers backing the rule

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerOutput:
    """Output from a single Worker agent."""
    worker_name: str  # "velocity" | "email" | "device" | "amount"
    proposed_rules: list[RuleProposal]
    summary: str  # 1-2 sentences: "what I found in this dimension"
    key_finding: str  # most important single insight from this worker
    runtime_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_name": self.worker_name,
            "proposed_rules": [r.to_dict() for r in self.proposed_rules],
            "summary": self.summary,
            "key_finding": self.key_finding,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
        }


@dataclass
class PlannerOutput:
    """Output from the Planner — which workers should run, with focused brief."""
    workers_to_run: list[str]  # subset of ["velocity", "email", "device", "amount"]
    velocity_brief: str  # focused instruction for velocity worker
    email_brief: str
    device_brief: str
    amount_brief: str
    overall_strategy: str  # high-level approach for this fraud set
    runtime_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesisOutput:
    """Final output from the Synthesizer — ranked rules across all workers."""
    ranked_rules: list[RuleProposal]
    coverage_summary: str  # "These N rules together catch ~X% of fraud"
    deployment_recommendation: str  # which to ship first, which to monitor
    runtime_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_rules": [r.to_dict() for r in self.ranked_rules],
            "coverage_summary": self.coverage_summary,
            "deployment_recommendation": self.deployment_recommendation,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
        }


@dataclass
class GenerationResult:
    """Full output of one Rule Generator run — what gets cached in session state."""
    aggregates: AggregateInput
    planner_output: PlannerOutput
    worker_outputs: dict[str, WorkerOutput]  # keyed by worker_name
    synthesis: SynthesisOutput
    total_runtime_s: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregates": self.aggregates.to_dict(),
            "planner_output": asdict(self.planner_output),
            "worker_outputs": {k: v.to_dict() for k, v in self.worker_outputs.items()},
            "synthesis": self.synthesis.to_dict(),
            "total_runtime_s": self.total_runtime_s,
            "error": self.error,
        }
    