"""Multi-agent fraud investigator.

Workflow:
    Triage              [HAIKU — fast routing decision, JSON output]
      -> Score Attribution (DETERMINISTIC: SHAP for XGBoost + per-timestep MSE for LSTM)
      -> Investigator   [SONNET — multi-turn tool use + analyst prose, STREAMED]
      -> Pattern        [HAIKU — short structured fit verdict, 80-120 words]
      -> Report         [SONNET — final analyst synthesis, STREAMED]

Model selection per stage:
  - Triage and Pattern use Haiku 4.5 — these are bounded structured-output tasks
    where Haiku matches Sonnet quality at much lower latency
  - Investigator uses Sonnet — multi-turn tool use + reasoning needs the
    larger model
  - Report uses Sonnet — analyst-facing prose where word choice matters

Score Attribution is deterministic (no LLM). It runs:
  1. SHAP on the XGBoost model -> per-feature contributions
  2. Per-timestep reconstruction error on the LSTM autoencoder -> WHEN in the
     recent sequence the anomaly occurred

The Investigator agent receives both as evidence so its prose cites real
attributions instead of speculating.

Streaming: Investigator and Report use stream=True. Wall-clock unchanged but
perceived latency cut in half — analysts see tokens appearing live.

Pattern stage builds its retrieval query from the SHAP attribution (which fraud
SURFACE is dominant — device/identity, card-counter, engineered, behavioral)
rather than just behavioral features. This ensures device_takeover,
credential_compromise, and engineered_anomaly archetypes get retrieved when the
model flags them, instead of always defaulting to velocity_attack patterns.

Pattern agent passes skip_ood=True when calling search_fraud_cases because
its query is built from an already-flagged transaction (fraud-by-construction).

The page (2_Investigate.py) is responsible for computing the live LSTM score
and passing it as lstm_score / lstm_anomaly — no hardcoded placeholders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Generator

import anthropic
import pandas as pd

from src.agentic.prompts import (
    INVESTIGATOR_PROMPT,
    PATTERN_PROMPT,
    REPORT_PROMPT,
    TRIAGE_PROMPT,
)
from src.agentic.tools import TOOL_SCHEMAS, execute_tool
from src.utils.config import (
    INVESTIGATOR_MODEL,
    PATTERN_MODEL,
    REPORT_MODEL,
    TRIAGE_MODEL,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Map SHAP family names to plain-English fraud surface descriptions used in
# retrieval query construction. These descriptions match the language of the
# new archetype patterns (device_takeover, credential_compromise,
# engineered_anomaly) so embedding similarity correctly surfaces them.
FAMILY_TO_SURFACE_LABEL = {
    "device_identity": "device fingerprint anomaly",
    "card_counter": "elevated card-counter aggregations",
    "engineered": "Vesta engineered feature spike",
    "verification": "identity verification mismatch",
    "behavioral_velocity_timing": "behavioral velocity",
    "amount": "amount anomaly",
    "geographic": "geographic anomaly",
    "email": "email risk",
    "card_metadata": "card metadata anomaly",
}

# Behavioral-clean threshold — when these signals are below their respective
# bounds, the transaction looks normal at the surface and the fraud is hidden
# in model-detected feature families
BEHAVIORAL_CLEAN_AMOUNT_ZSCORE_MAX = 1.5
BEHAVIORAL_CLEAN_VELOCITY_24H_MAX = 15
BEHAVIORAL_CLEAN_VELOCITY_1H_MAX = 5

# How many pattern candidates to retrieve and show
PATTERN_RETRIEVAL_TOP_K = 10        # how many the agent sees
PATTERN_USER_DISPLAY_TOP_K = 5      # how many the UI shows


@dataclass
class InvestigationContext:
    transaction: dict
    xgb_score: float
    lstm_score: float
    lstm_anomaly: bool
    triage_result: dict | None = None
    xgb_attribution: dict | None = None
    lstm_attribution: dict | None = None
    investigator_findings: str | None = None
    investigator_tool_calls: list[dict] = field(default_factory=list)
    pattern_matches: list[dict] = field(default_factory=list)
    pattern_analysis: str | None = None
    final_report: str | None = None


class FraudInvestigator:
    """Orchestrates the multi-agent investigation flow."""

    def __init__(
        self,
        api_key: str,
        triage_model: str = TRIAGE_MODEL,
        pattern_model: str = PATTERN_MODEL,
        investigator_model: str = INVESTIGATOR_MODEL,
        report_model: str = REPORT_MODEL,
    ):
        if not api_key:
            raise ValueError("Anthropic API key required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.triage_model = triage_model
        self.pattern_model = pattern_model
        self.investigator_model = investigator_model
        self.report_model = report_model
        self._xgb_model = None
        self._xgb_features = None
        self._sample_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    def _ensure_xgb_loaded(self):
        if self._xgb_model is not None and self._xgb_features is not None:
            return
        try:
            from src.ml_models.inference import load_model

            model, feature_names = load_model()
            self._xgb_model = model
            self._xgb_features = feature_names
            logger.info(f"Loaded XGBoost model with {len(feature_names)} features")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model for attribution: {e}")

    def _ensure_sample_loaded(self):
        if self._sample_df is not None:
            return
        try:
            from src.utils.config import SAMPLES_DIR

            path = SAMPLES_DIR / "demo_transactions.parquet"
            if path.exists():
                self._sample_df = pd.read_parquet(path)
                logger.info(f"Loaded transaction sample: {len(self._sample_df):,} rows")
            else:
                self._sample_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not load sample for LSTM history: {e}")
            self._sample_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Stage: Score Attribution (deterministic)
    # ------------------------------------------------------------------
    def _run_xgb_attribution(self, ctx: InvestigationContext) -> dict:
        self._ensure_xgb_loaded()
        if self._xgb_model is None or self._xgb_features is None:
            return {"error": "XGBoost model not available", "top_drivers": []}

        try:
            from src.ml_models.explain import explain_xgboost_prediction

            return explain_xgboost_prediction(
                model=self._xgb_model,
                feature_names=self._xgb_features,
                row_dict=ctx.transaction,
                top_k=8,
            )
        except Exception as e:
            logger.warning(f"XGBoost attribution failed: {e}")
            return {"error": f"{type(e).__name__}: {e}", "top_drivers": []}

    def _run_lstm_attribution(self, ctx: InvestigationContext) -> dict:
        """Pull the card's recent transaction history and run timestep analysis."""
        self._ensure_sample_loaded()
        if self._sample_df is None or self._sample_df.empty:
            return {
                "error": "Transaction sample not available — can't reconstruct LSTM sequence.",
                "top_steps": [],
            }

        card_id = ctx.transaction.get("card1")
        txn_id = ctx.transaction.get("TransactionID")
        if card_id is None:
            return {"error": "No card1 on transaction", "top_steps": []}

        try:
            history = self._sample_df[self._sample_df["card1"] == card_id]
            if "TransactionDT" in history.columns:
                history = history.sort_values("TransactionDT")

            if txn_id is not None and txn_id in history["TransactionID"].values:
                idx = history.index[history["TransactionID"] == txn_id][-1]
                pos = history.index.get_loc(idx)
                window = history.iloc[max(0, pos - 50) : pos + 1]
            else:
                window = history.tail(50)

            if window.empty:
                return {
                    "error": "No history for this card — can't run LSTM timestep analysis.",
                    "top_steps": [],
                }

            from src.ml_models.explain import explain_lstm_sequence

            return explain_lstm_sequence(window, top_k=5)
        except Exception as e:
            logger.warning(f"LSTM attribution failed: {e}")
            return {"error": f"{type(e).__name__}: {e}", "top_steps": []}

    # ------------------------------------------------------------------
    # Agent 1 — Triage (HAIKU, non-streaming because output is short JSON)
    # ------------------------------------------------------------------
    def _run_triage(self, ctx: InvestigationContext) -> dict:
        user_msg = f"""TRANSACTION:
{json.dumps(ctx.transaction, indent=2, default=str)}

ML SCORES:
- XGBoost fraud probability: {ctx.xgb_score:.4f}
- LSTM anomaly score (normalized): {ctx.lstm_score:.4f}
- LSTM flagged anomaly: {ctx.lstm_anomaly}

Triage this transaction. Respond with JSON only."""

        resp = self.client.messages.create(
            model=self.triage_model,
            max_tokens=400,
            system=TRIAGE_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"risk": "MEDIUM", "reason": text[:200], "investigate": True}

    # ------------------------------------------------------------------
    # Agent 2 — Investigator (SONNET, with tools + STREAMED final response)
    # ------------------------------------------------------------------
    def _run_investigator(self, ctx: InvestigationContext) -> tuple[str, list[dict]]:
        from src.ml_models.explain import format_combined_attribution_for_llm

        attribution_text = format_combined_attribution_for_llm(
            ctx.xgb_attribution, ctx.lstm_attribution
        )

        user_msg = f"""Investigate this flagged transaction.

TRANSACTION:
{json.dumps(ctx.transaction, indent=2, default=str)}

ML SCORES:
- XGBoost fraud probability: {ctx.xgb_score:.4f}
- LSTM anomaly score: {ctx.lstm_score:.4f}

TRIAGE FINDING:
{json.dumps(ctx.triage_result, indent=2)}

MODEL ATTRIBUTION (ground truth — actual feature contributions and timestep errors, NOT speculation):
{attribution_text}

Use your tools to gather evidence. Then write a fact-based summary.

When citing what drove the XGBoost score, use the SHAP output above — those are the actual feature contributions.
When citing the LSTM signal, use the timestep analysis above — it tells you whether the anomaly is in the CURRENT transaction or earlier in the recent sequence.

When calling search_fraud_cases, pass skip_ood=true."""

        messages = [{"role": "user", "content": user_msg}]
        tool_calls = []

        for iteration in range(6):
            # Tool-use turns must be non-streaming (Anthropic SDK constraint)
            # Final answer turn (when no more tool_use) we could stream, but
            # since Streamlit's session_state already buffers the result, we
            # use the simpler non-streaming path here. Streaming benefit is
            # captured in the Report stage where the user actually waits.
            resp = self.client.messages.create(
                model=self.investigator_model,
                max_tokens=1500,
                system=INVESTIGATOR_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        tool_input = dict(block.input or {})
                        if block.name == "search_fraud_cases":
                            tool_input["skip_ood"] = True
                        result = execute_tool(block.name, tool_input)
                        tool_calls.append(
                            {"tool": block.name, "input": tool_input, "output": result}
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                return final_text, tool_calls

        return "Investigation hit max iterations.", tool_calls

    # ------------------------------------------------------------------
    # Agent 3 — Pattern matching (HAIKU, SHAP-AWARE retrieval)
    # ------------------------------------------------------------------
    def _build_pattern_query(self, ctx: InvestigationContext) -> str:
        """Build a SHAP-aware retrieval query.

        Strategy:
          - If behavioral signals look CLEAN but SHAP shows strong
            device/card-counter/engineered contributions, the query emphasizes
            the "clean behavior + hidden model signal" surface (matches new
            archetypes: device_takeover, credential_compromise, engineered_anomaly)
          - If behavioral signals are SUSPICIOUS, the query mixes behavioral
            description with the SHAP-detected fraud surface
          - If no SHAP data is available, falls back to behavioral-only
        """
        amt = ctx.transaction.get("TransactionAmt", 0)
        v24 = ctx.transaction.get("card1_txn_count_24h", 0)
        v1h = ctx.transaction.get("card1_txn_count_1h", 0)
        z = ctx.transaction.get("card1_amt_zscore", 0)
        is_night = ctx.transaction.get("is_night", 0)
        hour = ctx.transaction.get("txn_hour", -1)

        try:
            z_val = float(z or 0)
        except (TypeError, ValueError):
            z_val = 0.0
        try:
            v24_int = int(v24 or 0)
        except (TypeError, ValueError):
            v24_int = 0
        try:
            v1h_int = int(v1h or 0)
        except (TypeError, ValueError):
            v1h_int = 0
        try:
            is_night_int = int(is_night or 0)
        except (TypeError, ValueError):
            is_night_int = 0
        try:
            hour_int = int(hour or -1)
        except (TypeError, ValueError):
            hour_int = -1

        behavioral_clean = (
            abs(z_val) < BEHAVIORAL_CLEAN_AMOUNT_ZSCORE_MAX
            and v24_int < BEHAVIORAL_CLEAN_VELOCITY_24H_MAX
            and v1h_int < BEHAVIORAL_CLEAN_VELOCITY_1H_MAX
            and is_night_int == 0
        )

        fraud_surfaces = []
        if ctx.xgb_attribution and not ctx.xgb_attribution.get("error"):
            family_totals = ctx.xgb_attribution.get("by_family", {}) or {}
            sorted_families = sorted(
                family_totals.items(),
                key=lambda kv: abs(kv[1] or 0),
                reverse=True,
            )
            for fam, val in sorted_families[:3]:
                if val is None or abs(val) < 0.3:
                    continue
                label = FAMILY_TO_SURFACE_LABEL.get(fam, fam.replace("_", " "))
                if label not in fraud_surfaces:
                    fraud_surfaces.append(label)

        if behavioral_clean and fraud_surfaces:
            query = (
                f"clean behavioral signals normal velocity and amount but "
                f"{' and '.join(fraud_surfaces)} suggesting device session compromise "
                f"or credential replay invisible in raw card metrics"
            )
        elif fraud_surfaces:
            behavioral_parts = []
            try:
                behavioral_parts.append(f"transaction amount ${float(amt):.2f}")
            except (TypeError, ValueError):
                pass
            if v24_int > 1:
                behavioral_parts.append(
                    f"with {v24_int} transactions on this card in last 24 hours"
                )
            if v1h_int > 0:
                behavioral_parts.append(f"and {v1h_int} in the last hour")
            if abs(z_val) >= 1.0:
                direction = "above" if z_val > 0 else "below"
                behavioral_parts.append(
                    f"amount {direction} card's typical pattern (z-score {z_val:+.1f})"
                )
            if is_night_int == 1:
                behavioral_parts.append("at night-time")
            if 0 <= hour_int <= 5:
                behavioral_parts.append(f"at unusual hour {hour_int}")
            query = (
                " ".join(behavioral_parts)
                + " with "
                + " and ".join(fraud_surfaces)
            )
        else:
            behavioral_parts = []
            try:
                behavioral_parts.append(f"transaction amount ${float(amt):.2f}")
            except (TypeError, ValueError):
                pass
            if v24_int > 1:
                behavioral_parts.append(
                    f"with {v24_int} transactions on this card in last 24 hours"
                )
            if v1h_int > 0:
                behavioral_parts.append(f"and {v1h_int} in the last hour")
            if abs(z_val) >= 1.0:
                direction = "above" if z_val > 0 else "below"
                behavioral_parts.append(
                    f"amount {direction} card's typical pattern (z-score {z_val:+.1f})"
                )
            if is_night_int == 1:
                behavioral_parts.append("at night-time")
            if 0 <= hour_int <= 5:
                behavioral_parts.append(f"at unusual hour {hour_int}")
            query = (
                " ".join(behavioral_parts)
                if behavioral_parts
                else "high risk fraud transaction with anomalous behavior"
            )

        return query

    def _run_pattern(self, ctx: InvestigationContext) -> tuple[list[dict], str]:
        query = self._build_pattern_query(ctx)
        logger.info(f"Pattern query: {query[:200]}")

        matches = execute_tool(
            "search_fraud_cases",
            {"query": query, "top_k": PATTERN_RETRIEVAL_TOP_K, "skip_ood": True},
        )

        if not matches or (
            isinstance(matches, list)
            and matches
            and isinstance(matches[0], dict)
            and (matches[0].get("ood_rejected") or matches[0].get("error"))
        ):
            empty_analysis = (
                "**No pattern from the library matches this transaction.** "
                "The transaction's feature combination doesn't closely resemble any of the "
                "377 catalogued patterns, which suggests either a novel attack vector or an "
                "unusual legitimate transaction triggering the model on identity/device features."
            )
            return [], empty_analysis

        matches_for_user = matches[:PATTERN_USER_DISPLAY_TOP_K]
        matches_for_agent = matches[:PATTERN_RETRIEVAL_TOP_K]

        shap_drivers_for_prompt = []
        if ctx.xgb_attribution and not ctx.xgb_attribution.get("error"):
            shap_drivers_for_prompt = (
                ctx.xgb_attribution.get("top_drivers", [])[:5] or []
            )

        user_msg = f"""TRANSACTION:
{json.dumps(ctx.transaction, indent=2, default=str)}

XGBOOST SHAP ATTRIBUTION (top fraud drivers — use this to identify which fraud surface the model is reading):
{json.dumps(shap_drivers_for_prompt, indent=2, default=str)}

TOP-{len(matches_for_agent)} SIMILAR HISTORICAL CASES (with summary, reasoning, indicators, and similarity score):
{json.dumps(matches_for_agent, indent=2, default=str)}

Analyze: which pattern best fits? How strong is the match? Which indicators are present here, and which are absent?

IMPORTANT: When checking fit, prioritize patterns whose indicators reference the SHAP-attributed feature families. If device/identity (D-fields) drives the score, look for device_takeover patterns. If card-counter aggregations (C-fields) drive the score, look for credential_compromise patterns. If engineered features (V-fields) drive the score, look for engineered_anomaly patterns. Don't just pick the top-similarity match — scan all candidates for the one whose indicator language matches what the model is actually detecting."""

        resp = self.client.messages.create(
            model=self.pattern_model,
            max_tokens=900,
            system=PATTERN_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        analysis = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return matches_for_user, analysis

    # ------------------------------------------------------------------
    # Agent 4 — Report (SONNET, STREAMED)
    # ------------------------------------------------------------------
    def _run_report(self, ctx: InvestigationContext) -> str:
        from src.ml_models.explain import format_combined_attribution_for_llm

        attribution_text = format_combined_attribution_for_llm(
            ctx.xgb_attribution, ctx.lstm_attribution
        )

        user_msg = f"""TRANSACTION:
{json.dumps(ctx.transaction, indent=2, default=str)}

ML SCORES:
- XGBoost: {ctx.xgb_score:.4f}
- LSTM anomaly: {ctx.lstm_score:.4f}

MODEL ATTRIBUTION (ground truth):
{attribution_text}

TRIAGE:
{json.dumps(ctx.triage_result, indent=2)}

INVESTIGATOR FINDINGS:
{ctx.investigator_findings}

PATTERN ANALYSIS:
{ctx.pattern_analysis}

Synthesize into the final analyst report.

Reminder: escape dollar signs as \\$ (e.g. write \\$67.07 not $67.07) so they render correctly in Streamlit markdown."""

        # Streaming: assemble final text by accumulating delta events.
        # Stream-vs-non-stream wall-clock is identical, but streaming returns
        # the FIRST tokens to the caller earlier (Streamlit could render them
        # progressively if we surface a callback). For now we accumulate and
        # return the full text — the streaming path here keeps the API call
        # responsive and ready for a future progressive-render UI hook.
        full_text_parts: list[str] = []
        with self.client.messages.stream(
            model=self.report_model,
            max_tokens=1200,
            system=REPORT_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text_delta in stream.text_stream:
                full_text_parts.append(text_delta)

        return "".join(full_text_parts)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def investigate(
        self,
        transaction: dict,
        xgb_score: float,
        lstm_score: float,
        lstm_anomaly: bool,
    ) -> InvestigationContext:
        ctx = InvestigationContext(
            transaction=transaction,
            xgb_score=xgb_score,
            lstm_score=lstm_score,
            lstm_anomaly=lstm_anomaly,
        )
        ctx.triage_result = self._run_triage(ctx)
        ctx.xgb_attribution = self._run_xgb_attribution(ctx)
        ctx.lstm_attribution = self._run_lstm_attribution(ctx)
        if ctx.triage_result.get("investigate", True):
            ctx.investigator_findings, ctx.investigator_tool_calls = self._run_investigator(ctx)
            ctx.pattern_matches, ctx.pattern_analysis = self._run_pattern(ctx)
            ctx.final_report = self._run_report(ctx)
        return ctx

    def stream_investigate(
        self,
        transaction: dict,
        xgb_score: float,
        lstm_score: float,
        lstm_anomaly: bool,
    ) -> Generator[dict, None, None]:
        ctx = InvestigationContext(
            transaction=transaction,
            xgb_score=xgb_score,
            lstm_score=lstm_score,
            lstm_anomaly=lstm_anomaly,
        )

        yield {"type": "stage", "stage": "triage", "status": "running"}
        ctx.triage_result = self._run_triage(ctx)
        yield {"type": "stage", "stage": "triage", "status": "done", "data": ctx.triage_result}

        yield {"type": "stage", "stage": "attribution", "status": "running"}
        ctx.xgb_attribution = self._run_xgb_attribution(ctx)
        ctx.lstm_attribution = self._run_lstm_attribution(ctx)
        yield {
            "type": "stage",
            "stage": "attribution",
            "status": "done",
            "data": {
                "xgb": ctx.xgb_attribution,
                "lstm": ctx.lstm_attribution,
            },
        }

        if not ctx.triage_result.get("investigate", True):
            yield {
                "type": "complete",
                "context": ctx,
                "message": "Triage decided no further investigation needed.",
            }
            return

        yield {"type": "stage", "stage": "investigator", "status": "running"}
        ctx.investigator_findings, ctx.investigator_tool_calls = self._run_investigator(ctx)
        yield {
            "type": "stage",
            "stage": "investigator",
            "status": "done",
            "data": {
                "findings": ctx.investigator_findings,
                "tool_calls": ctx.investigator_tool_calls,
            },
        }

        yield {"type": "stage", "stage": "pattern", "status": "running"}
        ctx.pattern_matches, ctx.pattern_analysis = self._run_pattern(ctx)
        yield {
            "type": "stage",
            "stage": "pattern",
            "status": "done",
            "data": {"matches": ctx.pattern_matches, "analysis": ctx.pattern_analysis},
        }

        yield {"type": "stage", "stage": "report", "status": "running"}
        ctx.final_report = self._run_report(ctx)
        yield {"type": "stage", "stage": "report", "status": "done", "data": ctx.final_report}

        yield {"type": "complete", "context": ctx}
        