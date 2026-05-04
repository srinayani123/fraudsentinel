"""Rule Generator orchestrator — Planner + parallel Workers + Synthesizer.

Pipeline:
    Planner (1 LLM call, ~1s)
        ↓
    [VelocityWorker, EmailWorker, DeviceWorker, AmountWorker]
        ↓ parallel via ThreadPoolExecutor (4 LLM calls, ~3s wall time)
        ↓
    Synthesizer (1 LLM call, ~2s)
        ↓
    GenerationResult (ranked rules, coverage summary, deployment recommendation)

Total: 6 LLM calls, ~5-7 seconds end-to-end via parallelism.

Mirrors the architecture of src.agentic.orchestrator.FraudInvestigator:
  - Class-based, takes api_key in __init__
  - Stream-style API (stream_generate yields stage events)
  - Fallback non-streaming API (generate returns final GenerationResult)
  - Uses anthropic.Anthropic client directly, no framework
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

import anthropic

from src.agentic.rule_generator.aggregates import compute_aggregates, filter_transactions
from src.agentic.rule_generator.prompts import (
    PLANNER_PROMPT,
    SYNTHESIZER_PROMPT,
)
from src.agentic.rule_generator.types import (
    AggregateInput,
    GenerationResult,
    PlannerOutput,
    RuleProposal,
    SynthesisOutput,
    WorkerOutput,
)
from src.agentic.rule_generator.workers import (
    amount_worker,
    device_worker,
    email_worker,
    velocity_worker,
)
from src.utils.config import DEFAULT_ANTHROPIC_MODEL
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Map worker names to their function so ThreadPoolExecutor can dispatch by name
_WORKER_DISPATCH = {
    "velocity": velocity_worker,
    "email": email_worker,
    "device": device_worker,
    "amount": amount_worker,
}


# ----------------------------------------------------------------------
# Helpers — JSON parsing
# ----------------------------------------------------------------------
def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _parse_rule_proposal(d: dict) -> RuleProposal:
    return RuleProposal(
        rule_name=str(d.get("rule_name", "unnamed-rule")),
        plain_english=str(d.get("plain_english", "")),
        rule_code_sql=str(d.get("rule_code_sql", "")),
        rule_code_pseudo=str(d.get("rule_code_pseudo", "")),
        feature_family=str(d.get("feature_family", "unknown")),
        severity=str(d.get("severity", "review")),
        estimated_catch_rate=str(d.get("estimated_catch_rate", "unknown")),
        estimated_false_positive_rate=str(d.get("estimated_false_positive_rate", "unknown")),
        rationale=str(d.get("rationale", "")),
        evidence=[str(e) for e in (d.get("evidence") or [])],
    )


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class RuleGenerator:
    """Multi-agent fraud rule generator.

    Usage:
        rg = RuleGenerator(api_key=...)
        result = rg.generate(filtered_df, risk_band="high_critical")

    Or streaming:
        for event in rg.stream_generate(filtered_df, risk_band="high_critical"):
            print(event)
    """

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL):
        if not api_key:
            raise ValueError("Anthropic API key required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    # ------------------------------------------------------------------
    # Stage 1 — Planner
    # ------------------------------------------------------------------
    def _run_planner(self, aggregates: AggregateInput) -> PlannerOutput:
        t0 = time.time()
        user_msg = f"""AGGREGATE STATISTICS FROM THE FLAGGED TRANSACTION SET:
{json.dumps(aggregates.to_dict(), indent=2, default=str)}

Plan which Workers should run and write a focused brief for each. Output JSON only."""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                system=PLANNER_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            text = _strip_json_fences(text)
            parsed = json.loads(text)

            return PlannerOutput(
                workers_to_run=list(parsed.get("workers_to_run", ["velocity", "email", "device", "amount"])),
                velocity_brief=str(parsed.get("velocity_brief", "")),
                email_brief=str(parsed.get("email_brief", "")),
                device_brief=str(parsed.get("device_brief", "")),
                amount_brief=str(parsed.get("amount_brief", "")),
                overall_strategy=str(parsed.get("overall_strategy", "")),
                runtime_ms=(time.time() - t0) * 1000,
            )
        except (json.JSONDecodeError, anthropic.APIError) as e:
            logger.warning(f"Planner failed, falling back to default plan: {e}")
            # Fallback: run all workers with a generic brief from the aggregates
            generic_brief = (
                f"Analyze the aggregate stats and propose 1-3 production rules. "
                f"Fraud rate in this set is {aggregates.fraud_rate*100:.1f}%."
            )
            return PlannerOutput(
                workers_to_run=["velocity", "email", "device", "amount"],
                velocity_brief=generic_brief,
                email_brief=generic_brief,
                device_brief=generic_brief,
                amount_brief=generic_brief,
                overall_strategy="Planner unavailable — running all workers with generic briefs.",
                runtime_ms=(time.time() - t0) * 1000,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        except Exception as e:
            logger.error(f"Planner unexpected error: {e}")
            return PlannerOutput(
                workers_to_run=["velocity", "email", "device", "amount"],
                velocity_brief="",
                email_brief="",
                device_brief="",
                amount_brief="",
                overall_strategy="",
                runtime_ms=(time.time() - t0) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    # ------------------------------------------------------------------
    # Stage 2 — Parallel Workers via ThreadPoolExecutor
    # ------------------------------------------------------------------
    def _run_workers_parallel(
        self,
        plan: PlannerOutput,
        aggregates: AggregateInput,
    ) -> dict[str, WorkerOutput]:
        """Run all planned workers in parallel.

        Each worker is a separate thread because they're I/O-bound (waiting
        on Anthropic API), so the GIL doesn't matter. ThreadPoolExecutor
        lets all 4 wall-clock-finish in ~3s instead of ~12s sequential.
        """
        # Map worker name → its brief
        briefs = {
            "velocity": plan.velocity_brief,
            "email": plan.email_brief,
            "device": plan.device_brief,
            "amount": plan.amount_brief,
        }

        # Filter to only workers actually planned to run
        active_workers = [w for w in plan.workers_to_run if w in _WORKER_DISPATCH]
        if not active_workers:
            # Defensive: planner returned no workers, run all anyway
            active_workers = list(_WORKER_DISPATCH.keys())

        outputs: dict[str, WorkerOutput] = {}

        with ThreadPoolExecutor(max_workers=len(active_workers)) as pool:
            future_to_worker = {
                pool.submit(
                    _WORKER_DISPATCH[worker_name],
                    self.client,
                    self.model,
                    briefs.get(worker_name, ""),
                    aggregates,
                ): worker_name
                for worker_name in active_workers
            }

            for future in as_completed(future_to_worker):
                worker_name = future_to_worker[future]
                try:
                    outputs[worker_name] = future.result()
                except Exception as e:
                    logger.error(f"{worker_name} worker raised: {e}")
                    outputs[worker_name] = WorkerOutput(
                        worker_name=worker_name,
                        proposed_rules=[],
                        summary="",
                        key_finding="",
                        runtime_ms=0,
                        error=f"{type(e).__name__}: {e}",
                    )

        return outputs

    # ------------------------------------------------------------------
    # Stage 3 — Synthesizer
    # ------------------------------------------------------------------
    def _run_synthesizer(
        self,
        plan: PlannerOutput,
        worker_outputs: dict[str, WorkerOutput],
        aggregates: AggregateInput,
    ) -> SynthesisOutput:
        t0 = time.time()

        # Collect all proposed rules from all workers
        all_proposed_rules = []
        for w_name, w_out in worker_outputs.items():
            for rule in w_out.proposed_rules:
                all_proposed_rules.append(rule.to_dict())

        if not all_proposed_rules:
            return SynthesisOutput(
                ranked_rules=[],
                coverage_summary="No rules were proposed by any worker. The aggregates may show too little signal to anchor production rules.",
                deployment_recommendation="No rules to deploy. Re-run with a different risk_band or date_range.",
                runtime_ms=(time.time() - t0) * 1000,
            )

        worker_summaries = {
            w_name: {
                "summary": w_out.summary,
                "key_finding": w_out.key_finding,
                "n_rules_proposed": len(w_out.proposed_rules),
                "error": w_out.error,
            }
            for w_name, w_out in worker_outputs.items()
        }

        user_msg = f"""OVERALL STRATEGY (from Planner):
{plan.overall_strategy}

WORKER SUMMARIES:
{json.dumps(worker_summaries, indent=2)}

ALL PROPOSED RULES FROM ALL WORKERS:
{json.dumps(all_proposed_rules, indent=2)}

AGGREGATE CONTEXT (for evaluating rule quality):
- Total transactions in set: {aggregates.n_transactions}
- Total fraud: {aggregates.n_fraud}
- Fraud rate: {aggregates.fraud_rate*100:.1f}%

Rank these rules and write the deployment recommendation. Output JSON only."""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=SYNTHESIZER_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            text = _strip_json_fences(text)
            parsed = json.loads(text)

            ranked_rules = [
                _parse_rule_proposal(r)
                for r in (parsed.get("ranked_rules") or [])
            ]

            return SynthesisOutput(
                ranked_rules=ranked_rules,
                coverage_summary=str(parsed.get("coverage_summary", "")),
                deployment_recommendation=str(parsed.get("deployment_recommendation", "")),
                runtime_ms=(time.time() - t0) * 1000,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Synthesizer JSON parse failed: {e}")
            # Fallback: return all proposed rules unranked
            fallback_rules = [
                _parse_rule_proposal(r) for r in all_proposed_rules
            ]
            return SynthesisOutput(
                ranked_rules=fallback_rules,
                coverage_summary="Synthesizer could not rank rules — showing all worker proposals unranked.",
                deployment_recommendation="Review each rule manually before deployment.",
                runtime_ms=(time.time() - t0) * 1000,
                error=f"JSON parse: {e}",
            )
        except anthropic.APIError as e:
            logger.warning(f"Synthesizer API error: {e}")
            fallback_rules = [
                _parse_rule_proposal(r) for r in all_proposed_rules
            ]
            return SynthesisOutput(
                ranked_rules=fallback_rules,
                coverage_summary="Synthesizer API error — showing all worker proposals unranked.",
                deployment_recommendation="Review each rule manually before deployment.",
                runtime_ms=(time.time() - t0) * 1000,
                error=f"API error: {str(e)[:200]}",
            )
        except Exception as e:
            logger.error(f"Synthesizer unexpected error: {e}")
            fallback_rules = [
                _parse_rule_proposal(r) for r in all_proposed_rules
            ]
            return SynthesisOutput(
                ranked_rules=fallback_rules,
                coverage_summary="",
                deployment_recommendation="",
                runtime_ms=(time.time() - t0) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def generate(
        self,
        filtered_df,
        risk_band: str = "high_critical",
    ) -> GenerationResult:
        """Run the full Rule Generator pipeline and return a GenerationResult.

        Args:
            filtered_df: pandas DataFrame already filtered to the analyst's
                         selected transaction set. Use filter_transactions()
                         from aggregates.py to produce this.
            risk_band: informational — passed into AggregateInput
        """
        t_start = time.time()

        try:
            aggregates = compute_aggregates(filtered_df, risk_band=risk_band)
        except Exception as e:
            logger.error(f"Aggregate computation failed: {e}")
            return GenerationResult(
                aggregates=AggregateInput(  # empty placeholder
                    n_transactions=0, n_fraud=0, fraud_rate=0.0,
                    date_range=("unknown", "unknown"), risk_band=risk_band,
                    velocity_1h_p50=0.0, velocity_1h_p95=0.0,
                    velocity_24h_p50=0.0, velocity_24h_p95=0.0,
                    velocity_7d_p50=0.0, velocity_7d_p95=0.0,
                    velocity_24h_fraud_mean=0.0, velocity_24h_legit_mean=0.0,
                    email_high_risk_fraud_rate=0.0, email_low_risk_fraud_rate=0.0,
                    email_mismatch_fraud_rate=0.0, email_null_fraud_rate=0.0,
                    top_fraud_email_domains=[],
                    fraud_card_distinct_addr1_p95=0.0,
                    legit_card_distinct_addr1_p95=0.0,
                    fraud_card_distinct_products_p95=0.0,
                    legit_card_distinct_products_p95=0.0,
                    d1_zero_fraud_rate=0.0, d1_nonzero_fraud_rate=0.0,
                    amount_p50=0.0, amount_p95=0.0, amount_p99=0.0,
                    amount_zscore_fraud_p95=0.0, amount_zscore_legit_p95=0.0,
                    amount_sum_24h_fraud_p95=0.0, amount_sum_24h_legit_p95=0.0,
                    products_with_high_fraud=[],
                    night_fraud_rate=0.0, day_fraud_rate=0.0,
                    hour_with_highest_fraud=0, hour_with_highest_fraud_rate=0.0,
                ),
                planner_output=PlannerOutput(
                    workers_to_run=[], velocity_brief="", email_brief="",
                    device_brief="", amount_brief="", overall_strategy="",
                    runtime_ms=0.0,
                ),
                worker_outputs={},
                synthesis=SynthesisOutput(
                    ranked_rules=[], coverage_summary="", deployment_recommendation="",
                    runtime_ms=0.0,
                ),
                total_runtime_s=time.time() - t_start,
                error=f"Aggregate computation failed: {e}",
            )

        plan = self._run_planner(aggregates)
        worker_outputs = self._run_workers_parallel(plan, aggregates)
        synthesis = self._run_synthesizer(plan, worker_outputs, aggregates)

        return GenerationResult(
            aggregates=aggregates,
            planner_output=plan,
            worker_outputs=worker_outputs,
            synthesis=synthesis,
            total_runtime_s=time.time() - t_start,
        )

    def stream_generate(
        self,
        filtered_df,
        risk_band: str = "high_critical",
    ) -> Generator[dict, None, None]:
        """Stream generator — yields stage events for live UI rendering.

        Event shapes:
            {"type": "stage", "stage": "aggregates", "status": "running"}
            {"type": "stage", "stage": "aggregates", "status": "done", "data": AggregateInput.to_dict()}
            {"type": "stage", "stage": "planner", "status": "running"}
            {"type": "stage", "stage": "planner", "status": "done", "data": PlannerOutput dict}
            {"type": "stage", "stage": "workers", "status": "running"}
            {"type": "stage", "stage": "workers", "status": "done", "data": {worker_name: WorkerOutput dict}}
            {"type": "stage", "stage": "synthesis", "status": "running"}
            {"type": "stage", "stage": "synthesis", "status": "done", "data": SynthesisOutput dict}
            {"type": "complete", "result": GenerationResult}
        """
        t_start = time.time()

        # Stage: aggregates
        yield {"type": "stage", "stage": "aggregates", "status": "running"}
        try:
            aggregates = compute_aggregates(filtered_df, risk_band=risk_band)
        except Exception as e:
            yield {
                "type": "complete",
                "result": None,
                "error": f"Aggregate computation failed: {e}",
            }
            return
        yield {
            "type": "stage",
            "stage": "aggregates",
            "status": "done",
            "data": aggregates.to_dict(),
        }

        # Stage: planner
        yield {"type": "stage", "stage": "planner", "status": "running"}
        plan = self._run_planner(aggregates)
        yield {
            "type": "stage",
            "stage": "planner",
            "status": "done",
            "data": {
                "workers_to_run": plan.workers_to_run,
                "overall_strategy": plan.overall_strategy,
                "runtime_ms": plan.runtime_ms,
                "error": plan.error,
            },
        }

        # Stage: workers (parallel)
        yield {"type": "stage", "stage": "workers", "status": "running"}
        worker_outputs = self._run_workers_parallel(plan, aggregates)
        yield {
            "type": "stage",
            "stage": "workers",
            "status": "done",
            "data": {k: v.to_dict() for k, v in worker_outputs.items()},
        }

        # Stage: synthesis
        yield {"type": "stage", "stage": "synthesis", "status": "running"}
        synthesis = self._run_synthesizer(plan, worker_outputs, aggregates)
        yield {
            "type": "stage",
            "stage": "synthesis",
            "status": "done",
            "data": synthesis.to_dict(),
        }

        # Complete
        result = GenerationResult(
            aggregates=aggregates,
            planner_output=plan,
            worker_outputs=worker_outputs,
            synthesis=synthesis,
            total_runtime_s=time.time() - t_start,
        )
        yield {"type": "complete", "result": result}
        