"""Worker agent functions for the Rule Generator.

Each worker is a single function that takes:
  - Anthropic client
  - The full AggregateInput
  - Its focused brief from the Planner
  - The model name

And returns a WorkerOutput.

Workers run in parallel via ThreadPoolExecutor in orchestrator.py.

These functions follow the same pattern as src.agentic.pattern_coach:
  - Module-level system prompt
  - Single client.messages.create call
  - JSON parsing with fence-stripping
  - Defensive error handling that returns a WorkerOutput with error field set
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from src.agentic.rule_generator.prompts import (
    AMOUNT_WORKER_PROMPT,
    DEVICE_WORKER_PROMPT,
    EMAIL_WORKER_PROMPT,
    VELOCITY_WORKER_PROMPT,
)
from src.agentic.rule_generator.types import (
    AggregateInput,
    RuleProposal,
    WorkerOutput,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _strip_json_fences(text: str) -> str:
    """Handle responses wrapped in ```json ... ``` fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _parse_rule_proposal(d: dict[str, Any]) -> RuleProposal:
    """Convert a dict from JSON into a RuleProposal, with defensive defaults."""
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


def _build_worker_user_msg(brief: str, aggregates: AggregateInput) -> str:
    """Construct the user-message payload that all four workers receive."""
    return f"""PLANNER BRIEF:
{brief}

AGGREGATE STATISTICS FROM THE SELECTED TRANSACTION SET:
{json.dumps(aggregates.to_dict(), indent=2, default=str)}

Propose 1-3 production-ready rules. Output JSON only."""


def _run_worker(
    client: anthropic.Anthropic,
    model: str,
    worker_name: str,
    system_prompt: str,
    brief: str,
    aggregates: AggregateInput,
    max_tokens: int = 1500,
) -> WorkerOutput:
    """Generic worker runner — used by all four worker functions."""
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": _build_worker_user_msg(brief, aggregates)}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        text = _strip_json_fences(text)
        parsed = json.loads(text)

        proposed_rules = [
            _parse_rule_proposal(r)
            for r in (parsed.get("proposed_rules") or [])
        ]

        return WorkerOutput(
            worker_name=worker_name,
            proposed_rules=proposed_rules,
            summary=str(parsed.get("summary", "")),
            key_finding=str(parsed.get("key_finding", "")),
            runtime_ms=(time.time() - t0) * 1000,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"{worker_name} worker: JSON parse failed — {e}")
        return WorkerOutput(
            worker_name=worker_name,
            proposed_rules=[],
            summary="",
            key_finding="",
            runtime_ms=(time.time() - t0) * 1000,
            error=f"JSON parse failed: {e}",
        )
    except anthropic.APIError as e:
        logger.warning(f"{worker_name} worker: API error — {e}")
        return WorkerOutput(
            worker_name=worker_name,
            proposed_rules=[],
            summary="",
            key_finding="",
            runtime_ms=(time.time() - t0) * 1000,
            error=f"API error: {str(e)[:200]}",
        )
    except Exception as e:
        logger.error(f"{worker_name} worker: unexpected error — {e}")
        return WorkerOutput(
            worker_name=worker_name,
            proposed_rules=[],
            summary="",
            key_finding="",
            runtime_ms=(time.time() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


# ----------------------------------------------------------------------
# Public worker functions — one per fraud surface
# ----------------------------------------------------------------------
def velocity_worker(
    client: anthropic.Anthropic,
    model: str,
    brief: str,
    aggregates: AggregateInput,
) -> WorkerOutput:
    """Propose velocity-based fraud rules."""
    return _run_worker(
        client=client,
        model=model,
        worker_name="velocity",
        system_prompt=VELOCITY_WORKER_PROMPT,
        brief=brief,
        aggregates=aggregates,
    )


def email_worker(
    client: anthropic.Anthropic,
    model: str,
    brief: str,
    aggregates: AggregateInput,
) -> WorkerOutput:
    """Propose email-related fraud rules."""
    return _run_worker(
        client=client,
        model=model,
        worker_name="email",
        system_prompt=EMAIL_WORKER_PROMPT,
        brief=brief,
        aggregates=aggregates,
    )


def device_worker(
    client: anthropic.Anthropic,
    model: str,
    brief: str,
    aggregates: AggregateInput,
) -> WorkerOutput:
    """Propose device/identity-based fraud rules."""
    return _run_worker(
        client=client,
        model=model,
        worker_name="device",
        system_prompt=DEVICE_WORKER_PROMPT,
        brief=brief,
        aggregates=aggregates,
    )


def amount_worker(
    client: anthropic.Anthropic,
    model: str,
    brief: str,
    aggregates: AggregateInput,
) -> WorkerOutput:
    """Propose amount-based fraud rules."""
    return _run_worker(
        client=client,
        model=model,
        worker_name="amount",
        system_prompt=AMOUNT_WORKER_PROMPT,
        brief=brief,
        aggregates=aggregates,
    )
