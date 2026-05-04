"""Rule Generator — multi-agent system that proposes production fraud rules.

Architecture:
    Planner (1 LLM call)
        ├── parallel: VelocityWorker, EmailWorker, DeviceWorker, AmountWorker (4 LLM calls)
        └── Synthesizer (1 LLM call) → ranked RuleProposals

Total: 6 LLM calls, ~5-7 seconds total runtime via ThreadPoolExecutor.

Distinct from the Investigation pipeline (one-transaction → decision).
This is the inverse: many-transactions → rule synthesis.

Direct Anthropic SDK + stdlib only — no CrewAI/LangGraph/LangChain — to keep
the codebase consistent with src.agentic.orchestrator and src.agentic.pattern_coach
which use the same direct-SDK pattern.
"""

from src.agentic.rule_generator.aggregates import (
    compute_aggregates,
    filter_transactions,
)
from src.agentic.rule_generator.orchestrator import RuleGenerator
from src.agentic.rule_generator.types import (
    AggregateInput,
    GenerationResult,
    PlannerOutput,
    RuleProposal,
    SynthesisOutput,
    WorkerOutput,
)

__all__ = [
    "RuleGenerator",
    "AggregateInput",
    "GenerationResult",
    "PlannerOutput",
    "RuleProposal",
    "SynthesisOutput",
    "WorkerOutput",
    "compute_aggregates",
    "filter_transactions",
]
