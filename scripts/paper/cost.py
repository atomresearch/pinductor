"""LLM price table and per-episode cost accounting.

``paper_runner`` calls :func:`estimate_cost_usd` after each subprocess and
stores the result in the registry's ``cost_usd`` column, which is what the
reported compute budget is summed from.

Prices are hard-coded rather than queried, so that re-running the pipeline
later reproduces the same figures instead of tracking provider changes.
They must be updated by hand when a provider's rates move.

The keys are provider model ids and must match the ids the runs actually
request, character for character. An id that is absent still yields 0.0,
because accounting must not abort a run that already produced results, but
it now warns once so that a missing price shows up as a warning rather than
as an understated budget.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Set

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    input: float   # USD / 1M prompt tokens
    output: float  # USD / 1M completion tokens


PRICING: Dict[str, Price] = {
    "qwen/qwen3.6-plus":           Price(input=0.50, output=1.50),
    "deepseek/deepseek-v3.2":      Price(input=0.15, output=0.60),
    "codestral-latest":            Price(input=0.30, output=0.90),
    "anthropic/claude-sonnet-4-6": Price(input=3.00, output=15.00),
    "anthropic/claude-opus-4-7":   Price(input=15.00, output=75.00),
    # The LLM ablation requests this id, with a dot rather than a dash. Both
    # spellings are listed because the lookup is exact and the run is what
    # decides which one reaches the provider.
    "anthropic/claude-opus-4.7":   Price(input=15.00, output=75.00),
    "openai/gpt-5":                Price(input=1.25, output=10.00),
}


# Model ids already reported as unpriced, so the warning fires once per id
# rather than once per run.
_UNPRICED_SEEN: Set[str] = set()


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int,
) -> float:
    """Cost in USD of one run's token usage under the frozen price table."""
    p = PRICING.get(model)
    if p is None:
        # Accounting must never abort a run that already produced results, so
        # an unpriced model is charged 0.0 — but silently charging zero is how
        # a whole arm ends up costing nothing on paper, so say it once.
        if model not in _UNPRICED_SEEN:
            _UNPRICED_SEEN.add(model)
            log.warning(
                "No price for %r; its runs are counted as costing 0.0. Add it "
                "to PRICING in this module to have it accounted for.", model,
            )
        return 0.0
    return (
        prompt_tokens * p.input + completion_tokens * p.output
    ) / 1_000_000.0
