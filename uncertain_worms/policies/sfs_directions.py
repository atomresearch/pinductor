"""Scattered Forest Search (SFS) direction generation for REx.

Adapted from Liu et al. (ICLR 2025) "Scattered Forest Search".
Three mechanisms:
  1. Scattering  -- generate K distinct improvement directions before code generation
  2. Scouting    -- track which directions succeeded/failed, share across iterations
  3. (Foresting is already handled by Thompson sampling over RexNodes)

This module is self-contained: it builds prompts and parses directions,
but never calls the LLM directly. The caller (partially_obs_planning_agent)
handles LLM calls via its existing requery_joint pipeline.

Usage:
    history = DirectionHistory()
    directions = generate_improvement_directions(
        current_code, model_fit_feedback, history, k=3
    )
    for d in directions:
        enriched_prompt = format_direction_prompt(base_prompt, d)
        # ... call LLM with enriched_prompt, score result ...
    history.record(direction=d, parent_score=-5.2, child_score=-4.1)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Direction history (Scouting)
# ---------------------------------------------------------------------------

@dataclass
class DirectionRecord:
    """One recorded attempt at a specific improvement direction."""
    direction: str
    parent_score: float
    child_score: float

    @property
    def delta(self) -> float:
        return self.child_score - self.parent_score

    @property
    def improved(self) -> bool:
        return self.delta > 0


@dataclass
class DirectionHistory:
    """Tracks improvement directions and their outcomes across REx iterations.

    Used by the Scouting mechanism: successful directions are shared as
    positive examples, failed directions as negative examples, in subsequent
    scattering prompts.
    """
    records: List[DirectionRecord] = field(default_factory=list)

    def record(
        self,
        direction: str,
        parent_score: float,
        child_score: float,
    ) -> None:
        """Record the outcome of attempting a direction."""
        rec = DirectionRecord(
            direction=direction,
            parent_score=parent_score,
            child_score=child_score,
        )
        self.records.append(rec)
        status = "improved" if rec.improved else "did not improve"
        log.info(
            f"[SFS] Direction {status} (delta={rec.delta:+.4f}): "
            f"{direction[:80]}..."
        )

    @property
    def successful_directions(self) -> List[DirectionRecord]:
        """Directions that led to score improvement, most recent first."""
        return [r for r in reversed(self.records) if r.improved]

    @property
    def failed_directions(self) -> List[DirectionRecord]:
        """Directions that did NOT lead to improvement, most recent first."""
        return [r for r in reversed(self.records) if not r.improved]

    def format_scouting_context(self, max_successes: int = 3, max_failures: int = 3) -> str:
        """Format direction history as context for the scattering prompt.

        Returns empty string if no history yet.
        """
        if not self.records:
            return ""

        lines: List[str] = []

        successes = self.successful_directions[:max_successes]
        if successes:
            lines.append("DIRECTIONS THAT IMPROVED THE SCORE:")
            for i, rec in enumerate(successes, 1):
                lines.append(
                    f"  {i}. \"{rec.direction}\" (score delta: {rec.delta:+.4f})"
                )

        failures = self.failed_directions[:max_failures]
        if failures:
            lines.append("DIRECTIONS THAT DID NOT IMPROVE THE SCORE:")
            for i, rec in enumerate(failures, 1):
                lines.append(
                    f"  {i}. \"{rec.direction}\" (score delta: {rec.delta:+.4f})"
                )

        if lines:
            lines.insert(0, "HISTORY OF PREVIOUS IMPROVEMENT ATTEMPTS:")
            lines.append("")  # trailing newline

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scattering: generate K distinct improvement directions
# ---------------------------------------------------------------------------

_SCATTERING_PROMPT_TEMPLATE = """\
You are analyzing code that models a POMDP environment. The code has been scored \
and you need to suggest {k} DISTINCT improvement directions.

CURRENT CODE:
```python
{current_code}
```

MODEL FIT FEEDBACK:
{model_fit_feedback}

{scouting_context}\
TASK: Propose exactly {k} distinct, concrete improvement directions. \
Each direction must target a DIFFERENT aspect of the code \
(e.g., transition logic, observation mapping, reward structure, edge cases, \
numerical precision).

FORMAT: Output exactly {k} lines, each starting with "DIRECTION:" followed by \
a concise description (1-2 sentences max). Do not include code.

Example format:
DIRECTION: Fix the wall collision logic -- currently the agent can pass through walls when moving left.
DIRECTION: Add handling for the goal tile observation -- the reward should be positive when the agent reaches the goal.
DIRECTION: Correct the observation rotation -- the 3x3 FOV should rotate with the agent's facing direction.
"""


def build_scattering_prompt(
    current_code: str,
    model_fit_feedback: str,
    history: DirectionHistory,
    k: int = 3,
) -> str:
    """Build the prompt that asks the LLM to propose K improvement directions.

    This prompt is sent as a lightweight call BEFORE the actual code generation.
    The LLM's response is parsed by ``parse_directions``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    scouting_context = history.format_scouting_context()
    return _SCATTERING_PROMPT_TEMPLATE.format(
        k=k,
        current_code=current_code,
        model_fit_feedback=model_fit_feedback,
        scouting_context=scouting_context,
    )


def parse_directions(llm_response: str, k: int = 3) -> List[str]:
    """Parse K directions from the LLM response to a scattering prompt.

    Expects lines starting with "DIRECTION:" (case-insensitive).
    Returns at most ``k`` directions. If fewer are found, pads with a
    generic fallback direction to ensure we always have k candidates.
    """
    pattern = re.compile(r"^\s*DIRECTION\s*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)
    matches = pattern.findall(llm_response)

    directions = [m.strip() for m in matches if m.strip()][:k]

    # Pad with generic fallback if the LLM returned fewer than k
    fallback = "General improvement: fix any remaining discrepancies between the model predictions and the observed data."
    while len(directions) < k:
        directions.append(fallback)
        log.warning(
            f"[SFS] LLM returned fewer than {k} directions, padding with fallback."
        )

    return directions


# ---------------------------------------------------------------------------
# Direction-enriched prompt construction
# ---------------------------------------------------------------------------

_DIRECTION_INJECTION = """
IMPROVEMENT DIRECTION (you MUST follow this specific direction):
{direction}

Focus your changes on this direction. Do not make unrelated changes.
"""


def format_direction_prompt(base_prompt: str, direction: str) -> str:
    """Inject an improvement direction into the base refining/starting prompt.

    Inserts the direction block just before the final instruction to produce code.
    The insertion point is the line "Explain what you believe is" which appears
    in po_joint_model_refining.txt.
    """
    injection = _DIRECTION_INJECTION.format(direction=direction)

    # Try to insert before the "Explain what you believe" instruction
    marker = "Explain what you believe is"
    if marker in base_prompt:
        return base_prompt.replace(marker, injection + "\n" + marker, 1)

    # Fallback: insert before the last code block instruction
    marker2 = "Please implement the code"
    if marker2 in base_prompt:
        return base_prompt.replace(marker2, injection + "\n" + marker2, 1)

    # Last resort: append before end
    return base_prompt + "\n" + injection


# ---------------------------------------------------------------------------
# Public API: convenience function for the REx loop
# ---------------------------------------------------------------------------

def needs_llm_scattering(current_code: Optional[str]) -> bool:
    """Whether we need an LLM call to generate directions.

    Returns False for the first iteration (no code to analyze),
    True otherwise.
    """
    return current_code is not None


def get_first_iter_directions(k: int = 3) -> List[str]:
    """Return k generic directions for the first iteration (no code yet).

    On the first iteration there is nothing to scatter from, so we return
    identical generic directions. The code generation prompt will still
    produce diversity because each call uses a different random seed.
    """
    return [
        "Write a complete initial implementation based on the environment "
        "description and observed data."
    ] * k
