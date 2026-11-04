"""Recovers metrics from the text logs written by ``main.py``.

The runner reads results out of the log rather than out of a structured
artefact, which keeps the two codebases free of any reporting hook: they
print, this module reads. It is therefore tolerant by design — a crashed
run yields a partially filled :class:`ParsedRun` instead of an exception,
and the runner infers the failure from how many episodes are missing.

Recognised lines (emitted by ``main.py``):
  "Episode {N}"                     start of episode N
  "Episode reward: {value}"         end of episode N
  "[TOKENS] {C} LLM calls, {I}in + {O}out tokens"
  "[OPENROUTER_TOKENS] prompt=... completion=... total=..."
  "Average Episode Reward: {mean}"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

_RE_EPISODE_START = re.compile(r"\bEpisode\s+(\d+)\s*$")
_RE_EPISODE_REWARD = re.compile(r"Episode reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_RE_TOKENS = re.compile(
    r"\[TOKENS\]\s+(\d+)\s+LLM\s+calls,\s+(\d+)\s*in\s*\+\s*(\d+)\s*out"
)
# Per-call token line. Preferred over the final [TOKENS] summary because
# main.py resets its token counters before the episode loop: a run whose
# LLM calls all happen during offline model induction reports zero there.
# Summing the per-call lines recovers the true total.
_RE_OPENROUTER_TOKENS = re.compile(
    r"\[OPENROUTER_TOKENS\]\s+prompt=(\d+)\s+completion=(\d+)\s+total=(\d+)"
)
_RE_AVG_REWARD = re.compile(
    r"Average Episode Reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_RE_LIKELIHOOD = re.compile(
    rf"(?:\[LIKELIHOOD\]\s+clamped=|Likelihood:\s*){_FLOAT}"
)
_RE_STEPS = re.compile(r"\bStep\s+(\d+)")


@dataclass
class ParsedRun:
    """Metrics recovered from one run's log.

    ``rewards`` is indexed by episode, and the runner uses its length to
    tell how far the subprocess got: fewer entries than expected means the
    remaining episodes were never produced.
    """

    rewards: List[float] = field(default_factory=list)
    avg_reward: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    steps_per_episode: List[int] = field(default_factory=list)
    likelihood_final: Optional[float] = None


def parse(log_text: str) -> ParsedRun:
    """Parse a run log (Hydra's output.log, or captured stdout).

    Never raises: patterns that the log does not contain leave their field
    at its default, which is how a crashed run is detected downstream.
    """
    out = ParsedRun()
    current_episode: Optional[int] = None
    current_max_step = 0

    for line in log_text.splitlines():
        m = _RE_EPISODE_START.search(line)
        if m:
            # The step count of an episode is only known once the next
            # episode starts, so it is committed here and again at EOF.
            if current_episode is not None:
                out.steps_per_episode.append(current_max_step)
            current_episode = int(m.group(1))
            current_max_step = 0
            continue

        m = _RE_STEPS.search(line)
        if m:
            current_max_step = max(current_max_step, int(m.group(1)))
            continue

        m = _RE_EPISODE_REWARD.search(line)
        if m:
            out.rewards.append(float(m.group(1)))
            continue

        m = _RE_OPENROUTER_TOKENS.search(line)
        if m:
            # Accumulated across every call of the run.
            out.prompt_tokens += int(m.group(1))
            out.completion_tokens += int(m.group(2))
            out.llm_calls += 1
            continue

        m = _RE_TOKENS.search(line)
        if m:
            # Fallback for providers that emit no per-call line. Only used
            # when nothing was accumulated above, since the summary would
            # otherwise overwrite the more reliable per-call total.
            if out.llm_calls == 0:
                out.llm_calls = int(m.group(1))
                out.prompt_tokens = int(m.group(2))
                out.completion_tokens = int(m.group(3))
            continue

        m = _RE_AVG_REWARD.search(line)
        if m:
            out.avg_reward = float(m.group(1))
            continue

        m = _RE_LIKELIHOOD.search(line)
        if m:
            out.likelihood_final = float(m.group(1))

    # Last episode: no following "Episode N" line closes it.
    if current_episode is not None:
        out.steps_per_episode.append(current_max_step)

    return out


def parse_file(path) -> ParsedRun:
    """Parse a log file. Undecodable bytes are replaced, not fatal."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read())
