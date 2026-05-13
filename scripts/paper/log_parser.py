"""Parseur robuste des logs de ``main.py`` — extrait rewards par épisode,
tokens, et wall-time.

Formats attendus (voir ``main.py``) :
  "Episode {N}"                     (début de l'épisode N)
  "Episode reward: {value}"         (fin de l'épisode N)
  "[TOKENS] {C} LLM calls, {I}in + {O}out tokens"
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
# OpenRouter per-call log. This is more reliable than the final [TOKENS]
# line when offline LLM calls happen before the episode loop, because the
# main loop resets token counters before evaluation.
_RE_OPENROUTER_TOKENS = re.compile(
    r"\[OPENROUTER_TOKENS\]\s+prompt=(\d+)\s+completion=(\d+)\s+total=(\d+)"
)
_RE_AVG_REWARD = re.compile(
    r"Average Episode Reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_RE_ELBO = re.compile(r"\[ELBO\].*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
_RE_STEPS = re.compile(r"\bStep\s+(\d+)")


@dataclass
class ParsedRun:
    rewards: List[float] = field(default_factory=list)
    avg_reward: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    steps_per_episode: List[int] = field(default_factory=list)
    elbo_final: Optional[float] = None


def parse(log_text: str) -> ParsedRun:
    """Parse un log (output.log de Hydra ou stdout) et retourne les metrics.

    Implémentation robuste — les patterns peuvent manquer si le run a crash.
    Les champs restent à leur default dans ce cas.
    """
    out = ParsedRun()
    current_episode: Optional[int] = None
    current_max_step = 0

    for line in log_text.splitlines():
        m = _RE_EPISODE_START.search(line)
        if m:
            # Commit le step-count de l'épisode précédent
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
            # Accumule tous les OpenRouter calls (plus fiable que [TOKENS] final)
            out.prompt_tokens += int(m.group(1))
            out.completion_tokens += int(m.group(2))
            out.llm_calls += 1
            continue

        m = _RE_TOKENS.search(line)
        if m:
            # Fallback seulement si aucun OpenRouter token capturé (providers
            # non-OpenRouter comme Mistral / ollama).
            if out.llm_calls == 0:
                out.llm_calls = int(m.group(1))
                out.prompt_tokens = int(m.group(2))
                out.completion_tokens = int(m.group(3))
            continue

        m = _RE_AVG_REWARD.search(line)
        if m:
            out.avg_reward = float(m.group(1))
            continue

        m = _RE_ELBO.search(line)
        if m:
            out.elbo_final = float(m.group(1))

    # Commit du dernier épisode
    if current_episode is not None:
        out.steps_per_episode.append(current_max_step)

    return out


def parse_file(path) -> ParsedRun:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read())
