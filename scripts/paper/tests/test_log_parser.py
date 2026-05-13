"""Tests du parseur de log."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from log_parser import parse


SAMPLE_LOG = """
2000-01-01 20:00:00 - INFO - Episode 0
2000-01-01 20:00:01 - INFO -  Step 0
2000-01-01 20:00:02 - INFO -  Step 1
2000-01-01 20:00:03 - INFO - Reward: 1.0
2000-01-01 20:00:04 - INFO - Episode reward: 0.97
2000-01-01 20:00:05 - INFO - Episode 1
2000-01-01 20:00:06 - INFO -  Step 0
2000-01-01 20:00:07 - INFO - Episode reward: 0.0
2000-01-01 20:00:08 - INFO - Episode 2
2000-01-01 20:00:09 - INFO -  Step 0
2000-01-01 20:00:10 - INFO -  Step 5
2000-01-01 20:00:11 - INFO - Episode reward: 0.85
2000-01-01 20:00:11 - INFO - [LIKELIHOOD] clamped=-3.2500 raw=-3.5000 n_collapsed=0/3 min_raw=-4.0000 max_raw=-2.0000
2000-01-01 20:00:12 - INFO - [TOKENS] 42 LLM calls, 125000in + 48000out tokens
2000-01-01 20:00:13 - INFO - Average Episode Reward: 0.6066666666666667
"""


def test_parse_full_log() -> None:
    r = parse(SAMPLE_LOG)
    assert r.rewards == [0.97, 0.0, 0.85]
    assert r.avg_reward is not None
    assert abs(r.avg_reward - 0.6066666666666667) < 1e-9
    assert r.llm_calls == 42
    assert r.prompt_tokens == 125000
    assert r.completion_tokens == 48000
    assert r.steps_per_episode == [1, 0, 5]
    assert r.likelihood_final == -3.25


def test_parse_partial_crash() -> None:
    partial = "\n".join(SAMPLE_LOG.splitlines()[:5])  # avant les rewards
    r = parse(partial)
    assert r.rewards == []
    assert r.avg_reward is None
    assert r.prompt_tokens == 0


def test_parse_empty() -> None:
    r = parse("")
    assert r.rewards == []
    assert r.avg_reward is None


if __name__ == "__main__":
    test_parse_full_log()
    test_parse_partial_crash()
    test_parse_empty()
    print("OK — all log_parser tests passed")
