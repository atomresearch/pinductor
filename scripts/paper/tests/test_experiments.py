"""Tests for experiments.py: atom construction and protocol compliance.

They pin the shape of each experiment — which environments, conditions,
seeds and episodes it covers — because those counts are what the reported
budget and error bars are computed from. An experiment that quietly loses
a condition or a seed still runs to completion.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import E1, E2_offline, E2_online, E2b_stochastic, E4_llm_variation, build_atoms


def test_E1_4envs_6conditions_10seeds_3ep() -> None:
    atoms = E1()
    envs = {a.env for a in atoms}
    conds = {a.condition for a in atoms}
    seeds = {a.seed for a in atoms}
    eps = {a.episode_idx for a in atoms}
    assert envs == {"lava", "corners", "four_rooms", "unlock"}
    assert conds == {
        "curtis_their", "curtis_hardcoded",
        "ours",
        "random", "tabular",
    }
    assert seeds == set(range(10))
    assert eps == {0, 1, 2}
    # 4 envs × 5 conditions × 10 seeds × 3 episodes = 600
    assert len(atoms) == 600


def test_E2_offline_sweeps_ND() -> None:
    atoms = E2_offline()
    nds = {a.extra.get("agent.num_input_examples") for a in atoms}
    assert nds == {2, 4, 6, 8, 10, 12}
    conds = {a.condition for a in atoms}
    assert conds == {"curtis_their", "ours", "tabular"}
    envs = {a.env for a in atoms}
    assert envs == {"lava", "unlock"}


def test_E2_online_sweeps_M() -> None:
    atoms = E2_online()
    ms = {a.extra.get("agent.num_model_attempts") for a in atoms}
    # A single attempt budget, not a sweep: the appendix reports that
    # larger budgets do not help weaker proposal models.
    assert ms == {5}


def test_E2b_stochastic_uses_stoch_envs() -> None:
    atoms = E2b_stochastic()
    envs = {a.env for a in atoms}
    assert envs == {"lava_stoch", "unlock_stoch"}, envs


def test_E4_three_tiers() -> None:
    atoms = E4_llm_variation()
    llms = {a.llm_model for a in atoms}
    # Two tiers, small and frontier. The default LLM is the third point of
    # the reported comparison, but its episodes come from the other
    # experiments rather than being re-run here.
    assert "qwen/qwen3-14b" in llms
    assert "anthropic/claude-opus-4.7" in llms
    # The ablation varies the LLM, so it applies to the LLM-based method.
    conds = {a.condition for a in atoms}
    assert conds == {"ours"}


def test_build_atoms_all() -> None:
    atoms = build_atoms("all")
    exp_ids = {a.exp_id for a in atoms}
    assert "E1" in exp_ids
    assert any(e.startswith("E2_offline") for e in exp_ids)
    assert any(e.startswith("E2_online") for e in exp_ids)
    assert "E2b" in exp_ids
    assert any(e.startswith("E4") for e in exp_ids)


def test_build_atoms_subset_envs() -> None:
    atoms = build_atoms("E1", envs=("lava",))
    envs = {a.env for a in atoms}
    assert envs == {"lava"}


if __name__ == "__main__":
    test_E1_4envs_6conditions_10seeds_3ep()
    test_E2_offline_sweeps_ND()
    test_E2_online_sweeps_M()
    test_E2b_stochastic_uses_stoch_envs()
    test_E4_three_tiers()
    test_build_atoms_all()
    test_build_atoms_subset_envs()
    print("OK — all experiments tests passed")
