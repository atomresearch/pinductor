"""Tests for ``hp_hash`` and ``resolve_overrides``.

These pin the boundary between the two places a run's configuration comes
from. Scientific hyperparameters belong to the generated YAML and must not
appear in the runtime overrides; conversely the swept parameters, the
tracker metadata and the per-condition filtering must appear there. A
regression on either side silently changes what an experiment ran with.

The hash tests matter for a second reason: ``hp_hash`` is the
deduplication key, so a hash that is unstable re-runs everything, and one
that is too coarse serves the wrong results.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperparams import hp_hash, resolve_overrides, DEFAULT_LLM, ENV_NAMES


def test_hp_hash_stable() -> None:
    d = {"a": 1, "b": "x", "c": True}
    assert hp_hash(d) == hp_hash(dict(d))
    # Insertion order must not matter.
    assert hp_hash({"b": "x", "a": 1, "c": True}) == hp_hash(d)


def test_hp_hash_different_on_any_change() -> None:
    base = resolve_overrides("lava", "ours")
    h0 = hp_hash(base)
    # Any change to any key must change the hash.
    mut = dict(base)
    mut["num_episodes"] = 1
    assert hp_hash(mut) != h0


def test_extras_always_present() -> None:
    """Tracker metadata is emitted for every condition."""
    for cond in ("ours", "curtis_their", "ours_hardcoded", "curtis_hardcoded",
                 "random", "tabular"):
        o = resolve_overrides("lava", cond)
        assert "+extras.approach" in o
        assert "+extras.environment" in o
        assert "+extras.experiment" in o
        assert o["+extras.experiment"] == "paper"
        assert o["+extras.environment"] == "lava"
    # ours and ours_hardcoded are reported under one approach name.
    assert resolve_overrides("lava", "ours")["+extras.approach"] == "ours"
    assert resolve_overrides("lava", "ours_hardcoded")["+extras.approach"] == "ours"
    # Other conditions keep their own name.
    assert resolve_overrides("lava", "curtis_their")["+extras.approach"] == "curtis_their"


def test_llm_never_in_hydra_overrides() -> None:
    """The LLM is passed by environment variable, never as an override."""
    for cond in ("ours", "curtis_their", "random", "tabular",
                 "curtis_hardcoded"):
        o = resolve_overrides("lava", cond, llm_model="anthropic/claude-opus-4-7")
        assert "agent.llm_model" not in o


def test_no_scientific_hyperparams_in_runtime_overrides() -> None:
    """Frozen hyperparameters come from the YAML, not from the overrides.

    If any of these reappeared here, the YAML would stop being an
    accurate record of what a run used.
    """
    forbidden = (
        "agent.max_attempts", "agent.num_particles",
        "agent.planner.entropy_coeff", "agent.planner.lambda_coeff",
        "agent.planner.max_expansions", "agent.use_ucb1_selection",
        "agent.ucb1_c", "agent.likelihood_softmax_temperature",
        "agent.fully_obs", "env.fully_obs",
    )
    for cond in ("ours", "curtis_their"):
        o = resolve_overrides("lava", cond)
        for k in forbidden:
            assert k not in o, f"{cond} received {k}; it should come from the YAML"


def test_e2_offline_sweep_dataset_path() -> None:
    """The demonstration count selects a dataset file."""
    o = resolve_overrides("lava", "ours",
                          extra={"agent.num_input_examples": 3})
    assert o["agent.dataset_path"].endswith(
        "MyMiniGrid-LavaWall-v0_paper_N3.pkl"
    )
    # Re-requesting the value the YAML already holds is idempotent.
    o10 = resolve_overrides("lava", "ours",
                            extra={"agent.num_input_examples": 10})
    assert o10["agent.dataset_path"].endswith("_paper_N10.pkl")
    # An N with no dataset must raise rather than fall back to another
    # one, which would make two sweep points read the same file.
    try:
        resolve_overrides("lava", "ours",
                          extra={"agent.num_input_examples": 7})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid N=7")
    # Same for the baseline condition.
    oc = resolve_overrides("lava", "curtis_their",
                           extra={"agent.num_input_examples": 4})
    assert oc["agent.dataset_path"].endswith("_paper_N4.pkl")


def test_e2_offline_sweep_filtered_for_no_dataset_consumers() -> None:
    """Conditions whose agent rejects dataset_path never receive it."""
    for cond in ("random", "ours_hardcoded", "curtis_hardcoded"):
        o = resolve_overrides("lava", cond,
                              extra={"agent.num_input_examples": 4})
        assert "agent.dataset_path" not in o, \
            f"{cond} ne doit pas recevoir dataset_path"
    # The tabular agent does accept it.
    ot = resolve_overrides("lava", "tabular",
                           extra={"agent.num_input_examples": 4})
    assert ot["agent.dataset_path"].endswith("_paper_N4.pkl")


def test_e2_online_sweep_num_model_attempts() -> None:
    """The attempt budget reaches the LLM conditions and only those."""
    for cond in ("ours", "curtis_their"):
        o = resolve_overrides("lava", cond,
                              extra={"agent.num_model_attempts": 10})
        assert o["agent.num_model_attempts"] == 10
    # Conditions with no LLM would reject the kwarg.
    for cond in ("random", "tabular", "ours_hardcoded", "curtis_hardcoded"):
        o = resolve_overrides("lava", cond,
                              extra={"agent.num_model_attempts": 10})
        assert "agent.num_model_attempts" not in o


def test_num_episodes_forced_via_extra() -> None:
    """num_episodes is overridden only when a caller asks for it."""
    o = resolve_overrides("lava", "ours", extra={"num_episodes": 1})
    assert o["num_episodes"] == 1
    # Otherwise the YAML decides.
    o_default = resolve_overrides("lava", "ours")
    assert "num_episodes" not in o_default


def test_unknown_env_or_condition_raises() -> None:
    try:
        resolve_overrides("NO_SUCH_ENV", "ours")
    except KeyError:
        pass
    else:
        raise AssertionError("env invalide devait lever KeyError")
    try:
        resolve_overrides("lava", "NO_SUCH_COND")
    except KeyError:
        pass
    else:
        raise AssertionError("condition invalide devait lever KeyError")


def test_env_names_covers_all_envs() -> None:
    expected = {"lava", "corners", "four_rooms", "unlock",
                "lava_stoch", "four_rooms_stoch", "unlock_stoch"}
    assert set(ENV_NAMES) == expected


if __name__ == "__main__":
    test_hp_hash_stable()
    test_hp_hash_different_on_any_change()
    test_extras_always_present()
    test_llm_never_in_hydra_overrides()
    test_no_scientific_hyperparams_in_runtime_overrides()
    test_e2_offline_sweep_dataset_path()
    test_e2_offline_sweep_filtered_for_no_dataset_consumers()
    test_e2_online_sweep_num_model_attempts()
    test_num_episodes_forced_via_extra()
    test_unknown_env_or_condition_raises()
    test_env_names_covers_all_envs()
    print("OK — all hp_hash tests passed")
