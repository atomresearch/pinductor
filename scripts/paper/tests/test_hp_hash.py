"""Tests pour hp_hash + resolve_overrides (post-refactor modular YAML).

Depuis le refactor modular YAML, ``resolve_overrides()`` ne retourne plus les
hyperparams scientifiques (qui sont figés dans les YAMLs
``scripts/paper/configs/<cond>/<env>.yaml``). Ces tests vérifient que le
runtime API reste stable : extras wandb, sweeps E2 (dataset_path et
num_model_attempts), filtrage random/tabular, dédup hp_hash.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperparams import hp_hash, resolve_overrides, DEFAULT_LLM, ENV_NAMES


def test_hp_hash_stable() -> None:
    d = {"a": 1, "b": "x", "c": True}
    assert hp_hash(d) == hp_hash(dict(d))
    # Ordre d'insertion ne change rien
    assert hp_hash({"b": "x", "a": 1, "c": True}) == hp_hash(d)


def test_hp_hash_different_on_any_change() -> None:
    base = resolve_overrides("lava", "ours")
    h0 = hp_hash(base)
    # Toucher 1 clé → hash différent
    mut = dict(base)
    mut["num_episodes"] = 1
    assert hp_hash(mut) != h0


def test_extras_always_present() -> None:
    """Les +extras.* sont injectés pour toutes les conditions."""
    for cond in ("ours", "curtis_their", "curtis_hardcoded",
                 "random", "tabular"):
        o = resolve_overrides("lava", cond)
        assert "+extras.approach" in o
        assert "+extras.environment" in o
        assert "+extras.experiment" in o
        assert o["+extras.experiment"] == "paper"
        assert o["+extras.environment"] == "lava"
    # ours → approach="ours"
    assert resolve_overrides("lava", "ours")["+extras.approach"] == "ours"
    # curtis_their reste "curtis_their"
    assert resolve_overrides("lava", "curtis_their")["+extras.approach"] == "curtis_their"


def test_llm_never_in_hydra_overrides() -> None:
    """Le modèle LLM transite via PAPER_LLM_MODEL env var, jamais Hydra."""
    for cond in ("ours", "curtis_their", "random", "tabular",
                 "curtis_hardcoded"):
        o = resolve_overrides("lava", cond, llm_model="anthropic/claude-opus-4-7")
        assert "agent.llm_model" not in o


def test_no_scientific_hyperparams_in_runtime_overrides() -> None:
    """Les hyperparams figés (max_attempts, entropy, UCB1…) sont dans les YAMLs,
    PAS injectés par resolve_overrides.
    """
    forbidden = (
        "agent.max_attempts", "agent.num_particles",
        "agent.planner.entropy_coeff", "agent.planner.lambda_coeff",
        "agent.planner.max_expansions", "agent.use_ucb1_selection",
        "agent.ucb1_c", "agent.elbo_softmax_temperature",
        "agent.fully_obs", "env.fully_obs",
    )
    for cond in ("ours", "curtis_their"):
        o = resolve_overrides("lava", cond)
        for k in forbidden:
            assert k not in o, f"{cond} a reçu {k}, devrait venir du YAML"


def test_e2_offline_sweep_dataset_path() -> None:
    """E2 offline : ``agent.num_input_examples`` → override dataset_path."""
    # N=3 pour ours
    o = resolve_overrides("lava", "ours",
                          extra={"agent.num_input_examples": 3})
    assert o["agent.dataset_path"].endswith(
        "MyMiniGrid-LavaWall-v0_paper_N3.pkl"
    )
    # N=10 pour ours (par défaut dans le YAML) → si demandé explicitement,
    # on réinjecte quand même (idempotent)
    o10 = resolve_overrides("lava", "ours",
                            extra={"agent.num_input_examples": 10})
    assert o10["agent.dataset_path"].endswith("_paper_N10.pkl")
    # N=7 (invalide) : raise loud rather than silently routing to N=10
    # (avoids ND10/ND12 hp_hash collisions seen in prior campaigns).
    try:
        resolve_overrides("lava", "ours",
                          extra={"agent.num_input_examples": 7})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid N=7")
    # curtis_their aussi
    oc = resolve_overrides("lava", "curtis_their",
                           extra={"agent.num_input_examples": 4})
    assert oc["agent.dataset_path"].endswith("_paper_N4.pkl")


def test_e2_offline_sweep_filtered_for_no_dataset_consumers() -> None:
    """random et curtis_hardcoded n'acceptent pas dataset_path.
    On ne doit PAS leur injecter d'override.
    """
    for cond in ("random", "curtis_hardcoded"):
        o = resolve_overrides("lava", cond,
                              extra={"agent.num_input_examples": 4})
        assert "agent.dataset_path" not in o, \
            f"{cond} ne doit pas recevoir dataset_path"
    # tabular accepte dataset_path
    ot = resolve_overrides("lava", "tabular",
                           extra={"agent.num_input_examples": 4})
    assert ot["agent.dataset_path"].endswith("_paper_N4.pkl")


def test_e2_online_sweep_num_model_attempts() -> None:
    """E2 online : ``agent.num_model_attempts`` override pour ours et curtis_their."""
    for cond in ("ours", "curtis_their"):
        o = resolve_overrides("lava", cond,
                              extra={"agent.num_model_attempts": 10})
        assert o["agent.num_model_attempts"] == 10
    # Pas d'injection pour non-LLM
    for cond in ("random", "tabular", "curtis_hardcoded"):
        o = resolve_overrides("lava", cond,
                              extra={"agent.num_model_attempts": 10})
        assert "agent.num_model_attempts" not in o


def test_num_episodes_forced_via_extra() -> None:
    """Smoke : num_episodes=1 via extra → réinjecté en override."""
    o = resolve_overrides("lava", "ours", extra={"num_episodes": 1})
    assert o["num_episodes"] == 1
    # Par défaut, pas d'override (YAML paper a num_episodes=3)
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
