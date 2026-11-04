"""Generates the frozen experiment configs under ``scripts/paper/configs/``.

One YAML per (condition, environment): 6 conditions x 7 environments = 42
files. Each starts from the config the target codebase ships for that
agent, and receives the hyperparameters the paper fixes.

Three rules shape the output:

  - The source configs are never modified. The generated file is
    self-contained, so reading it tells you everything a run used.
  - A hyperparameter is written only if the target agent's constructor
    accepts it. Hydra instantiates by keyword, so an extra key is a crash
    at construction, and the accepted set differs between conditions.
  - Values the paper reports are frozen in the file rather than passed on
    the command line, which is what lets the runner hash the config and
    detect that a run predates a change to it.

Rerun after editing anything here; the runner refuses to launch a
condition whose YAML is missing, and hashes the ones that exist.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "configs"


# ---------------------------------------------------------------------------
# Where each (condition, env) takes its source config from. The layouts
# and filename conventions differ between the two codebases, hence one
# resolver per condition rather than a single naming rule.
# ---------------------------------------------------------------------------
ENV_KEYS = ("lava", "corners", "four_rooms", "unlock",
            "lava_stoch", "four_rooms_stoch", "unlock_stoch")

ENV_NATIVE_NAMES = {
    "lava":              "MyMiniGrid-LavaWall-v0",
    "corners":           "CornerGoalRandom-Empty-10x10-v0",
    "four_rooms":        "MyMiniGrid-FourRooms-v0",
    "unlock":            "MyUnlockEnv-v0",
    "lava_stoch":        "MyMiniGrid-StochasticLavaWall-v0",
    "four_rooms_stoch":  "MyMiniGrid-StochasticFourRooms-v0",
    "unlock_stoch":      "MyMiniGrid-StochasticUnlock-v0",
}


def _ours_source(env: str) -> Path:
    base = REPO / "uncertain_worms" / "config" / "approaches" / "ours"
    return base / f"{env}_ours.yaml"


def _ours_hardcoded_source(env: str) -> Path:
    base = REPO / "uncertain_worms" / "config" / "approaches" / "hardcoded"
    # The hand-written models have no stochastic variant: the same model
    # is used, and only env_name below differs.
    base_env = env.replace("_stoch", "")
    return base / f"{base_env}_hardcoded.yaml"


def _curtis_their_source(env: str) -> Path:
    base = REPO / "curtis_baseline" / "uncertain_worms" / "config" / "approaches" / "ours"
    return base / f"{env}_llm_TROI_po_planning_agent.yaml"


def _curtis_hardcoded_source(env: str) -> Path:
    base = REPO / "curtis_baseline" / "uncertain_worms" / "config" / "approaches" / "hardcoded"
    base_env = env.replace("_stoch", "")
    mapping = {
        "lava": "lavawall_hardcoded_po_planning_agent",
        "corners": "corners_hardcoded_po_planning_agent",
        "four_rooms": "four_rooms_hardcoded_po_planning_agent",
        "unlock": "unlock_hardcoded_po_planning_agent",
    }
    return base / f"{mapping[base_env]}.yaml"


def _random_source(env: str) -> Path:
    base = REPO / "curtis_baseline" / "uncertain_worms" / "config" / "approaches" / "random"
    base_env = env.replace("_stoch", "")
    if base_env == "corners":
        return base / "corners_random_po_agent.yaml"
    return base / "random_po_agent.yaml"


def _tabular_source(env: str) -> Path:
    base = REPO / "curtis_baseline" / "uncertain_worms" / "config" / "approaches" / "tabular"
    base_env = env.replace("_stoch", "")
    if base_env == "corners":
        return base / "corners_tabular_I_po_planning_agent.yaml"
    return base / "tabular_TROI_po_planning_agent.yaml"


SOURCE_PATH: Dict[Tuple[str, str], Any] = {}
for e in ENV_KEYS:
    SOURCE_PATH[("ours", e)]             = _ours_source(e)
    SOURCE_PATH[("curtis_their", e)]     = _curtis_their_source(e)
    SOURCE_PATH[("curtis_hardcoded", e)] = _curtis_hardcoded_source(e)
    SOURCE_PATH[("random", e)]           = _random_source(e)
    SOURCE_PATH[("tabular", e)]          = _tabular_source(e)


# ---------------------------------------------------------------------------
# Evaluation protocol. The step budget is per environment: four rooms is
# larger than the others and needs a longer horizon to be solvable.
# ---------------------------------------------------------------------------
MAX_STEPS = {
    "lava": 40, "corners": 40, "four_rooms": 60, "unlock": 40,
    "lava_stoch": 40, "four_rooms_stoch": 60, "unlock_stoch": 40,
}

NUM_EPISODES = 3  # 10 seeds x 3 ep = 30 ep/condition


def _nested_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set a dotted key in a nested dict, creating the intermediate levels.

    Overrides are written in Hydra's dotted notation, while the config is
    a tree; this bridges the two.
    """
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# The hyperparameters written into each generated config, as
# {dotted_key: value}. Split into blocks that are combined per condition,
# because a key an agent does not accept cannot be written at all.
# ---------------------------------------------------------------------------
def paper_overrides(condition: str, env: str) -> Dict[str, Any]:
    """Hyperparameters to freeze into the config for one (condition, env)."""
    env_name = ENV_NATIVE_NAMES[env]

    # Evaluation protocol, identical for every condition.
    common: Dict[str, Any] = {
        "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS[env],
        "env_name": env_name,
        "env.fully_obs": False,   # partially observed unless overridden
    }

    # Shared by every planning agent, which is everything except random.
    # The particle count and expansion budget are taken from the baseline
    # paper, so that neither method gets more planning compute than the
    # other. ``kernel_bandwidth`` is not in this block because the
    # baseline has no such parameter: its particle distance is binary.
    #
    # ``max_attempts`` caps particle rejuvenation. A generated model with
    # a latent bug can reject nearly every sample, and the loop then burns
    # its full budget without making progress. A successful filter
    # converges an order of magnitude below this cap, so the value bounds
    # the pathological case while leaving the normal one untouched.
    po_planning: Dict[str, Any] = {
        "agent.max_attempts": 50_000,             # rejuvenation cap, see above
        "agent.num_particles": 10,                # from the baseline paper
        "agent.fully_obs": False,
        "agent.planner.max_expansions": 5000,     # from the baseline paper
        "agent.planner.lambda_coeff": 0.1,        # alpha in the baseline paper
    }
    # Soft-kernel particle weighting: exists only in the proposed method,
    # at the bandwidth the paper reports.
    ours_kb: Dict[str, Any] = {
        "agent.kernel_bandwidth": 0.2,
    }

    # Accepted only by the two conditions that query an LLM.
    llm_agent: Dict[str, Any] = {
        "agent.num_model_attempts": 5,
        "agent.num_online_model_attempts": 5,
        "agent.dataset_path":
            f"environments/minigrid/trajectory_data/{env_name}_paper_N10.pkl",
    }

    if condition == "ours":
        o = {**common, **po_planning, **ours_kb, **llm_agent}
        # Components of the proposed method that its shipped config does
        # not enable by default: information-gain planning, UCB1 model
        # selection, and the temperature of the likelihood-based softmax.
        o["agent.planner.entropy_coeff"] = 1.0
        o["agent.use_ucb1_selection"] = True
        o["agent.ucb1_c"] = 1.0
        o["agent.likelihood_softmax_temperature"] = 0.1
        return o

    if condition == "ours_hardcoded":
        o = {**common, **po_planning, **ours_kb}
        # Information-gain planning is kept on, so that this ablation
        # differs from `ours` only in where the model comes from.
        o["agent.planner.entropy_coeff"] = 1.0
        return o

    if condition == "curtis_their":
        o = {**common, **po_planning, **llm_agent}
        # Left partially observed, as its own config is. Its environment
        # returns a 3x3 field of view together with the agent position,
        # and the demonstration files are regenerated in exactly that
        # format (see po_obs_fn_curtis in regenerate_datasets.py), so no
        # observation conversion happens at run time.
        o["agent.planner.entropy_coeff"] = 0.0
        return o

    if condition == "curtis_hardcoded":
        o = {**common, **po_planning}
        # Same observability as the condition above, for the same reason.
        o["agent.planner.entropy_coeff"] = 0.0
        return o

    if condition == "random":
        # The random policy accepts none of the pipeline keyword
        # arguments, so it gets the protocol block and nothing else.
        return {**common}

    if condition == "tabular":
        # Given the same particle count and expansion budget as the two
        # LLM conditions. Its own configs use smaller values, which would
        # make it lose on planning compute rather than on model quality.
        o = {**common, **po_planning}
        o["agent.dataset_path"] = (
            f"environments/minigrid/trajectory_data/{env_name}_paper_N10.pkl"
        )
        # No induced model, so there is nothing for information-gain
        # planning to reduce uncertainty about.
        o["agent.planner.entropy_coeff"] = 0.0
        return o

    raise ValueError(f"Unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main() -> int:
    """Write every (condition, env) config; non-zero if any is missing."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (cond, env), src_path in SOURCE_PATH.items():
        if not src_path.exists():
            print(f"[skip] {cond}/{env}: source not found ({src_path})")
            continue
        cfg = yaml.safe_load(src_path.read_text()) or {}
        cfg = deepcopy(cfg)
        for key, val in paper_overrides(cond, env).items():
            _nested_set(cfg, key, val)
        out_sub = OUT_DIR / cond
        out_sub.mkdir(parents=True, exist_ok=True)
        out_path = out_sub / f"{env}.yaml"
        # The header records which source config the file came from; the
        # generated files are committed, so this is how a reader tells
        # them apart from hand-written ones.
        header = (
            f"# Auto-generated by scripts/paper/build_configs.py\n"
            f"# source: {src_path.relative_to(REPO)}\n"
            f"# condition={cond} env={env}\n"
        )
        out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        written += 1
        print(f"✓ {cond}/{env} ← {src_path.name}")
    print(f"\n[build] {written} YAMLs generated in {OUT_DIR}")
    # 6 conditions x 7 environments. Anything less means a source config
    # was not found, which would otherwise pass unnoticed until a run
    # failed on a missing file.
    return 0 if written == 42 else 1


if __name__ == "__main__":
    sys.exit(main())
