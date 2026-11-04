"""Declarative definition of the experiments reported in the paper.

An experiment expands into a flat list of *atoms*. One atom is one
evaluation episode, identified by
``(exp_id, env, condition, seed, episode_idx, llm_model, extra_overrides)``.
``paper_runner`` groups the atoms that share a launch configuration,
resolves their Hydra overrides through ``hyperparams.resolve_overrides``,
hashes the result into an ``hp_hash``, and submits them to the SQLite
registry, which deduplicates identical atoms across experiments.

Atom construction is pure data: it touches neither the filesystem nor the
network, so ``paper_runner.py run <exp> --dry-run`` prints the exact plan
without launching anything.

Unless an experiment states otherwise, the protocol is 10 seeds x 3
episodes = 30 evaluation episodes per (env, condition).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:  # Support both package-style imports and direct script execution.
    from .hyperparams import DEFAULT_LLM
except ImportError:  # pragma: no cover - exercised by direct CLI usage
    from hyperparams import DEFAULT_LLM

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
SEEDS_DEFAULT: Sequence[int] = tuple(range(10))  # 10 seeds x 3 ep = 30 ep/condition
EPISODES_PER_SEED = 3

ENVS_E1 = ("lava", "four_rooms", "unlock", "corners")
ENVS_PRIO = ("lava", "unlock")  # subset used by E2 / E2b / E4

E4_LLMS = (
    # Small tier: 14B with native thinking, same model family as the
    # default LLM, so the comparison isolates model scale. Its reasoning
    # budget is capped at the same value as the default through
    # PAPER_REASONING_MAX_TOKENS=5000 (read by uncertain_worms/utils.py).
    "qwen/qwen3-14b",
    # Frontier tier, same 5000-token reasoning budget for comparability.
    # Provider pricing at the time of the runs was $5/M input and $25/M
    # output tokens; the 60 atoms of this arm were budgeted at roughly
    # $233 by extrapolating the token counts measured on the small tier.
    "anthropic/claude-opus-4.7",
)

_default_sweep = (2, 4, 6, 8, 10, 12)
# Opt-in narrowing of the offline sweep without editing the default:
# ``E2_OFFLINE_SWEEP_OVERRIDE=10,12`` restricts a run to those two demo
# counts. A malformed value falls back to the full sweep instead of
# aborting, so a bad env var cannot silently truncate a campaign to an
# empty sweep.
import os as _os
_sweep_override = _os.environ.get("E2_OFFLINE_SWEEP_OVERRIDE", "").strip()
if _sweep_override:
    try:
        E2_OFFLINE_SWEEP = tuple(int(x) for x in _sweep_override.split(","))
    except ValueError:
        E2_OFFLINE_SWEEP = _default_sweep
else:
    E2_OFFLINE_SWEEP = _default_sweep  # N_D: number of offline demonstrations
# Declared for reference: no code reads this tuple, since E2_online below
# fixes M=5.
E2_ONLINE_SWEEP = (1, 3, 5, 10)      # M (agent.num_model_attempts)


@dataclass(frozen=True)
class Atom:
    """One evaluation episode of one experiment.

    ``extra`` holds the free-form Hydra overrides that distinguish this
    atom from the condition's frozen YAML defaults (for instance the
    demonstration count swept by E2 offline). It is what makes two atoms
    of the same (env, condition, seed, episode) hash differently.
    """

    exp_id: str
    env: str
    condition: str
    seed: int
    episode_idx: int
    llm_model: Optional[str]
    extra: Dict[str, Any]


def _materialise(
    exp_id: str,
    envs: Sequence[str],
    conditions: Sequence[str],
    *,
    llm_model: Optional[str] = DEFAULT_LLM,
    extra: Optional[Dict[str, Any]] = None,
    seeds: Sequence[int] = SEEDS_DEFAULT,
    episodes: int = EPISODES_PER_SEED,
) -> List[Atom]:
    """Expand the cross-product (env x condition x seed x episode) into atoms.

    Conditions that never call a language model carry ``llm_model=None``,
    which keeps them out of the per-model dedup and cost accounting.
    """
    atoms: List[Atom] = []
    for env in envs:
        for cond in conditions:
            uses_llm = cond in ("curtis_their", "ours")
            effective_llm = llm_model if uses_llm else None
            for seed in seeds:
                for ep in range(episodes):
                    atoms.append(Atom(
                        exp_id=exp_id,
                        env=env,
                        condition=cond,
                        seed=seed,
                        episode_idx=ep,
                        llm_model=effective_llm,
                        extra=dict(extra or {}),
                    ))
    return atoms


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def E1(envs: Sequence[str] = ENVS_E1) -> List[Atom]:
    """E1 — Reward / Success Rate, 4 envs, 5 conditions."""
    conditions = (
        "curtis_their", "curtis_hardcoded",
        "ours",
        "random", "tabular",
    )
    return _materialise("E1", envs, conditions)


def E2_offline(envs: Sequence[str] = ENVS_PRIO) -> List[Atom]:
    """E2 offline — sample efficiency in the number of demonstrations N_D.

    One sub-experiment per value of ``E2_OFFLINE_SWEEP``, each tagged with
    its own ``exp_id`` (``E2_offline_ND<n>``) so the sweep points stay
    separable in the registry.
    """
    atoms: List[Atom] = []
    for n_d in E2_OFFLINE_SWEEP:
        if n_d == 5:
            # No-op branch: a sweep point that duplicates an already-run
            # configuration is still emitted as an atom, so the registry
            # records it as skipped_dedup instead of leaving a hole in the
            # sweep. Deduplication happens in the registry, not here.
            pass
        extra = {"agent.num_input_examples": int(n_d)}
        atoms.extend(_materialise(
            f"E2_offline_ND{n_d}",
            envs,
            ("curtis_their", "ours", "tabular"),
            extra=extra,
        ))
    return atoms


def E2_online(envs: Sequence[str] = ENVS_PRIO) -> List[Atom]:
    """E2 online — learning curve over K=10 episodes, M=5 attempts, N_D=2.

    The plotted curve is ``result.json["rewards"][k]`` for k = 0..9, that
    is the reward of each successive episode within a single run, which is
    what makes online model refinement visible. The attempt budget M is
    held fixed rather than swept.

    The 10 episodes belong to one subprocess, so ``episodes=10`` here also
    forces ``num_episodes=10`` through ``extra``.
    """
    extra = {
        "agent.num_model_attempts": 5,
        "agent.num_online_model_attempts": 5,
        "agent.num_input_examples": 2,
        "num_episodes": 10,
    }
    return _materialise(
        "E2_online_K10",
        envs,
        ("curtis_their", "ours", "tabular"),
        extra=extra,
        episodes=10,
    )


def E2b_stochastic(envs: Sequence[str] = ("lava_stoch", "unlock_stoch")) -> List[Atom]:
    """E2b — stochastic variants that re-randomise the layout on every reset.

    In these environments the agent position, goal and direction are drawn
    again at each reset, so a model that memorised one layout cannot carry
    over. Three such environments exist (``lava_stoch``,
    ``four_rooms_stoch``, ``unlock_stoch``); the paper reports the first
    and the third.
    """
    conditions = ("curtis_their", "ours", "random", "tabular")
    return _materialise("E2b", envs, conditions)


def E4_llm_variation(envs: Sequence[str] = ENVS_PRIO) -> List[Atom]:
    """E4 — the proposed method under a small and a frontier LLM.

    The default LLM arm is not re-run here: its episodes already exist
    under E1/E2 with an identical configuration, and the plotting script
    reads them from there.
    """
    atoms: List[Atom] = []
    for llm in E4_LLMS:
        atoms.extend(_materialise(
            f"E4_{_slug(llm)}",
            envs,
            ("ours",),
            llm_model=llm,
        ))
    return atoms


# ---------------------------------------------------------------------------
# Experiment lookup table
# ---------------------------------------------------------------------------
def all_experiments() -> Dict[str, callable]:
    """Map the experiment names accepted on the command line to builders."""
    return {
        "E1":          E1,
        "E2_offline":  E2_offline,
        "E2_online":   E2_online,
        "E2b":         E2b_stochastic,
        "E4":          E4_llm_variation,
    }


def build_atoms(
    exp_name: str,
    envs: Optional[Sequence[str]] = None,
) -> List[Atom]:
    """Build the atoms of one experiment, or of every experiment for "all".

    ``envs``, when given, replaces the experiment's own default
    environment list; this is how ``--envs`` narrows a run.
    """
    table = all_experiments()
    if exp_name == "all":
        out: List[Atom] = []
        for name, fn in table.items():
            out.extend(fn() if envs is None else fn(envs))
        return out
    if exp_name not in table:
        raise KeyError(
            f"Unknown experiment {exp_name!r}. Known: {sorted(table) + ['all']}"
        )
    fn = table[exp_name]
    return fn() if envs is None else fn(envs)


def _slug(name: str) -> str:
    """Turn a model id into an identifier safe for exp_ids and directory names."""
    return name.replace("/", "__").replace(".", "_").replace("-", "_")
