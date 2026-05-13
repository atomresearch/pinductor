"""Définition des expériences papier.

Chaque expérience produit une liste d'« atomes » = tuple
``(exp_id, env, condition, seed, episode_idx, llm_model, extra_overrides)``.
Le runner itère ces atomes, calcule leur ``hp_hash`` et les soumet au registre
(avec dédup inter-exp).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:  # Support both package-style imports and direct script execution.
    from .hyperparams import DEFAULT_LLM
except ImportError:  # pragma: no cover - exercised by direct CLI usage
    from hyperparams import DEFAULT_LLM

# ---------------------------------------------------------------------------
# Constantes de protocole.
# ---------------------------------------------------------------------------
SEEDS_DEFAULT: Sequence[int] = tuple(range(10))  # 10 seeds × 3 ep = 30 ep/condition
EPISODES_PER_SEED = 3

ENVS_E1 = ("lava", "four_rooms", "unlock", "corners")
ENVS_PRIO = ("lava", "unlock")  # E2/E2b/E4 : prio lava+unlock

E4_LLMS = (
    # Petit baseline.
    # qwen/qwen3-14b : 14B, thinking native, meme famille que baseline qwen3.6-plus
    # Cap reasoning identique baseline via PAPER_REASONING_MAX_TOKENS=5000.
    "qwen/qwen3-14b",
    # Frontier comparison model for E4 with the same reasoning budget
    # as the baseline model.
    "anthropic/claude-opus-4.7",
)

_default_sweep = (2, 4, 6, 8, 10, 12)
# Opt-in env override for E2_OFFLINE_SWEEP. Set
# ``E2_OFFLINE_SWEEP_OVERRIDE=10,12`` to restrict.
import os as _os
_sweep_override = _os.environ.get("E2_OFFLINE_SWEEP_OVERRIDE", "").strip()
if _sweep_override:
    try:
        E2_OFFLINE_SWEEP = tuple(int(x) for x in _sweep_override.split(","))
    except ValueError:
        E2_OFFLINE_SWEEP = _default_sweep
else:
    E2_OFFLINE_SWEEP = _default_sweep  # N_D
E2_ONLINE_SWEEP = (1, 3, 5, 10)      # M (agent.num_model_attempts)


@dataclass(frozen=True)
class Atom:
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
# Expériences
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
    atoms: List[Atom] = []
    for n_d in E2_OFFLINE_SWEEP:
        if n_d == 5:
            # 5 est le default E1 → dédup automatique, on génère quand même
            # l'atome pour que le skipped_dedup soit tracé.
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
    """E2_online — K=10 episodes, M=5 attempts fixe, dataset offline N=2.

    On lit ``result.json[rewards][k]`` pour tracer reward(k) k=0..9
    (apprentissage online sur la duree). Pas de sweep M (peu d'info en plus).
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
    """E2b — variantes stochastiques (agent+goal/dir random init à chaque reset).

    Envs stoch disponibles (merge from upstream) :
      - lava_stoch → MyMiniGrid-StochasticLavaWall-v0
      - four_rooms_stoch → MyMiniGrid-StochasticFourRooms-v0
      - unlock_stoch → MyMiniGrid-StochasticUnlock-v0
    Priorité : lava+unlock.
    """
    conditions = ("curtis_their", "ours", "random", "tabular")
    return _materialise("E2b", envs, conditions)


def E4_llm_variation(envs: Sequence[str] = ENVS_PRIO) -> List[Atom]:
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
# Registre
# ---------------------------------------------------------------------------
def all_experiments() -> Dict[str, callable]:
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
    return name.replace("/", "__").replace(".", "_").replace("-", "_")
