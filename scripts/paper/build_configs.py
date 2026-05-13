"""Génère les 42 YAMLs paper-ready dans ``scripts/paper/configs/``.

Pour chaque (condition × env), on part du YAML **natif** du code ciblé (notre
repo pour ours, curtis_baseline pour le reste) puis on applique
**uniquement** les overrides qui font sens pour cette condition + la cible.

Règles :
  - Le YAML natif reste intact ; on crée un YAML self-contained dans
    scripts/paper/configs/<condition>/<env>.yaml.
  - Un kwarg n'est écrit dans le YAML paper que si l'agent-cible l'accepte
    (sinon on casse silencieusement l'instantiation Hydra).
  - Les valeurs "paper" alignées sur leur tableau sont figées dans le YAML,
    pas passées en override CLI.

Après ce script, le runner n'a plus qu'à faire :
  python main.py --config-path scripts/paper/configs/<cond> \
                 --config-name <env> seed=X +extras.*=...
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
# Mapping (condition, env) → YAML natif source
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
    # Les stoch YAML ours ont été créés plus tôt
    base = REPO / "uncertain_worms" / "config" / "approaches" / "ours"
    return base / f"{env}_ours.yaml"


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
# Protocole paper : max_steps par env + num_episodes
# ---------------------------------------------------------------------------
MAX_STEPS = {
    "lava": 40, "corners": 40, "four_rooms": 60, "unlock": 40,
    "lava_stoch": 40, "four_rooms_stoch": 60, "unlock_stoch": 40,
}

NUM_EPISODES = 3  # 10 seeds × 3 ep = 30 ep/condition


# ---------------------------------------------------------------------------
# Helpers pour setter une clé imbriquée dans un dict yaml
# ---------------------------------------------------------------------------
def _nested_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# Overrides paper par condition — seulement les hyperparams acceptés par la
# cible. Chaque dict = {dotted_key: value}.
# ---------------------------------------------------------------------------
def paper_overrides(condition: str, env: str) -> Dict[str, Any]:
    env_name = ENV_NATIVE_NAMES[env]

    # Protocole, commun à toutes les conditions
    common: Dict[str, Any] = {
        "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS[env],
        "env_name": env_name,
        "env.fully_obs": False,   # default PO ; seul curtis_their l'override à True
    }

    # Pour les agents PO planning (non random/tabular)
    # Valeurs alignées sur la baseline publiée pour
    # garantir l'équité de la comparaison budget-planner × particle-filter.
    # ``kernel_bandwidth`` n'existe pas côté curtis_baseline (leur distance
    # est binaire 0/inf), donc ajouté uniquement sur les branches ours plus
    # bas dans la fonction.
    #
    # ``max_attempts=50_000`` caps wasted rejuvenation time when generated
    # code has latent bugs while leaving a comfortable margin for typical
    # non-collapsed PF runs.
    po_planning: Dict[str, Any] = {
        "agent.max_attempts": 50_000,             # cap rejuv (cf comment above)
        "agent.num_particles": 10,                # baseline paper
        "agent.fully_obs": False,                  # tous PO sauf override
        "agent.planner.max_expansions": 5000,     # baseline paper
        "agent.planner.lambda_coeff": 0.1,        # baseline table
    }
    # Kernel bandwidth (soft-kernel PF) — ours only.
    ours_kb: Dict[str, Any] = {
        "agent.kernel_bandwidth": 0.2,
    }

    # Pour les agents LLM (ours + curtis_their)
    llm_agent: Dict[str, Any] = {
        "agent.num_model_attempts": 5,
        "agent.num_online_model_attempts": 5,
        "agent.dataset_path":
            f"environments/minigrid/trajectory_data/{env_name}_paper_N10.pkl",
    }

    if condition == "ours":
        o = {**common, **po_planning, **ours_kb, **llm_agent}
        # Specs méthodo ours (non présents dans YAML natif)
        o["agent.planner.entropy_coeff"] = 1.0
        o["agent.use_ucb1_selection"] = True
        o["agent.ucb1_c"] = 1.0
        o["agent.elbo_softmax_temperature"] = 0.1
        return o

    if condition == "curtis_their":
        o = {**common, **po_planning, **llm_agent}
        # PO natif curtis : leur YAML natif a fully_obs=False, leur env
        # renvoie des obs 3x3 FOV + agent_pos revele. Nos datasets
        # _paper_NX.pkl sont maintenant regeneres au meme format natif
        # (cf regenerate_datasets.py:po_obs_fn_curtis). Pas de FO post-hoc.
        o["agent.planner.entropy_coeff"] = 0.0
        return o

    if condition == "curtis_hardcoded":
        o = {**common, **po_planning}
        # Baseline hardcoded reste PO : leur métrique coverage gère les obs
        # shape discrepancies, mais par cohérence avec leur YAML natif on
        # laisse env.fully_obs=False (YAML natif curtis l'a aussi à false).
        o["agent.planner.entropy_coeff"] = 0.0
        return o

    if condition == "random":
        # RandomPolicy : n'accepte quasiment aucun kwarg pipeline
        return {**common}

    if condition == "tabular":
        # Tabular : on aligne sur ours/curtis_their pour les kwargs du PF et du
        # planner → comparabilité directe (budget planificateur identique).
        # Rappel : leurs YAML natifs ont num_particles=10 / max_expansions=1000
        # (voire 100/absent pour corners). On force l'homogénéité ici.
        o = {**common, **po_planning}
        # Les classes tabular_learners.Tabular_*_PartiallyObsPlanningAgent
        # acceptent max_attempts/num_particles/fully_obs/dataset_path
        # (voir curtis_baseline/uncertain_worms/policies/tabular_learners.py).
        o["agent.dataset_path"] = (
            f"environments/minigrid/trajectory_data/{env_name}_paper_N10.pkl"
        )
        # Tabular n'apprend pas de modèles via LLM → pas d'info gain côté
        # planner (entropy_coeff=0 pour rester cohérent avec curtis_hardcoded).
        o["agent.planner.entropy_coeff"] = 0.0
        return o

    raise ValueError(f"Unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (cond, env), src_path in SOURCE_PATH.items():
        if not src_path.exists():
            print(f"[skip] {cond}/{env}: source manquante ({src_path})")
            continue
        # Load natif
        cfg = yaml.safe_load(src_path.read_text()) or {}
        cfg = deepcopy(cfg)
        # Apply paper overrides
        for key, val in paper_overrides(cond, env).items():
            _nested_set(cfg, key, val)
        # Ensure env_name anchor updated (remove any YAML anchor &env_name)
        out_sub = OUT_DIR / cond
        out_sub.mkdir(parents=True, exist_ok=True)
        out_path = out_sub / f"{env}.yaml"
        # Écrire avec un en-tête informatif
        header = (
            f"# Auto-generated by scripts/paper/build_configs.py\n"
            f"# source: {src_path.relative_to(REPO)}\n"
            f"# condition={cond} env={env}\n"
        )
        out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        written += 1
        print(f"✓ {cond}/{env} ← {src_path.name}")
    print(f"\n[build] {written} YAMLs generated in {OUT_DIR}")
    return 0 if written == 42 else 1


if __name__ == "__main__":
    sys.exit(main())
