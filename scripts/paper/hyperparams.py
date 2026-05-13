"""Runtime overrides paper — minimal par design.

Depuis le refactor "modular YAML", **tous** les hyperparams scientifiques
(``max_attempts``, ``num_particles``, ``entropy_coeff``, ``use_ucb1_selection``,
``num_model_attempts``, ``lambda_coeff``, ``max_expansions``, ``fully_obs``,
``dataset_path``, ``num_episodes``, ``max_steps``, ``env_name``…) sont figés
dans les YAMLs auto-générés sous ``scripts/paper/configs/<condition>/<env>.yaml``
par ``build_configs.py``.

Ce module ne gère plus QUE le runtime :
  - métadonnées ``+extras.*`` pour wandb (approach, environment, experiment).
  - réinjection ``agent.dataset_path`` pour les **sweeps E2 offline** qui varient
    le N_D (2/4/6/8/10).
  - réinjection ``agent.num_model_attempts`` pour les **sweeps E2 online** qui
    varient le M (1/3/5/10).
  - ``num_episodes`` forcé pour les smoke tests (sinon hérité du YAML = 3).
  - filtrage pour random/tabular qui n'acceptent pas les kwargs LLM.
  - ``hp_hash`` reste la clé de dédup inter-exp du registre SQLite.

Le modèle LLM (``llm_model``) n'est JAMAIS passé en override Hydra car
curtis_baseline ne l'accepte pas dans son constructeur. Il est injecté via
la variable d'environnement ``PAPER_LLM_MODEL`` lue par ``utils.py`` côté
ours et curtis.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Mapping env logique → nom MiniGrid (utilisé pour résoudre dataset_path)
# ---------------------------------------------------------------------------
ENV_NAMES: Dict[str, str] = {
    "lava":              "MyMiniGrid-LavaWall-v0",
    "corners":           "CornerGoalRandom-Empty-10x10-v0",
    "four_rooms":        "MyMiniGrid-FourRooms-v0",
    "unlock":            "MyUnlockEnv-v0",
    "lava_stoch":        "MyMiniGrid-StochasticLavaWall-v0",
    "four_rooms_stoch":  "MyMiniGrid-StochasticFourRooms-v0",
    "unlock_stoch":      "MyMiniGrid-StochasticUnlock-v0",
}

CONDITIONS = ("ours", "curtis_their", "curtis_hardcoded", "random", "tabular")

# Conditions qui consomment dataset_path via leur constructeur agent.
_DATASET_CONSUMERS = ("curtis_their", "ours", "tabular")
# Conditions LLM (reçoivent num_model_attempts via sweep E2_online)
_LLM_CONDITIONS = ("curtis_their", "ours")

# Tous les N_D valides pour les datasets générés par regenerate_datasets.py.
_VALID_N_DEMOS = (1, 2, 3, 4, 5, 6, 8, 10, 12)

# ---------------------------------------------------------------------------
# Défaut LLM (overridé par E4)
# ---------------------------------------------------------------------------
DEFAULT_LLM = "qwen/qwen3.6-plus"


def resolve_overrides(
    env: str,
    condition: str,
    *,
    llm_model: str = DEFAULT_LLM,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Retourne le dict plat des overrides **runtime** pour un atome.

    Le YAML ``scripts/paper/configs/<condition>/<env>.yaml`` fournit déjà tous
    les hyperparams scientifiques. On ne rajoute ici que :
      - ``+extras.*`` pour le logging wandb.
      - les paramètres **swept** (``agent.dataset_path`` pour E2 offline,
        ``agent.num_model_attempts`` pour E2 online).
      - ``num_episodes`` si explicitement forcé dans ``extra`` (smoke test).

    ``llm_model`` est **non-injecté** (Hydra) — il transite via la variable
    d'environnement ``PAPER_LLM_MODEL`` (cf ``paper_runner.launch_group``).
    """
    if env not in ENV_NAMES:
        raise KeyError(f"Unknown env: {env!r}. Known: {sorted(ENV_NAMES)}")
    if condition not in CONDITIONS:
        raise KeyError(
            f"Unknown condition: {condition!r}. Known: {list(CONDITIONS)}"
        )

    overrides: Dict[str, Any] = {}
    extra = dict(extra or {})
    env_name = ENV_NAMES[env]

    # Métadonnées wandb (toujours)
    overrides["+extras.approach"] = (
        "ours" if condition.startswith("ours") else condition
    )
    overrides["+extras.environment"] = env
    overrides["+extras.experiment"] = "paper"

    # Sweep E2 offline : N_D → dataset_path (uniquement pour conditions qui
    # acceptent dataset_path). Si non-swept, le dataset_path du YAML (pointant
    # vers _paper_N10.pkl) est conservé.
    n_demos = int(extra.pop("agent.num_input_examples", 0)) if \
        "agent.num_input_examples" in extra else None
    if n_demos is not None and n_demos not in _VALID_N_DEMOS:
        # Fail loud: if a new N_D is added to E2_OFFLINE_SWEEP, the
        # dataset .pkl and ``_VALID_N_DEMOS`` must both be updated.
        raise ValueError(
            f"agent.num_input_examples={n_demos} not in _VALID_N_DEMOS="
            f"{_VALID_N_DEMOS}. Generate the corresponding _paper_N{n_demos}.pkl "
            f"and add the value to _VALID_N_DEMOS in scripts/paper/hyperparams.py."
        )
    if n_demos is not None and condition in _DATASET_CONSUMERS:
        overrides["agent.dataset_path"] = (
            f"environments/minigrid/trajectory_data/{env_name}_paper_N{n_demos}.pkl"
        )

    # Sweep E2 online : num_model_attempts (uniquement LLM conditions)
    M = extra.pop("agent.num_model_attempts", None)
    if M is not None and condition in _LLM_CONDITIONS:
        overrides["agent.num_model_attempts"] = M

    # Smoke test / mini : num_episodes forcé
    if "num_episodes" in extra:
        overrides["num_episodes"] = extra.pop("num_episodes")

    # Toute autre clé dans ``extra`` est passée telle-quelle (Hydra override
    # libre). À utiliser avec précaution : le YAML du paper est la source de
    # vérité pour les hyperparams figés.
    for k, v in extra.items():
        overrides[k] = v

    return overrides


def hp_hash(overrides: Dict[str, Any]) -> str:
    """SHA256 JSON-canonicalisé des overrides → clé de dédup inter-exp.

    Stable : même dict → même hash. Sensible à toute modification (clé, valeur,
    type). Utilisé comme 6e composante de la clé unique du registre.
    """
    canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def as_hydra_args(overrides: Dict[str, Any]) -> list[str]:
    """Convertit les overrides en liste d'args CLI Hydra.

    Les YAML du projet sont en mode struct (refusent les clés non déclarées).
    On utilise donc ``++key=value`` (override-or-append) pour les clés
    standard.

    Les clés préfixées ``+`` (ex. ``+extras.approach``) utilisent la syntaxe
    append-pure. Les clés ``hydra.*`` et certaines clés natives
    (``env_name``, ``seed``, ``num_episodes``, ``max_steps``) sont toujours
    présentes dans les YAML paper → ``key=value`` simple.

    ``True/False`` → ``true/false`` (convention Hydra/OmegaConf).
    """
    out: list[str] = []
    for k, v in overrides.items():
        if isinstance(v, bool):
            val = "true" if v else "false"
        else:
            val = str(v)
        if k.startswith("+"):
            prefix = ""
        elif k.startswith(("hydra.",)) or k in (
            "env_name", "seed", "num_episodes", "max_steps",
        ):
            prefix = ""
        else:
            prefix = "++"
        out.append(f"{prefix}{k}={val}")
    return out
