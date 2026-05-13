"""Backend : mapping (env, condition) → (config-path, config-name, cwd, main.py).

**Refactor modular YAML** : les YAML natifs des deux repos (ours + curtis)
ne sont plus lancés directement. À la place, on utilise les **YAMLs
auto-générés paper** sous ``scripts/paper/configs/<condition>/<env>.yaml``
qui sont self-contained et figent tous les hyperparams paper.

On garde cependant ``cwd`` et ``main.py`` **par repo** car le ``_target_``
Hydra des YAMLs pointe vers des classes dans des codebases différentes :
  - ``ours`` → ``uncertain_worms.*`` dans REPO_ROOT/
  - ``curtis_*`` + ``random`` + ``tabular`` → ``uncertain_worms.*`` dans
    REPO_ROOT/curtis_baseline/

Le ``config-path`` est résolu en chemin absolu vers ``scripts/paper/configs/<cond>/``
par ``paper_runner.py`` avant le subprocess Hydra.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Racine : le parent des dossiers scripts/paper/
REPO_ROOT = Path(__file__).resolve().parents[2]
# Répertoire des YAMLs paper auto-générés (1 par atome condition × env).
PAPER_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class LaunchSpec:
    main_py: Path          # chemin vers main.py
    cwd: Path              # working dir pour le process
    config_path: Path      # chemin **absolu** vers scripts/paper/configs/<cond>
    config_name: str       # passé à --config-name (sans .yaml, = <env>)


# -----------------------------------------------------------------------
# cwd / main.py mapping par condition (codebase à charger via PYTHONPATH)
# -----------------------------------------------------------------------
_OURS_REPO = REPO_ROOT
_CURTIS_REPO = REPO_ROOT / "curtis_baseline"

_CONDITION_TO_REPO = {
    "ours":              _OURS_REPO,
    "curtis_their":      _CURTIS_REPO,
    "curtis_hardcoded":  _CURTIS_REPO,
    "random":            _CURTIS_REPO,
    "tabular":           _CURTIS_REPO,
}


def spec_for(env: str, condition: str) -> LaunchSpec:
    """Résout (env, condition) en ``LaunchSpec`` prêt à être lancé.

    Tous les YAMLs paper sont self-contained ; le runner n'a plus qu'à
    passer ``seed=X`` + ``+extras.*`` (et, pour les sweeps E2, un override
    ``agent.dataset_path`` ou ``agent.num_model_attempts``).
    """
    if condition not in _CONDITION_TO_REPO:
        raise KeyError(f"Unknown condition {condition!r}")
    cwd = _CONDITION_TO_REPO[condition]

    paper_yaml = PAPER_CONFIGS_DIR / condition / f"{env}.yaml"
    if not paper_yaml.exists():
        raise FileNotFoundError(
            f"Paper YAML manquant : {paper_yaml}. "
            f"Lance `python3 scripts/paper/build_configs.py` pour (re)générer."
        )

    return LaunchSpec(
        main_py=cwd / "main.py",
        cwd=cwd,
        config_path=PAPER_CONFIGS_DIR / condition,  # absolu
        config_name=env,
    )
