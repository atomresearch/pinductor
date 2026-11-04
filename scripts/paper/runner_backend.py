"""Resolves which codebase and config a given (env, condition) launches from.

The repository ships two implementations of the ``uncertain_worms``
package: the proposed method at the repository root, and the reproduced
prior-work baseline under ``curtis_baseline/``. They use the same module
paths, so which one a subprocess imports is decided by its working
directory, not by an import path. That is the only reason ``cwd`` and
``main.py`` are tracked per condition here.

The generated YAML under ``scripts/paper/configs/<condition>/<env>.yaml``
is self-contained and freezes the hyperparameters, but its Hydra
``_target_`` names classes that exist in one codebase or the other, so
config and working directory have to agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repository root, two levels above scripts/paper/.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Generated configs: one YAML per (condition, env).
PAPER_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class LaunchSpec:
    main_py: Path          # entry point of the selected codebase
    cwd: Path              # working dir, which selects that codebase
    config_path: Path      # absolute, since Hydra resolves it from cwd
    config_name: str       # passed to --config-name, without .yaml


# -----------------------------------------------------------------------
# Which codebase each condition runs from.
# -----------------------------------------------------------------------
_OURS_REPO = REPO_ROOT
_CURTIS_REPO = REPO_ROOT / "curtis_baseline"

_CONDITION_TO_REPO = {
    "ours":              _OURS_REPO,
    "curtis_their":      _CURTIS_REPO,
    "curtis_hardcoded":  _CURTIS_REPO,
    "random":            _CURTIS_REPO,
    "tabular":           _CURTIS_REPO,
    "direct_llm_po":     _CURTIS_REPO,
    "direct_llm_fo":     _CURTIS_REPO,
}

# Per-(condition, env) exceptions to the table above. The baseline on
# Blocks runs from the repository root because the agent its config names,
# `curtis_original_agent`, is a faithful copy of the baseline that lives in
# the top-level package, so that baseline and method share one runner, one
# environment and one set of demonstrations on that domain.
_CONDITION_ENV_TO_REPO = {
    ("curtis_their", "blocks"): _OURS_REPO,
}


def spec_for(env: str, condition: str) -> LaunchSpec:
    """Resolve (env, condition) into a launchable ``LaunchSpec``.

    Raises immediately if the generated YAML is missing, so a campaign
    fails at planning time rather than after some atoms have already run.
    """
    if condition not in _CONDITION_TO_REPO:
        raise KeyError(f"Unknown condition {condition!r}")
    cwd = _CONDITION_ENV_TO_REPO.get((condition, env), _CONDITION_TO_REPO[condition])

    paper_yaml = PAPER_CONFIGS_DIR / condition / f"{env}.yaml"
    if not paper_yaml.exists():
        raise FileNotFoundError(
            f"Missing paper YAML: {paper_yaml}. "
            f"Run `python3 scripts/paper/build_configs.py` to (re)generate it."
        )

    return LaunchSpec(
        main_py=cwd / "main.py",
        cwd=cwd,
        config_path=PAPER_CONFIGS_DIR / condition,  # absolute: cwd varies
        config_name=env,
    )
