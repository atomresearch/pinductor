"""Runtime Hydra overrides, kept deliberately minimal.

Every scientific hyperparameter (``max_attempts``, ``num_particles``,
``entropy_coeff``, ``use_ucb1_selection``, ``num_model_attempts``,
``lambda_coeff``, ``max_expansions``, ``fully_obs``, ``dataset_path``,
``num_episodes``, ``max_steps``, ``env_name``, ...) is frozen in the
generated YAML at ``scripts/paper/configs/<condition>/<env>.yaml``, which
``build_configs.py`` produces. Those YAML files are the source of truth
for what an experiment ran with; this module only adds what cannot be
known until launch time:

  - ``+extras.*`` metadata used to tag the experiment tracker run;
  - ``agent.dataset_path``, re-derived for the E2 offline sweep, whose
    whole point is to vary the demonstration count N_D;
  - ``agent.num_model_attempts`` for the E2 online sweep over M;
  - ``num_episodes`` when a caller forces a shorter run (smoke tests);
  - filtering: the random and tabular agents reject LLM-related kwargs,
    so those overrides are dropped rather than passed and crashed on.

The resulting dict is hashed by :func:`hp_hash` into the deduplication
key of the SQLite registry, so anything added here changes run identity.

The LLM id is never passed as a Hydra override: the baseline agent's
constructor does not accept it. It travels through the ``PAPER_LLM_MODEL``
environment variable, which the ``utils.py`` of both codebases reads when
selecting an engine.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Short env key -> registered MiniGrid id. The long form appears in
# dataset filenames, which is why the mapping lives here.
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

# Every condition that has configurations under scripts/paper/configs/. Keep
# this in step with runner_backend, which decides where each of them runs; a
# condition missing here is rejected at launch, and one missing there plans
# zero runs without saying why.
CONDITIONS = ("ours", "ours_hardcoded", "curtis_their", "curtis_hardcoded",
              "random", "tabular", "direct_llm_po", "direct_llm_fo")

# Conditions whose agent constructor accepts dataset_path.
_DATASET_CONSUMERS = ("curtis_their", "ours", "tabular")
# Conditions that query an LLM, and therefore accept num_model_attempts.
_LLM_CONDITIONS = ("curtis_their", "ours")

# Demonstration counts for which a dataset file exists. Must be kept in
# step with what regenerate_datasets.py writes: a value listed here but
# missing on disk fails at launch, and a value present on disk but missing
# here is rejected by resolve_overrides below.
_VALID_N_DEMOS = (1, 2, 3, 4, 5, 6, 8, 10, 12)

# ---------------------------------------------------------------------------
# LLM used everywhere except the E4 ablation, which overrides it per atom.
# ---------------------------------------------------------------------------
DEFAULT_LLM = "qwen/qwen3.6-plus"


def resolve_overrides(
    env: str,
    condition: str,
    *,
    llm_model: str = DEFAULT_LLM,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return the flat dict of runtime Hydra overrides for one atom.

    The condition's YAML already carries the scientific hyperparameters,
    so the returned dict holds only tracker metadata, the parameters a
    sweep varies, and any key an experiment passed through ``extra``.

    ``llm_model`` is accepted for symmetry but never emitted as an
    override; see the module docstring and ``paper_runner.launch_group``,
    which exports it as an environment variable instead.

    Raises:
        KeyError: on an unknown env or condition, before anything is
            launched, rather than letting Hydra fail inside a subprocess.
        ValueError: when a requested demonstration count has no dataset.
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

    # Tracker metadata, emitted for every condition. ours and
    # ours_hardcoded are reported under a single approach name.
    overrides["+extras.approach"] = (
        "ours" if condition.startswith("ours") else condition
    )
    overrides["+extras.environment"] = env
    overrides["+extras.experiment"] = "paper"

    # E2 offline sweep: N_D selects a dataset file. When the atom does not
    # sweep N_D, the YAML's own dataset_path is left untouched.
    n_demos = int(extra.pop("agent.num_input_examples", 0)) if \
        "agent.num_input_examples" in extra else None
    if n_demos is not None and n_demos not in _VALID_N_DEMOS:
        # Fail loudly instead of falling back to a default dataset. A
        # silent fallback is undetectable downstream: the atom keeps the
        # N_D it asked for in its exp_id while reading another dataset,
        # and the registry then holds sweep points that are duplicates of
        # each other. Adding a sweep value means adding both the .pkl and
        # an entry in _VALID_N_DEMOS.
        raise ValueError(
            f"agent.num_input_examples={n_demos} not in _VALID_N_DEMOS="
            f"{_VALID_N_DEMOS}. Generate the corresponding _paper_N{n_demos}.pkl "
            f"and add the value to _VALID_N_DEMOS in scripts/paper/hyperparams.py."
        )
    if n_demos is not None and condition in _DATASET_CONSUMERS:
        overrides["agent.dataset_path"] = (
            f"environments/minigrid/trajectory_data/{env_name}_paper_N{n_demos}.pkl"
        )

    # E2 online sweep. Popped for every condition, but only re-emitted for
    # the LLM ones: the others would reject the kwarg at construction.
    M = extra.pop("agent.num_model_attempts", None)
    if M is not None and condition in _LLM_CONDITIONS:
        overrides["agent.num_model_attempts"] = M

    # Only forced when a caller asks for it; otherwise the YAML decides.
    if "num_episodes" in extra:
        overrides["num_episodes"] = extra.pop("num_episodes")

    # Remaining keys pass through as free Hydra overrides. They change the
    # hp_hash and therefore run identity, while leaving no trace in the
    # YAML that documents the condition, so prefer editing the YAML for
    # anything meant to be permanent.
    for k, v in extra.items():
        overrides[k] = v

    return overrides


def hp_hash(overrides: Dict[str, Any]) -> str:
    """Hash an override dict into the registry's deduplication key.

    Canonical JSON, so key insertion order does not matter, but any change
    of key, value or value type yields a different hash. This is the sixth
    component of the registry's unique key: two experiments that resolve
    to the same overrides share their episodes instead of re-running them.
    """
    canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def as_hydra_args(overrides: Dict[str, Any]) -> list[str]:
    """Render an override dict as Hydra command-line arguments.

    The prefix carries the meaning:

      - ``++key=value`` (default) is override-or-append. The configs are
        loaded in struct mode, which rejects a plain assignment to a key
        the YAML does not declare, so this form is used for anything that
        may or may not already exist.
      - keys written with a leading ``+`` in the dict are pure appends and
        keep their own prefix.
      - ``hydra.*`` and the few keys always present in the generated YAML
        are emitted as plain ``key=value``.

    Booleans are lowercased to match OmegaConf's literals.
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
