"""Hydra entrypoint shared by Pinductor and the POMDP Coder baseline.

This single file is invoked by `scripts/paper/paper_runner.py` once per
(env, condition, seed) atom. The runner picks the working directory based
on the condition (repository root for ``ours*``, ``curtis_baseline/`` for
``curtis_*``, ``random``, ``tabular`` and ``direct_llm_*``), which causes
Python to import the correct ``uncertain_worms`` package via the standard
CWD-on-path rule.

Lifecycle of a single subprocess
--------------------------------
1. Hydra resolves a YAML under ``scripts/paper/configs/<cond>/<env>.yaml``.
2. The environment is instantiated via ``hydra.utils.instantiate(cfg.env)``.
3. The agent (``cfg.agent._target_``) is instantiated with offline demos
   loaded from ``cfg.agent.dataset_path``.
4. WandB logging is opt-in (auto-disabled when ``WANDB_API_KEY`` is unset).
5. The agent runs ``cfg.num_episodes`` episodes, each capped at
   ``cfg.max_steps``. Per-episode rewards and per-call LLM token usage
   are streamed to ``stdout.log`` for ``scripts/paper/log_parser.py``.
6. On exit, the rewards and token totals are emitted as machine-readable
   markers (``[REWARDS] ...``, ``[TOKENS] ...``) that the runner reads
   back into the SQLite registry.

To launch a single atom manually, prefer the runner; if you must call
``main.py`` directly, see the README §5.3 and §8.1 (working-directory
gotcha for the POMDP Coder baseline conditions).
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import importlib.util
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from uncertain_worms.policies import Policy
from uncertain_worms.structs import Environment, Observation, ReplayBuffer, State
from uncertain_worms.utils import discounted_reward, get_log_dir, DEFAULT_MODELS, get_token_usage, reset_token_usage

import wandb

log = logging.getLogger(__name__)


class StreamToLogger:
    """File-like object that forwards everything written to it to a logger.

    Installed over ``sys.stdout`` / ``sys.stderr`` by :func:`setup_logger`,
    so that anything printed rather than logged — by a library, or by code
    the pipeline executes at runtime — still reaches the run's log file.
    """

    def __init__(self, logger: logging.Logger, log_level: int) -> None:
        self.logger: logging.Logger = logger
        self.log_level: int = log_level
        self.linebuf: str = ""

    def write(self, buf: str) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self) -> None:
        pass


def setup_logger() -> None:
    """Send all logging to ``<run dir>/output.log`` and to stdout.

    Enabled through ``cfg.save_log``. Handlers already attached to the
    root logger are removed first, so that a process which has already
    configured logging does not emit every line twice.
    """
    log_level = logging.INFO

    # Get the Hydra log directory
    log_dir = get_log_dir()
    log_file = os.path.join(log_dir, "output.log")

    # Set up the logger
    logger = logging.getLogger()
    logging.getLogger("matplotlib.font_manager").disabled = True
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)
    logger.setLevel(log_level)

    # Remove any existing handlers to prevent duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add FileHandler to logger to output logs to a file
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Add StreamHandler to logger to output logs to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logger, log_level)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


@dataclass
class Config:
    """Shape of the configuration :func:`run_app` expects.

    The values come from the YAML files Hydra composes, not from the
    defaults below; ``env`` and ``agent`` hold un-instantiated Hydra
    nodes (with a ``_target_`` key) rather than objects. ``extras`` is
    free-form metadata added by the experiment runner — it names the
    environment and the approach for logging, and is absent when
    ``main.py`` is invoked directly.
    """

    env: Any = None
    agent: Any = None
    num_episodes: int = 0
    belief: Any = None
    max_steps: int = 0
    seed: int = 0
    gamma: float = 0.9
    save_log: bool = False
    replay_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def run_app(cfg: Config) -> None:
    """Build environment and agent from ``cfg`` and run the evaluation.

    One call is one experimental condition: a fixed seed, one
    environment, one agent, ``cfg.num_episodes`` episodes played in
    sequence with an online model update after each of them. Returns
    nothing; everything the analysis needs is written to the run
    directory.
    """
    if cfg.save_log:
        setup_logger()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        import torch
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except ImportError:
        pass

    # The LLM sampling seed follows the run seed unless the agent config
    # pins it, which allows replaying the same environment draws while
    # varying the model's sampling, or the reverse.
    llm_seed = cfg.seed
    if "llm_seed" in cfg.agent and cfg.agent.llm_seed is not None:
        llm_seed = int(cfg.agent.llm_seed)
    # Configs disagree on where the environment name lives, so look for it
    # from the most specific location to the least. Only the LLM planning
    # agent is given the value (see the `agent_kwargs` block below), so an
    # empty string is harmless for every other approach.
    prompt_env_name = ""
    if "extras" in cfg and cfg.extras is not None and "environment" in cfg.extras:
        prompt_env_name = str(cfg.extras.environment)
    elif "env_name" in cfg and cfg.env_name is not None:
        prompt_env_name = str(cfg.env_name)
    elif "env" in cfg and cfg.env is not None and "env_name" in cfg.env:
        prompt_env_name = str(cfg.env.env_name)
    elif "env" in cfg and cfg.env is not None and "_target_" in cfg.env:
        prompt_env_name = str(cfg.env._target_)

    writer: SummaryWriter = SummaryWriter(log_dir=os.path.join(get_log_dir(), "tensorboard"))  # type: ignore
    env: Environment = hydra.utils.instantiate(cfg.env, max_steps=cfg.max_steps)
    agent_kwargs: Dict[str, Any] = {
        "writer": writer,
        "max_steps": cfg.max_steps,
        "replay_path": cfg.replay_path,
        "llm_seed": llm_seed,
    }
    agent_target = ""
    if "_target_" in cfg.agent:
        agent_target = str(cfg.agent._target_)
    if "LLMPartiallyObsPlanningAgent" in agent_target:
        agent_kwargs["env_name"] = prompt_env_name

    agent: Policy = hydra.utils.instantiate(
        cfg.agent,
        **agent_kwargs,
    )

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    llm_provider = cfg.agent.get("llm_provider", "openrouter") if "llm_provider" in cfg.agent else "openrouter"
    # `extras` is injected by the experiment runner but is absent from most
    # configs, which run fine on their own — read it defensively so that a
    # direct `python main.py --config-name=...` works for every condition.
    extras = config_dict.get("extras") or {}
    approach = extras.get("approach")
    config_dict["llm_engine"] = DEFAULT_MODELS.get(llm_provider, "") if approach == "ours" else None
    # Weights & Biases is optional: with no API key in the environment the
    # run is created in "disabled" mode, so the calls below become no-ops
    # and the experiment still produces its local logs.
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pinductor"),
        entity=os.getenv("WANDB_ENTITY") or None,
        mode="disabled" if not os.getenv("WANDB_API_KEY") else "online",
        config=config_dict,
        name=f"{extras.get('environment', 'run')}_{approach or 'direct'}_seed{cfg.seed}",
        notes="Learning obs, trans and reward models jointly ; known initial model ; NO STATE SPACE INFO",
        dir=get_log_dir()
    )

    reset_token_usage()

    # Evaluate
    eval_replay_buffer = ReplayBuffer[State, int, Observation]()
    episode_rewards = []
    for episode in range(cfg.num_episodes):
        previous_obs: Optional[Observation] = None
        previous_action: Optional[int] = None
        log.info(f"Episode {episode}")
        # Reset ALL RNGs before each episode so that the same seed always
        # produces the same grid, regardless of prior LLM code generation.
        # Multiply by num_episodes to ensure non-overlapping ep_seeds across seeds.
        ep_seed = cfg.seed * cfg.num_episodes + episode
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        previous_state = env.reset(seed=ep_seed)
        # The reset emits an observation, and the belief has to be conditioned on
        # it before the first action is chosen. Without this the agent starts the
        # episode from the prior alone and throws away the one observation it gets
        # for free.
        if hasattr(env, "get_last_obs"):
            previous_obs = env.get_last_obs()
        log.info("Starting state: " + str(previous_state))
        # Grid environments only: the guard keeps this loop usable with
        # state classes that have no notion of object positions.
        if hasattr(previous_state, 'get_type_indices'):
            goal_pos = previous_state.get_type_indices(10)  # ObjectTypes.goal = 10
            log.info(f"Goal position: {goal_pos}")

        agent.reset()
        terminated = False
        step = 0
        while not terminated:
            log.info(f" Step {step}")
            # In the partially observable setting the agent is handed the
            # last observation and nothing else: the environment state is
            # kept out of its reach for the whole episode.
            if agent.fully_obs:
                action = agent.get_next_action(previous_state)
            else:
                action = agent.get_next_action(previous_obs)

            # Diagnostics only. Everything below compares the agent's belief
            # against the environment state, which the agent itself never
            # sees; it is written to the log and read by nothing else. The
            # `hasattr` guards let the same loop run with agents and state
            # classes that expose none of these attributes.
            if hasattr(agent, 'current_belief') and agent.current_belief is not None:
                belief = agent.current_belief
                real_pos = tuple(previous_state.agent_pos) if hasattr(previous_state, 'agent_pos') else None
                if real_pos is not None and hasattr(belief, 'particles'):
                    # Find closest belief particle to real position
                    belief_positions = {}
                    for state_p, count in belief.particles.items():
                        pos = tuple(state_p.agent_pos) if hasattr(state_p, 'agent_pos') else None
                        if pos is not None:
                            belief_positions[pos] = belief_positions.get(pos, 0) + count
                    total = sum(belief_positions.values())
                    if total > 0 and belief_positions:
                        sorted_pos = sorted(belief_positions.items(), key=lambda x: -x[1])
                        top_pos, top_count = sorted_pos[0]
                        dist_to_real = abs(top_pos[0]-real_pos[0]) + abs(top_pos[1]-real_pos[1])
                        real_in_belief = real_pos in belief_positions
                        real_weight = belief_positions.get(real_pos, 0) / total
                        log.info(
                            f"[BELIEF_VS_REAL] real={real_pos} top_belief={top_pos}({top_count/total:.2f}) "
                            f"dist={dist_to_real} real_in_belief={real_in_belief}({real_weight:.2f})"
                        )

                # [FOV_VS_BELIEF]: for every cell of the current field of
                # view, compare the true cell value against the particle
                # distribution over that cell. This is how one checks that
                # the belief update actually kills particles contradicted by
                # the observation — for instance, if particles place a goal
                # at (8,8) and the agent observes (8,8) empty, the mass on
                # "goal at (8,8)" is expected to collapse.
                if (
                    hasattr(previous_state, 'agent_pos')
                    and hasattr(previous_state, 'agent_dir')
                    and hasattr(previous_state, 'grid')
                    and hasattr(belief, 'particles')
                    and previous_state.grid is not None
                ):
                    try:
                        # Diagnostic only. This sweeps a 5x5 window while the
                        # environments are built with agent_view_size=3, so it
                        # also reports cells the agent cannot see. It informs no
                        # decision; the belief update uses the real view.
                        view_size = 5
                        ax, ay = int(previous_state.agent_pos[0]), int(previous_state.agent_pos[1])
                        ad = int(previous_state.agent_dir)
                        half = view_size // 2
                        if ad == 0:
                            topX, topY = ax, ay - half
                        elif ad == 1:
                            topX, topY = ax - half, ay
                        elif ad == 2:
                            topX, topY = ax - view_size + 1, ay - half
                        elif ad == 3:
                            topX, topY = ax - half, ay - view_size + 1
                        else:
                            topX, topY = ax, ay
                        gw, gh = previous_state.grid.shape[0], previous_state.grid.shape[1]
                        non_empty = []
                        contradictions = []
                        n_match = 0
                        n_eval = 0
                        for dx in range(view_size):
                            for dy in range(view_size):
                                x, y = topX + dx, topY + dy
                                if not (0 <= x < gw and 0 <= y < gh):
                                    continue
                                real_v = int(previous_state.grid[x, y])
                                cell_dist: dict = {}
                                for state_p, cnt in belief.particles.items():
                                    if not hasattr(state_p, 'grid'):
                                        continue
                                    pv = int(state_p.grid[x, y])
                                    cell_dist[pv] = cell_dist.get(pv, 0) + cnt
                                total = sum(cell_dist.values())
                                if total == 0:
                                    continue
                                n_eval += 1
                                top_v = max(cell_dist, key=cell_dist.get)
                                top_pct = cell_dist[top_v] / total
                                if top_v == real_v:
                                    n_match += 1
                                if real_v != 0:
                                    non_empty.append((x, y, real_v))
                                if top_v != real_v and top_pct > 0.5:
                                    real_w = cell_dist.get(real_v, 0) / total
                                    contradictions.append({
                                        "pos": (x, y), "real": real_v,
                                        "bel_top": top_v, "bel_pct": round(top_pct, 2),
                                        "real_w": round(real_w, 2),
                                    })
                        if n_eval > 0:
                            log.info(
                                f"[FOV_VS_BELIEF] agent=({ax},{ay}) dir={ad} view={view_size} "
                                f"top_left=({topX},{topY}) seen_non_empty={non_empty} "
                                f"match={n_match}/{n_eval} ({100.0*n_match/n_eval:.0f}%) "
                                f"contradictions={contradictions[:5]}"
                            )
                    except Exception as _fov_err:
                        log.debug(f"[FOV_VS_BELIEF] failed: {_fov_err}")

            log.info("Executing action: " + str(action))
            next_obs, next_state, reward, terminated, truncated, _ = env.step(action)
            

            log.info("Next state: " + str(next_state))
            log.info("Next obs: " + str(next_obs))
            log.info("Reward: " + str(reward))
            log.info("Terminated: " + str(terminated))

            eval_replay_buffer.append_episode_step(
                previous_state,
                next_state,
                action,
                previous_obs,
                next_obs,
                reward,
                terminated,
            )

            previous_state = next_state
            previous_obs = next_obs
            if terminated or truncated:
                assert eval_replay_buffer.current_episode is not None

                e = eval_replay_buffer.current_episode
                all_states = [e.previous_states[0]] + e.next_states
                all_obs = e.next_observations
                # Episode GIFs are opt-in (ENABLE_VIZ=1). Rendering them
                # with matplotlib and imageio dominated the wall-clock of
                # short episodes (measured at roughly three quarters of a
                # run) while no result depends on them.
                if os.environ.get("ENABLE_VIZ", "").strip().lower() in (
                    "1", "true", "yes", "on"
                ):
                    env.visualize_episode(
                        all_states,
                        all_obs,
                        actions=e.actions,
                        episode_num=episode,
                    )
                # Order matters: `wrap_up_episode` moves the episode being
                # recorded into `episodes`, and that is the list the agent
                # reads when it refits its models. Updating before the call
                # would show it every episode except the one that ended.
                eval_replay_buffer.wrap_up_episode()

                agent.online_update_models(eval_replay_buffer, episode)
                break
            step += 1

        # `discounted_reward` overwrites the list it is given, so it is fed
        # a copy: the buffer keeps the raw per-step rewards it will be
        # pickled with.
        ep_reward = discounted_reward(
            copy.deepcopy(eval_replay_buffer.episodes[-1].rewards), gamma=cfg.gamma
        )
        episode_rewards.append(ep_reward)
        log.info("Episode reward: " + str(ep_reward))

        writer.add_scalar("Episode Reward", ep_reward, episode)  # type: ignore
        wandb.log({f"Episode {episode} Reward": ep_reward}) # type: ignore

    # The summary scalar below is added after this close, which is harmless:
    # SummaryWriter reopens its event file on demand, and the value is written
    # and read back like any other.
    writer.close()  # type: ignore
    eval_replay_buffer.save_to_file(os.path.join(get_log_dir(), "replay_buffer.pkl"))  # type: ignore
    writer.add_scalar("Average Episode Reward", np.mean(episode_rewards), 0)  # type: ignore
    wandb.log({"Average Episode Reward": np.mean(episode_rewards)})  # type: ignore
    wandb.save("replay_buffer.pkl")  # type: ignore

    # Log token usage
    usage = get_token_usage()
    log.info(f"[TOKENS] {usage['calls']} LLM calls, {usage['input']}in + {usage['output']}out tokens")
    wandb.log(usage)  # type: ignore

    wandb.finish()  # type: ignore
    log.info("Average Episode Reward: " + str(np.mean(episode_rewards)))


# Two packages named ``uncertain_worms`` live in this repository: this one, and
# the reused baseline under ``curtis_baseline/``. They expose classes of the
# same name under the same import path, and which one an interpreter sees
# depends only on the working directory. A configuration launched from the
# wrong one therefore does not fail: it silently runs a different
# implementation and reports numbers as if nothing were wrong.
#
# The condition a configuration belongs to is the name of its directory under
# scripts/paper/configs/, and runner_backend holds the mapping from condition
# to the package it must run against. Reading that table here keeps one source
# of truth rather than restating it.
def _refuse_wrong_package() -> None:
    import uncertain_worms

    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.dirname(os.path.realpath(uncertain_worms.__file__))
    if os.path.dirname(pkg) != here:
        raise RuntimeError(
            f"This entry point is {__file__}, but `uncertain_worms` was "
            f"imported from {pkg}. The two are different implementations. "
            f"Run this file from {here}, or use the main.py next to the "
            f"package you meant."
        )

    for condition in sorted(_conditions_from_argv()):
        expected = _expected_repo_for(condition)
        if expected is not None and os.path.realpath(expected) != here:
            raise RuntimeError(
                f"Condition {condition!r} runs against the package under "
                f"{expected}, but this entry point is {__file__}. Launch it "
                f"with that directory as the working directory; running it "
                f"here would use a different implementation without saying so."
            )


def _conditions_from_argv() -> "set[str]":
    """Every condition named by a config path on the command line.

    Hydra accepts the path under several spellings and has its own precedence
    when more than one is given. Rather than restate that precedence, collect
    all of them: if any path names a condition belonging to the other package,
    the command is refused. A command that names two different conditions is
    ambiguous anyway.
    """
    flags = ("--config-dir", "-cd", "--config-path", "-cp")
    paths = []
    for i, arg in enumerate(sys.argv):
        for flag in flags:
            if arg.startswith(flag + "="):
                paths.append(arg.split("=", 1)[1])
            elif arg == flag and i + 1 < len(sys.argv):
                paths.append(sys.argv[i + 1])

    found = set()
    for path in paths:
        # realpath, not abspath: a symlink named after some other condition
        # would otherwise report that other name and the check would look at
        # the wrong entry.
        parent, name = os.path.split(os.path.realpath(path.rstrip("/")))
        if os.path.basename(parent) == "configs":
            found.add(name)
        elif name == "configs":
            config_name = _env_from_argv() or ""
            if "/" in config_name:
                found.add(config_name.split("/", 1)[0])
    return found


def _expected_repo_for(condition: str) -> Optional[str]:
    """Where runner_backend says this condition runs, or None if unknown."""
    root = os.path.dirname(os.path.abspath(__file__))
    for candidate in (root, os.path.dirname(root)):
        backend = os.path.join(candidate, "scripts", "paper", "runner_backend.py")
        if not os.path.isfile(backend):
            continue
        spec = importlib.util.spec_from_file_location("_runner_backend", backend)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            # dataclasses resolves field types through sys.modules, so the
            # module has to be registered before it executes.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            env = _env_from_argv()
            if env is not None:
                return str(module.spec_for(env, condition).cwd)
            return str(module._CONDITION_TO_REPO[condition])
        except KeyError:
            # The path names a directory under configs/ that the table does not
            # know. Nothing to check against, so let the run proceed and fail on
            # its own terms if the configuration is wrong.
            return None
        except Exception as exc:
            # The table exists but could not be read. Refusing here rather than
            # returning None keeps a broken check from silently disabling
            # itself, which is the failure mode this whole function exists to
            # prevent.
            raise RuntimeError(
                f"Cannot determine which package condition {condition!r} runs "
                f"against: reading {backend} raised {type(exc).__name__}: "
                f"{exc}. Fix that before running, or the wrong implementation "
                f"could be used without warning."
            ) from exc
    return None


def _env_from_argv() -> Optional[str]:
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config-name="):
            return arg.split("=", 1)[1]
        if arg in ("--config-name", "-cn") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


@hydra.main(
    version_base=None,
)
def main(cfg: Config) -> None:
    """Hydra entry point.

    The decorator declares no config path or name, so both are supplied
    on the command line and every run states which condition it is::

        python main.py --config-dir=<dir> --config-name=<config>
    """
    _refuse_wrong_package()
    run_app(cfg)


if __name__ == "__main__":
    main()
