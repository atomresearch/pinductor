from __future__ import annotations

import copy
import logging
import importlib.util
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import hydra
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from uncertain_worms.policies import Policy
from uncertain_worms.structs import Environment, Observation, ReplayBuffer, State
from uncertain_worms.utils import discounted_reward, get_log_dir

log = logging.getLogger(__name__)


class StreamToLogger:
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
    if cfg.save_log:
        setup_logger()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    writer: SummaryWriter = SummaryWriter(log_dir=os.path.join(get_log_dir(), "tensorboard"))  # type: ignore
    env: Environment = hydra.utils.instantiate(cfg.env, max_steps=cfg.max_steps)
    agent: Policy = hydra.utils.instantiate(
        cfg.agent, writer=writer, max_steps=cfg.max_steps, replay_path=cfg.replay_path
    )

    # Evaluate
    eval_replay_buffer = ReplayBuffer[State, int, Observation]()
    episode_rewards = []
    for episode in range(cfg.num_episodes):
        previous_obs: Optional[Observation] = None
        log.info(f"Episode {episode}")
        # The evaluation instances have to be the same ones the other conditions
        # see, so the episode seed is derived the same way here as in the
        # top-level entry point. This baseline originally drew a fresh random
        # seed per episode, which gave it a different evaluation grid on every
        # run and made any comparison against it meaningless.
        ep_seed = cfg.seed * cfg.num_episodes + episode
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        previous_state = env.reset(seed=ep_seed)
        log.info("Starting state: " + str(previous_state))

        agent.reset()
        terminated = False
        step = 0
        while not terminated:
            log.info(f"   Step {step}")
            if agent.fully_obs:
                action = agent.get_next_action(previous_state)
            else:
                action = agent.get_next_action(previous_obs)

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
                # Skip GIF rendering by default; opt in with ENABLE_VIZ=1.
                if os.environ.get("ENABLE_VIZ", "").strip().lower() in (
                    "1", "true", "yes", "on"
                ):
                    env.visualize_episode(
                        all_states,
                        all_obs,
                        actions=e.actions,
                        episode_num=episode,
                    )
                eval_replay_buffer.wrap_up_episode()

                agent.online_update_models(eval_replay_buffer, episode)
                break
            step += 1

        ep_reward = discounted_reward(
            copy.deepcopy(eval_replay_buffer.episodes[-1].rewards), gamma=cfg.gamma
        )
        episode_rewards.append(ep_reward)
        log.info("Episode reward: " + str(ep_reward))

        writer.add_scalar("Episode Reward", ep_reward, episode)  # type: ignore

    writer.close()  # type: ignore
    eval_replay_buffer.save_to_file(os.path.join(get_log_dir(), "replay_buffer.pkl"))  # type: ignore
    writer.add_scalar("Average Episode Reward", np.mean(episode_rewards), 0)  # type: ignore
    log.info(" Average Episode Reward: " + str(np.mean(episode_rewards)))


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
    _refuse_wrong_package()
    run_app(cfg)


if __name__ == "__main__":
    main()
