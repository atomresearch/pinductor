from __future__ import annotations

import copy
import logging
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
        # Paper-run reproducibility : mêmes ep_seed que ours main.py (ligne 161+).
        # Sans cela, curtis utilisait `np.random.randint(0, 10000)` → grille
        # d'éval différente à chaque run → comparaison ours↔curtis non atomique.
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


@hydra.main(
    version_base=None,
)
def main(cfg: Config) -> None:
    run_app(cfg)


if __name__ == "__main__":
    main()
