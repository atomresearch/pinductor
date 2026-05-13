"""Hydra entrypoint shared by Pinductor and the POMDP Coder baseline.

This single file is invoked by `scripts/paper/paper_runner.py` once per
(env, condition, seed) atom. The runner picks the working directory based
on the condition (repository root for ``ours*``, ``curtis_baseline/`` for
``curtis_*``, ``random``, ``tabular``), which causes Python to import the
correct ``uncertain_worms`` package via the standard CWD-on-path rule.

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
``main.py`` directly, see the README §5.3 and §9.2 (working-directory
gotcha for the POMDP Coder baseline conditions).
"""
from __future__ import annotations

import copy
import dataclasses
import logging
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
    try:
        import torch
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except ImportError:
        pass

    llm_seed = cfg.seed
    if "llm_seed" in cfg.agent and cfg.agent.llm_seed is not None:
        llm_seed = int(cfg.agent.llm_seed)
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
    ) # this is where we go into partially_obs_planning_agent.py, when we intialize the agent using the config file
        
    # TODO W&B logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    llm_provider = cfg.agent.get("llm_provider", "openrouter") if "llm_provider" in cfg.agent else "openrouter"
    config_dict["llm_engine"] = DEFAULT_MODELS.get(llm_provider, "") if cfg.extras.approach == "ours" else None
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pinductor"),
        entity=os.getenv("WANDB_ENTITY") or None,
        mode="disabled" if not os.getenv("WANDB_API_KEY") else "online",
        config=config_dict,
        name=f"{cfg.extras.environment}_{cfg.extras.approach}_seed{cfg.seed}",
        notes="Learning obs, trans and reward models jointly ; known initial model ; NO STATE SPACE INFO",
        dir=get_log_dir()
    )

    reset_token_usage()

    # Evaluate
    eval_replay_buffer = ReplayBuffer[State, int, Observation]()
    episode_rewards = []
    for episode in range(cfg.num_episodes): # for each episode
        previous_obs: Optional[Observation] = None # we set "previous obs" to none at start 
        previous_action: Optional[int] = None
        log.info(f"Episode {episode}")
        # Reset ALL RNGs before each episode so that the same seed always
        # produces the same grid, regardless of prior LLM code generation.
        # Multiply by num_episodes to ensure non-overlapping ep_seeds across seeds.
        ep_seed = cfg.seed * cfg.num_episodes + episode
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        previous_state = env.reset(seed=ep_seed)  # deterministic per episode for reproducibility
        log.info("Starting state: " + str(previous_state))
        # Log goal position for analysis
        if hasattr(previous_state, 'get_type_indices'):
            goal_pos = previous_state.get_type_indices(10)  # ObjectTypes.goal = 10
            log.info(f"Goal position: {goal_pos}")

        agent.reset()
        terminated = False
        step = 0
        while not terminated:
            log.info(f" Step {step}")
            if agent.fully_obs: # fully observable setting (I don't think we use that)
                action = agent.get_next_action(previous_state) # choose action using the previous state
            else:
                action = agent.get_next_action(previous_obs) # in partialy observable setting, use obs instead of state to decide action

            # Log belief vs reality comparison
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

                # Compare the current field of view against the particle
                # belief so observation/update contradictions are visible in
                # logs while debugging.
                if (
                    hasattr(previous_state, 'agent_pos')
                    and hasattr(previous_state, 'agent_dir')
                    and hasattr(previous_state, 'grid')
                    and hasattr(belief, 'particles')
                    and previous_state.grid is not None
                ):
                    try:
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
            next_obs, next_state, reward, terminated, truncated, _ = env.step(action) # we obtain next obs, state and reward after the action is done
            

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

            previous_state = next_state # for the next step
            previous_obs = next_obs # for the next step
            if terminated or truncated:
                assert eval_replay_buffer.current_episode is not None

                e = eval_replay_buffer.current_episode
                all_states = [e.previous_states[0]] + e.next_states
                all_obs = e.next_observations
                # Skip GIF rendering by default. It is useful for debugging
                # but adds avoidable overhead to experiment runs.
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

                agent.online_update_models(eval_replay_buffer, episode) # online learning after the episode is terminated
                break
            step += 1

        ep_reward = discounted_reward(
            copy.deepcopy(eval_replay_buffer.episodes[-1].rewards), gamma=cfg.gamma
        ) # compute reward for the episode
        episode_rewards.append(ep_reward)
        log.info("Episode reward: " + str(ep_reward))

        writer.add_scalar("Episode Reward", ep_reward, episode)  # type: ignore
        wandb.log({f"Episode {episode} Reward": ep_reward}) # type: ignore

    # why do we close the writer ? we still try to write some things in it after that line
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


@hydra.main(
    version_base=None,
)
def main(cfg: Config) -> None:
    run_app(cfg)


if __name__ == "__main__":
    main()
