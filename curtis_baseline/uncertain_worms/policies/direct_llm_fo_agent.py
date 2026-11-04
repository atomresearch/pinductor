import logging
import os
import random
from typing import Any, List

from uncertain_worms.environments import *
from uncertain_worms.policies.base_policy import PROMPT_DIR, Policy, requery
from uncertain_worms.structs import ActType, Episode, ObsType, ReplayBuffer, StateType
from uncertain_worms.utils import PROJECT_ROOT

log = logging.getLogger(__name__)

NEXT_ACTION_NAME = "next_action"


class DirectLLMFullyObsPolicy(Policy[StateType, ActType, ObsType]):
    def __init__(
        self,
        *args: Any,
        history_length: int = 1,
        env_code_path: str = "",
        env_description: str = "",
        dataset_path: str = "",
        **kwargs: Any,
    ) -> None:
        self.last_file_episodes: List[Episode] = []
        self.history_length = history_length
        self.env_description = env_description
        self.env_code_path = env_code_path
        with open(os.path.join(PROJECT_ROOT, env_code_path, "api.py")) as f:
            self.code_api = f.read().strip()

        self.state_history: List[StateType] = []
        self.action_history: List[ActType] = []
        self.iter_num = 0

        # Train
        train_replay_buffer = ReplayBuffer[StateType, int, ObsType].load_from_file(
            os.path.join(PROJECT_ROOT, dataset_path)
        )
        self.init_types(train_replay_buffer)
        self.update_models(replay_buffer=train_replay_buffer, iter_num=0)

        super().__init__(*args, **kwargs)

    def print_episode(self, episode: Episode) -> str:
        return_str = "Initial State: \n"
        return_str += "{}\n".format(str(episode.previous_states[0]))
        for action, state in zip(episode.actions, episode.next_states):
            return_str += "Action: \n"
            return_str += "```python\nnext_action = {}\n```".format(str(action))
            return_str += "State: \n"
            return_str += "{}\n".format(str(state))
        return return_str

    def get_next_action(self, state: StateType) -> ActType:
        prompt_file = os.path.join(PROMPT_DIR, "direct_llm_prompt.txt")

        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_template = file.read()

        example_string = "".join(
            [self.print_episode(e) for e in self.last_file_episodes]
        )

        state_history = self.state_history + [state]
        current_episode_str = self.print_episode(
            Episode(state_history, state_history[1:], self.action_history)
        )

        direct_llm_templates = {
            "exp": example_string,
            "current_episode": current_episode_str,
            "code_api": self.code_api,
            "env_description": self.env_description,
        }

        prompt = prompt_template.format(**direct_llm_templates)
        messages = [{"role": "system", "content": prompt}]

        code_str, local_scope = requery(
            messages,
            NEXT_ACTION_NAME,
            self.iter_num,
            exec_attempt=0,
            step_num=len(self.action_history),
            replay_path=self.replay_path,
        )
        if (
            code_str is not None
            and NEXT_ACTION_NAME in local_scope
            and local_scope[NEXT_ACTION_NAME] in self.actions
        ):
            action = local_scope[NEXT_ACTION_NAME]
        else:
            log.info("Couldn't parse next_action, selecting random action.")
            action = random.choice(self.actions)

        self.state_history.append(state)
        self.action_history.append(action)

        return action

    def reset(self) -> None:
        self.state_history = []
        self.action_history = []

    def init_types(self, replay_buffer: ReplayBuffer) -> None:
        assert len(replay_buffer.episodes) > 0

        self.act_type = type(replay_buffer.episodes[0].actions[0])
        self.obs_type = type(replay_buffer.episodes[0].next_observations[0])
        self.state_type = type(replay_buffer.episodes[0].next_states[0])

    def update_models(
        self,
        replay_buffer: ReplayBuffer,
        iter_num: int,
    ) -> None:
        self.iter_num = iter_num
        start_index = max(0, len(replay_buffer.episodes) - self.history_length)
        self.last_file_episodes = replay_buffer.episodes[start_index:]
