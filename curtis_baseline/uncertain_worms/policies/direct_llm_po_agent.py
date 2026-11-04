import logging
import os
import random
from typing import Any, List

from uncertain_worms.environments import *
from uncertain_worms.policies.base_policy import PROMPT_DIR, Policy, requery
from uncertain_worms.policies.direct_llm_fo_agent import NEXT_ACTION_NAME
from uncertain_worms.structs import ActType, Episode, ObsType, ReplayBuffer, StateType
from uncertain_worms.utils import PROJECT_ROOT

log = logging.getLogger(__name__)


class DirectLLMPartiallyObsPolicy(Policy[StateType, ActType, ObsType]):
    def __init__(
        self,
        *args: Any,
        history_length: int = 3,
        env_code_path: str = "",
        env_description: str = "",
        dataset_path: str = "",
        **kwargs: Any,
    ) -> None:
        self.last_file_episodes: List[Episode] = []
        self.history_length = history_length
        self.env_description = env_description
        self.steps = 0
        with open(os.path.join(PROJECT_ROOT, env_code_path, "api.py")) as f:
            self.code_api = f.read().strip()

        self.initial_state = None
        self.action_history: List[ActType] = []
        self.observation_history: List[ObsType] = []
        self.iter_num = 0

        # Train
        train_replay_buffer = ReplayBuffer[StateType, int, ObsType].load_from_file(
            os.path.join(PROJECT_ROOT, dataset_path)
        )
        self.init_types(train_replay_buffer)
        self.update_models(replay_buffer=train_replay_buffer, iter_num=0)

        super().__init__(*args, **kwargs)

    def print_po_episode(self, episode: Episode, current_episode: bool = True) -> str:
        return_str = ""
        if not current_episode:
            return_str += "============\n"
            return_str += "Initial State: \n"
            return_str += "{}\n".format(str(episode.previous_states[0]))

        for i, (action, observation) in enumerate(
            zip(episode.actions, episode.next_observations)
        ):
            return_str += "\nAction: \n"
            return_str += "```python\nnext_action = {}\n```".format(str(action))
            return_str += "\nObservation: \n"
            return_str += "{}\n".format(str(observation))
            if not current_episode:
                return_str += "\nReward: \n"
                return_str += "{}\n".format(str(episode.rewards[i]))

        return return_str

    def get_next_action(self, observation: Optional[ObsType]) -> ActType:
        prompt_file = os.path.join(PROMPT_DIR, "direct_llm_prompt.txt")

        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_template = file.read()

        example_string = "\n".join(
            [
                self.print_po_episode(e, current_episode=False)
                for e in self.last_file_episodes
            ]
        )

        if observation is not None:
            observation_history = self.observation_history + [observation]
        else:
            observation_history = self.observation_history

        current_episode_str = self.print_po_episode(
            Episode(
                [],
                [],
                self.action_history,
                [],
                [],
                observation_history,
            ),
            current_episode=True,
        )

        direct_llm_templates = {
            "code_api": self.code_api,
            "exp": example_string,
            "current_episode": current_episode_str,
            "env_description": self.env_description,
        }

        prompt = prompt_template.format(**direct_llm_templates)
        messages = [{"role": "system", "content": prompt}]

        prompt = prompt_template.format(**direct_llm_templates)
        messages = [{"role": "system", "content": prompt}]

        code_str, local_scope = requery(
            messages,
            NEXT_ACTION_NAME,
            self.iter_num,
            exec_attempt=self.steps,
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

        if observation is not None:
            self.observation_history.append(observation)

        self.action_history.append(action)
        self.steps += 1
        return action

    def reset(self) -> None:
        self.initial_state = None
        self.observation_history = []
        self.action_history = []
        self.steps = 0
        self.iter_num += 1

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
