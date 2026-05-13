import os
from typing import Any

from uncertain_worms.policies.partially_obs_planning_agent import (
    PartiallyObsPlanningAgent,
)
from uncertain_worms.structs import (
    ActType,
    ObsType,
    ReplayBuffer,
    StateType,
    tabular_initial_model_gen,
    tabular_observation_model_gen,
    tabular_reward_model_gen,
    tabular_transition_model_gen,
)
from uncertain_worms.utils import PROJECT_ROOT


class TabularAgent(PartiallyObsPlanningAgent[StateType, ActType, ObsType]):
    def __init__(
        self,
        *args: Any,
        dataset_path: str = "",
        num_online_model_attempts: int = 0,
        num_model_attempts: int = 0,
        num_input_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        # Tabular agents don't refine online — they fit closed-form on
        # the offline buffer. Accept the LLM-agent kwargs the E2_online
        # campaign passes uniformly (num_online_model_attempts,
        # num_model_attempts, num_input_examples) so Hydra
        # instantiation doesn't crash, and ignore them here.
        del num_online_model_attempts, num_model_attempts, num_input_examples
        super().__init__(*args, **kwargs)

        # Train
        train_replay_buffer = ReplayBuffer[StateType, int, ObsType].load_from_file(
            os.path.join(PROJECT_ROOT, dataset_path)
        )
        self.update_models(replay_buffer=train_replay_buffer, iter_num=0)

    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        raise NotImplementedError


class Tabular_I_PartiallyObsPlanningAgent(TabularAgent[StateType, ActType, ObsType]):
    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        self.init_types(replay_buffer)

        _ = self.evaluate_model("initial_model", replay_buffer)
        self.planner.initial_model = tabular_initial_model_gen(replay_buffer)


class Tabular_R_PartiallyObsPlanningAgent(TabularAgent[StateType, ActType, ObsType]):
    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        self.init_types(replay_buffer)
        _ = self.evaluate_model("reward_model", replay_buffer)
        self.planner.reward_model = tabular_reward_model_gen(replay_buffer)


class Tabular_T_PartiallyObsPlanningAgent(TabularAgent[StateType, ActType, ObsType]):
    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        self.init_types(replay_buffer)
        _ = self.evaluate_model("transition_model", replay_buffer)
        self.planner.transition_model = tabular_transition_model_gen(replay_buffer)


class Tabular_O_PartiallyObsPlanningAgent(TabularAgent[StateType, ActType, ObsType]):
    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        self.init_types(replay_buffer)
        assert self.type_tuple is not None
        self.planner.observation_model = tabular_observation_model_gen(
            replay_buffer, type_tuple=self.type_tuple
        )


class Tabular_TR_PartiallyObsPlanningAgent(TabularAgent[StateType, ActType, ObsType]):
    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        self.init_types(replay_buffer)
        _ = self.evaluate_model("transition_model", replay_buffer)
        _ = self.evaluate_model("reward_model", replay_buffer)

        self.planner.transition_model = tabular_transition_model_gen(replay_buffer)
        self.planner.reward_model = tabular_reward_model_gen(replay_buffer)


class Tabular_TROI_PartiallyObsPlanningAgent(TabularAgent[StateType, ActType, ObsType]):
    def update_models(self, replay_buffer: ReplayBuffer, iter_num: int) -> None:
        self.init_types(replay_buffer)
        assert self.type_tuple is not None

        _ = self.evaluate_model("transition_model", replay_buffer)
        _ = self.evaluate_model("reward_model", replay_buffer)
        _ = self.evaluate_model("observation_model", replay_buffer)
        _ = self.evaluate_model("initial_model", replay_buffer)

        self.planner.transition_model = tabular_transition_model_gen(replay_buffer)
        self.planner.reward_model = tabular_reward_model_gen(replay_buffer)
        self.planner.observation_model = tabular_observation_model_gen(
            replay_buffer, type_tuple=self.type_tuple
        )
        self.planner.initial_model = tabular_initial_model_gen(replay_buffer)
