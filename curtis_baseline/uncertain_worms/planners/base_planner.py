from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Tuple

from uncertain_worms.structs import (
    ActType,
    BeliefType,
    Heuristic,
    InitialModel,
    ObservationModel,
    ObsType,
    RewardModel,
    StateType,
    TransitionModel,
    zero_heuristic,
)


class FullyObservablePlanner(ABC, Generic[StateType, ActType]):
    def __init__(
        self,
        actions: List[ActType],
        transition_model: TransitionModel,
        reward_model: RewardModel,
        *args: Any,
        heuristic: Heuristic = zero_heuristic,
        **kwargs: Any,
    ) -> None:
        self.actions = actions
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.heuristic = heuristic

    @abstractmethod
    def plan_next_action(
        self, current_state: StateType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        pass


class PartiallyObservablePlanner(ABC, Generic[StateType, ActType, ObsType, BeliefType]):
    def __init__(
        self,
        actions: List[ActType],
        transition_model: TransitionModel,
        reward_model: RewardModel,
        observation_model: ObservationModel,
        initial_model: InitialModel,
        *args: Any,
        heuristic: Heuristic = zero_heuristic,
        **kwargs: Any,
    ):
        self.actions = actions
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.observation_model = observation_model
        self.initial_model = initial_model
        self.heuristic = heuristic

    @abstractmethod
    def plan_next_action(
        self, current_belief: BeliefType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        pass
