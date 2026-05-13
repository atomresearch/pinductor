import logging
import random
from typing import Any, Optional, Union

from uncertain_worms.policies.base_policy import Policy
from uncertain_worms.structs import ActType, ObsType, StateType

log = logging.getLogger(__name__)


class RandomPolicy(Policy[StateType, ActType, ObsType]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_next_action(self, obs: Union[StateType, Optional[ObsType]]) -> ActType:
        return random.choice(self.actions)

    def reset(self) -> None:
        pass

    def update_models(self, *args: Any, **kwargs: Any) -> None:
        pass
