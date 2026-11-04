from __future__ import annotations

from typing import Callable

from uncertain_worms.environments.popgym.popgym_env import PopGymState


def make_popgym_state_sampler(reference_state: PopGymState) -> Callable[[], PopGymState]:
    """Return a conservative sampler for generic PopGym states.

    Most useful PopGym planning runs should learn/provide an initial model.
    This sampler exists so the agent can be configured uniformly across
    environment families before a task-specific sampler is available.
    """

    def _sample() -> PopGymState:
        return reference_state.copy()

    return _sample
