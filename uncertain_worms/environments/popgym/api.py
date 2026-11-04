# type: ignore
# LLM context template for PopGym wrappers. The authoritative runtime
# implementation lives in popgym_env.py.
#
# IMPORTANT FOR GENERATED CODE:
# - Always return PopGymState(...) from initial_func / transition_func.
# - Always return PopGymObservation(...) from observation_func.
# - reward_func must return (float_reward, bool_done).
# - Do NOT return raw dict/list/tuple state objects directly.
# - Do NOT use numpy arrays in generated states/observations.
# - State/observation values should be simple Python scalars or nested
#   tuples made from int, float, str, bool, or None.
# - If you need a dictionary-like value, convert it to a sorted tuple of
#   key/value pairs with freeze_value.
# - initial_func should sample plausible initial hidden states rather than
#   always returning the all-zero state.
# - observation_func must handle action=None for reset observations.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


SimpleValue = int | float | str | bool | None | Tuple[Any, ...]


class State:
    pass


class Observation:
    pass


def freeze_value(value: Any) -> SimpleValue:
    """Return a hashable, deterministic representation.

    Use this helper when copying or modifying ``state.value`` or
    ``obs.value``. The runtime will also normalize values, but using this
    in generated code prevents common syntax/type errors.

    Examples:
        freeze_value([1, 2, 3]) -> (1, 2, 3)
        freeze_value({"card": 2, "prev": 1})
            -> (("card", 2), ("prev", 1))
    """
    if isinstance(value, dict):
        return tuple(
            (str(k), freeze_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(v) for v in value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True)
class PopGymState(State):
    """Hidden Markov state exposed by PopGym's Markovian info wrapper.

    The only field is ``value``.

    Example:
        PopGymState(value=(current_card, previous_card, t))

    Avoid:
        {"current": current_card}          # raw dict is not a State
        PopGymState(value=np.array(...))   # numpy arrays are fragile
    """

    value: SimpleValue


@dataclass(frozen=True)
class PopGymObservation(Observation):
    """Flattened/discrete PopGym observation.

    The only field is ``value``. In this config, observations are already
    flattened/discrete by wrappers. Preserve the observed structure when
    possible.

    Example:
        PopGymObservation(value=(0, 3, 1))
    """

    value: SimpleValue
