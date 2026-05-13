import copy
import linecache
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

from uncertain_worms.environments import *
from uncertain_worms.structs import (
    ActType,
    InitialModel,
    ObservationModel,
    ObsType,
    ReplayBuffer,
    RewardModel,
    StateType,
    TransitionModel,
    TypeTuple,
)
from uncertain_worms.utils import (
    PROJECT_ROOT,
    get_log_dir,
    parse_code,
    parse_prompt,
    query_llm,
    save_log,
    write_prompt,
)

log = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(PROJECT_ROOT, "policies/prompts")


class Policy(ABC, Generic[StateType, ActType, ObsType]):
    def __init__(
        self,
        actions: List[ActType],
        fully_obs: bool,
        writer: SummaryWriter,
        max_steps: int = -1,
        replay_path: Optional[str] = None,
    ):
        self.actions = actions
        self.fully_obs = fully_obs
        self.writer = writer
        self.max_steps = max_steps
        self.replay_path = replay_path
        self.type_tuple: Optional[TypeTuple] = None

    def online_update_models(self, replay_buffer: ReplayBuffer, episode: int) -> None:
        pass

    @abstractmethod
    def get_next_action(self, arg: Any) -> ActType:
        ...

    @abstractmethod
    def reset(self) -> None:
        pass

    def init_types(self, replay_buffer: ReplayBuffer) -> None:
        assert len(replay_buffer.episodes) > 0
        self.type_tuple = (
            type(replay_buffer.episodes[0].next_states[0]),
            type(replay_buffer.episodes[0].next_observations[0]),
        )


### Translators: The LLM may use a different state/observation representation than we use in code.
# These wrappers help transform the LLM-generated models into models that our planners can use


def initial_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> InitialModel[StateType]:
    def initial_model(empty_state: StateType) -> StateType:
        locals().update(locals_scope)
        initial_state = locals()[func_name](empty_state)
        return type_tuple[0].decode(copy.deepcopy(initial_state))

    return initial_model


def transition_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> TransitionModel[StateType, ActType]:
    def transition_model(state: StateType, action: ActType) -> StateType:
        locals().update(locals_scope)
        next_state = locals()[func_name](copy.deepcopy(state).encode(), action)
        return type_tuple[0].decode(copy.deepcopy(next_state))

    return transition_model


def reward_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> RewardModel[StateType, ActType]:
    def reward_model(
        state: StateType, action: ActType, next_state: StateType
    ) -> Tuple[float, bool]:
        locals().update(locals_scope)
        reward, done = locals()[func_name](
            copy.deepcopy(state).encode(), action, copy.deepcopy(next_state).encode()
        )
        return reward, done

    return reward_model


def observation_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> ObservationModel[StateType, ActType, ObsType]:
    def observation_model(
        state: StateType, action: ActType, empty_obs: ObsType
    ) -> ObsType:
        locals().update(locals_scope)
        obs = locals()[func_name](copy.deepcopy(state).encode(), action, empty_obs)
        return type_tuple[1].decode(copy.deepcopy(obs))

    return observation_model


def requery(
    messages: List[Dict[str, str]],
    function_name: str,
    iter_num: int,
    exec_attempt: int,
    step_num: int = 0,
    max_attempts: int = 5,
    replay_path: Optional[str] = None,
    api: Optional[str] = None,
    episode:int = 0
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Persistent querying until the code parses and executes without error
    Returns the code string if successful, and None otherwise."""

    for i in range(max_attempts):
        input_fn = f"episode_{episode}_iter_{iter_num}_step_{step_num}_{function_name}_exec_{exec_attempt}_attempt_{i}_llm_input.txt"
        output_fn = f"episode_{episode}_iter_{iter_num}_step_{step_num}_{function_name}_exec_{exec_attempt}_attempt_{i}_llm_output.txt"
        log.info(f"Code Gen Attempt {i} ...")

        if replay_path is not None and os.path.exists(
            os.path.join(replay_path, input_fn)
        ):
            replay_input_fn = os.path.join(replay_path, input_fn)
            log.info(f"Replaying {replay_input_fn}")
            messages = parse_prompt(replay_input_fn)

        write_prompt(input_fn, messages)

        code = None
        if replay_path is not None:
            full_replay_output_fn = os.path.join(get_log_dir(), replay_path, output_fn)
            if os.path.isfile(full_replay_output_fn):
                with open(full_replay_output_fn, "r") as file:
                    code = file.read()

        if code is None:
            code, _ = query_llm(messages)

        save_log(output_fn, code)

        messages.append(
            {
                "role": "assistant",
                "content": code,
            }
        )

        code_str = parse_code(code)

        if code_str is None:
            log.info("Parse fail.")
            messages.append(
                {
                    "role": "user",
                    "content": f"Failed to parse python code block for {function_name}",
                }
            )
            continue
        else:
            try:
                log.info(os.path.join(get_log_dir(), output_fn.replace(".txt", ".py")))

                local_scope: Any = {}
                uid = "_".join([function_name + str(iter_num) + str(step_num) + str(i)])

                filename = f"generated_code_{uid}.py"
                code_obj = compile(code_str, filename=filename, mode="exec")
                linecache.cache[filename] = (
                    len(code_str),
                    None,
                    code_str.splitlines(keepends=True),
                    filename,
                )
                exec(code_obj, globals(), local_scope)

                # Check if the desired function is in the generated names
                if function_name not in local_scope.keys():
                    not_generated_message = f"Warning: The desired function '{function_name}' was not generated."
                    messages.append({"role": "user", "content": not_generated_message})
                    log.info(not_generated_message)

                # Check for conflicts
                module_name = "environments"

                # Get only functions and classes that come from the specified module
                conflicting_names = [
                    name
                    for name, obj in globals().items()
                    if getattr(obj, "__module__", "").startswith(module_name)
                ]

                if conflicting_names:
                    conflict_error_message = f"Error: The following function/class names already exist: {conflicting_names}"
                    messages.append({"role": "user", "content": conflict_error_message})
                    log.info(conflict_error_message)
                    continue

                log.info("Parse and exec success.")
                return code_str, local_scope
            except:
                log.info("Exec fail.")
                messages.append({"role": "user", "content": traceback.format_exc()})
                continue
    return None, {}
