"""Abstract policy interface + LLM code-generation helpers.

This module groups three responsibilities:

1. :class:`Policy` — abstract base inherited by every agent (random,
   tabular, hardcoded, Pinductor, POMDP Coder baseline). Subclasses must
   implement :meth:`Policy.get_action` and :meth:`Policy.update_belief`.

2. :func:`requery` and :func:`requery_joint` — drive the LLM proposal loop
   used by Pinductor and the POMDP Coder baseline:
     * ``requery`` issues one prompt per POMDP component (transition /
       observation / reward / initial) and is used by the per-component
       baseline.
     * ``requery_joint`` issues a single prompt that returns all four
       components in one shot and is used by Pinductor (paper §4.2).

3. Translator helpers (``*_translator``) — wrap LLM-returned ``def
   transition_func(...)`` Python source into closures that the particle
   filter can call as regular functions, with diagnostic logging on
   exceptions.

Prompt templates loaded from :data:`PROMPT_DIR`
(``uncertain_worms/policies/prompts/``) are mixed with environment-specific
``po_inserts.json`` snippets at runtime.
"""
import copy
import linecache
import logging
import math
import os
import random  # exposed to LLM-generated code via exec(code_obj, globals(), local_scope)
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter

from uncertain_worms.environments import *
from uncertain_worms.structs import (
    ActType,
    InitialModel,
    Observation,
    ObservationModel,
    ObsType,
    ReplayBuffer,
    RewardModel,
    State,
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
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_top_p: Optional[float] = None,
        llm_seed: Optional[int] = None,
    ):
        self.actions = actions
        self.fully_obs = fully_obs
        self.writer = writer
        self.max_steps = max_steps
        self.replay_path = replay_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.llm_seed = llm_seed
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
    # P9 (perf): capture ``func`` once at translator-build time instead of
    # rebuilding ``locals()`` on every call. Bit-exact with the previous
    # ``locals().update(locals_scope); locals()[func_name](...)`` pattern,
    # but avoids an O(N_locals) dict copy per invocation. On a typical PF
    # eval the translator is called ~10M+ times — profiled gain ~5% wall.
    func = locals_scope[func_name]

    def initial_model(empty_state: StateType) -> StateType:
        initial_state = func(empty_state)
        return type_tuple[0].decode(copy.deepcopy(initial_state))

    return initial_model


def transition_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> TransitionModel[StateType, ActType]:
    # P9 (perf): closure direct, see ``initial_model_translator`` above.
    func = locals_scope[func_name]

    def transition_model(state: StateType, action: ActType) -> StateType:
        next_state = func(state.copy().encode(), action)
        return type_tuple[0].decode(next_state)  # decode already creates a new object

    return transition_model


def reward_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> RewardModel[StateType, ActType]:
    # P9 (perf): closure direct, see ``initial_model_translator`` above.
    func = locals_scope[func_name]

    def reward_model(
        state: StateType, action: ActType, next_state: StateType
    ) -> Tuple[float, bool]:
        # reward_func only reads state, no mutation risk
        reward, done = func(state.encode(), action, next_state.encode())
        return reward, done

    return reward_model


def observation_model_translator(
    func_name: str, locals_scope: Dict[str, Any], type_tuple: TypeTuple
) -> ObservationModel[StateType, ActType, ObsType]:
    # P9 (perf): closure direct, see ``initial_model_translator`` above.
    func = locals_scope[func_name]

    def observation_model(
        state: StateType, action: ActType, empty_obs: ObsType
    ) -> ObsType:
        obs = func(state.copy().encode(), action, empty_obs)
        return type_tuple[1].decode(obs)  # decode already creates a new object

    return observation_model


def requery(
    messages: List[Dict[str, str]],
    function_name: str,
    iter_num: int,
    exec_attempt: int,
    step_num: int = 0,
    max_attempts: int = 5, # was 20
    replay_path: Optional[str] = None,
    api: Optional[str] = None,
    episode: int = 0,
    llm_provider: str = "openrouter",
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_top_p: Optional[float] = None,
    llm_seed: Optional[int] = None,
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
            code, _ = query_llm(
                messages,
                llm_provider=llm_provider,
                llm_model=llm_model,
                temperature=llm_temperature,
                top_p=llm_top_p,
                seed=llm_seed,
            )

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
            except Exception:
                log.info("Exec fail.")
                messages.append({"role": "user", "content": traceback.format_exc()})
                continue
    return None, {}

def requery_joint(
    messages: List[Dict[str, str]],
    function_name: str,
    iter_num: int,
    exec_attempt: int,
    step_num: int = 0,
    max_attempts: int = 5, # was 20
    replay_path: Optional[str] = None,
    api: Optional[str] = None,
    episode: int = 0,
    llm_provider: str = "openrouter",
    llm_model: Optional[str] = None,
    required_functions: Optional[List[str]] = None,
    llm_temperature: Optional[float] = None,
    llm_top_p: Optional[float] = None,
    llm_seed: Optional[int] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Persistent querying until the code parses and executes without error
    Returns the code string if successful, and None otherwise."""

    effective_required_functions = (
        required_functions
        if required_functions is not None
        else ["observation_func", "transition_func", "reward_func"]
    )

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
            code, _ = query_llm(
                messages,
                llm_provider=llm_provider,
                llm_model=llm_model,
                temperature=llm_temperature,
                top_p=llm_top_p,
                seed=llm_seed,
            ) # retunrs the entire LLM response (not just the code) and the execution time

        save_log(output_fn, code) # we save the entire LLM response to the output txt file 

        messages.append(
            {
                "role": "assistant",
                "content": code,
            }
        )

        code_str = parse_code(code) # we parse the code from the entire LLM response to get the code string ; single python block format

        if code_str is None:
            log.info("Parse fail.")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Failed to parse python code block. Expected functions: "
                        + ", ".join(effective_required_functions)
                    ),
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

                # Check if the any of the desired functions is missing from the generated names
                missing_functions = [
                    f for f in effective_required_functions if f not in local_scope.keys()
                ]
                if missing_functions:
                    not_generated_message = f"Warning: The following required functions were not generated: {missing_functions}"
                    messages.append({"role": "user", "content": not_generated_message})
                    log.info(not_generated_message)
                    continue

                # Check for conflicts
                module_name = "environments"

                # Get only functions and classes that come from the specified module
                conflicting_names = [
                    name
                    for name, obj in globals().items()
                    if getattr(obj, "__module__", "").startswith(module_name)
                ]

                # conflicting_names = [name for name in local_scope if name in globals()]


                if conflicting_names:
                    conflict_error_message = f"Error: The following function/class names already exist: {conflicting_names}"
                    messages.append({"role": "user", "content": conflict_error_message})
                    log.info(conflict_error_message)
                    continue

                log.info("Parse and exec success.")
                return code_str, local_scope
            except Exception:
                log.info("Exec fail.")
                messages.append({"role": "user", "content": traceback.format_exc()})
                continue
    return None, {}
