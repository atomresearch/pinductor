from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pyvis.network import Network  # type:ignore

from uncertain_worms.planners.base_planner import PartiallyObservablePlanner
from uncertain_worms.policies.base_policy import (
    PROMPT_DIR,
    Policy,
    initial_model_translator,
    observation_model_translator,
    requery,
    reward_model_translator,
    transition_model_translator,
)
from uncertain_worms.structs import (
    ActType,
    InitialModel,
    ObsType,
    ParticleBelief,
    ReplayBuffer,
    StateType,
)
from uncertain_worms.utils import PROJECT_ROOT, get_log_dir

log = logging.getLogger(__name__)


# Define type variables for conditions and outcomes.

RewardCondition = Tuple[StateType, ActType, StateType]
RewardOutcome = Tuple[float, bool]
TransitionCondition = Tuple[StateType, ActType]
TransitionOutcome = Tuple[StateType]
ObservationCondition = Tuple[StateType]
ObservationOutcome = Tuple[ObsType]
InitialModelCondition = Tuple
InitialModelOutcome = Tuple[StateType]

Condition = Union[
    RewardCondition,
    TransitionCondition,
    ObservationCondition,
    InitialModelCondition,
]
Outcome = Union[
    RewardOutcome, TransitionOutcome, ObservationOutcome, InitialModelOutcome
]


def jsd(p: Dict[Any, float], q: Dict[Any, float], base: float = 2.0) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    JSD is symmetric and bounded between 0 and 1.
    """
    # Get union of all states
    all_states = set(p.keys()) | set(q.keys())

    # Create the midpoint distribution M = (P + Q)/2
    m: Dict[Any, float] = {}
    for state in all_states:
        p_val = p.get(state, 0.0)
        q_val = q.get(state, 0.0)
        m[state] = (p_val + q_val) / 2.0

    # Compute JSD = 0.5 * (KL(P||M) + KL(Q||M))
    kl_pm = 0.0
    kl_qm = 0.0

    for state in all_states:
        p_val = p.get(state, 0.0)
        q_val = q.get(state, 0.0)
        m_val = m[state]

        if p_val > 0:
            kl_pm += p_val * math.log(p_val / m_val, base)
        if q_val > 0:
            kl_qm += q_val * math.log(q_val / m_val, base)

    return 0.5 * (kl_pm + kl_qm)


def normalize_outcome_list(outcome_list: List[Outcome]) -> Dict[Outcome, float]:
    """Converts a list of outcomes into a normalized frequency distribution."""
    counts = Counter(outcome_list)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def conditional_jsd(
    p_dict: Dict[Condition, List[Outcome]], q_dict: Dict[Condition, List[Outcome]]
) -> float:
    """Computes the conditional Jensen-Shannon divergence, taking expectation
    over P(X).

    Args:
        p_dict: Dictionary mapping conditions (X) to lists of observed outcomes.
        q_dict: Dictionary mapping conditions (X) to lists of predicted outcomes.

    Returns:
        The expectation of JSD under P(X), i.e.,
        E_{X ~ P(X)}[D_JS(P(Y|X) || Q(Y|X))].
    """
    js_divs: List[float] = []
    # Compute P(X) from the empirical counts (each condition's weight)
    total_count = sum(len(outcomes) for outcomes in p_dict.values())
    p_x_distribution: Dict[Condition, float] = {
        condition: len(outcomes) / total_count for condition, outcomes in p_dict.items()
    }

    for condition, p_x in p_x_distribution.items():
        if condition in q_dict:
            # Convert outcome lists into normalized frequency distributions
            p_dist = normalize_outcome_list(p_dict[condition])
            q_dist = normalize_outcome_list(q_dict[condition])

            # Compute JSD for this condition
            js_div = jsd(p_dist, q_dist, base=2)

            # Weight the JSD by P(X)
            js_divs.append(p_x * js_div)

    return float(np.sum(js_divs)) if js_divs else 0.0


class PartiallyObsPlanningAgent(Policy[StateType, ActType, ObsType]):
    def __init__(
        self,
        planner: PartiallyObservablePlanner[
            StateType, ActType, ObsType, ParticleBelief
        ],
        empty_state: StateType,
        empty_observation: ObsType,
        *args: Any,
        ground_truth_initial_model: Optional[InitialModel] = None,
        num_initial_model_samples: int = 20000,
        num_observation_model_samples: int = 25,
        num_transition_model_samples: int = 25,
        num_reward_model_samples: int = 1,
        num_particles: int = 1000,
        resampling: bool = True,
        max_attempts: int = 1000000,
        num_bug_fix_attempts: int = 5,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.empty_state = empty_state
        self.empty_observation = empty_observation
        self.num_particles = num_particles
        self.num_initial_model_samples = num_initial_model_samples
        self.num_observation_model_samples = num_observation_model_samples
        self.num_transition_model_samples = num_transition_model_samples
        self.num_reward_model_samples = num_reward_model_samples
        self.ground_truth_initial_model = ground_truth_initial_model
        self.planner = planner
        self.action_hist: List[ActType] = []
        self.observation_hist: List[ObsType] = []
        self.steps_taken = 0
        self.resampling = resampling
        self.num_bug_fix_attempts = num_bug_fix_attempts
        self.max_attempts = max_attempts
        self.num_attempts = 0
        self.current_belief: Optional[ParticleBelief] = None

    def reset(self) -> None:
        self.observation_hist = []
        self.action_hist = []
        self.steps_taken = 0
        self.num_attempts = 0
        self.current_belief = None

    def _evaluate_coverage(
        self,
        empirical_dist: Dict[Condition, List[Outcome]],
        model_dist: Dict[Condition, List[Outcome]],
    ) -> float:
        impossible_io: List[Tuple[Condition, Outcome]] = []
        correct_io: List[Tuple[Condition, Outcome]] = []
        for condition, outcomes in empirical_dist.items():
            for outcome in outcomes:
                if outcome not in model_dist[condition]:
                    impossible_io.append((condition, outcome))
                else:
                    correct_io.append((condition, outcome))

        return float(len(correct_io)) / float(len(impossible_io) + len(correct_io))

    def evaluate_initial_model(
        self, replay_buffer: ReplayBuffer
    ) -> Tuple[Tuple, Optional[str]]:
        log.info("[agent] Evaluating initial model")
        # Evaluate initial state model

        err_str = None
        empirical_initials = {
            (): [(e.previous_states[0],) for e in replay_buffer.episodes]
        }
        model_initials = defaultdict(list)
        try:
            # Compute a model distribution from the current estimated initial state distribution
            model_initials[()] = [
                (self.planner.initial_model(copy.deepcopy(self.empty_state)),)
                for _ in range(self.num_initial_model_samples)
            ]
        except Exception:
            log.info("Bug during initial state evaluation")
            err_str = str(traceback.format_exc())
            log.info(err_str)

        return (empirical_initials, model_initials), err_str

    def evaluate_transition_model(
        self, replay_buffer: ReplayBuffer
    ) -> Tuple[Tuple, Optional[str]]:
        log.info("[agent] Evaluating transition model")
        # Evaluate transition model
        empirical_transitions = defaultdict(list)
        model_transitions = defaultdict(list)
        err_str = None
        for episode in replay_buffer.episodes:
            for previous_state, action, next_state in zip(
                episode.previous_states, episode.actions, episode.next_states
            ):
                try:
                    empirical_transitions[(previous_state, action)].append(
                        (next_state,)
                    )
                    for _ in range(self.num_transition_model_samples):
                        # Sample a transition from the model
                        pred_next_state = self.planner.transition_model(
                            previous_state, action
                        )

                        model_transitions[(previous_state, action)].append(
                            (pred_next_state,)
                        )
                except Exception:
                    log.info("Bug during transition evaluation")
                    err_str = str(traceback.format_exc())
                    log.info(err_str)
                    break
            if err_str is not None:
                break

        return (empirical_transitions, model_transitions), err_str

    def evaluate_reward_model(
        self, replay_buffer: ReplayBuffer
    ) -> Tuple[Tuple, Optional[str]]:
        log.info("[agent] Evaluating reward model")
        # Evaluate reward model
        empirical_rewards = defaultdict(list)
        model_rewards = defaultdict(list)
        err_str = None
        for episode in replay_buffer.episodes:
            for previous_state, action, next_state, reward, terminated in zip(
                episode.previous_states,
                episode.actions,
                episode.next_states,
                episode.rewards,
                episode.terminated,
            ):
                try:
                    empirical_rewards[(previous_state, action, next_state)].append(
                        (reward, terminated)
                    )
                    for _ in range(self.num_reward_model_samples):
                        # Sample a reward from the model
                        pred_reward, pred_terminated = self.planner.reward_model(
                            previous_state, action, next_state
                        )
                        model_rewards[(previous_state, action, next_state)].append(
                            (pred_reward, pred_terminated)
                        )
                except Exception:
                    log.info("Bug during reward evaluation")
                    err_str = str(traceback.format_exc())
                    log.info(err_str)
                    break
            if err_str is not None:
                break

        return (empirical_rewards, model_rewards), err_str

    def evaluate_observation_model(
        self, replay_buffer: ReplayBuffer
    ) -> Tuple[Tuple, Optional[str]]:
        log.info("[agent] Evaluating observation model")
        # Evaluate observation model
        empirical_observations = defaultdict(list)
        model_observations = defaultdict(list)
        err_str = None
        for episode in replay_buffer.episodes:
            for state, action, obs in zip(
                episode.next_states, episode.actions, episode.next_observations
            ):
                try:
                    empirical_observations[(state, action)].append((obs,))
                    for _ in range(self.num_observation_model_samples):
                        # Sample an observation from the model
                        pred_obs = self.planner.observation_model(
                            state, action, self.empty_observation
                        )
                        model_observations[(state, action)].append((pred_obs,))
                except Exception:
                    log.info("Bug during observation evaluation")
                    err_str = str(traceback.format_exc())
                    log.info(err_str)
                    break
            if err_str is not None:
                break

        return (empirical_observations, model_observations), err_str

    def evaluate_model(
        self, model_name: str, replay_buffer: ReplayBuffer
    ) -> Tuple[Tuple, Optional[str]]:
        """Dispatch function to evaluate a specific model by name.

        :param model_name: One of "initial_model", "transition_model",
            "reward_model", or "observation_model".
        :param replay_buffer: The replay buffer containing episodes.
        :return: A dictionary with the evaluated empirical and model
            data.
        """
        if model_name == "initial_model":
            return self.evaluate_initial_model(replay_buffer)
        elif model_name == "transition_model":
            return self.evaluate_transition_model(replay_buffer)
        elif model_name == "reward_model":
            return self.evaluate_reward_model(replay_buffer)
        elif model_name == "observation_model":
            return self.evaluate_observation_model(replay_buffer)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def update_belief(
        self,
        current_belief: ParticleBelief,
        observation_history: List[ObsType],
        action_history: List[ActType],
        *,
        distance_threshold: float | None = None,
        kernel_bandwidth: float = 1.0,
    ) -> Tuple[Optional[ParticleBelief], Dict[str, str]]:
        """Soft-particle belief update using object-defined distances.

        • dist == ∞  ➜ reject particle  (old exact-match behaviour) •
        distance_threshold given ➜ keep particles with dist ≤ threshold
        (equal weight) • else ➜ weight = exp(-dist / kernel_bandwidth)
        """
        assert len(observation_history) == len(action_history)

        weighted_particles: Dict[StateType, float] = {}

        # ------------------------------------------------------------------ #
        # 1) Propagate existing particles and weight them                    #
        # ------------------------------------------------------------------ #
        if current_belief:
            a_last = action_history[-1]
            o_last = observation_history[-1]

            for particle in current_belief.to_list():
                try:
                    next_state = self.planner.transition_model(
                        copy.deepcopy(particle), a_last
                    )
                    model_obs = self.planner.observation_model(
                        copy.deepcopy(next_state), a_last, self.empty_observation
                    )
                except Exception:
                    return None, {"model": traceback.format_exc()}

                dist = model_obs.distance(o_last)

                if math.isinf(dist):
                    continue  # impossible

                if distance_threshold is not None:
                    if dist <= distance_threshold:
                        weighted_particles[next_state] = (
                            weighted_particles.get(next_state, 0) + 1.0
                        )
                else:
                    w = math.exp(-dist / max(kernel_bandwidth, 1e-12))
                    if w > 0.0:
                        weighted_particles[next_state] = (
                            weighted_particles.get(next_state, 0) + w
                        )

        log.info(
            "Num valid initial particles pre-rejuvenation: "
            f"{len(weighted_particles)}"
        )

        # ------------------------------------------------------------------ #
        # 2) Rejuvenate / initialise until we hit self.num_particles         #
        # ------------------------------------------------------------------ #
        while (
            sum(weighted_particles.values()) < self.num_particles
            and self.num_attempts < self.max_attempts
        ):
            self.num_attempts += 1
            try:
                candidate = self.planner.initial_model(copy.deepcopy(self.empty_state))
            except Exception:
                return None, {"initial_model": traceback.format_exc()}

            ok = True
            next_state = candidate
            for a, obs in zip(action_history, observation_history):
                try:
                    next_state = self.planner.transition_model(
                        copy.deepcopy(next_state), a
                    )
                    cand_obs = self.planner.observation_model(
                        copy.deepcopy(next_state), a, self.empty_observation
                    )
                except Exception:
                    return None, {"model": traceback.format_exc()}

                dist = cand_obs.distance(obs)
                if math.isinf(dist) or (
                    distance_threshold is not None and dist > distance_threshold
                ):
                    ok = False
                    break

            if ok:
                weighted_particles[next_state] = (
                    weighted_particles.get(next_state, 0) + 1.0
                )

        log.info(f"Num attempts: {self.num_attempts}/{self.max_attempts}")
        log.info("Num valid initial particles: " + str(len(weighted_particles)))

        if not weighted_particles:
            return None, {
                "observation_model": "No particles consistent with observation"
            }

        # ------------------------------------------------------------------ #
        # 3) Normalise weights and convert to integer counts                 #
        # ------------------------------------------------------------------ #
        total_w = sum(weighted_particles.values())
        if total_w == 0.0:
            return None, {"observation_model": "All particle weights zero"}

        int_particles: Dict[StateType, int] = Counter()
        for s, w in weighted_particles.items():
            count = max(1, int(round(w / total_w * self.num_particles)))
            int_particles[s] += count

        log.info("Num unique initial particles: " + str(len(int_particles)))

        new_belief = ParticleBelief[StateType, ObsType](particles=int_particles)
        return new_belief, {}

    def get_next_action(self, obs: Optional[ObsType]) -> ActType:
        if obs is not None:
            self.observation_hist.append(obs)

        # If no belief has been established yet, initialize it
        if self.current_belief is None:
            log.info("Initializing belief")
            belief, error = self.update_belief(
                ParticleBelief(), observation_history=[], action_history=[]
            )
            if belief is None:
                log.info("Belief initialization error: " + str(error))
                action = random.choice(self.actions)
                self.action_hist.append(action)
                return action
            self.current_belief = belief
        else:
            log.info("Updating belief")
            # Otherwise update the belief using the last action and the new observation
            belief, error = self.update_belief(
                self.current_belief,
                observation_history=self.observation_hist,
                action_history=self.action_hist,
            )
            if belief is None:
                log.info("Belief update error: " + str(error))
                action = random.choice(self.actions)
                self.action_hist.append(action)
                return action
            self.current_belief = belief

        steps_left = self.max_steps - self.steps_taken
        action, error = self.planner.plan_next_action(belief, max_steps=steps_left)

        self.action_hist.append(action)
        if len(error) != 0:
            log.info("Planning error, taking random action")
            action = random.choice(self.actions)

        self.steps_taken += 1
        return action


def save_rex_tree_html(
    model_name: str,
    generated_nodes: List[RexNode],
    parent_mapping: Dict[RexNode, RexNode],
    log_dir: str,
    iteration: int,
) -> None:
    """Saves the Rex tree visualization as an HTML file in the specified log
    directory.

    Args:
        generated_nodes: A list of generated RexNode objects.
        parent_mapping: Mapping from a child RexNode to its parent's RexNode.
        log_dir: Directory where HTML files will be saved.
        iteration: Current iteration number for unique file naming.
    """
    # Create a directed network graph.
    net = Network(height="750px", width="100%", directed=True)

    # Add nodes for each RexNode.
    for node in generated_nodes:
        label = (
            f"Iter: {node.iter_num}\n"
            f"Depth: {node.depth}\n"
            f"Train/Test: {node.train_coverage:.2f}, {node.test_coverage:.2f}\n"
            f"Alpha,Beta: ({node.alpha:.2f}, {node.beta:.2f})\n"
        )

        # Compute total coverage as the average of train and test coverage.
        total_coverage = (node.train_coverage + node.test_coverage) / 2.0

        # Create a red-to-green gradient:
        # When total_coverage is 0, color is red (#FF0000);
        # when it is 1, color is green (#00FF00).
        red = int((1 - total_coverage) * 255)
        green = int(total_coverage * 255)
        blue = 0
        color = f"#{red:02x}{green:02x}{blue:02x}"

        net.add_node(str(node.iter_num), label=label, color=color)

    # Add edges from parent to child.
    for child, parent in parent_mapping.items():
        net.add_edge(str(parent.iter_num), str(child.iter_num))

    # Construct the file name and save the HTML file in log_dir.
    # filename = os.path.join(log_dir, f"rex_tree_iter_{model_name}_{iteration}.html")
    filename = os.path.join(log_dir, f"rex_tree_iter_{model_name}.html")

    net.write_html(filename)
    print(f"Rex tree for iteration {iteration} saved to {filename}")


@dataclass
class RexNode:
    model: Any
    to_update: bool
    train_coverage: float
    test_coverage: float
    train_empirical_dist: Dict[Condition, List[Outcome]]
    train_model_dist: Dict[Condition, List[Outcome]]
    test_empirical_dist: Dict[Condition, List[Outcome]]
    test_model_dist: Dict[Condition, List[Outcome]]
    alpha: float
    beta: float
    depth: int
    iter_num: int
    messages: List[Dict[str, str]] = field(default_factory=list)
    previous_code: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"RexNode(iter={self.iter_num}, depth={self.depth},\n"
            f"        train_coverage={self.train_coverage:.4f}, test_coverage={self.test_coverage:.4f},\n"
            f"        alpha={self.alpha:.4f}, beta={self.beta:.4f}, to_update={self.to_update})"
        )

    def __hash__(self) -> int:
        return hash(self.iter_num)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RexNode):
            return False
        return self.iter_num == other.iter_num


class LLMPartiallyObsPlanningAgent(
    PartiallyObsPlanningAgent[StateType, ActType, ObsType]
):
    def __init__(
        self,
        *args: Any,
        env_code_path: str = "",
        env_description: str = "",
        goal_description: str = "",
        num_model_attempts: int = 25,
        num_online_model_attempts: int = 25,
        num_input_examples: int = 5,
        learn_transition: bool = False,
        learn_reward: bool = False,
        learn_observation: bool = False,
        learn_initial: bool = False,
        use_offline: bool = True,
        use_online: bool = False,
        dataset_path: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model_translators = {
            "transition_model": transition_model_translator,
            "reward_model": reward_model_translator,
            "observation_model": observation_model_translator,
            "initial_model": initial_model_translator,
        }
        self.should_learn = {
            "transition_model": learn_transition,
            "reward_model": learn_reward,
            "observation_model": learn_observation,
            "initial_model": learn_initial,
        }

        self.model_names = list(self.should_learn.keys())
        self.num_input_examples = num_input_examples
        self.env_description = env_description
        self.goal_description = goal_description
        self.replay_buffer = ReplayBuffer[StateType, ActType, ObsType]()
        self.previous_state: Optional[StateType] = None
        self.num_model_attempts = num_model_attempts
        self.num_online_model_attempts = num_online_model_attempts
        self.current_code = {model_name: "" for model_name in self.model_names}
        self.use_offline = use_offline
        self.use_online = use_online
        self.previous_coverage = {model_name: 0.0 for model_name in self.model_names}

        with open(os.path.join(PROMPT_DIR, "po_inserts.json")) as f:
            self.inserts = json.load(f)

        self.templates: Dict[str, str] = {}
        for model_name in self.model_names:
            with open(
                os.path.join(
                    PROJECT_ROOT,
                    env_code_path,
                    f"model_templates/po_{model_name}_template.txt",
                )
            ) as f:
                self.templates[model_name] = f.read().strip()

        with open(os.path.join(PROJECT_ROOT, env_code_path, "api.py")) as f:
            self.code_api = f.read().strip()

        if self.use_offline:
            # Train
            print(os.path.join(PROJECT_ROOT, dataset_path))
            train_replay_buffer = ReplayBuffer[StateType, int, ObsType].load_from_file(
                os.path.join(PROJECT_ROOT, dataset_path)
            )
            self.init_types(train_replay_buffer)
            self.offline_replay_buffer = train_replay_buffer
            self.offline_update_models()

    def get_next_action(self, obs: Optional[ObsType]) -> ActType:
        if obs is not None:
            self.observation_hist.append(obs)

        # If no belief has been established yet, initialize it
        in_error_state = False
        if self.current_belief is None:
            belief, error = self.update_belief(
                ParticleBelief(), observation_history=[], action_history=[]
            )
            if belief is None:
                log.info("Belief initialization error: " + str(error))
                in_error_state = True
            else:
                self.current_belief = belief
        else:
            # Otherwise update the belief using the last action and the new observation
            belief, error = self.update_belief(
                self.current_belief,
                observation_history=self.observation_hist,
                action_history=self.action_hist,
            )
            if belief is None:
                log.info("Belief update error: " + str(error))
                in_error_state = True
            else:
                self.current_belief = belief

        if in_error_state == 0:
            steps_left = self.max_steps - self.steps_taken
            assert belief is not None
            action, error = self.planner.plan_next_action(belief, max_steps=steps_left)
        else:
            log.info("In error state, returning random action")
            action = random.choice(self.actions)

        self.action_hist.append(action)
        self.steps_taken += 1
        return action

    def gpt_update_model(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        iter_num: int,
        exec_attempt: int,
        episode: int = 0,
    ) -> Optional[str]:
        model_func_name = model_name.replace("model", "func")
        code_str, local_scope = requery(
            messages,
            model_func_name,
            iter_num,
            exec_attempt,
            replay_path=self.replay_path,
            episode=episode,
        )

        if code_str is not None:
            assert self.type_tuple is not None
            new_model = self.model_translators[model_name](
                model_func_name, local_scope, type_tuple=self.type_tuple
            )
            if model_name == "initial_model":
                self.planner.initial_model = new_model  # type:ignore
            elif model_name == "transition_model":
                self.planner.transition_model = new_model  # type:ignore
            elif model_name == "reward_model":
                self.planner.reward_model = new_model  # type:ignore
            elif model_name == "observation_model":
                self.planner.observation_model = new_model  # type:ignore

            return code_str
        return None

    def print_io(self, input_outputs: List[Tuple[Condition, Outcome]]) -> str:
        io_str = ""
        for condition, outcome in input_outputs:
            if len(condition) > 0:
                for condition_item in list(condition):
                    io_str += f"Input {type(condition_item).__name__}: "
                    io_str += f"{str(condition_item)}\n"
            for outcome_item in list(outcome):
                io_str += f"Output {type(outcome_item).__name__}: "
                io_str += f"{str(outcome_item)}\n"
            io_str += "\n\n"
        return io_str

    def get_starting_prompt(
        self, model_name: str, empirical_dist: Dict[Condition, List[Outcome]]
    ) -> str:
        starting_prompt_fn = os.path.join(PROMPT_DIR, "po_model_prompt.txt")
        empirical_io: List[Tuple[Condition, Outcome]] = []
        for condition, outcomes in empirical_dist.items():
            for outcome in outcomes:
                empirical_io.append((condition, outcome))

        empirical_io = random.sample(
            empirical_io, min(self.num_input_examples, len(empirical_io))
        )

        initial_exp_str = self.print_io(empirical_io)

        with open(starting_prompt_fn, "r", encoding="utf-8") as file:
            starting_prompt = file.read()

        starting_templates = {
            "exp": initial_exp_str,
            "code_template": self.templates[model_name],
            "code_api": self.code_api,
            "env_description": self.env_description,
            "goal_description": self.goal_description,
        } | self.inserts[model_name]
        starting_prompt = starting_prompt.format(**starting_templates)

        return starting_prompt

    def get_feedback_prompt(
        self,
        model_name: str,
        previous_code: str,
        model_dist: Dict[Condition, List[Outcome]],
        empirical_dist: Dict[Condition, List[Outcome]],
    ) -> Optional[str]:
        experiences = ""
        # Identify conditions where at least one empirical outcome is not generated by the model.
        candidate_conditions = []
        for condition, outcomes in empirical_dist.items():
            model_outcomes = model_dist.get(condition, [])
            if any(outcome not in model_outcomes for outcome in outcomes):
                candidate_conditions.append(condition)

        if not candidate_conditions:
            return None  # No condition found where the model fails to capture an empirical outcome.

        # Pick one condition (e.g., at random) from those candidates.
        random.shuffle(candidate_conditions)
        for selected_condition in candidate_conditions[: self.num_input_examples]:
            # Get unique empirical outcomes and model outcomes for this condition.
            empirical_unique = list(set(empirical_dist[selected_condition]))
            model_unique = list(set(model_dist.get(selected_condition, [])))

            # Filter empirical outcomes to only those that are impossible under the model.
            filtered_empirical = [
                outcome for outcome in empirical_unique if outcome not in model_unique
            ]

            # Sample up to 5 outcomes from each without duplicates.
            empirical_samples = (
                random.sample(
                    filtered_empirical,
                    min(self.num_input_examples, len(filtered_empirical)),
                )
                if len(filtered_empirical) > self.num_input_examples
                else filtered_empirical
            )
            model_samples = (
                random.sample(
                    model_unique, min(self.num_input_examples, len(model_unique))
                )
                if len(model_unique) > self.num_input_examples
                else model_unique
            )

            # Format the samples into strings.
            empirical_io_str = self.print_io(
                [(selected_condition, outcome) for outcome in empirical_samples]
            )
            model_io_str = self.print_io(
                [(selected_condition, outcome) for outcome in model_samples]
            )

            experiences += (
                "Here are some samples from the real world that were impossible under your model\n"
                + empirical_io_str
                + "\n"
            )
            experiences += (
                "And here are some samples from your code under the same conditions\n"
                + model_io_str
                + "\n"
            )

        # Load the initial prompt template.
        initial_file_path = os.path.join(PROMPT_DIR, "po_model_refining.txt")
        with open(initial_file_path, "r", encoding="utf-8") as file:
            initial_prompt = file.read()

        # Create the dictionary of template variables.
        initial_templates = {
            "code_api": self.code_api,
            "code": previous_code,
            "empirical_experience": empirical_io_str,
            "experiences": experiences,
            "env_description": self.env_description,
            "goal_description": self.goal_description,
        } | self.inserts[model_name]

        return initial_prompt.format(**initial_templates)

    def online_update_models(
        self,
        replay_buffer: ReplayBuffer,
        episode: int,
    ) -> None:
        log.info("Online updating models")

        if not self.use_online:
            return

        if self.use_offline:
            # Split replay buffer into two equal parts.
            total_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                self.offline_replay_buffer.episodes + replay_buffer.episodes
            )
            test_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                self.offline_replay_buffer.episodes[
                    : len(self.offline_replay_buffer.episodes) // 2
                ]
            )
            offline_train = self.offline_replay_buffer.episodes[
                len(self.offline_replay_buffer.episodes) // 2 :
            ]
            online_train = replay_buffer.episodes
            train_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                offline_train + online_train
            )
        else:
            total_replay_buffer = replay_buffer
            train_replay_buffer = replay_buffer
            test_replay_buffer = replay_buffer

        for model_name in self.model_names:
            (emperical_dist, model_dist), error = self.evaluate_model(
                model_name, total_replay_buffer
            )
            assert error is None  # Should have not been added as a node if errors
            coverage = self._evaluate_coverage(emperical_dist, model_dist)
            log.info(f"Previous Total Model {model_name} Coverage: {coverage:.4f}")

            eps = 0.001
            if self.previous_coverage[model_name] > 0 and coverage + eps >= self.previous_coverage[model_name]:
                log.info(
                    f"Coverage on new data is not worse than previous coverage, skipping update"
                )
                continue

            if not self.should_learn[model_name]:
                log.info(f"Skipping update for {model_name} as learning is disabled")
                continue

            node, num_rex_nodes = self.update_models_rex(
                train_replay_buffer,
                test_replay_buffer,
                model_name,
                num_model_attempts=self.num_online_model_attempts,
                episode=episode
            )
            self.writer.add_scalar(
                "online_num_rex_nodes/" + model_name, num_rex_nodes, episode
            )  # type:ignore
            self.writer.add_scalar(
                "online_test/" + model_name, node.test_coverage, episode
            )  # type:ignore
            self.writer.add_scalar(
                "online_train/" + model_name, node.train_coverage, episode
            )  # type:ignore

        self.reset()
        for model_name in self.model_names:
            (emperical_dist, model_dist), error = self.evaluate_model(
                model_name, total_replay_buffer
            )
            assert error is None  # Should have not been added as a node if errors
            coverage = self._evaluate_coverage(emperical_dist, model_dist)
            self.previous_coverage[model_name] = coverage
            log.info(f"Total Model {model_name} Coverage: {coverage:.4f}")
            self.writer.add_scalar(
                "online_total/" + model_name, coverage, episode
            )  # type:ignore

    def offline_update_models(
        self,
    ) -> None:
        log.info("Offline updating models")

        # Split replay buffer into two equal parts.
        # N=1 edge case (sample efficiency sweep) : half == 0 → test buffer
        # vide → ZeroDivisionError dans _evaluate_coverage. On reuse the
        # single trajectory for both train & test in that case.
        n_ep = len(self.offline_replay_buffer.episodes)
        if n_ep >= 2:
            test_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                list(self.offline_replay_buffer.episodes[: n_ep // 2])
            )
            train_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                list(self.offline_replay_buffer.episodes[n_ep // 2 :])
            )
        else:
            test_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                list(self.offline_replay_buffer.episodes)
            )
            train_replay_buffer = ReplayBuffer[StateType, ActType, ObsType](
                list(self.offline_replay_buffer.episodes)
            )

        for model_name in self.model_names:
            if not self.should_learn[model_name]:
                log.info(f"Skipping update for {model_name} as learning is disabled")
                continue
            node, num_rex_nodes = self.update_models_rex(
                train_replay_buffer,
                test_replay_buffer,
                model_name,
                num_model_attempts=self.num_model_attempts,
                episode=-1,
            )
            self.writer.add_scalar(
                "offline_test/" + model_name, node.test_coverage, 0
            )  # type:ignore
            self.writer.add_scalar(
                "offline_train/" + model_name, node.train_coverage, 0
            )  # type:ignore
            self.writer.add_scalar(
                "offline_num_rex_nodes/" + model_name, num_rex_nodes, 0
            )  # type:ignore

        self.reset()
        for model_name in self.model_names:
            (emperical_dist, model_dist), error = self.evaluate_model(
                model_name, self.offline_replay_buffer
            )
            assert error is None  # Should have not been added as a node if errors
            coverage = self._evaluate_coverage(emperical_dist, model_dist)
            self.previous_coverage[model_name] = coverage
            log.info(f"Total Model {model_name} Coverage: {coverage:.4f}")
            self.writer.add_scalar(
                "offline_total/" + model_name, coverage, 0
            )  # type:ignore

    def update_models_rex(
        self,
        train_replay_buffer: ReplayBuffer,
        test_replay_buffer: ReplayBuffer,
        model_name: str,
        rex_C: float = 20,
        num_model_attempts: int = 5,
        episode: int = 0,
    ) -> Tuple[RexNode, int]:
        log.info(f"Updating models using REx for {model_name}")

        # Evaluate initial coverage on both train and test replay buffers.
        (train_empirical_dist, train_model_dist), train_error = self.evaluate_model(
            model_name, train_replay_buffer
        )
        (test_empirical_dist, test_model_dist), test_error = self.evaluate_model(
            model_name, test_replay_buffer
        )

        # These are from the initial (default) model, so they should be fine.
        assert train_error is None
        assert test_error is None

        train_coverage = self._evaluate_coverage(train_empirical_dist, train_model_dist)
        test_coverage = self._evaluate_coverage(test_empirical_dist, test_model_dist)
        to_update = True

        # Create the initial RexNode.
        initial_node = RexNode(
            model=self._get_model(model_name),
            to_update=to_update,
            train_coverage=train_coverage,
            test_coverage=test_coverage,
            train_empirical_dist=train_empirical_dist,
            train_model_dist=train_model_dist,
            test_empirical_dist=test_empirical_dist,
            test_model_dist=test_model_dist,
            alpha=1,
            beta=1,
            depth=0,
            iter_num=-1,
        )
        # Use a set to store generated nodes.
        generated_nodes: Set[RexNode] = set()
        generated_nodes.add(initial_node)

        # Maintain a parent mapping from child RexNode to parent RexNode.
        parent_mapping: Dict[RexNode, RexNode] = {}

        for iter_num in range(num_model_attempts):
            log.info("-" * 20)
            log.info(f"REx Step {iter_num}: Attempting to generate new code")
            self.step_num = iter_num

            # Sample a node based on random beta sampling.
            to_update_nodes = [node for node in generated_nodes if node.to_update]
            gcode = max(
                to_update_nodes,
                key=lambda node: (
                    np.random.beta(node.alpha, node.beta) * int(node.to_update)
                ),
            )
            log.info(f"Sampling new code:\n{gcode!r}")

            error = None
            for attempt in range(self.num_bug_fix_attempts):
                log.info(f"Attempt {attempt} to generate valid code from sampled node")
                to_update, new_messages, model_code = self._update_model(
                    model_name,
                    iter_num,
                    attempt,
                    train_replay_buffer,
                    gcode,
                    error=error,
                    episode=episode
                )
                gcode.messages = new_messages

                # Evaluate updated model on both train and test replay buffers.
                (
                    train_empirical_dist,
                    train_model_dist,
                ), train_error = self.evaluate_model(model_name, train_replay_buffer)
                (
                    test_empirical_dist,
                    test_model_dist,
                ), test_error = self.evaluate_model(model_name, test_replay_buffer)
                if train_error is not None or test_error is not None:
                    error = (train_error or "") + "\n" + (test_error or "")
                else:
                    break
            else:
                log.info("Failed to generate valid code after maximum attempts.")
                continue

            new_train_coverage = self._evaluate_coverage(
                train_empirical_dist, train_model_dist
            )
            new_test_coverage = self._evaluate_coverage(
                test_empirical_dist, test_model_dist
            )
            combined_coverage = (new_train_coverage + new_test_coverage) / 2

            # Create a new RexNode for the generated code.
            new_node = RexNode(
                model=self._get_model(model_name),
                to_update=to_update,
                train_coverage=new_train_coverage,
                test_coverage=new_test_coverage,
                train_empirical_dist=train_empirical_dist,
                train_model_dist=train_model_dist,
                test_empirical_dist=test_empirical_dist,
                test_model_dist=test_model_dist,
                alpha=1 + rex_C * combined_coverage,
                beta=1 + rex_C * (1.0 - combined_coverage),
                depth=gcode.depth + 1,
                iter_num=iter_num,
                messages=new_messages,
                previous_code=model_code,  # Store the previous code for feedback
            )
            # Add the new node and record its parent.
            generated_nodes.add(new_node)
            parent_mapping[new_node] = gcode
            log.info(f"New node generated:\n{new_node!r}")

            if new_test_coverage == 1.0 and new_train_coverage == 1.0:
                log.info(
                    f"Model Update Step {self.step_num}: Generated code is correct"
                )
                break
            else:
                # Increase beta for the chosen node if the update was not successful.
                gcode.beta += 1

            self.reset()

            # Optionally, you can now save the current Rex tree to an HTML file.
            # (See the pyvis snippet below for saving.)
            save_rex_tree_html(
                model_name,
                list(generated_nodes),
                parent_mapping,
                get_log_dir(),
                iter_num,
            )

        # Select the best node based on test and train coverage.
        best_node = max(
            generated_nodes, key=lambda node: (node.test_coverage, node.train_coverage)
        )
        log.info(f"Best code found after REx iterations:\n{best_node!r}")
        self._set_model(model_name, best_node.model)
        return best_node, len(generated_nodes)

    def _update_model(
        self,
        model_name: str,
        iter_num: int,
        exec_attempt: int,
        replay_buffer: ReplayBuffer,
        rex_node: RexNode,
        error: Optional[str] = None,
        episode: int = 0,
    ) -> Tuple[bool, List[Dict[str, str]], Optional[str]]:
        self.init_types(replay_buffer)
        self._set_model(model_name, rex_node.model)
        messages = copy.deepcopy(rex_node.messages)
        to_update = True
        if rex_node.to_update:
            log.info(f"Updating {model_name}")

            # Rank the probabilities of empirical distribution under the model
            if error is not None:
                log.info(
                    f"[{model_name}] Episode {episode} Iter {iter_num}, step {self.step_num} bug: {error}"
                )

                messages.append({"role": "user", "content": error})
                code_str = self.gpt_update_model(
                    messages, model_name, iter_num=iter_num, exec_attempt=exec_attempt, episode=episode
                )
                assert code_str is not None
                messages.append(
                    {
                        "role": "assistant",
                        "content": code_str,
                    }
                )
            else:
                log.info(f"[{model_name}] no execution errors")
                prompt: Optional[str] = None
                if rex_node.previous_code is None:
                    log.info(f"[{model_name}] First model update")
                    # First model update. LLM has not generated any code yet
                    prompt = self.get_starting_prompt(
                        model_name, rex_node.train_empirical_dist
                    )
                else:
                    log.info(f"[{model_name}] Refining model")
                    # Check for cases where the LLM generated models were wrong and feed back
                    prompt = self.get_feedback_prompt(
                        model_name,
                        rex_node.previous_code,
                        rex_node.train_model_dist,
                        rex_node.train_empirical_dist,
                    )

                if prompt is not None:
                    log.info(f"[{model_name}] Prompting for feedback")
                    messages = [{"role": "system", "content": prompt}]
                    code_str = self.gpt_update_model(
                        messages,
                        model_name=model_name,
                        iter_num=iter_num,
                        exec_attempt=exec_attempt,
                        episode=episode,
                    )
                    assert code_str is not None
                    messages.append(
                        {
                            "role": "assistant",
                            "content": code_str,
                        }
                    )
                else:
                    log.info(f"[{model_name}] No feedback needed")
                    # Model has full support, no need to update
                    to_update = False
                    code_str = rex_node.previous_code
                    messages = copy.deepcopy(rex_node.messages)

        return to_update, messages, code_str

    def _get_model(self, model_name: str) -> Dict[str, Any]:
        return self.planner.__getattribute__(model_name)

    def _set_model(self, model_name: str, model: Any) -> None:
        self.planner.__setattr__(model_name, model)
