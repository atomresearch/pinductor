"""Core types every environment, agent and planner in this package shares.

Two things live here.

**The interfaces.** :class:`State`, :class:`Observation`,
:class:`Environment` and :class:`Belief` are the abstract classes an
environment implements; :class:`ParticleBelief` and
:class:`CategoricalBelief` are the two concrete belief representations
the planners consume. Because these are the only types the planners
know about, a new domain becomes usable by subclassing them and nothing
else.

**The POMDP as four callables.** ``InitialModel``, ``TransitionModel``,
``ObservationModel`` and ``RewardModel`` are plain function types. A model
is therefore interchangeable regardless of where it came from: written by
hand for a reference environment, generated as Python source by a
language model, or estimated from data by the ``tabular_*_model_gen``
functions below, which read counts out of a :class:`ReplayBuffer` and
sample from the empirical distributions. The ``empty_*_model_gen``
counterparts return the degenerate models (no reward, identity
transition) used as placeholders before anything has been learned.

Experience is recorded in :class:`Episode` objects held by a
:class:`ReplayBuffer`, which is what the evaluation loop fills, pickles
at the end of a run, and hands to an agent when it refits its models.

States and observations are used as dictionary keys throughout (particle
counts, tabular models), so subclasses must implement ``__hash__``
consistently with equality.
"""

from __future__ import annotations

import copy
import pickle
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np
import math
from gymnasium.core import ActType


def tabular_initial_model_gen(
    replay_buffer: ReplayBuffer[StateType, ActType, ObsType]
) -> InitialModel:
    """Build an initial-state model from the starts observed in the buffer.

    The returned model samples a first state in proportion to how often it
    began an episode, so it can only ever produce a start that has already
    been seen.
    """
    counter = Counter[StateType]()  # initialize the counter
    for episode in replay_buffer.episodes:
        initial_state = episode.previous_states[0]
        counter[initial_state] += 1  # add to the counter

    def tabular_initial_model(_: StateType) -> StateType:
        # Convert counts to probabilities and sample
        total = sum(counter.values())
        if total == 0:
            raise ValueError("No episodes in replay buffer")

        # Sample from the counter according to frequency
        states = list(counter.keys())
        probs = [count / total for count in counter.values()]
        return random.choices(states, weights=probs, k=1)[
            0
        ]  # choices returns a list, so take first element

    return tabular_initial_model


def empty_reward_model_gen() -> RewardModel:
    """Reward model that pays nothing and never terminates an episode."""

    def empty_reward_model(
        state: StateType, action: ActType, next_state: StateType
    ) -> Tuple[float, bool]:
        return 0.0, False

    return empty_reward_model


def empty_observation_model_gen() -> ObservationModel:
    """Observation model that always returns the caller's empty observation.

    Every observation model takes an ``empty_obs`` argument, which is a
    correctly shaped observation for the domain; returning it unchanged
    gives a model that reveals nothing about the state.
    """

    def empty_observation_model(
        state: StateType, action: ActType, empty_obs: ObsType
    ) -> ObsType:
        return empty_obs

    return empty_observation_model


def empty_transition_model_gen() -> TransitionModel:
    """Transition model under which no action changes anything."""

    def empty_transition_model(state: StateType, action: ActType) -> StateType:
        return state

    return empty_transition_model


class State(ABC):
    """A domain state, hashable so it can be counted inside a belief.

    Subclasses provide ``encode`` / ``decode``, the pair that converts
    between the Python object and the representation shown to (and
    produced by) a language model.

    Equality is defined as hash equality, which makes ``__hash__`` the
    single definition of state identity: whatever a subclass leaves out
    of its hash, the rest of the pipeline treats as irrelevant.
    """

    def copy(self) -> Self:
        return copy.deepcopy(self)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def distance(self, other: State) -> float:
        """Zero for equal states, infinite otherwise.

        The default is deliberately all-or-nothing; domains where partial
        agreement is meaningful override it.
        """
        if other == self:
            return 0.0
        else:
            return float("inf")

    @abstractmethod
    def encode(self) -> Any:
        """Converts the state to a representation that the language model is
        expecting."""
        pass

    @classmethod
    @abstractmethod
    def decode(cls: Type[Self], encoded: Any) -> Self:
        """Converts the encoding back to the state representation."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    def __eq__(self, other: object) -> bool:
        return self.__hash__() == other.__hash__()


StateType = TypeVar("StateType", bound=State)


class Observation(ABC, Generic[StateType]):
    """What an agent perceives after a step — never the state itself.

    Partially observable agents are given instances of this class and
    nothing else, so a subclass defines exactly how much of the world the
    method under evaluation is allowed to know.

    Two comparisons are available: :meth:`distance`, which is all-or-
    nothing, and :meth:`distance_soft`, which grades partial agreement and
    is what particle weighting uses.
    """

    def copy(self) -> Self:
        return copy.deepcopy(self)

    @abstractmethod
    def __hash__(self) -> int:
        pass

    def distance(self, other: "Observation") -> float:
        if other == self:
            return 0.0
        else:
            return float("inf")

    def distance_soft(self, other: "Observation") -> float:
        """Graded distance between two grid-shaped observations.

        Cells are compared one by one; the match ratio in [0, 1] is turned
        into a distance by ``-log`` so that it can be used as a negative
        log-likelihood when weighting particles. Raising the ratio to a
        power greater than one before the log makes weak agreement
        markedly more expensive than the raw ratio would.

        Returns ``+inf`` when the shapes differ or nothing matches, which
        eliminates the particle outright. The smoothing constant only
        keeps the ``log`` away from zero.

        This default implementation reads ``self.image``: it applies to
        the grid-like domains, and subclasses whose observations are not
        images override it.
        """
        if other is self:
            return 0.0

        a = np.asarray(self.image)
        b = np.asarray(other.image)

        if a.shape != b.shape:
            return float("inf")

        # 12 is the agent marker in the grid encoding (``ObjectTypes.agent``
        # in the MiniGrid environment module; repeated as a literal here
        # because the core types cannot import an environment without
        # creating an import cycle). That cell sits at a fixed slot of the
        # field of view and therefore always matches, so counting it would
        # flatter every comparison.
        mask = (a != 12)

        total = np.count_nonzero(mask)
        if total == 0:
            return 0.0  # nothing to compare

        matches = np.count_nonzero((a == b) & mask)
        if matches == 0:
            return float("inf")
        match_ratio = matches / total  # in [0, 1]

        eps = 1e-6  # smoothing parameter
        gamma = 3 # >1 punishes weak matches more
        smoothed_ratio = eps + (1.0 - eps) * match_ratio ** gamma

        return -math.log(smoothed_ratio)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def encode(self) -> Any:
        """Converts the state to a representation that the language model is
        expecting."""
        pass

    @classmethod
    @abstractmethod
    def decode(cls: Type[Self], encoded: Any) -> Self:
        """Converts the encoding back to the state representation."""
        pass


ObsType = TypeVar("ObsType", bound=Observation)


class Environment(ABC, Generic[StateType, ActType, ObsType]):
    """The real system an agent is evaluated against.

    :meth:`step` returns the observation, the true next state, the reward,
    the termination and truncation flags, and an info dictionary. The
    evaluation loop keeps the state for logging and for the replay buffer;
    a partially observable agent is never handed it.

    The two ``visualize_*`` hooks do nothing by default: rendering is
    optional and only some domains implement it.
    """

    def __init__(self, *args: Any, max_steps: int = 0, **kwargs: Any):
        self.max_steps = max_steps

    @abstractmethod
    def step(
        self, action: ActType
    ) -> Tuple[Observation, State, float, bool, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def reset(self, seed: int = 0) -> State:
        pass

    def get_last_obs(self) -> Optional[ObsType]:
        """The observation the last :meth:`reset` or :meth:`step` produced.

        The evaluation loop conditions the initial belief on the observation a
        reset emits, and the recurrent baselines read their input through it.
        The default returns ``None``, which costs an agent the first
        observation of every episode rather than raising, so implement it in
        any environment meant for those two paths.
        """
        return None

    def visualize_episode(
        self,
        states: List[StateType],
        observations: List[ObsType],
        actions: List[ActType],
        episode_num: int,
    ) -> None:
        pass

    def visualize_belief(self, belief: Belief, episode_num: int) -> None:
        pass


class Belief(ABC, Generic[StateType, ObsType]):
    """A distribution over states, the agent's substitute for the state.

    Planners only ever need to draw states from it and to score one, hence
    the two abstract methods.
    """

    def copy(self) -> Self:
        return copy.deepcopy(self)

    @abstractmethod
    def sample(self) -> Optional[StateType]:
        pass

    @abstractmethod
    def log_prob(self, state: StateType) -> float:
        pass


class ParticleBelief(Belief[StateType, ObsType]):
    """Belief held as a multiset of states: state -> number of particles.

    Storing counts rather than a list keeps duplicated particles cheap,
    which matters because a filter concentrates its particles on a few
    states as evidence accumulates. Probabilities are the counts divided
    by :attr:`num_particles`.
    """

    def __init__(self, particles: Optional[Dict[StateType, int]] = None) -> None:
        # Default to None rather than {}: a mutable default would be
        # created once and shared by every belief built without arguments.
        self.particles = particles if particles is not None else {}
        super(ParticleBelief, self).__init__()

    @property
    def num_particles(self) -> int:
        return sum(self.particles.values())

    def sample(self) -> StateType:
        probs = [v / float(self.num_particles) for v in self.particles.values()]
        return random.choices(list(self.particles.keys()), weights=probs, k=1)[0]

    def sample_k(self, k: int) -> List[StateType]:
        probs = [v / float(self.num_particles) for v in self.particles.values()]
        return random.choices(list(self.particles.keys()), weights=probs, k=k)

    def log_prob(self, state: StateType) -> float:
        return np.log(self.particles[state] / float(self.num_particles))

    def to_list(self) -> List[StateType]:
        """Return a flat list of states where each state appears according to
        its count."""
        return [state for state, count in self.particles.items() for _ in range(count)]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ParticleBelief):
            return False
        return hash(other) == hash(self)

    def __hash__(self) -> int:
        return hash(frozenset(self.particles.items()))

    def __len__(self) -> int:
        return sum(self.particles.values())

    @classmethod
    def from_list(cls, states: List[StateType]) -> ParticleBelief:
        """Create a ParticleBelief from a list of state samples by counting how
        many times each state appears."""
        counts = Counter(states)
        return cls(particles=dict(counts))


class CategoricalBelief(Belief[StateType, ObsType]):
    """Belief held as explicit probabilities: state -> probability.

    Probabilities are rounded to :attr:`quantization` decimals before
    beliefs are compared or hashed, so that two beliefs which differ only
    by floating-point noise count as the same one — the belief-space
    search hashes its nodes through the belief they carry, and would
    otherwise re-expand nodes it has already visited.
    Entropy and hash are computed once and cached, which assumes the
    distribution is not mutated after construction.
    """

    def __init__(self, dist: Optional[Dict[StateType, float]] = None) -> None:
        # Default to None rather than {}: a mutable default would be
        # created once and shared by every belief built without arguments.
        self.dist = dist if dist is not None else {}
        self.quantization = 9
        assert round(sum(self.dist.values()), self.quantization) == 1.0 or not self.dist
        self._cached_entropy: Optional[float] = None
        self._cached_hash: Optional[int] = None
        super(CategoricalBelief, self).__init__()

    def sample(self) -> StateType:
        return random.choices(
            list(self.dist.keys()), weights=list(self.dist.values()), k=1
        )[0]

    def sample_k(self, k: int) -> List[StateType]:
        return random.choices(
            list(self.dist.keys()), weights=list(self.dist.values()), k=k
        )

    def get_entropy(self) -> float:
        """Calculate and return the entropy of the categorical belief
        distribution.

        Entropy is defined as: H = -∑ p * log(p) for each probability p in the distribution.
        This measures the uncertainty of the belief.
        """
        if self._cached_entropy is None:
            self._cached_entropy = float(
                -sum(p * np.log(p) for p in self.dist.values() if p > 0)
            )
        return self._cached_entropy

    def log_prob(self, state: StateType) -> float:
        return np.log(self.dist[state])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CategoricalBelief):
            return False

        quant_self = {k: round(v, self.quantization) for k, v in self.dist.items()}
        quant_other = {k: round(v, self.quantization) for k, v in other.dist.items()}
        return quant_self == quant_other

    def __hash__(self) -> int:
        if self._cached_hash is None:
            quantized_items = frozenset(
                (key, round(prob, self.quantization)) for key, prob in self.dist.items()
            )
            self._cached_hash = hash(quantized_items)
        return self._cached_hash

    @classmethod
    def from_list(cls, states: List[StateType]) -> CategoricalBelief:
        """Build a CategoricalBelief from state samples.

        Each distinct state gets the fraction of the samples it accounts
        for, so a set of particles becomes a normalised distribution.
        """
        dist = {k: v / len(states) for k, v in dict(Counter(states)).items()}
        return cls(dist=dist)


BeliefType = TypeVar("BeliefType", bound=Belief)
# The four components of a POMDP, as function types. Anything matching a
# signature can be used as that component — hand-written reference model,
# code generated by a language model, or a model estimated from a replay
# buffer. Each is a *sampler*, not a distribution: it returns one drawn
# outcome per call, and stochasticity is expressed by returning different
# outcomes across calls. `RewardModel` returns the reward together with
# the termination flag, so a domain defines both in one place. The lone
# argument of `InitialModel` and the last argument of `ObservationModel`
# are blank objects of the right shape for the domain, passed in as
# templates; several implementations ignore them entirely.
RewardModel = Callable[[StateType, ActType, StateType], Tuple[float, bool]]
InitialModel = Callable[[StateType], StateType]
ObservationModel = Callable[[StateType, ActType, ObsType], ObsType]
TransitionModel = Callable[[StateType, ActType], StateType]
Heuristic = Callable[[StateType], float]
TypeTuple = Tuple[type[StateType], type[ObsType]]


def zero_heuristic(state: StateType) -> float:
    """Uninformed heuristic: zero everywhere, so search stays uniform-cost."""
    return 0


@dataclass
class Episode(Generic[StateType, ActType, ObsType, BeliefType]):
    """One episode, stored as parallel per-step lists.

    Index ``t`` of every list describes the same step, so a transition is
    read as ``previous_states[t] --actions[t]--> next_states[t]`` with the
    matching observations, reward and termination flag. ``length`` is the
    number of steps recorded.
    """

    previous_states: List[StateType] = field(default_factory=list)
    next_states: List[StateType] = field(default_factory=list)
    actions: List[ActType] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminated: List[bool] = field(default_factory=list)
    previous_observations: List[ObsType] = field(default_factory=list)
    next_observations: List[ObsType] = field(default_factory=list)
    length: int = 0


@dataclass
class Transition(Generic[StateType, ActType]):
    """A single step, flattened out of the parallel lists of an Episode.

    This is the shape the count-based model estimators consume; it carries
    no observation, since those models are built over states.
    """

    prev_state: StateType
    action: ActType
    next_state: StateType
    reward: float
    terminated: bool


class ReplayBuffer(Generic[StateType, ActType, ObsType]):
    """The record of everything that happened during a run.

    Steps are appended to a single episode under construction; calling
    :meth:`wrap_up_episode` closes it and moves it into :attr:`episodes`,
    which is the only list the model estimators and the analysis scripts
    read. An episode still in progress is therefore invisible to them.

    The buffer is pickled at the end of a run and re-read by the offline
    tooling, so its layout is part of the on-disk format.
    """

    def __init__(self, episodes: Optional[List[Episode]] = None) -> None:
        # Default to None rather than []: a mutable default would be
        # created once and shared by every buffer built without arguments,
        # silently merging the histories of separate runs.
        self.episodes = episodes if episodes is not None else []
        self.current_episode: Optional[Episode] = None

    def append_episode_step(
        self,
        prev_state: StateType,
        state: StateType,
        action: ActType,
        prev_obs: ObsType,
        obs: ObsType,
        true_reward: float,
        terminated: bool,
    ) -> None:
        """Record one step into the episode currently being built.

        The episode is created on the first step, so a caller never has to
        open one explicitly; it has to close it with
        :meth:`wrap_up_episode` once the episode ends.
        """
        if self.current_episode is None:
            self.current_episode = Episode()

        self.current_episode.previous_states.append(prev_state)
        self.current_episode.next_states.append(state)
        self.current_episode.actions.append(action)
        self.current_episode.previous_observations.append(prev_obs)
        self.current_episode.next_observations.append(obs)
        self.current_episode.rewards.append(true_reward)
        self.current_episode.terminated.append(terminated)
        self.current_episode.length += 1

    @property
    def transitions(self) -> List[Transition[StateType, ActType]]:
        """Every step of every completed episode, as flat transitions.

        The list is rebuilt from scratch on each access, so callers that
        need it more than once should hold on to the result.
        """
        transitions_list = []
        for episode in self.episodes:
            for i in range(episode.length):
                transition = Transition(
                    prev_state=episode.previous_states[i],
                    action=episode.actions[i],
                    next_state=episode.next_states[i],
                    reward=episode.rewards[i],
                    terminated=episode.terminated[i],
                )
                transitions_list.append(transition)
        return transitions_list

    def wrap_up_episode(self) -> None:
        """Close the episode under construction and file it in `episodes`.

        Until this is called the episode is not part of the buffer's
        history and no model estimator can see it.
        """
        assert isinstance(self.current_episode, Episode)
        self.episodes.append(self.current_episode)
        self.current_episode = None

    def get_episode(self, eps_number: int) -> Episode:
        """Return one completed episode, with a readable error if absent.

        ``eps_number`` indexes :attr:`episodes` directly and therefore
        counts from zero.
        """
        try:
            return self.episodes[eps_number]
        except IndexError as exc:
            raise ValueError(f" eps_number {eps_number} is out of bounds.") from exc

    def sample_transitions(self, n: int) -> List[Transition]:
        """Samples `n` transitions uniformly from replay buffer.

        Returns all the transitions if number of transitions is less than `n`.
        """
        # If we have fewer transitions than requested, return all transitions
        if len(self.transitions) <= n:
            return self.transitions

        # Assuming Transition is a defined type
        transitions = np.array(self.transitions)
        sampled_transitions = np.random.choice(transitions, size=n).tolist()
        return list(sampled_transitions)  # Explicitly cast to a Python list

    def save_to_file(self, file_path: str) -> None:
        """Saves the current ReplayBuffer instance to a file using pickle."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ReplayBuffer":
        """Loads a ReplayBuffer instance from a pickle file.

        Raises a ValueError if the loaded object is not an instance of
        ReplayBuffer.
        """
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError("Loaded object is not of type ReplayBuffer")
        return obj


def tabular_reward_model_gen(
    replay_buffer: ReplayBuffer[StateType, ActType, ObsType]
) -> RewardModel:
    """Estimate a reward model by counting outcomes per transition.

    The returned model answers with the mean reward recorded for exactly
    that ``(state, action, next_state)`` triple, and draws termination
    from the frequency with which the triple ended an episode. Triples
    absent from the buffer are worth nothing and never terminate, which
    is what makes a purely tabular model blind outside its data.
    """
    # Track (state, action, next_state) -> (total_reward, count, termination_count)
    rewards_dict: Dict[Tuple[StateType, ActType, StateType], List[float]] = {}
    for transition in replay_buffer.transitions:
        key = (transition.prev_state, transition.action, transition.next_state)
        if key not in rewards_dict:
            rewards_dict[key] = [0.0, 0, 0]
        rewards_dict[key][0] += transition.reward
        rewards_dict[key][1] += 1
        rewards_dict[key][2] += 1 if transition.terminated else 0

    def tabular_reward_model(
        state: StateType, action: ActType, next_state: StateType
    ) -> Tuple[float, bool]:
        key = (state, action, next_state)
        if key not in rewards_dict or rewards_dict[key][1] == 0:
            return 0.0, False

        avg_reward = rewards_dict[key][0] / rewards_dict[key][1]
        termination_prob = rewards_dict[key][2] / rewards_dict[key][1]
        is_terminated = random.random() < termination_prob

        return avg_reward, is_terminated

    return tabular_reward_model


def tabular_transition_model_gen(
    replay_buffer: ReplayBuffer[StateType, ActType, ObsType]
) -> TransitionModel:
    """Estimate a transition model by counting successors per (state, action).

    Successors are drawn in proportion to how often they were observed.
    An unseen pair leaves the state unchanged, which is a stand-in for
    "no idea", not something observed in the data.
    """
    # Track (state, action) -> {next_state: count}
    transitions_dict: Dict[Tuple[StateType, ActType], Counter[StateType]] = {}
    for transition in replay_buffer.transitions:
        key = (transition.prev_state, transition.action)
        if key not in transitions_dict:
            transitions_dict[key] = Counter[StateType]()
        transitions_dict[key][transition.next_state] += 1

    def tabular_transition_model(state: StateType, action: ActType) -> StateType:
        key = (state, action)
        if key not in transitions_dict:
            return state  # Return current state if transition not seen

        # Sample next state according to empirical distribution
        next_states = list(transitions_dict[key].keys())
        counts = list(transitions_dict[key].values())
        total = sum(counts)
        probs = [count / total for count in counts]

        return random.choices(next_states, weights=probs, k=1)[0]

    return tabular_transition_model


def empty_initial_model_gen() -> InitialModel:
    """Initial model that hands back the blank state it is given."""

    def empty_initial_model(empty_state: State) -> State:
        return empty_state

    return empty_initial_model


def tabular_observation_model_gen(
    replay_buffer: ReplayBuffer[StateType, ActType, ObsType],
    type_tuple: Tuple[type[StateType], type[ObsType]],
) -> ObservationModel:
    """Estimate an observation model by counting observations per state.

    Observations are drawn in proportion to how often they followed the
    same ``(state, action)`` pair; an unseen pair yields the blank
    observation supplied by the caller.

    Counting is done over ``next_states`` and ``next_observations``, which
    pairs each observation with the state it was emitted from.
    """
    # Track state -> {observation: count}
    observation_dict: Dict[Tuple[StateType, ActType], Dict[ObsType, int]] = {}
    for episode in replay_buffer.episodes:
        for state, action, obs in zip(
            episode.next_states, episode.actions, episode.next_observations
        ):
            key = (state, action)
            if key not in observation_dict:
                observation_dict[key] = Counter()
            observation_dict[key][obs] += 1

    def tabular_observation_model(
        state: StateType, action: ActType, empty_obs: ObsType
    ) -> ObsType:
        key = (state, action)
        if key not in observation_dict:
            # Return empty observation
            return empty_obs

        # Sample observation according to empirical distribution
        observations = list(observation_dict[key].keys())
        counts = list(observation_dict[key].values())
        total = sum(counts)
        probs = [count / total for count in counts]

        return random.choices(observations, weights=probs, k=1)[0]

    return tabular_observation_model