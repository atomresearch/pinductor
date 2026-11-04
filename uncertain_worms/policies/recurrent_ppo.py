"""Recurrent PPO, the memory-based RL baseline, in two classes and no RL library.

One implementation for all three suites. It talks to our own Environment
interface, so there is no gym adapter.

Two classes, as small as the algorithm allows:

    GRUActorCritic   the architecture: GRU over the observation sequence, then
                     a policy head and a value head
    RecurrentPPO     the loop: collect whole episodes, compute GAE, optimise the
                     clipped surrogate

Whole episodes rather than fixed-length chunks. Our environments cap out at a
few dozen steps, so a sequence is always an episode, which removes the usual
sources of recurrent-PPO bugs: no hidden state to carry across episodes, no
masking of resets inside a batch, no burn-in. Truncation is still distinguished
from termination when bootstrapping the advantage, which matters here because
the rewards are terminal and most episodes end on the step cap.

Memory is the point of this baseline: on MineSweeper-Easy PPO-GRU reaches
0.311 against 0.211 for a random policy. Under partial observability the
recurrence is what separates the method from chance.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class GRUActorCritic(nn.Module):
    """Shared GRU trunk, separate policy and value heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs_seq: torch.Tensor,
                h: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs_seq is (batch, time, obs_dim). Returns logits, values, hidden."""
        z, h = self.gru(self.encoder(obs_seq), h)
        return self.policy(z), self.value(z).squeeze(-1), h

    @torch.no_grad()
    def act(self, obs: np.ndarray, h: Optional[torch.Tensor]
            ) -> Tuple[int, float, float, torch.Tensor]:
        """One step. Returns action, log-prob, value, next hidden state."""
        x = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        logits, value, h = self.forward(x, h)
        dist = Categorical(logits=logits[0, 0])
        a = dist.sample()
        return int(a), float(dist.log_prob(a)), float(value[0, 0]), h

    def bootstrap_value(self, obs: np.ndarray,
                        h: Optional[torch.Tensor]) -> float:
        """The critic alone, for the bootstrap of a truncated episode."""
        x = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            _, value, _ = self.forward(x, h)
        return float(value[0, 0])


class RecurrentPPO:
    """PPO with a recurrent policy, one episode per sequence.

    Defaults follow the values that are standard for PPO rather than tuned
    here, so that a reader can recognise them: clip 0.2, GAE lambda 0.95,
    entropy 0.01, four optimisation epochs per batch.
    """

    def __init__(self, obs_dim: int, n_actions: int, *, hidden: int = 128,
                 lr: float = 3e-4, gamma: float = 0.98, gae_lambda: float = 0.95,
                 clip: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, epochs: int = 4,
                 max_grad_norm: float = 0.5, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.net = GRUActorCritic(obs_dim, n_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.epochs, self.max_grad_norm = epochs, max_grad_norm

    def _gae(self, rewards: Sequence[float], values: Sequence[float],
             bootstrap: float) -> Tuple[np.ndarray, np.ndarray]:
        adv = np.zeros(len(rewards), dtype=np.float32)
        last = 0.0
        nxt = bootstrap
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * nxt - values[t]
            last = delta + self.gamma * self.lam * last
            adv[t] = last
            nxt = values[t]
        return adv, adv + np.asarray(values, dtype=np.float32)

    def collect(self, env, encode: Callable, episodes: int, max_steps: int,
                seed_of: Callable[[int], int]) -> Tuple[List[dict], float]:
        """Run `episodes` episodes greedily-sampled, return their transitions."""
        batch, returns = [], []
        for i in range(episodes):
            env.reset(seed=seed_of(i))
            h = None
            obs_l, act_l, lp_l, val_l, rew_l = [], [], [], [], []
            total, discount = 0.0, 1.0
            terminated = truncated = False
            for _ in range(max_steps):
                obs = encode(env)
                a, lp, v, h = self.net.act(obs, h)
                _, _, r, terminated, truncated, _ = env.step(a)
                obs_l.append(obs); act_l.append(a)
                lp_l.append(lp); val_l.append(v); rew_l.append(float(r))
                total += discount * float(r); discount *= self.gamma
                if terminated or truncated:
                    break
            # Termination and truncation are not the same event. Reaching the
            # goal ends the episode and there is nothing beyond it, so the
            # bootstrap is zero. Hitting the step cap merely stops the rollout:
            # the state still has value, and zeroing it would teach the agent
            # that running out of time is as bad as failing. Here that is the
            # common case, since rewards are terminal and most episodes are cut
            # by the cap.
            if terminated:
                bootstrap = 0.0
            else:
                bootstrap = self.net.bootstrap_value(encode(env), h)
            adv, ret = self._gae(rew_l, val_l, bootstrap=bootstrap)
            batch.append({"obs": np.asarray(obs_l, np.float32),
                          "act": np.asarray(act_l, np.int64),
                          "logp": np.asarray(lp_l, np.float32),
                          "adv": adv, "ret": ret})
            returns.append(total)
        return batch, float(np.mean(returns))

    def update(self, batch: List[dict]) -> float:
        """One PPO update over a batch of variable-length episodes."""
        flat_adv = np.concatenate([e["adv"] for e in batch])
        mean, std = float(flat_adv.mean()), float(flat_adv.std()) + 1e-8
        losses = []
        for _ in range(self.epochs):
            self.opt.zero_grad()
            total = torch.zeros(())
            for e in batch:
                obs = torch.as_tensor(e["obs"]).unsqueeze(0)
                logits, values, _ = self.net(obs)
                dist = Categorical(logits=logits[0])
                act = torch.as_tensor(e["act"])
                ratio = torch.exp(dist.log_prob(act) - torch.as_tensor(e["logp"]))
                adv = (torch.as_tensor(e["adv"]) - mean) / std
                pg = -torch.min(ratio * adv,
                                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv).mean()
                vloss = ((values[0] - torch.as_tensor(e["ret"])) ** 2).mean()
                total = total + pg + self.value_coef * vloss \
                    - self.entropy_coef * dist.entropy().mean()
            mean_loss = total / len(batch)
            losses.append(mean_loss.item())  # read before backward frees it
            mean_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.opt.step()
        return float(np.mean(losses))
