"""Custom value-based and actor-critic agents for the EMS candidate-action
environment.

WHY THIS FILE EXISTS
---------------------
`train_multi_algo_ems.py` imports four names from this module:
    DQNAgent, DQNConfig, DiscreteSACAgent, DiscreteSACConfig
and drives them through a small, fixed contract:
    agent.act(obs, deterministic: bool) -> int
    agent.replay_buffer.add(obs, action, reward, next_obs, done)
    agent.train_step() -> dict (metrics, may include "epsilon")
    agent.save(path: str)

That contract is intentionally *not* the Stable-Baselines3 API (SB3's
PPO is used elsewhere in this project for the MaskablePPO baseline),
because DQN/Double DQN/Dueling DQN/discrete SAC all need to share one
extremely small, readable implementation that's easy to defend line by
line, rather than depending on SB3 internals for the off-policy side.

ALGORITHMS IMPLEMENTED
-----------------------
1. DQN (Mnih et al., 2015): a single Q-network, target network updated
   by hard copy every `target_update_interval` steps, epsilon-greedy
   exploration that linearly decays from 1.0 to 0.05 over
   `epsilon_decay_steps`.
2. Double DQN (van Hasselt et al., 2016): identical network and replay
   buffer, but the TD target uses the *online* network to choose the
   greedy next action and the *target* network only to evaluate that
   action's value. This decouples action selection from action
   evaluation and is the standard fix for DQN's well-known
   overestimation bias. Selected via `DQNConfig(double_dqn=True)`.
3. Dueling DQN (Wang et al., 2016): same training loop, different
   network head. Splits the Q-function into a scalar state-value
   V(s) and a per-action advantage A(s,a), recombined as
   Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)). The mean-subtraction is
   required for identifiability (otherwise V and A could each absorb
   an arbitrary constant and Q would still be correct, but V would no
   longer be interpretable nor would gradients be well-conditioned).
   Selected via `DQNConfig(dueling=True)`. Double DQN and Dueling DQN
   are orthogonal and can be combined (train_multi_algo_ems.py keeps
   them as separate CLI choices "ddqn" / "dueling_dqn" for clarity in
   experiment naming, but the flags themselves can co-occur).
4. Discrete SAC (Christodoulou, 2019 -- "Soft Actor-Critic for
   Discrete Action Settings"): the maximum-entropy actor-critic
   formulation of SAC adapted to a Discrete action space. Instead of
   a Gaussian policy over a continuous action, the actor outputs a
   categorical distribution over the `num_candidates` discrete
   actions. Two critics (Q1, Q2) are trained with the standard
   "take the min to fight overestimation" trick from TD3/SAC, and the
   policy is trained to maximize Q(s,a) + alpha * H(pi(.|s)), i.e. to
   prefer high-value actions while staying as stochastic as the
   entropy temperature alpha allows. We use automatic entropy-
   temperature tuning (log_alpha is itself an optimized parameter)
   targeting a entropy close to log(num_actions) * target_entropy_ratio,
   which removes the need to hand-pick a fixed alpha per environment.

WHY THESE FOUR AND NOT, SAY, RAINBOW OR A3C
---------------------------------------------
The brief was to compare a masked on-policy method (MaskablePPO, in
train_ppo_ems.py) against a family of off-policy value-based learners
(DQN and two of its most common, well-cited single-trick variants)
plus one off-policy actor-critic method that natively reasons about
exploration via entropy (SAC) instead of via an epsilon schedule. This
gives an interview-ready 2x2-ish design: {on-policy, off-policy} x
{actor-critic, value-based} x {epsilon-greedy exploration, entropy-
regularized exploration} without going as far as a full Rainbow stack
(prioritized replay, n-step returns, distributional Q, noisy nets),
which would add many more hyperparameters to defend without changing
the core comparison being made: does explicit safety/action masking
(PPO) outperform a fixed small candidate menu (DQN family / SAC) on
this dispatch task, and does entropy regularization (SAC) explore the
candidate menu more effectively than epsilon-greedy (DQN family)?

REPLAY BUFFER
--------------
A single flat, NumPy-backed circular buffer (`ReplayBuffer`) is shared
in spirit (same implementation) by all four algorithms; SAC and DQN
only differ in how they *consume* sampled batches, not in how
transitions are stored. This keeps memory bounded (`buffer_size`
transitions of float32 obs/next_obs + small scalars) and sampling O(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "custom_rl_agents.py requires PyTorch. Install with "
        "`pip install torch --break-system-packages`."
    ) from exc


# ----------------------------------------------------------------------
# Replay buffer (shared by DQN family and Discrete SAC)
# ----------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s', done) transitions.

    Stored as pre-allocated NumPy arrays (not a Python deque of tuples)
    so that sampling a batch is a single vectorized fancy-index
    operation rather than `random.sample` + per-item stacking. This
    matters here because obs_dim can be in the thousands of dimensions
    (EMSEnv's base observation alone is large; CandidateActionEnv adds
    num_candidates * 7 more floats on top), so avoiding per-transition
    Python-object overhead is worth the upfront allocation cost.
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        idx = self._ptr
        self.obs[idx] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


def _mlp(input_dim: int, output_dim: int, hidden_sizes=(256, 256)) -> nn.Sequential:
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------
# DQN family (vanilla / Double / Dueling)
# ----------------------------------------------------------------------
@dataclass
class DQNConfig:
    obs_dim: int
    action_dim: int
    lr: float = 1e-4
    gamma: float = 0.97
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_interval: int = 2_000
    epsilon_decay_steps: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    double_dqn: bool = False
    dueling: bool = False
    hidden_sizes: tuple = (256, 256)
    grad_clip_norm: float = 10.0


class _DuelingQNetwork(nn.Module):
    """Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)).

    The shared torso encodes the state once; two small heads branch off
    it for the scalar value and the per-action advantage. Subtracting
    the mean advantage (rather than e.g. the max) is the formulation
    from Wang et al. (2016) -- it is smoother to optimize and avoids
    letting a single action's advantage estimate dominate V's identity.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        *trunk_sizes, last_hidden = hidden_sizes
        trunk_layers = []
        prev = obs_dim
        for h in (trunk_sizes or [hidden_sizes[0]]):
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        trunk_out = prev if trunk_layers else obs_dim

        self.value_head = nn.Sequential(
            nn.Linear(trunk_out, last_hidden), nn.ReLU(), nn.Linear(last_hidden, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(trunk_out, last_hidden), nn.ReLU(), nn.Linear(last_hidden, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        value = self.value_head(features)                       # (B, 1)
        advantage = self.advantage_head(features)                # (B, A)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DQNAgent:
    """Vanilla DQN, optionally Double-DQN and/or Dueling-DQN.

    The two variants are deliberately implemented as flags on a single
    class rather than three separate classes: the only thing that
    changes between DQN / Double DQN is one line in `_compute_targets`
    (which network proposes the greedy next action), and the only
    thing that changes for Dueling DQN is the network architecture
    used for `self.q_net` / `self.target_q_net`. Keeping them unified
    makes it obvious in an interview that "Double" and "Dueling" are
    orthogonal, independently-explainable tricks layered on the same
    base algorithm, not three unrelated algorithms.
    """

    def __init__(self, config: DQNConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.action_dim = config.action_dim

        net_cls = _DuelingQNetwork if config.dueling else None
        if net_cls is not None:
            self.q_net = net_cls(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
            self.target_q_net = net_cls(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
        else:
            self.q_net = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
            self.target_q_net = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.obs_dim)

        self._step_count = 0

    # -- exploration schedule --------------------------------------------------
    def _epsilon(self) -> float:
        frac = min(self._step_count / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        self._step_count += 1
        epsilon = 0.0 if deterministic else self._epsilon()
        if not deterministic and np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))

        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
            q_values = self.q_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    # -- learning ---------------------------------------------------------------
    def train_step(self) -> Dict[str, Any]:
        if len(self.replay_buffer) < max(self.config.learning_starts, self.config.batch_size):
            return {"epsilon": self._epsilon()}

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        obs, actions, rewards, next_obs, dones = (
            batch["obs"], batch["actions"], batch["rewards"], batch["next_obs"], batch["dones"]
        )

        with torch.no_grad():
            if self.config.double_dqn:
                # Online network SELECTS the greedy next action...
                next_q_online = self.q_net(next_obs)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                # ...target network EVALUATES that action's value.
                next_q_target = self.target_q_net(next_obs)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_q_net(next_obs).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.config.gamma * next_q

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        if self._step_count % max(self.config.target_update_interval, 1) == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": float(loss.item()),
            "epsilon": self._epsilon(),
            "mean_q": float(q_values.mean().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])


# ----------------------------------------------------------------------
# Discrete SAC
# ----------------------------------------------------------------------
@dataclass
class DiscreteSACConfig:
    obs_dim: int
    action_dim: int
    lr: float = 1e-4
    gamma: float = 0.97
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_freq: int = 4
    tau: float = 0.005                 # Polyak averaging coefficient for target critics
    target_entropy_ratio: float = 0.98  # target entropy = ratio * log(action_dim)
    hidden_sizes: tuple = (256, 256)
    grad_clip_norm: float = 10.0


class DiscreteSACAgent:
    """Discrete-action Soft Actor-Critic (Christodoulou, 2019).

    Differences from continuous SAC that are worth being able to
    explain out loud:
    - The actor is a categorical policy: a softmax over `action_dim`
      logits, not a Gaussian mean/std over a continuous vector. No
      reparameterization trick (rsample) is needed because we can
      compute the exact expectation over the (small, discrete) action
      set in closed form -- see `_actor_loss` and `_critic_targets`
      below, both of which sum over `action_dim` rather than drawing a
      Monte-Carlo sample of actions.
    - Two critics Q1, Q2 (and their Polyak-averaged targets) are kept,
      following the double-Q trick that both TD3 and continuous SAC
      use to counter overestimation bias; each outputs a full
      (action_dim,) vector of Q-values per state rather than a single
      scalar for a given (s,a) pair, since with a discrete action
      space it is cheap to just evaluate every action at once.
    - Entropy temperature alpha is learned automatically by gradient
      descent on log_alpha, targeting a fixed entropy level
      `target_entropy_ratio * log(action_dim)`. A ratio just under 1.0
      keeps the policy close to (but not exactly at) maximum entropy
      early in training, which in this environment is desirable: the
      candidate list per decision point is already pre-filtered to
      "reasonable" (ambulance, hospital) pairs by
      `CandidateActionEnv`, so encouraging near-uniform exploration
      over that *already curated* shortlist is cheap and helps the
      critic see a wide range of (state, action) pairs early on.
    """

    def __init__(self, config: DiscreteSACConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.action_dim = config.action_dim

        self.actor = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
        self.critic_1 = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
        self.critic_2 = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
        self.target_critic_1 = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
        self.target_critic_2 = _mlp(config.obs_dim, config.action_dim, config.hidden_sizes).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=config.lr
        )

        self.target_entropy = config.target_entropy_ratio * float(np.log(config.action_dim))
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr)

        self.replay_buffer = ReplayBuffer(config.buffer_size, config.obs_dim)
        self._step_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        self._step_count += 1
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
            logits = self.actor(obs_t)
            probs = F.softmax(logits, dim=1)
            if deterministic:
                return int(torch.argmax(probs, dim=1).item())
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    def train_step(self) -> Dict[str, Any]:
        if len(self.replay_buffer) < max(self.config.learning_starts, self.config.batch_size):
            return {"alpha": float(self.alpha.item())}

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        obs, actions, rewards, next_obs, dones = (
            batch["obs"], batch["actions"], batch["rewards"], batch["next_obs"], batch["dones"]
        )

        # ---- Critic update ----
        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_probs = F.softmax(next_logits, dim=1)                  # (B, A)
            next_log_probs = F.log_softmax(next_logits, dim=1)          # (B, A)

            next_q1 = self.target_critic_1(next_obs)                    # (B, A)
            next_q2 = self.target_critic_2(next_obs)                    # (B, A)
            next_q_min = torch.min(next_q1, next_q2)                    # (B, A)

            # Exact expectation over the discrete action set (no sampling
            # needed): E_{a~pi}[Q(s',a) - alpha*log pi(a|s')]
            soft_state_value = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=1)
            targets = rewards + (1.0 - dones) * self.config.gamma * soft_state_value

        q1_pred = self.critic_1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.critic_2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1_pred, targets) + F.mse_loss(q2_pred, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.config.grad_clip_norm,
        )
        self.critic_optimizer.step()

        # ---- Actor update ----
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            q1 = self.critic_1(obs)
            q2 = self.critic_2(obs)
            q_min = torch.min(q1, q2)
        # Maximize E_{a~pi}[Q(s,a) - alpha*log pi(a|s)]  <=>  minimize the negative.
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()

        # ---- Temperature update ----
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=1)     # (B,)
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy.detach())).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Polyak-average target critics ----
        with torch.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.mul_(1.0 - self.config.tau).add_(self.config.tau * param)
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.mul_(1.0 - self.config.tau).add_(self.config.tau * param)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(entropy.mean().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.log_alpha = checkpoint["log_alpha"].clone().to(self.device).requires_grad_(True)
