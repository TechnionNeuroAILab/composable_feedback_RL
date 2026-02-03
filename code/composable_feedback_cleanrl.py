"""
composable_feedback_cleanrl.py

Single-file, CleanRL-style DQN trainer with five learning options:
  (standard) Standard cleanRL DQN: exact implementation of cleanRL's standard DQN algorithm.
  (a) Scalar DQN: classic single scalar TD error (corresponds to Algorithm 1 from paper).
  (b) Outcome-specific multi-head DQN: multiple TD errors for different outcome targets o^(k) (corresponds to Algorithm 2 from paper).
  (c) Feature-specific PE (global update): decompose TD target into feature-wise components, but learn from summed scalar error (corresponds to Algorithm 3 from paper).
  (c_local) Feature-specific PE (local vectorial errors): modified C algorithm that explicitly computes vectorial errors δ_i = y_i - q_i and uses each element locally on its corresponding feature-specific branch.
  (d) Feature-specific PE (local update): learn with per-feature TD targets (local vector-PE learning).

Designed to run on a simple Gymnasium task (default CartPole-v1).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


# -----------------------------
# Args
# -----------------------------


@dataclass
class Args:
    # experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False

    # env / algo
    env_id: str = "CartPole-v1"
    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 50_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    learning_starts: int = 10_000
    train_frequency: int = 10

    # epsilon-greedy
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5

    # option selector
    option: str = "standard"  # one of: standard,a,b,c,c_local,d

    # option (b): number of outcome heads and how to weight them for action selection
    num_outcomes: int = 2
    policy_outcome_idx: int = 0  # which outcome-head to use for behavior policy (default: reward head)

    # option (c,d): feature decomposition size
    num_features: int = 32  # size of learned feature vector h(s)

    # logging
    log_interval: int = 100


# -----------------------------
# Utils
# -----------------------------


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def set_seed(seed: int, torch_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


# -----------------------------
# Replay Buffer (minimal)
# -----------------------------


class ReplayBuffer:
    def __init__(self, obs_shape, buffer_size: int, device: torch.device):
        self.device = device
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

        self.obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int):
        max_idx = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, max_idx, size=batch_size)
        obs = torch.tensor(self.obs[idx], device=self.device)
        next_obs = torch.tensor(self.next_obs[idx], device=self.device)
        actions = torch.tensor(self.actions[idx], device=self.device)
        rewards = torch.tensor(self.rewards[idx], device=self.device)
        dones = torch.tensor(self.dones[idx], device=self.device)
        return obs, next_obs, actions, rewards, dones


# -----------------------------
# Networks
# -----------------------------


class StandardQNetwork(nn.Module):
    """
    Standard cleanRL DQN: exact implementation matching cleanRL's QNetwork.
    Uses standard TD error computation: r + γ * max_a' Q(s', a') * (1 - done)
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ScalarQNetwork(nn.Module):
    """Option (a): classic DQN Q(s, a)."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class OutcomeSpecificQNetwork(nn.Module):
    """
    Option (b): outcome-specific heads.
    A shared trunk + K heads, each head k predicts Q_k(s, a) for outcome o^(k).
    """

    def __init__(self, obs_dim: int, n_actions: int, num_outcomes: int):
        super().__init__()
        self.num_outcomes = num_outcomes
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(64, n_actions) for _ in range(num_outcomes)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns Q heads stacked: [B, K, A]
        """
        h = self.trunk(obs)
        qs = [head(h) for head in self.heads]
        return torch.stack(qs, dim=1)


class FeatureDecomposedQNetwork(nn.Module):
    """
    Options (c,d): feature-specific decomposition.

    Learn a feature vector h(s) in R^F, then decompose Q(s, a) as a sum over feature-channels:
      Q(s, a) = sum_{i=1..F} Q_i(s, a) + b_a
      Q_i(s, a) = h_i(s) * W_{i,a}

    This makes per-feature prediction errors δ_i meaningful:
      δ_i = r_i + γ Q_i^-(s', a*) - Q_i(s, a)
    where a* = argmax_a Q^-(s', a).
    """

    def __init__(self, obs_dim: int, n_actions: int, num_features: int):
        super().__init__()
        self.n_actions = n_actions
        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.ReLU(),
        )
        # Per-feature action weights W_{i,a}
        self.W = nn.Parameter(torch.empty(num_features, n_actions))
        nn.init.xavier_uniform_(self.W)
        # Action bias b_a
        self.b = nn.Parameter(torch.zeros(n_actions))

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)  # [B, F]

    def q_components(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns per-feature contributions: [B, F, A] with Q_i(s,a) = h_i(s)*W_{i,a}
        """
        h = self.features(obs)  # [B, F]
        # broadcast multiply: [B,F,1] * [1,F,A] -> [B,F,A]
        return h.unsqueeze(-1) * self.W.unsqueeze(0)

    def q_total(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns total Q(s,a): [B, A]
        """
        comps = self.q_components(obs)  # [B,F,A]
        return comps.sum(dim=1) + self.b.unsqueeze(0)


# -----------------------------
# Option-specific outcomes / targets
# -----------------------------


def compute_outcomes_b(
    rewards: torch.Tensor, next_obs: torch.Tensor, num_outcomes: int
) -> torch.Tensor:
    """
    Option (b): define outcome vector o^(k).

    Minimal example for CartPole:
      o^0 = environment reward
      o^1 = shaping signal based on next state's pole angle (encourages small angle)

    Returns o: [B, K]
    """
    B = rewards.shape[0]
    o = torch.zeros((B, num_outcomes), device=rewards.device, dtype=torch.float32)
    o[:, 0] = rewards
    if num_outcomes >= 2:
        # CartPole obs: [x, x_dot, theta, theta_dot]
        theta = next_obs[:, 2]
        o[:, 1] = -theta.abs().clamp(max=0.5)  # bounded auxiliary outcome
    # if more heads requested, leave as zeros (user can customize)
    return o


def uniform_reward_split(reward: torch.Tensor, num_parts: int) -> torch.Tensor:
    """
    Split scalar reward into K parts r_i such that sum_i r_i = reward.
    Returns [B, K].
    """
    return reward.unsqueeze(1).repeat(1, num_parts) / float(num_parts)


# -----------------------------
# Main
# -----------------------------


def main():
    args = tyro.cli(Args)
    assert args.option in {"standard", "a", "b", "c", "c_local", "d"}, "args.option must be one of: standard,a,b,c,c_local,d"

    run_name = f"{args.env_id}__{args.exp_name}__opt{args.option}__seed{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]))

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.env_id, args.seed)
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action spaces supported."

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    # build model(s)
    if args.option == "standard":
        q_net: nn.Module = StandardQNetwork(obs_dim, n_actions).to(device)
        target_net: nn.Module = StandardQNetwork(obs_dim, n_actions).to(device)
    elif args.option == "a":
        q_net: nn.Module = ScalarQNetwork(obs_dim, n_actions).to(device)
        target_net: nn.Module = ScalarQNetwork(obs_dim, n_actions).to(device)
    elif args.option == "b":
        q_net = OutcomeSpecificQNetwork(obs_dim, n_actions, args.num_outcomes).to(device)
        target_net = OutcomeSpecificQNetwork(obs_dim, n_actions, args.num_outcomes).to(device)
    else:  # c, c_local, or d
        q_net = FeatureDecomposedQNetwork(obs_dim, n_actions, args.num_features).to(device)
        target_net = FeatureDecomposedQNetwork(obs_dim, n_actions, args.num_features).to(device)

    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(env.observation_space.shape, args.buffer_size, device=device)

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.total_timesteps), global_step)

        # action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                if args.option == "standard":
                    q_vals = q_net(obs_t)  # [1,A]
                elif args.option == "a":
                    q_vals = q_net(obs_t)  # [1,A]
                elif args.option == "b":
                    q_heads = q_net(obs_t)  # [1,K,A]
                    q_vals = q_heads[:, args.policy_outcome_idx, :]  # policy uses one head
                else:  # c, c_local, or d
                    q_vals = q_net.q_total(obs_t)  # [1,A]
                action = int(torch.argmax(q_vals, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        rb.add(obs, next_obs, action, reward, done)

        episode_return += float(reward)
        episode_length += 1

        obs = next_obs

        if done:
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            obs, _ = env.reset(seed=args.seed)
            episode_return = 0.0
            episode_length = 0

        # training
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)

            if args.option == "standard":
                # Standard cleanRL DQN: exact implementation from cleanRL
                with torch.no_grad():
                    target_max, _ = target_net(b_next_obs).max(dim=1)
                    td_target = b_rewards + args.gamma * target_max * (1.0 - b_dones)
                old_val = q_net(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

            elif args.option == "a":
                with torch.no_grad():
                    target_max = target_net(b_next_obs).max(dim=1).values
                    y = b_rewards + args.gamma * target_max * (1.0 - b_dones)
                q_sa = q_net(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_sa, y)

            elif args.option == "b":
                # outcome-specific TD for each head
                o = compute_outcomes_b(b_rewards, b_next_obs, args.num_outcomes)  # [B,K]
                with torch.no_grad():
                    q_next_heads = target_net(b_next_obs)  # [B,K,A]
                    target_max = q_next_heads.max(dim=2).values  # [B,K]
                    y = o + args.gamma * target_max * (1.0 - b_dones.unsqueeze(1))  # [B,K]

                q_heads = q_net(b_obs)  # [B,K,A]
                q_sa = q_heads.gather(2, b_actions.view(-1, 1, 1).expand(-1, args.num_outcomes, 1)).squeeze(2)  # [B,K]
                loss = F.mse_loss(q_sa, y)

            else:
                # feature-specific decomposition for DQN target using a* from total Q
                # 1) compute next greedy action a* using target total Q(s',a)
                with torch.no_grad():
                    q_next_total = target_net.q_total(b_next_obs)  # [B,A]
                    a_star = torch.argmax(q_next_total, dim=1)  # [B]

                    # 2) compute per-feature target components for chosen a*
                    r_parts = uniform_reward_split(b_rewards, args.num_features)  # [B,F]
                    q_next_comps = target_net.q_components(b_next_obs)  # [B,F,A]
                    q_next_i = q_next_comps.gather(2, a_star.view(-1, 1, 1).expand(-1, args.num_features, 1)).squeeze(2)  # [B,F]
                    y_i = r_parts + args.gamma * q_next_i * (1.0 - b_dones.unsqueeze(1))  # [B,F]

                q_comps = q_net.q_components(b_obs)  # [B,F,A]
                q_i = q_comps.gather(2, b_actions.view(-1, 1, 1).expand(-1, args.num_features, 1)).squeeze(2)  # [B,F]

                if args.option == "c":
                    # global update: sum targets and predictions -> scalar loss
                    y = y_i.sum(dim=1)  # [B]
                    q_sa = (q_i.sum(dim=1) + q_net.b[b_actions])  # [B] (add bias for taken action)
                    loss = F.mse_loss(q_sa, y)
                elif args.option == "c_local":
                    # local vectorial errors: explicitly compute δ_i = y_i - q_i and use each element locally
                    delta_i = y_i - q_i  # [B, F] - explicit vectorial error
                    loss_i = delta_i ** 2  # [B, F] - per-feature squared errors
                    loss = loss_i.sum(dim=1).mean()  # sum over features, mean over batch
                else:  # d
                    # local update: per-feature loss uses vector PEs directly
                    loss = F.mse_loss(q_i, y_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.log_interval == 0:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # target update
        if global_step % args.target_network_frequency == 0:
            for tp, p in zip(target_net.parameters(), q_net.parameters()):
                tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

    env.close()
    writer.close()


if __name__ == "__main__":
    main()

