"""
run_cleanrl_vs_mixed_dqn.py

Clean implementation of CleanRL DQN and comparable mixed-backprop DQN.
Same architecture (120 -> ReLU -> 84 -> ReLU -> n_actions), same hyperparameters,
step-based training. Train both and compare results.

- CleanRL DQN: from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
  Single MLP, standard TD loss, full backprop.
- Mixed backprop: same layout with first layer as linear features z; update z with
  fixed sparse B and delta_z = B*e; trunk (ReLU->84->ReLU->n_actions) with full backprop.

No cleanrl_utils dependency: minimal replay buffer and env loop in this file.

Usage:
  python run_cleanrl_vs_mixed_dqn.py [--algos dqn,mixed] [--seeds 1,2,3] [--total-timesteps 200000] [--output-dir results]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ReplayBufferSamples: observations, actions, next_observations, dones, rewards (all tensors)
ReplayBufferSamples = namedtuple(
    "ReplayBufferSamples",
    ["observations", "actions", "next_observations", "dones", "rewards"],
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- CleanRL defaults (from cleanrl/dqn.py) ----------
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAMMA_SHORT = 0.9  # for multihead head 3 (short-term critic)
BUFFER_SIZE = 10_000
BATCH_SIZE = 128
START_E = 1.0
END_E = 0.05
EXPLORATION_FRACTION = 0.5
LEARNING_STARTS = 10_000
TRAIN_FREQUENCY = 10
TARGET_NETWORK_FREQUENCY = 500
TAU = 1.0
# Mixed-backprop specific: smaller LR for linear layer for stability; more frequent target sync
LR_LINEAR_MIXED = 1e-4
TARGET_NETWORK_FREQUENCY_MIXED = 250
FEEDBACK_GRAD_CLIP = 1.0  # max norm for W_z gradient update (0 = no clip)
# Multihead: weight combined loss higher so the policy used for actions is driven to true return
MULTIHEAD_COMBINED_LOSS_WEIGHT = 2.0  # loss = sum(head losses) + this * loss_combined
MULTIHEAD_UNIFIED_TARGETS = True  # if True, all heads use reward return (no cos(theta) or gamma_short)
TARGET_NETWORK_FREQUENCY_MULTIHEAD = 500  # same as DQN for more stable targets (was 250)
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_N_EPISODES = 20_000
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_ALGOS = ["dqn", "mixed", "mixed3f"]
MULTIPLIER_3F = 3  # 3F output channels; each action Q = mean of 3F/A channels

# Architecture: match CleanRL exactly
OBS_TO_120 = 120
HIDDEN_84 = 84


def set_seed(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ---------- Minimal Replay Buffer (no cleanrl_utils) ----------
class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.pos = 0
        self.full = False
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: float,
    ) -> None:
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        return ReplayBufferSamples(
            observations=torch.tensor(self.observations[idx], device=self.device),
            actions=torch.tensor(self.actions[idx], device=self.device).unsqueeze(1),
            next_observations=torch.tensor(self.next_observations[idx], device=self.device),
            dones=torch.tensor(self.dones[idx], device=self.device).unsqueeze(1),
            rewards=torch.tensor(self.rewards[idx], device=self.device).unsqueeze(1),
        )


# ---------- CleanRL DQN network (120 -> ReLU -> 84 -> ReLU -> n_actions) ----------
class CleanRLQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, OBS_TO_120),
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
            nn.Linear(HIDDEN_84, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward_q_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ---------- Mixed backprop: same architecture, first layer = z, rest = trunk ----------
def make_fixed_sparse_B(
    F: int, A: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Fixed sparse B in R^{F x A}: row i has one non-zero at column i % A."""
    B = torch.zeros(F, A, device=device, dtype=dtype)
    for i in range(F):
        B[i, i % A] = 1.0
    return B


def make_fixed_sparse_B_3f(
    F: int, device: torch.device, dtype: torch.dtype, n_channels_per_feature: int = 3
) -> torch.Tensor:
    """B in R^{F x 3F}: row i has exactly n_channels_per_feature non-zeros at columns [3i, 3i+1, 3i+2], value 1/3 so delta_z_i = mean of those 3 errors."""
    K = n_channels_per_feature * F  # 3F
    B = torch.zeros(F, K, device=device, dtype=dtype)
    for i in range(F):
        for j in range(n_channels_per_feature):
            c = n_channels_per_feature * i + j
            B[i, c] = 1.0 / n_channels_per_feature
    return B


def make_fixed_sparse_B_5heads(
    F: int, n_heads: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """B in R^{F x n_heads}: row i has one non-zero at column i % n_heads with value 1."""
    B = torch.zeros(F, n_heads, device=device, dtype=dtype)
    for i in range(F):
        B[i, i % n_heads] = 1.0
    return B


class MixedBackpropQNetwork(nn.Module):
    """
    Same layout as CleanRL: obs -> 120 (z) -> ReLU -> 84 -> ReLU -> n_actions.
    Split as: linear_feature (obs -> 120), trunk (ReLU, Linear(120,84), ReLU, Linear(84, n_actions)).
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)  # z
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
            nn.Linear(HIDDEN_84, n_actions),
        )

    def forward(self, x: torch.Tensor, detach_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.linear_feature(x)
        h = z.detach() if detach_z else z
        q = self.trunk(h)
        return q, z

    def forward_q_only(self, x: torch.Tensor) -> torch.Tensor:
        q, _ = self.forward(x, detach_z=False)
        return q


class MixedBackprop3FQNetwork(nn.Module):
    """
    3F output: trunk produces 3*F channels. Q(a) = mean of channels in group a.
    Groups: action a gets channels [a*(3F/A) : (a+1)*(3F/A)]. F = OBS_TO_120.
    B is F x 3F with 3 non-zeros per row (cols 3i, 3i+1, 3i+2), value 1/3.
    """

    def __init__(self, obs_dim: int, n_actions: int, F: int = OBS_TO_120, mult: int = MULTIPLIER_3F):
        super().__init__()
        self.n_actions = n_actions
        self.F = F
        self.K = K = mult * F  # 3F
        assert K % n_actions == 0, "3F must be divisible by n_actions"
        self.group_size = K // n_actions
        self.linear_feature = nn.Linear(obs_dim, F)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(F, HIDDEN_84),
            nn.ReLU(),
            nn.Linear(HIDDEN_84, K),
        )

    def forward_raw(self, x: torch.Tensor, detach_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_raw [B, K], z [B, F])."""
        z = self.linear_feature(x)
        h = z.detach() if detach_z else z
        q_raw = self.trunk(h)
        return q_raw, z

    def q_from_raw(self, q_raw: torch.Tensor) -> torch.Tensor:
        """Q(a) = mean of group a. Returns [B, A]."""
        B = q_raw.shape[0]
        q_actions = q_raw.new_empty(B, self.n_actions)
        for a in range(self.n_actions):
            start = a * self.group_size
            end = start + self.group_size
            q_actions[:, a] = q_raw[:, start:end].mean(dim=1)
        return q_actions

    def forward_q_only(self, x: torch.Tensor) -> torch.Tensor:
        q_raw, _ = self.forward_raw(x, detach_z=False)
        return self.q_from_raw(q_raw)


# ---------- Multihead: 5 critics (full, uprightness, gamma_short, cart view, pole view) + trainable w ----------
N_HEADS = 5


class MultiHeadMixedQNetwork(nn.Module):
    """
    Shared linear_feature (obs -> 120) and trunk (ReLU -> 84 -> ReLU) outputting h.
    Five heads: head1..head5 each Linear(84, n_actions). Heads 1-3 use full state; head4 uses masked cart (x, xdot, 0, 0); head5 uses masked pole (0, 0, theta, thetadot).
    Trainable weights w = softmax(logits) for Q_combined = sum_k w_k Q_k.
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        # Trunk outputs h in R^84 (no final readout to actions)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(HIDDEN_84, n_actions)
        self.head2 = nn.Linear(HIDDEN_84, n_actions)
        self.head3 = nn.Linear(HIDDEN_84, n_actions)
        self.head4 = nn.Linear(HIDDEN_84, n_actions)
        self.head5 = nn.Linear(HIDDEN_84, n_actions)
        # Bias combination toward head 1 (full-state reward) so early policy ≈ DQN and can reach 500
        self.weight_logits = nn.Parameter(torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32))

    @staticmethod
    def _mask_cart(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone()
        out[..., 2:4] = 0.0
        return out

    @staticmethod
    def _mask_pole(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone()
        out[..., 0:2] = 0.0
        return out

    @staticmethod
    def cos_theta_next(next_obs: torch.Tensor) -> torch.Tensor:
        """cos(theta') for uprightness target; next_obs[..., 2] is theta in radians. Returns (B,) for batch."""
        return torch.cos(next_obs[..., 2])

    def forward(
        self, obs: torch.Tensor, detach_z: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (Q1, Q2, Q3, Q4, Q5, Q_combined, z) with z from full-state pass."""
        # Full state
        z = self.linear_feature(obs)
        h = z.detach() if detach_z else z
        h = self.trunk(h)
        Q1 = self.head1(h)
        Q2 = self.head2(h)
        Q3 = self.head3(h)
        # Masked cart
        obs_cart = self._mask_cart(obs)
        z_cart = self.linear_feature(obs_cart)
        h_cart = self.trunk(z_cart)
        Q4 = self.head4(h_cart)
        # Masked pole
        obs_pole = self._mask_pole(obs)
        z_pole = self.linear_feature(obs_pole)
        h_pole = self.trunk(z_pole)
        Q5 = self.head5(h_pole)
        # Combined
        w = F.softmax(self.weight_logits, dim=0)
        Q_combined = (
            w[0] * Q1 + w[1] * Q2 + w[2] * Q3 + w[3] * Q4 + w[4] * Q5
        )
        return Q1, Q2, Q3, Q4, Q5, Q_combined, z

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, Q_combined, _ = self.forward(obs, detach_z=False)
        return Q_combined


# ---------- Training loop (step-based or episode-based; algo switches update rule) ----------
def run_training(
    algo: str,
    env_id: str,
    seed: int,
    device: torch.device,
    total_timesteps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[List[float], List[int], List[float]]:
    """
    Run 'dqn', 'mixed', or 'mixed3f'. If n_episodes is set, run until that many episodes; else use total_timesteps.
    Returns: episode_returns, episode_steps (global step when episode ended), episode_losses (mean loss in that episode).
    """
    assert (total_timesteps is not None) != (n_episodes is not None), "Set exactly one of total_timesteps or n_episodes"
    set_seed(seed)
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    if algo == "dqn":
        q_network = CleanRLQNetwork(obs_dim, n_actions).to(device)
        target_network = CleanRLQNetwork(obs_dim, n_actions).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
        B_fixed = None
        B_3f = None
    elif algo == "mixed":
        q_network = MixedBackpropQNetwork(obs_dim, n_actions).to(device)
        target_network = MixedBackpropQNetwork(obs_dim, n_actions).to(device)
        B_fixed = make_fixed_sparse_B(OBS_TO_120, n_actions, device, torch.float32)
        B_3f = None
        optimizer = optim.Adam(q_network.trunk.parameters(), lr=LEARNING_RATE)
    elif algo == "mixed3f":
        q_network = MixedBackprop3FQNetwork(obs_dim, n_actions).to(device)
        target_network = MixedBackprop3FQNetwork(obs_dim, n_actions).to(device)
        B_fixed = None
        B_3f = make_fixed_sparse_B_3f(OBS_TO_120, device, torch.float32)
        optimizer = optim.Adam(q_network.trunk.parameters(), lr=LEARNING_RATE)
    elif algo == "multihead":
        q_network = MultiHeadMixedQNetwork(obs_dim, n_actions).to(device)
        target_network = MultiHeadMixedQNetwork(obs_dim, n_actions).to(device)
        B_fixed = None
        B_3f = None
        B_multi = make_fixed_sparse_B_5heads(OBS_TO_120, N_HEADS, device, torch.float32)
        optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size=BUFFER_SIZE,
        obs_shape=env.observation_space.shape,
        device=device,
    )

    # Match CleanRL: epsilon decays over first 250k steps so learning can exploit in second half
    if total_timesteps is not None:
        exploration_duration = int(EXPLORATION_FRACTION * total_timesteps)
        max_steps = total_timesteps
    else:
        # ~200 steps/episode on average -> decay over 250k steps (like CleanRL 500k run)
        exploration_duration = int(EXPLORATION_FRACTION * 500_000)
        max_steps = 10_000_000
    episode_returns: List[float] = []
    episode_steps: List[int] = []
    episode_losses: List[float] = []
    loss_list_this_episode: List[float] = []

    obs, _ = env.reset(seed=seed)
    global_step = 0
    while global_step < max_steps:
        if n_episodes is not None and len(episode_returns) >= n_episodes:
            break
        if total_timesteps is not None and global_step >= total_timesteps:
            break
        epsilon = linear_schedule(
            START_E, END_E, exploration_duration, global_step
        )
        if random.random() < epsilon:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_network.forward_q_only(x)
            action = int(q_values.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        real_next_obs = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_next_obs = infos["final_observation"]

        rb.add(obs, real_next_obs, action, float(reward), float(done))

        # Episode stats
        if "episode" in infos:
            ep_info = infos["episode"]
            episode_returns.append(ep_info["r"])
            episode_steps.append(global_step)
            mean_loss = (
                np.mean(loss_list_this_episode) if loss_list_this_episode else 0.0
            )
            episode_losses.append(mean_loss)
            loss_list_this_episode.clear()

        obs = next_obs
        if done:
            obs, _ = env.reset()

        # Training
        if global_step > LEARNING_STARTS and global_step % TRAIN_FREQUENCY == 0:
            data = rb.sample(BATCH_SIZE)
            if algo != "multihead":
                with torch.no_grad():
                    target_q = target_network.forward_q_only(data.next_observations)
                    target_max, _ = target_q.max(dim=1)
                    td_target = (
                        data.rewards.flatten()
                        + GAMMA * target_max * (1 - data.dones.flatten())
                    )
            if algo == "dqn":
                old_val = q_network.forward_q_only(data.observations).gather(
                    1, data.actions
                ).squeeze()
                loss = F.mse_loss(td_target, old_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif algo == "mixed":
                # Mixed: Q and z from current net (trunk sees z.detach())
                q_sa_b, z_b = q_network(data.observations, detach_z=True)
                q_sa = q_sa_b.gather(1, data.actions).squeeze()
                loss_td = F.mse_loss(td_target, q_sa)
                e = torch.zeros_like(q_sa_b, device=device)
                e.scatter_(1, data.actions, (td_target - q_sa).unsqueeze(1))
                delta_z = e @ B_fixed.t()  # (B, A) @ (A, F) = (B, F)
                grad_W_z = (delta_z.t() @ data.observations) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                optimizer.zero_grad()
                loss_td.backward()
                optimizer.step()
                loss = loss_td
            elif algo == "mixed3f":
                # mixed3f: 3F output; vectorized e: (B,K) = (td_err * spread[action])
                q_raw, _ = q_network.forward_raw(data.observations, detach_z=True)
                q_actions = q_network.q_from_raw(q_raw)
                q_sa = q_actions.gather(1, data.actions).squeeze()
                loss_td = F.mse_loss(td_target, q_sa)
                group_size = q_network.group_size
                K = q_network.K
                A = q_network.n_actions
                spread = torch.zeros(A, K, device=device, dtype=q_raw.dtype)
                for a in range(A):
                    spread[a, a * group_size : (a + 1) * group_size] = 1.0 / group_size
                td_err = (td_target - q_sa).unsqueeze(1)  # (B, 1)
                e = td_err * spread[data.actions.squeeze()]  # (B, K)
                delta_z = e @ B_3f.t()  # (B, K) @ (K, F) = (B, F)
                grad_W_z = (delta_z.t() @ data.observations) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                optimizer.zero_grad()
                loss_td.backward()
                optimizer.step()
                loss = loss_td
            elif algo == "multihead":
                # Five heads; five targets; e (B,5); delta_z = e @ B_multi.t(); combined loss for w
                Q1, Q2, Q3, Q4, Q5, Q_combined, z = q_network(data.observations, detach_z=True)
                with torch.no_grad():
                    tQ1, tQ2, tQ3, tQ4, tQ5, tQ_combined, _ = target_network(data.next_observations)
                    r = data.rewards.flatten()
                    d = data.dones.flatten()
                    max1, _ = tQ1.max(dim=1)
                    max2, _ = tQ2.max(dim=1)
                    max3, _ = tQ3.max(dim=1)
                    max4, _ = tQ4.max(dim=1)
                    max5, _ = tQ5.max(dim=1)
                    y1 = r + GAMMA * (1 - d) * max1
                    if MULTIHEAD_UNIFIED_TARGETS:
                        # All heads learn same reward return; avoids scale/objective conflict
                        y2 = y1.clone()
                        y3 = y1.clone()
                    else:
                        cos_next = MultiHeadMixedQNetwork.cos_theta_next(data.next_observations)
                        y2 = cos_next + GAMMA * (1 - d) * max2
                        y3 = r + GAMMA_SHORT * (1 - d) * max3
                    y4 = r + GAMMA * (1 - d) * max4
                    y5 = r + GAMMA * (1 - d) * max5
                    max_combined, _ = tQ_combined.max(dim=1)
                    y_combined = r + GAMMA * (1 - d) * max_combined
                q1_sa = Q1.gather(1, data.actions).squeeze()
                q2_sa = Q2.gather(1, data.actions).squeeze()
                q3_sa = Q3.gather(1, data.actions).squeeze()
                q4_sa = Q4.gather(1, data.actions).squeeze()
                q5_sa = Q5.gather(1, data.actions).squeeze()
                q_combined_sa = Q_combined.gather(1, data.actions).squeeze()
                e1 = (y1 - q1_sa).unsqueeze(1)
                e2 = (y2 - q2_sa).unsqueeze(1)
                e3 = (y3 - q3_sa).unsqueeze(1)
                e4 = (y4 - q4_sa).unsqueeze(1)
                e5 = (y5 - q5_sa).unsqueeze(1)
                e = torch.cat([e1, e2, e3, e4, e5], dim=1)
                delta_z = e @ B_multi.t()
                grad_W_z = (delta_z.t() @ data.observations) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                loss_td_heads = (
                    F.mse_loss(y1, q1_sa) + F.mse_loss(y2, q2_sa) + F.mse_loss(y3, q3_sa)
                    + F.mse_loss(y4, q4_sa) + F.mse_loss(y5, q5_sa)
                )
                loss_combined = F.mse_loss(y_combined, q_combined_sa)
                loss = loss_td_heads + MULTIHEAD_COMBINED_LOSS_WEIGHT * loss_combined
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = torch.tensor(0.0, device=device)

            loss_list_this_episode.append(loss.item())

        # Target network update
        target_freq = (
            TARGET_NETWORK_FREQUENCY_MULTIHEAD
            if algo == "multihead"
            else (TARGET_NETWORK_FREQUENCY_MIXED if algo in ("mixed", "mixed3f") else TARGET_NETWORK_FREQUENCY)
        )
        if global_step % target_freq == 0:
            for tp, qp in zip(
                target_network.parameters(), q_network.parameters()
            ):
                tp.data.copy_(
                    TAU * qp.data + (1.0 - TAU) * tp.data
                )

        if (global_step + 1) % 100_000 == 0 or global_step == 0:
            cap = total_timesteps if total_timesteps is not None else n_episodes
            log(
                f"  [{algo}] seed={seed} step={global_step + 1} (cap={cap}) "
                f"episodes={len(episode_returns)}"
            )
        global_step += 1

    env.close()

    # Checkpoint so training can be extended later
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "algo": algo,
            "seed": seed,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "q_network": q_network.state_dict(),
            "target_network": target_network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "n_episodes": len(episode_returns),
            "episode_returns": episode_returns,
            "episode_steps": episode_steps,
            "episode_losses": episode_losses,
        }
        path = checkpoint_dir / f"{algo}_seed{seed}.pt"
        torch.save(ckpt, path)
        log(f"  Checkpoint: {path}")

    return episode_returns, episode_steps, episode_losses


def plot_and_save(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not found; skipping figures.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    algos_seen = list(dict.fromkeys(r["algo"] for r in all_results))
    colors = {"dqn": "#1f77b4", "mixed": "#d62728", "mixed3f": "#2ca02c", "multihead": "#9467bd"}
    labels = {"dqn": "CleanRL DQN", "mixed": "Mixed backprop", "mixed3f": "Mixed 3F", "multihead": "Multihead (5 critics + trainable w)"}

    # Learning curves: episode index vs episodic return (mean across seeds per algo)
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in algos_seen:
        runs = [r for r in all_results if r["algo"] == algo]
        color = colors.get(algo, "#333")
        label = labels.get(algo, algo)
        for r in runs:
            returns = r["episode_returns"]
            ep = np.arange(1, len(returns) + 1)
            ax.plot(ep, returns, color=color, alpha=0.3, linewidth=0.8)
        if runs:
            max_len = max(len(r["episode_returns"]) for r in runs)
            arr = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                arr[i, : len(r["episode_returns"])] = r["episode_returns"]
            mean_ret = np.nanmean(arr, axis=0)
            ep_mean = np.arange(1, max_len + 1)
            w = min(50, max_len // 4)
            if w >= 1:
                smooth = np.convolve(mean_ret, np.ones(w) / w, mode="valid")
                ax.plot(ep_mean[w - 1 :], smooth, color=color, linewidth=2, label=label)
            else:
                ax.plot(ep_mean, mean_ret, color=color, linewidth=2, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.legend(loc="lower right")
    ax.set_title("CleanRL DQN vs Mixed backprop (CartPole-v1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cleanrl_vs_mixed_learning_curves.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / 'cleanrl_vs_mixed_learning_curves.pdf'}")

    # Loss (episode-wise mean loss)
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in algos_seen:
        runs = [r for r in all_results if r["algo"] == algo]
        color = colors.get(algo, "#333")
        label = labels.get(algo, algo)
        for r in runs:
            losses = r["episode_losses"]
            ep_idx = np.arange(1, len(losses) + 1)
            ax.plot(ep_idx, losses, color=color, alpha=0.3, linewidth=0.8)
        if runs:
            max_len = max(len(r["episode_losses"]) for r in runs)
            arr = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                arr[i, : len(r["episode_losses"])] = r["episode_losses"]
            mean_loss = np.nanmean(arr, axis=0)
            ep_idx = np.arange(1, len(mean_loss) + 1)
            w = min(20, max_len // 4)
            if w >= 1:
                smooth = np.convolve(mean_loss, np.ones(w) / w, mode="valid")
                ax.plot(ep_idx[w - 1 :], smooth, color=color, linewidth=2, label=label)
            else:
                ax.plot(ep_idx, mean_loss, color=color, linewidth=2, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean TD loss (per episode)")
    ax.legend(loc="upper right")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cleanrl_vs_mixed_loss.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / 'cleanrl_vs_mixed_loss.pdf'}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CleanRL DQN vs Mixed backprop, comparable setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS), help="dqn,mixed,mixed3f,multihead")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--total-timesteps", type=int, default=None, help="If set, run for this many steps")
    p.add_argument("--n-episodes", type=int, default=None, help="If set, run until this many episodes")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Save final model checkpoints here (e.g. results/checkpoints)")
    p.add_argument("--save-figures", action="store_true", default=True)
    p.add_argument("--no-save-figures", action="store_false", dest="save_figures")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    algos = [a for a in algos if a in ("dqn", "mixed", "mixed3f", "multihead")] or list(DEFAULT_ALGOS)
    if args.total_timesteps is None and args.n_episodes is None:
        args.n_episodes = DEFAULT_N_EPISODES
    if args.total_timesteps is not None and args.n_episodes is not None:
        args.n_episodes = None  # prefer total_timesteps
    seeds = []
    for s in args.seeds.split(","):
        try:
            seeds.append(int(s.strip()))
        except ValueError:
            pass
    if not seeds:
        seeds = DEFAULT_SEEDS

    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = root / checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 60)
    log("CleanRL DQN vs Mixed backprop (comparable)")
    log("=" * 60)
    log(f"  Algos: {algos}")
    log(f"  Env: {args.env}")
    log(f"  Seeds: {seeds}")
    log(f"  Total timesteps: {args.total_timesteps}")
    log(f"  N episodes: {args.n_episodes}")
    log(f"  Output: {output_dir}")
    if checkpoint_dir:
        log(f"  Checkpoints: {checkpoint_dir}")
    log("=" * 60)

    all_results: List[Dict] = []
    for algo in algos:
        for seed in seeds:
            log(f"\n[run] {algo} seed={seed} ...")
            t0 = time.perf_counter()
            episode_returns, episode_steps, episode_losses = run_training(
                algo=algo,
                env_id=args.env,
                seed=seed,
                device=device,
                total_timesteps=args.total_timesteps,
                n_episodes=args.n_episodes,
                checkpoint_dir=checkpoint_dir,
            )
            elapsed = time.perf_counter() - t0
            all_results.append({
                "algo": algo,
                "seed": seed,
                "episode_returns": episode_returns,
                "episode_steps": episode_steps,
                "episode_losses": episode_losses,
                "training_time_sec": elapsed,
                "n_episodes": len(episode_returns),
                "final_return": episode_returns[-1] if episode_returns else None,
                "mean_last_100": (
                    np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else None
                ),
            })
            log(f"  Done in {elapsed:.1f}s | episodes={len(episode_returns)} | final_return={all_results[-1]['final_return']}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_cleanrl_vs_mixed_{ts}.json"
    summary = {
        "config": {
            "algos": algos,
            "env": args.env,
            "seeds": seeds,
            "total_timesteps": args.total_timesteps,
            "n_episodes": args.n_episodes,
        },
        "runs": [
            {
                "algo": r["algo"],
                "seed": r["seed"],
                "n_episodes": r["n_episodes"],
                "final_return": r["final_return"],
                "mean_last_100": r["mean_last_100"],
                "training_time_sec": r["training_time_sec"],
            }
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary: {summary_path}")

    if args.save_figures:
        log("Saving figures ...")
        plot_and_save(all_results, output_dir)

    log("\n" + "=" * 60)
    log("SUMMARY")
    for algo in algos:
        runs = [r for r in all_results if r["algo"] == algo]
        rets = [r["final_return"] for r in runs if r["final_return"] is not None]
        if rets:
            log(f"  {algo}: final_return {np.mean(rets):.1f} ± {np.std(rets):.1f} (n={len(rets)})")
    log("=" * 60)


if __name__ == "__main__":
    main()
