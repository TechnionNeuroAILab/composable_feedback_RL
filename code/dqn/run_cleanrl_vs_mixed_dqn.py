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
import shutil
import sys
import time
from collections import namedtuple
from pathlib import Path

# Allow `python code/run_cleanrl_vs_mixed_dqn.py` from repo root
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from feature_contributions import (
    contribution_grad_times_h_cleanrl,
    contribution_grad_times_h_mixed,
    contribution_grad_times_h_multihead_combined,
    contribution_grad_times_h_multihead_head1,
    contribution_grad_times_h_threehead_full,
    numpy_dict_from_contrib,
)
from typing import Any, Dict, List, Optional, Tuple

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
MULTIHEAD_LOG_EVERY = 5000  # log softmax combination weights w during training (paper figures)
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_N_EPISODES = 20_000
DEFAULT_SEEDS = list(range(1, 11))  # seeds 1–10 for cross-seed consistency analysis
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
N_HEADS_THREE = 3  # full + cart + pole (proposal-style; no aux full-state heads)

# Multi-head algorithms that support confidence gating on W_z compartments (CartPole 4-state, 2 actions).
MULTIHEAD_CONFIDENCE_GATING_ALGOS = frozenset({"multihead", "mh_a", "mh_b", "mh_c", "mh_3a"})


def scaled_obs_for_manual_delta_W(
    obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor
) -> torch.Tensor:
    """
    When Q1 uses gated z = k4 * (W_cart @ x_cart) + k5 * (W_pole @ x_pole) + b with detached k,
    dL/dW_cart ∝ k4 * x_cart (per sample). This returns x scaled so (delta_z.T @ gx) / B matches
    the manual B-feedback outer product for CartPole-sized obs (last dim 4).
    """
    if obs.shape[-1] < 4:
        return obs
    gx = obs.clone()
    k4c = k4.to(dtype=gx.dtype, device=gx.device)
    k5c = k5.to(dtype=gx.dtype, device=gx.device)
    gx[:, :2] = gx[:, :2] * k4c
    gx[:, 2:4] = gx[:, 2:4] * k5c
    return gx


class ThreeHeadMixedQNetwork(nn.Module):
    """
    Three critics matching the cart/pole proposal: Q_full (full obs), Q_cart (pole masked),
    Q_pole (cart masked). Shared linear_feature and trunk; margins for gating from Q_cart and
    Q_pole (same |Q(right)-Q(left)| setup as the five-head head4/head5).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        confidence_gating: bool = False,
        gate_k_min: float = 0.8,
        gate_k_max: float = 1.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.confidence_gating = confidence_gating
        self.gate_k_min = float(gate_k_min)
        self.gate_k_max = float(gate_k_max)
        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head_full = nn.Linear(HIDDEN_84, n_actions)
        self.head_cart = nn.Linear(HIDDEN_84, n_actions)
        self.head_pole = nn.Linear(HIDDEN_84, n_actions)

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

    def _gating_compatible(self) -> bool:
        return self.confidence_gating and self.obs_dim == 4 and self.n_actions == 2

    def _confidence_gate_k(
        self, Q_cart: torch.Tensor, Q_pole: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_c = (Q_cart[:, 1] - Q_cart[:, 0]).abs()
        m_p = (Q_pole[:, 1] - Q_pole[:, 0]).abs()
        logits = torch.stack([m_c, m_p], dim=1)
        C = F.softmax(logits, dim=1).detach()
        km, kx = self.gate_k_min, self.gate_k_max
        k4 = km + (kx - km) * C[:, 0:1]
        k5 = km + (kx - km) * C[:, 1:2]
        return k4, k5

    def _z_full_from_gates(
        self, obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor
    ) -> torch.Tensor:
        W = self.linear_feature.weight
        b = self.linear_feature.bias
        cart_part = obs[..., 0:2] @ W[:, 0:2].T
        pole_part = obs[..., 2:4] @ W[:, 2:4].T
        return k4 * cart_part + k5 * pole_part + b

    def forward(
        self, obs: torch.Tensor, detach_z: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (Q_full, Q_cart, Q_pole, z_full, k_cart, k_pole). k_* are (B,1), ones if no gating."""
        obs_cart_m = self._mask_cart(obs)
        z_cart = self.linear_feature(obs_cart_m)
        h_cart = self.trunk(z_cart)
        Q_cart = self.head_cart(h_cart)

        obs_pole_m = self._mask_pole(obs)
        z_pole = self.linear_feature(obs_pole_m)
        h_pole = self.trunk(z_pole)
        Q_pole = self.head_pole(h_pole)

        z_full = self.linear_feature(obs)
        B = obs.shape[0]
        device, dtype = obs.device, obs.dtype
        k4 = torch.ones(B, 1, device=device, dtype=dtype)
        k5 = torch.ones(B, 1, device=device, dtype=dtype)
        z_qfull = z_full
        if self._gating_compatible():
            k4, k5 = self._confidence_gate_k(Q_cart, Q_pole)
            z_qfull = self._z_full_from_gates(obs, k4, k5)

        h_in = z_qfull.detach() if detach_z else z_qfull
        h_full = self.trunk(h_in)
        Q_full = self.head_full(h_full)
        return Q_full, Q_cart, Q_pole, z_full, k4, k5

    @torch.no_grad()
    def gate_k_for_manual_grad(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._gating_compatible():
            B = obs.shape[0]
            o = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
            return o, o
        obs_cart_m = self._mask_cart(obs)
        z_cart = self.linear_feature(obs_cart_m)
        h_cart = self.trunk(z_cart)
        Q_cart = self.head_cart(h_cart)
        obs_pole_m = self._mask_pole(obs)
        z_pole = self.linear_feature(obs_pole_m)
        h_pole = self.trunk(z_pole)
        Q_pole = self.head_pole(h_pole)
        return self._confidence_gate_k(Q_cart, Q_pole)

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        Q_full, _, _, _, _, _ = self.forward(obs, detach_z=False)
        return Q_full


class MultiHeadMixedQNetwork(nn.Module):
    """
    Shared linear_feature (obs -> 120) and trunk (ReLU -> 84 -> ReLU) outputting h.
    Five heads: head1..head5 each Linear(84, n_actions). Heads 1-3 use full state; head4 uses masked cart (x, xdot, 0, 0); head5 uses masked pole (0, 0, theta, thetadot).
    Action selection and forward_q_only use Q1 only (same logic as Model 1 / DQN).
    Heads 2-5 provide diverse learning signals to the feature layer via B but do NOT influence action selection.
    Trainable weights w = softmax(logits) are kept for analysis/logging but not used for action selection.

    If confidence_gating is True (CartPole: obs_dim==4, n_actions==2), Q1 uses the gated first layer
    from the proposal: margins from Q4/Q5, softmax (T=1), k in [k_min,k_max], detached from backprop
    through Q4/Q5; Q2/Q3 still use the ungated full linear_feature(obs). B-feedback error vector is
    unchanged (no optional k weighting on e4,e5).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        confidence_gating: bool = False,
        gate_k_min: float = 0.8,
        gate_k_max: float = 1.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.confidence_gating = confidence_gating
        self.gate_k_min = float(gate_k_min)
        self.gate_k_max = float(gate_k_max)
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

    def _gating_compatible(self) -> bool:
        return (
            self.confidence_gating
            and self.obs_dim == 4
            and self.n_actions == 2
        )

    def _confidence_gate_k(self, Q4: torch.Tensor, Q5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """T=1: softmax on raw margins (no temperature scaling). k is detached from Q4/Q5."""
        m4 = (Q4[:, 1] - Q4[:, 0]).abs()
        m5 = (Q5[:, 1] - Q5[:, 0]).abs()
        logits = torch.stack([m4, m5], dim=1)
        C = F.softmax(logits, dim=1).detach()
        km, kx = self.gate_k_min, self.gate_k_max
        k4 = km + (kx - km) * C[:, 0:1]
        k5 = km + (kx - km) * C[:, 1:2]
        return k4, k5

    def _z_q1_from_gates(self, obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
        W = self.linear_feature.weight
        b = self.linear_feature.bias
        cart_part = obs[..., 0:2] @ W[:, 0:2].T
        pole_part = obs[..., 2:4] @ W[:, 2:4].T
        return k4 * cart_part + k5 * pole_part + b

    def forward(
        self, obs: torch.Tensor, detach_z: bool = False
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns (Q1..Q5, Q_combined, z_full, k4, k5).
        z_full is linear_feature(obs). k4,k5 are (B,1); all-ones when gating is off.
        Q1 uses gated z when _gating_compatible(); Q2/Q3 always use trunk(z_full).
        """
        obs_cart = self._mask_cart(obs)
        z_cart = self.linear_feature(obs_cart)
        h_cart = self.trunk(z_cart)
        Q4 = self.head4(h_cart)

        obs_pole = self._mask_pole(obs)
        z_pole = self.linear_feature(obs_pole)
        h_pole = self.trunk(z_pole)
        Q5 = self.head5(h_pole)

        z_full = self.linear_feature(obs)
        B = obs.shape[0]
        device, dtype = obs.device, obs.dtype
        k4 = torch.ones(B, 1, device=device, dtype=dtype)
        k5 = torch.ones(B, 1, device=device, dtype=dtype)
        z_q1 = z_full
        if self._gating_compatible():
            k4, k5 = self._confidence_gate_k(Q4, Q5)
            z_q1 = self._z_q1_from_gates(obs, k4, k5)

        h1_in = z_q1.detach() if detach_z else z_q1
        z23_in = z_full.detach() if detach_z else z_full

        if self._gating_compatible():
            h1 = self.trunk(h1_in)
            h23 = self.trunk(z23_in)
            Q1 = self.head1(h1)
            Q2 = self.head2(h23)
            Q3 = self.head3(h23)
        else:
            h = self.trunk(h1_in)
            Q1 = self.head1(h)
            Q2 = self.head2(h)
            Q3 = self.head3(h)

        w = F.softmax(self.weight_logits, dim=0)
        Q_combined = (
            w[0] * Q1 + w[1] * Q2 + w[2] * Q3 + w[3] * Q4 + w[4] * Q5
        )
        return Q1, Q2, Q3, Q4, Q5, Q_combined, z_full, k4, k5

    @torch.no_grad()
    def gate_k_for_manual_grad(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cheap recompute of (k4,k5) for B-feedback outer product scaling (training loop)."""
        if not self._gating_compatible():
            B = obs.shape[0]
            o = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
            return o, o
        obs_cart = self._mask_cart(obs)
        z_cart = self.linear_feature(obs_cart)
        h_cart = self.trunk(z_cart)
        Q4 = self.head4(h_cart)
        obs_pole = self._mask_pole(obs)
        z_pole = self.linear_feature(obs_pole)
        h_pole = self.trunk(z_pole)
        Q5 = self.head5(h_pole)
        return self._confidence_gate_k(Q4, Q5)

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q1 — same interface as DQN (gated z when compatibility holds)."""
        if not self._gating_compatible():
            z = self.linear_feature(obs)
            return self.head1(self.trunk(z))
        obs_cart = self._mask_cart(obs)
        z_cart = self.linear_feature(obs_cart)
        h_cart = self.trunk(z_cart)
        Q4 = self.head4(h_cart)
        obs_pole = self._mask_pole(obs)
        z_pole = self.linear_feature(obs_pole)
        h_pole = self.trunk(z_pole)
        Q5 = self.head5(h_pole)
        k4, k5 = self._confidence_gate_k(Q4, Q5)
        z_q1 = self._z_q1_from_gates(obs, k4, k5)
        return self.head1(self.trunk(z_q1))


# ---------- Training loop (step-based or episode-based; algo switches update rule) ----------
def run_training(
    algo: str,
    env_id: str,
    seed: int,
    device: torch.device,
    total_timesteps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    paper_dir: Optional[Path] = None,
    multihead_action_mode: str = "combined",
    confidence_gating: bool = False,
    paper_save_algo: Optional[str] = None,
) -> Tuple[List[float], List[int], List[float], Dict[str, Any]]:
    """
    Run 'dqn', 'mixed', 'mixed3f', or 'multihead'. If n_episodes is set, run until that many episodes; else use total_timesteps.
    Returns: episode_returns, episode_steps, episode_losses, paper_extra (dict for figures: multihead_w_history, contributions, etc.).
    multihead_action_mode: 'combined' (argmax Q_combined) or 'head1' (argmax Q1 only) for epsilon-greedy exploitation.
    confidence_gating: If True for multi-head algos, Q1 uses margin-based gated W_z when obs_dim==4 and n_actions==2.
    paper_save_algo: Optional filename tag for paper_extra JSON (e.g. "mh_a_cg" when using --confidence-gating).
    """
    assert (total_timesteps is not None) != (n_episodes is not None), "Set exactly one of total_timesteps or n_episodes"
    assert multihead_action_mode in ("combined", "head1"), "multihead_action_mode must be 'combined' or 'head1'"
    assert algo in (
        "dqn", "mixed", "mixed3f", "multihead", "mh_a", "mh_b", "mh_c", "mh_3a"
    ), f"Unknown algo: {algo}"
    paper_extra: Dict[str, Any] = {
        "algo": algo,
        "seed": seed,
        "multihead_action_mode": multihead_action_mode if algo == "multihead" else None,
        "multihead_w_history": [],  # list of {"step": int, "w": [5 floats]}
        "confidence_gating": False,
        "confidence_gate_history": [],  # list of {"step", "episode", "mean_k4", "mean_k5"}
    }
    set_seed(seed)
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    gate_effective = (
        confidence_gating
        and algo in MULTIHEAD_CONFIDENCE_GATING_ALGOS
        and obs_dim == 4
        and n_actions == 2
    )
    paper_extra["confidence_gating"] = bool(gate_effective)
    if (
        confidence_gating
        and algo in MULTIHEAD_CONFIDENCE_GATING_ALGOS
        and not gate_effective
    ):
        log(
            f"  [!] confidence_gating skipped for {algo}: needs obs_dim==4 and 2 actions "
            f"(this env: obs_dim={obs_dim}, n_actions={n_actions})"
        )

    optimizer_aux: Optional[optim.Adam] = None  # secondary optimizer for mh_a heads 2-5
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
        q_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        target_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        B_fixed = None
        B_3f = None
        B_multi = make_fixed_sparse_B_5heads(OBS_TO_120, N_HEADS, device, torch.float32)
        optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    elif algo == "mh_a":
        # Option A: trunk+head1 updated by Q1 loss only; heads 2-5 updated by their own losses
        # (trunk gradient blocked); linear_feature updated only by B-feedback from all 5 errors.
        q_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        target_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        B_fixed = None
        B_3f = None
        B_multi = make_fixed_sparse_B_5heads(OBS_TO_120, N_HEADS, device, torch.float32)
        optimizer = optim.Adam(
            list(q_network.trunk.parameters()) + list(q_network.head1.parameters()),
            lr=LEARNING_RATE,
        )
        optimizer_aux = optim.Adam(
            list(q_network.head2.parameters()) + list(q_network.head3.parameters())
            + list(q_network.head4.parameters()) + list(q_network.head5.parameters()),
            lr=LEARNING_RATE,
        )
    elif algo == "mh_3a":
        # Three heads (full / cart / pole) with Option A mechanics: trunk+head_full from Q_full loss only;
        # cart and pole heads separately; W_z only via B from three TD errors.
        q_network = ThreeHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        target_network = ThreeHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        B_fixed = None
        B_3f = None
        B_multi = make_fixed_sparse_B_5heads(
            OBS_TO_120, N_HEADS_THREE, device, torch.float32
        )
        optimizer = optim.Adam(
            list(q_network.trunk.parameters()) + list(q_network.head_full.parameters()),
            lr=LEARNING_RATE,
        )
        optimizer_aux = optim.Adam(
            list(q_network.head_cart.parameters()) + list(q_network.head_pole.parameters()),
            lr=LEARNING_RATE,
        )
    elif algo == "mh_b":
        # Option B: only Q1 used for everything. Heads 2-5 not trained.
        # B-feedback uses only Q1 error (e2-e5 = 0); linear_feature not in any optimizer.
        q_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        target_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        B_fixed = None
        B_3f = None
        B_multi = make_fixed_sparse_B_5heads(OBS_TO_120, N_HEADS, device, torch.float32)
        optimizer = optim.Adam(
            list(q_network.trunk.parameters()) + list(q_network.head1.parameters()),
            lr=LEARNING_RATE,
        )
    elif algo == "mh_c":
        # Option C (fix): W_z co-adapted by standard backprop AND receives B-feedback as a bonus.
        # linear_feature IS in the optimizer so end-to-end gradient flows through the full network
        # (same as DQN). B-feedback is applied on top with a smaller LR as an additive local signal.
        # Heads 2-5 are dropped (same as mh_b). Only Q1 drives action selection and backprop.
        q_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        target_network = MultiHeadMixedQNetwork(
            obs_dim, n_actions, confidence_gating=gate_effective
        ).to(device)
        B_fixed = None
        B_3f = None
        B_multi = make_fixed_sparse_B_5heads(OBS_TO_120, N_HEADS, device, torch.float32)
        optimizer = optim.Adam(
            list(q_network.linear_feature.parameters())
            + list(q_network.trunk.parameters())
            + list(q_network.head1.parameters()),
            lr=LEARNING_RATE,
        )
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
                if algo == "multihead" and multihead_action_mode == "head1":
                    Q1, _, _, _, _, _, _, _, _ = q_network(x, detach_z=False)
                    action = int(Q1.argmax(dim=1).item())
                else:
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
            if algo not in ("multihead", "mh_a", "mh_3a"):
                # For dqn, mixed, mixed3f, mh_b: compute a single TD target via Q1 (forward_q_only)
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
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q": float(old_val.detach().mean().item()),
                        "mean_y": float(td_target.mean().item()),
                    })
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
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                optimizer.zero_grad()
                loss_td.backward()
                optimizer.step()
                loss = loss_td
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q": float(q_sa.detach().mean().item()),
                        "mean_y": float(td_target.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
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
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                optimizer.zero_grad()
                loss_td.backward()
                optimizer.step()
                loss = loss_td
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q": float(q_sa.detach().mean().item()),
                        "mean_y": float(td_target.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
            elif algo == "multihead":
                # Five heads; each has its own TD target. Action selection uses Q1 only (same as DQN/Model 1).
                # Heads 2-5 provide extra learning signals routed to the feature layer via B_multi.
                # No combined loss: Q_combined is not used for action selection.
                Q1, Q2, Q3, Q4, Q5, _, _z_full, k4, k5 = q_network(data.observations, detach_z=True)
                with torch.no_grad():
                    tQ1, tQ2, tQ3, tQ4, tQ5, _, _, _, _ = target_network(
                        data.next_observations
                    )
                    r = data.rewards.flatten()
                    d = data.dones.flatten()
                    max1, _ = tQ1.max(dim=1)
                    max2, _ = tQ2.max(dim=1)
                    max3, _ = tQ3.max(dim=1)
                    max4, _ = tQ4.max(dim=1)
                    max5, _ = tQ5.max(dim=1)
                    # y1 is the standard reward-discount TD target (same formula as DQN)
                    y1 = r + GAMMA * (1 - d) * max1
                    if MULTIHEAD_UNIFIED_TARGETS:
                        # All heads use same reward-return target; cleanest comparison with DQN
                        y2 = y1.clone()
                        y3 = y1.clone()
                    else:
                        cos_next = MultiHeadMixedQNetwork.cos_theta_next(data.next_observations)
                        y2 = cos_next + GAMMA * (1 - d) * max2
                        y3 = r + GAMMA_SHORT * (1 - d) * max3
                    y4 = r + GAMMA * (1 - d) * max4
                    y5 = r + GAMMA * (1 - d) * max5
                q1_sa = Q1.gather(1, data.actions).squeeze()
                q2_sa = Q2.gather(1, data.actions).squeeze()
                q3_sa = Q3.gather(1, data.actions).squeeze()
                q4_sa = Q4.gather(1, data.actions).squeeze()
                q5_sa = Q5.gather(1, data.actions).squeeze()
                e1 = (y1 - q1_sa).unsqueeze(1)
                e2 = (y2 - q2_sa).unsqueeze(1)
                e3 = (y3 - q3_sa).unsqueeze(1)
                e4 = (y4 - q4_sa).unsqueeze(1)
                e5 = (y5 - q5_sa).unsqueeze(1)
                e = torch.cat([e1, e2, e3, e4, e5], dim=1)  # (B, 5)
                delta_z = e @ B_multi.t()  # (B, 5) @ (5, F) = (B, F)
                gx_obs = scaled_obs_for_manual_delta_W(data.observations, k4, k5)
                grad_W_z = (delta_z.t() @ gx_obs) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                # Backprop: trunk + all five heads via their individual TD losses
                loss = (
                    F.mse_loss(y1, q1_sa) + F.mse_loss(y2, q2_sa) + F.mse_loss(y3, q3_sa)
                    + F.mse_loss(y4, q4_sa) + F.mse_loss(y5, q5_sa)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q": float(q1_sa.detach().mean().item()),
                        "mean_y": float(y1.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
                    if paper_extra.get("confidence_gating"):
                        paper_extra.setdefault("confidence_gate_history", []).append({
                            "step": int(global_step),
                            "episode": len(episode_returns),
                            "mean_k4": float(k4.mean().item()),
                            "mean_k5": float(k5.mean().item()),
                        })
            elif algo == "mh_a":
                # Option A: Q1 drives trunk; heads 2-5 learn separately; B-feedback uses all 5 errors.
                z_full = q_network.linear_feature(data.observations)
                obs_cart = MultiHeadMixedQNetwork._mask_cart(data.observations)
                z_cart = q_network.linear_feature(obs_cart)
                h_cart = q_network.trunk(z_cart.detach())
                Q4 = q_network.head4(h_cart.detach())
                obs_pole = MultiHeadMixedQNetwork._mask_pole(data.observations)
                z_pole = q_network.linear_feature(obs_pole)
                h_pole = q_network.trunk(z_pole.detach())
                Q5 = q_network.head5(h_pole.detach())

                if q_network._gating_compatible():
                    k4, k5 = q_network._confidence_gate_k(Q4, Q5)
                    z_q1 = q_network._z_q1_from_gates(data.observations, k4, k5)
                    h_q1 = q_network.trunk(z_q1.detach())
                    Q1 = q_network.head1(h_q1)
                    h_full = q_network.trunk(z_full.detach())
                    Q2 = q_network.head2(h_full.detach())
                    Q3 = q_network.head3(h_full.detach())
                else:
                    B_sz = data.observations.shape[0]
                    k4 = torch.ones(B_sz, 1, device=device, dtype=torch.float32)
                    k5 = torch.ones_like(k4)
                    h = q_network.trunk(z_full.detach())
                    Q1 = q_network.head1(h)
                    Q2 = q_network.head2(h.detach())
                    Q3 = q_network.head3(h.detach())

                # TD targets from target network (all 5 heads)
                with torch.no_grad():
                    tQ1, tQ2, tQ3, tQ4, tQ5, _, _, _, _ = target_network(
                        data.next_observations, detach_z=False
                    )
                    r_mha = data.rewards.flatten()
                    d_mha = data.dones.flatten()
                    max1, _ = tQ1.max(dim=1)
                    max2, _ = tQ2.max(dim=1)
                    max3, _ = tQ3.max(dim=1)
                    max4, _ = tQ4.max(dim=1)
                    max5, _ = tQ5.max(dim=1)
                    y1 = r_mha + GAMMA * (1 - d_mha) * max1
                    y2 = y1.clone() if MULTIHEAD_UNIFIED_TARGETS else (
                        MultiHeadMixedQNetwork.cos_theta_next(data.next_observations) + GAMMA * (1 - d_mha) * max2)
                    y3 = y1.clone() if MULTIHEAD_UNIFIED_TARGETS else (r_mha + GAMMA_SHORT * (1 - d_mha) * max3)
                    y4 = r_mha + GAMMA * (1 - d_mha) * max4
                    y5 = r_mha + GAMMA * (1 - d_mha) * max5
                q1_sa = Q1.gather(1, data.actions).squeeze()
                q2_sa = Q2.gather(1, data.actions).squeeze()
                q3_sa = Q3.gather(1, data.actions).squeeze()
                q4_sa = Q4.gather(1, data.actions).squeeze()
                q5_sa = Q5.gather(1, data.actions).squeeze()
                # B-feedback: all 5 errors → manual update to linear_feature
                e1 = (y1 - q1_sa).detach().unsqueeze(1)
                e2 = (y2 - q2_sa).detach().unsqueeze(1)
                e3 = (y3 - q3_sa).detach().unsqueeze(1)
                e4 = (y4 - q4_sa).detach().unsqueeze(1)
                e5 = (y5 - q5_sa).detach().unsqueeze(1)
                e = torch.cat([e1, e2, e3, e4, e5], dim=1)
                delta_z = e @ B_multi.t()
                gx_obs = scaled_obs_for_manual_delta_W(data.observations, k4, k5)
                grad_W_z = (delta_z.t() @ gx_obs) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                # Backprop Q1 loss → trunk + head1 only
                loss_q1 = F.mse_loss(y1, q1_sa)
                optimizer.zero_grad()
                loss_q1.backward()
                optimizer.step()
                # Backprop Q2-Q5 losses → heads 2-5 only (trunk/linear_feature already detached)
                loss_q2 = F.mse_loss(y2, q2_sa)
                loss_q3 = F.mse_loss(y3, q3_sa)
                loss_q4 = F.mse_loss(y4, q4_sa)
                loss_q5 = F.mse_loss(y5, q5_sa)
                loss_aux_val = loss_q2 + loss_q3 + loss_q4 + loss_q5
                optimizer_aux.zero_grad()
                loss_aux_val.backward()
                optimizer_aux.step()
                loss = loss_q1  # log Q1 loss for fair comparison with DQN
                # Log per-head losses AND mean Q-values periodically for analysis
                # Use global_step % MULTIHEAD_LOG_EVERY so it aligns with TRAIN_FREQUENCY=10
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("head_loss_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "loss_q1": float(loss_q1.item()),
                        "loss_q2": float(loss_q2.item()),
                        "loss_q3": float(loss_q3.item()),
                        "loss_q4": float(loss_q4.item()),
                        "loss_q5": float(loss_q5.item()),
                    })
                    # Log mean Q-value + grad norm for Q1 decay analysis
                    paper_extra.setdefault("head_qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q1": float(q1_sa.detach().mean().item()),
                        "mean_q2": float(q2_sa.detach().mean().item()),
                        "mean_q3": float(q3_sa.detach().mean().item()),
                        "mean_q4": float(q4_sa.detach().mean().item()),
                        "mean_q5": float(q5_sa.detach().mean().item()),
                        "mean_y1": float(y1.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
                    if paper_extra.get("confidence_gating"):
                        paper_extra.setdefault("confidence_gate_history", []).append({
                            "step": int(global_step),
                            "episode": len(episode_returns),
                            "mean_k4": float(k4.mean().item()),
                            "mean_k5": float(k5.mean().item()),
                        })
            elif algo == "mh_3a":
                # Option A–style with 3 critics; B in R^{F x 3}; gating from Q_cart,Q_pole margins.
                z_full = q_network.linear_feature(data.observations)
                obs_cart = ThreeHeadMixedQNetwork._mask_cart(data.observations)
                z_cart = q_network.linear_feature(obs_cart)
                h_cart = q_network.trunk(z_cart.detach())
                Q_cart = q_network.head_cart(h_cart.detach())
                obs_pole = ThreeHeadMixedQNetwork._mask_pole(data.observations)
                z_pole = q_network.linear_feature(obs_pole)
                h_pole = q_network.trunk(z_pole.detach())
                Q_pole = q_network.head_pole(h_pole.detach())

                if q_network._gating_compatible():
                    k4, k5 = q_network._confidence_gate_k(Q_cart, Q_pole)
                    z_qf = q_network._z_full_from_gates(data.observations, k4, k5)
                    h_qf = q_network.trunk(z_qf.detach())
                    Q_full = q_network.head_full(h_qf)
                else:
                    B_sz = data.observations.shape[0]
                    k4 = torch.ones(B_sz, 1, device=device, dtype=torch.float32)
                    k5 = torch.ones_like(k4)
                    h_qf = q_network.trunk(z_full.detach())
                    Q_full = q_network.head_full(h_qf)

                with torch.no_grad():
                    tQf, tQc, tQp, _, _, _ = target_network(
                        data.next_observations, detach_z=False
                    )
                    r3 = data.rewards.flatten()
                    d3 = data.dones.flatten()
                    max_f, _ = tQf.max(dim=1)
                    max_c, _ = tQc.max(dim=1)
                    max_p, _ = tQp.max(dim=1)
                    y_full = r3 + GAMMA * (1 - d3) * max_f
                    y_cart = r3 + GAMMA * (1 - d3) * max_c
                    y_pole = r3 + GAMMA * (1 - d3) * max_p

                qf_sa = Q_full.gather(1, data.actions).squeeze()
                qc_sa = Q_cart.gather(1, data.actions).squeeze()
                qp_sa = Q_pole.gather(1, data.actions).squeeze()

                e1 = (y_full - qf_sa).detach().unsqueeze(1)
                e2 = (y_cart - qc_sa).detach().unsqueeze(1)
                e3 = (y_pole - qp_sa).detach().unsqueeze(1)
                e = torch.cat([e1, e2, e3], dim=1)
                delta_z = e @ B_multi.t()
                gx_obs = scaled_obs_for_manual_delta_W(data.observations, k4, k5)
                grad_W_z = (delta_z.t() @ gx_obs) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)

                loss_qf = F.mse_loss(y_full, qf_sa)
                optimizer.zero_grad()
                loss_qf.backward()
                optimizer.step()
                loss_cart = F.mse_loss(y_cart, qc_sa)
                loss_pole = F.mse_loss(y_pole, qp_sa)
                loss_aux_val = loss_cart + loss_pole
                optimizer_aux.zero_grad()
                loss_aux_val.backward()
                optimizer_aux.step()
                loss = loss_qf
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("head_loss_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "loss_q_full": float(loss_qf.item()),
                        "loss_q_cart": float(loss_cart.item()),
                        "loss_q_pole": float(loss_pole.item()),
                    })
                    paper_extra.setdefault("head_qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q_full": float(qf_sa.detach().mean().item()),
                        "mean_q_cart": float(qc_sa.detach().mean().item()),
                        "mean_q_pole": float(qp_sa.detach().mean().item()),
                        "mean_y_full": float(y_full.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
                    if paper_extra.get("confidence_gating"):
                        paper_extra.setdefault("confidence_gate_history", []).append({
                            "step": int(global_step),
                            "episode": len(episode_returns),
                            "mean_k4": float(k4.mean().item()),
                            "mean_k5": float(k5.mean().item()),
                        })
            elif algo == "mh_b":
                # Option B: only Q1 used everywhere; B-feedback uses only Q1 error (e2-e5 = 0).
                z_full = q_network.linear_feature(data.observations)
                if q_network._gating_compatible():
                    obs_cart = MultiHeadMixedQNetwork._mask_cart(data.observations)
                    z_cart = q_network.linear_feature(obs_cart)
                    h_cart = q_network.trunk(z_cart.detach())
                    Q4 = q_network.head4(h_cart.detach())
                    obs_pole = MultiHeadMixedQNetwork._mask_pole(data.observations)
                    z_pole = q_network.linear_feature(obs_pole)
                    h_pole = q_network.trunk(z_pole.detach())
                    Q5 = q_network.head5(h_pole.detach())
                    k4, k5 = q_network._confidence_gate_k(Q4, Q5)
                    z_q1 = q_network._z_q1_from_gates(data.observations, k4, k5)
                    h = q_network.trunk(z_q1.detach())
                else:
                    B_sz = data.observations.shape[0]
                    k4 = torch.ones(B_sz, 1, device=device, dtype=torch.float32)
                    k5 = torch.ones_like(k4)
                    h = q_network.trunk(z_full.detach())

                Q1 = q_network.head1(h)
                q1_sa = Q1.gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, q1_sa)
                # B-feedback: only slot 0 (head1 error); slots 1-4 stay zero → only i%5==0 features get a signal
                e_full = torch.zeros(data.observations.shape[0], N_HEADS, device=device, dtype=torch.float32)
                e_full[:, 0] = (td_target - q1_sa).detach()
                delta_z = e_full @ B_multi.t()
                gx_obs = scaled_obs_for_manual_delta_W(data.observations, k4, k5)
                grad_W_z = (delta_z.t() @ gx_obs) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q": float(q1_sa.detach().mean().item()),
                        "mean_y": float(td_target.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
                    if paper_extra.get("confidence_gating"):
                        paper_extra.setdefault("confidence_gate_history", []).append({
                            "step": int(global_step),
                            "episode": len(episode_returns),
                            "mean_k4": float(k4.mean().item()),
                            "mean_k5": float(k5.mean().item()),
                        })
            elif algo == "mh_c":
                # Option C: end-to-end backprop (W_z in Adam, NO z.detach) + B-feedback bonus.
                # W_z is co-adapted with the trunk via the Q1 gradient (same as DQN), PLUS it
                # receives an additive B-feedback nudge at a smaller LR (LR_LINEAR_MIXED).
                Q1 = q_network.forward_q_only(data.observations)
                q1_sa = Q1.gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, q1_sa)
                # B-feedback bonus: apply BEFORE backward so the manual update does not interfere
                # with the Adam step (Adam sees .grad from backward; B directly updates .data).
                k4, k5 = q_network.gate_k_for_manual_grad(data.observations)
                e_full = torch.zeros(data.observations.shape[0], N_HEADS, device=device, dtype=torch.float32)
                e_full[:, 0] = (td_target - q1_sa).detach()
                delta_z = e_full @ B_multi.t()
                gx_obs = scaled_obs_for_manual_delta_W(data.observations, k4, k5)
                grad_W_z = (delta_z.t() @ gx_obs) / BATCH_SIZE
                grad_b_z = delta_z.mean(0)
                _grad_wz_norm_raw = float(grad_W_z.norm().item())
                if FEEDBACK_GRAD_CLIP > 0:
                    grad_norm = grad_W_z.norm()
                    if grad_norm > FEEDBACK_GRAD_CLIP:
                        grad_W_z = grad_W_z * (FEEDBACK_GRAD_CLIP / grad_norm.item())
                _grad_wz_norm_clipped = float(grad_W_z.norm().item())
                with torch.no_grad():
                    q_network.linear_feature.weight.data.add_(grad_W_z, alpha=LR_LINEAR_MIXED)
                    q_network.linear_feature.bias.data.add_(grad_b_z, alpha=LR_LINEAR_MIXED)
                # Standard end-to-end Adam step (W_z gradient flows all the way back)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                    paper_extra.setdefault("qval_history", []).append({
                        "step": int(global_step),
                        "episode": len(episode_returns),
                        "mean_q": float(q1_sa.detach().mean().item()),
                        "mean_y": float(td_target.mean().item()),
                        "grad_wz_norm_raw": _grad_wz_norm_raw,
                        "grad_wz_norm_clipped": _grad_wz_norm_clipped,
                    })
                    if paper_extra.get("confidence_gating"):
                        paper_extra.setdefault("confidence_gate_history", []).append({
                            "step": int(global_step),
                            "episode": len(episode_returns),
                            "mean_k4": float(k4.mean().item()),
                            "mean_k5": float(k5.mean().item()),
                        })
            else:
                loss = torch.tensor(0.0, device=device)

            loss_list_this_episode.append(loss.item())
            # Step-level loss snapshot for peak analysis (all algos)
            if global_step > LEARNING_STARTS and global_step % MULTIHEAD_LOG_EVERY == 0:
                paper_extra.setdefault("step_loss_history", []).append({
                    "step": int(global_step),
                    "episode": len(episode_returns),
                    "loss": float(loss.item()),
                    "epsilon": float(linear_schedule(START_E, END_E, exploration_duration, global_step)),
                })

        # Target network update
        target_freq = (
            TARGET_NETWORK_FREQUENCY_MULTIHEAD
            if algo in ("multihead", "mh_a", "mh_b", "mh_c", "mh_3a")
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
        if (
            algo in ("multihead", "mh_a")
            and global_step > LEARNING_STARTS
            and (global_step + 1) % MULTIHEAD_LOG_EVERY == 0
            and hasattr(q_network, "weight_logits")
        ):
            with torch.no_grad():
                w = F.softmax(q_network.weight_logits, dim=0).cpu().tolist()
            paper_extra["multihead_w_history"].append({
                "step": int(global_step + 1),
                "episode": len(episode_returns),
                "w": w,
            })
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

    # Optional: gradient×hidden attribution sample for paper (one observation from env space)
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)

        q_network.eval()
        sample_obs = torch.tensor(
            obs, dtype=torch.float32, device=device
        )  # last seen obs
        with torch.no_grad():
            x = sample_obs.unsqueeze(0)
            if algo == "dqn":
                qv = q_network.forward_q_only(x)
            elif algo == "multihead":
                qv = q_network.forward_q_only(x)
            else:
                qv = q_network.forward_q_only(x)
            greedy_a = int(qv.argmax(dim=1).item())
        q_network.train()
        obs_np = obs.copy() if hasattr(obs, "copy") else np.array(obs)

        try:
            if algo == "dqn":
                h, c = contribution_grad_times_h_cleanrl(q_network, sample_obs, greedy_a)
                paper_extra["contribution_sample"] = numpy_dict_from_contrib(
                    obs_np, greedy_a, c, h
                )
            elif algo == "mixed":
                h, c = contribution_grad_times_h_mixed(q_network, sample_obs, greedy_a)
                paper_extra["contribution_sample"] = numpy_dict_from_contrib(
                    obs_np, greedy_a, c, h
                )
            elif algo == "mh_3a":
                h, c = contribution_grad_times_h_threehead_full(
                    q_network, sample_obs, greedy_a
                )
                paper_extra["contribution_sample"] = numpy_dict_from_contrib(
                    obs_np, greedy_a, c, h
                )
            elif algo in ("multihead", "mh_a", "mh_b", "mh_c"):
                h, c_c = contribution_grad_times_h_multihead_combined(
                    q_network, sample_obs, greedy_a
                )
                _, c1 = contribution_grad_times_h_multihead_head1(
                    q_network, sample_obs, greedy_a
                )
                paper_extra["contribution_sample"] = {
                    "observation": obs_np.tolist(),
                    "greedy_action_combined": greedy_a,
                    "contrib_h_dim120_Q_combined": c_c.cpu().numpy().tolist(),
                    "contrib_h_dim120_Q1_only": c1.cpu().numpy().tolist(),
                    "sum_contrib_combined": float(c_c.sum().item()),
                    "sum_contrib_head1": float(c1.sum().item()),
                }
        except Exception as ex:
            paper_extra["contribution_sample"] = {"error": str(ex)}

        def _json_default(o: Any) -> Any:
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(f"not JSON serializable: {type(o)!r}")

        out_json = paper_dir / f"paper_extra_{paper_save_algo or algo}_seed{seed}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(paper_extra, f, indent=2, default=_json_default)
        log(f"  Paper extras: {out_json}")

    return episode_returns, episode_steps, episode_losses, paper_extra


def _plot_confidence_gate_analysis(
    all_results: List[Dict],
    output_dir: Path,
    fig_prefix: str,
    colors: Dict[str, str],
    labels: Dict[str, str],
) -> None:
    """Mean gate factors k4, k5 over training for runs with confidence_gating."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plotted = []
    for r in all_results:
        pe = r.get("paper_extra") or {}
        if not pe.get("confidence_gate_history"):
            continue
        plotted.append((r["algo"], r["seed"], pe["confidence_gate_history"]))
    if not plotted:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, seed, hist in plotted:
        steps = np.array([h["step"] for h in hist])
        m4 = np.array([h["mean_k4"] for h in hist])
        m5 = np.array([h["mean_k5"] for h in hist])
        color = colors.get(algo, "#333")
        base_l = labels.get(algo, algo)
        ax.plot(steps, m4, color=color, linestyle="-", linewidth=1.6, alpha=0.85, label=f"{base_l} (k cart) s{seed}")
        ax.plot(steps, m5, color=color, linestyle="--", linewidth=1.2, alpha=0.55, label=f"{base_l} (k pole) s{seed}")
    ax.set_xlabel("Environment step")
    ax.set_ylabel(r"Mean $k_{\mathrm{cart}}$ / $k_{\mathrm{pole}}$ (batch mean)")
    ax.set_title("Confidence gate factors on $W_z$ compartments (margins softmax, then $k\\in[k_{\\min},k_{\\max}]$)")
    ax.set_ylim(0.74, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    name = f"{fig_prefix}confidence_gate_k_trace.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def plot_and_save(
    all_results: List[Dict],
    output_dir: Path,
    fig_prefix: str = "",
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
    colors = {
        "dqn": "#1f77b4", "mixed": "#d62728", "mixed3f": "#2ca02c", "multihead": "#9467bd",
        "mh_a": "#ff7f0e", "mh_b": "#8c564b", "mh_c": "#17becf",
        "mh_3a": "#bcbd22",
    }
    labels = {
        "dqn": "DQN (Model 1)",
        "mixed": "Mixed backprop",
        "mixed3f": "Mixed 3F",
        "multihead": "Multihead (5 critics + combined Q)",
        "mh_a": "Option A: Q1 trunk + 5-signal B",
        "mh_b": "Option B: Q1 only + 1-signal B",
        "mh_c": "Option C: end-to-end + B bonus",
        "mh_3a": "3-head (full / cart / pole) Option A",
    }
    for _bk in ("multihead", "mh_a", "mh_b", "mh_c", "mh_3a"):
        _cgk = _bk + "_cg"
        labels[_cgk] = labels[_bk] + r" (+ conf. gate)"
        colors[_cgk] = colors[_bk]

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
    _lc_parts = [labels.get(a, a) for a in algos_seen]
    ax.set_title("Learning curves — " + " vs ".join(_lc_parts) if len(_lc_parts) <= 3
                 else "Episodic return — " + ", ".join(_lc_parts))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    lc_name = f"{fig_prefix}cleanrl_vs_mixed_learning_curves.pdf"
    fig.savefig(output_dir / lc_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / lc_name}")

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
    ax.set_title("Training loss — all models (episode-averaged TD loss)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_name = f"{fig_prefix}cleanrl_vs_mixed_loss.pdf"
    fig.savefig(output_dir / loss_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / loss_name}")

    # DQN vs multihead only (Model 1 vs Model 4 comparison)
    subset = [r for r in all_results if r["algo"] in ("dqn", "multihead")]
    if len(subset) >= 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sub_algos = list(dict.fromkeys(r["algo"] for r in subset))
        for algo in sub_algos:
            runs = [r for r in subset if r["algo"] == algo]
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
        ax.set_title(f"{labels.get('dqn','DQN (Model 1)')} vs {labels.get('multihead','Multihead')} — episodic return")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dm_name = f"{fig_prefix}dqn_vs_multihead_learning_curves.pdf"
        fig.savefig(output_dir / dm_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / dm_name}")

    # Option A vs B comparison (dqn as baseline)
    option_subset = [r for r in all_results if r["algo"] in ("dqn", "mh_a", "mh_b", "mh_c")]
    if option_subset:
        fig, ax = plt.subplots(figsize=(7, 4))
        for algo in ["dqn", "mh_a", "mh_b", "mh_c"]:
            runs = [r for r in option_subset if r["algo"] == algo]
            if not runs:
                continue
            color = colors.get(algo, "#333")
            label = labels.get(algo, algo)
            for r in runs:
                ep = np.arange(1, len(r["episode_returns"]) + 1)
                ax.plot(ep, r["episode_returns"], color=color, alpha=0.25, linewidth=0.7)
            max_len = max(len(r["episode_returns"]) for r in runs)
            arr = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                arr[i, : len(r["episode_returns"])] = r["episode_returns"]
            mean_ret = np.nanmean(arr, axis=0)
            ep_mean = np.arange(1, max_len + 1)
            w_sm = min(50, max_len // 4)
            if w_sm >= 1:
                smooth = np.convolve(mean_ret, np.ones(w_sm) / w_sm, mode="valid")
                ax.plot(ep_mean[w_sm - 1 :], smooth, color=color, linewidth=2, label=label)
            else:
                ax.plot(ep_mean, mean_ret, color=color, linewidth=2, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episodic return")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("DQN vs Options A / B / C (episodic return)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        opt_name = f"{fig_prefix}options_ab_learning_curves.pdf"
        fig.savefig(output_dir / opt_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / opt_name}")

        # Also a loss comparison for the options
        fig, ax = plt.subplots(figsize=(7, 4))
        for algo in ["dqn", "mh_a", "mh_b", "mh_c"]:
            runs = [r for r in option_subset if r["algo"] == algo]
            if not runs:
                continue
            color = colors.get(algo, "#333")
            label = labels.get(algo, algo)
            for r in runs:
                ep_idx = np.arange(1, len(r["episode_losses"]) + 1)
                ax.plot(ep_idx, r["episode_losses"], color=color, alpha=0.25, linewidth=0.7)
            max_len = max(len(r["episode_losses"]) for r in runs)
            arr = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                arr[i, : len(r["episode_losses"])] = r["episode_losses"]
            mean_loss = np.nanmean(arr, axis=0)
            ep_idx = np.arange(1, len(mean_loss) + 1)
            w_sm = min(20, max_len // 4)
            if w_sm >= 1:
                smooth = np.convolve(mean_loss, np.ones(w_sm) / w_sm, mode="valid")
                ax.plot(ep_idx[w_sm - 1 :], smooth, color=color, linewidth=2, label=label)
            else:
                ax.plot(ep_idx, mean_loss, color=color, linewidth=2, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean TD loss (per episode)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Training loss: DQN vs Options A / B / C (episode-averaged)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        opt_loss_name = f"{fig_prefix}options_ab_loss.pdf"
        fig.savefig(output_dir / opt_loss_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / opt_loss_name}")

    # Per-head loss evolution plot for mh_a (Model 4 head analysis)
    mh_a_runs = [r for r in all_results if r["algo"] == "mh_a"]
    head_runs_with_data = [r for r in mh_a_runs if (r.get("paper_extra") or {}).get("head_loss_history")]
    if head_runs_with_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        head_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        head_labels = [
            r"$Q_1$ (full obs, drives policy)",
            r"$Q_2$ (full obs, auxiliary)",
            r"$Q_3$ (full obs, short-$\gamma$)",
            r"$Q_4$ (cart-only masked)",
            r"$Q_5$ (pole-only masked)",
        ]
        for r_idx, r in enumerate(head_runs_with_data):
            hist = r["paper_extra"]["head_loss_history"]
            episodes = np.array([h.get("episode", h["step"]) for h in hist])
            for k, key in enumerate(["loss_q1", "loss_q2", "loss_q3", "loss_q4", "loss_q5"]):
                vals = np.array([h[key] for h in hist])
                lw = 2.2 if r_idx == 0 else 0.8
                alpha = 1.0 if r_idx == 0 else 0.45
                label = head_labels[k] if r_idx == 0 else None
                ax.plot(episodes, vals, color=head_colors[k], linewidth=lw, alpha=alpha, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("TD loss (MSE)")
        ax.set_title(r"Option A (mh\_a): per-head TD loss by episode")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        hl_name = f"{fig_prefix}mha_head_losses.pdf"
        fig.savefig(output_dir / hl_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / hl_name}")

    # Per-head Q-value and episodic return for mh_a
    qval_runs = [r for r in mh_a_runs if (r.get("paper_extra") or {}).get("head_qval_history")]
    if qval_runs:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax_q, ax_r = axes
        head_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        head_labels = [
            r"$Q_1$: full obs (policy head)",
            r"$Q_2$: full obs (auxiliary)",
            r"$Q_3$: short-$\gamma$ (aux.)",
            r"$Q_4$: cart-only (aux.)",
            r"$Q_5$: pole-only (aux.)",
        ]
        ref_color = "#7f7f7f"
        for r_idx, r in enumerate(qval_runs):
            hist = r["paper_extra"]["head_qval_history"]
            episodes = np.array([h.get("episode", h["step"]) for h in hist])
            lw = 2.0 if r_idx == 0 else 0.7
            alpha = 1.0 if r_idx == 0 else 0.4
            # Q-value subplot
            for k, key in enumerate(["mean_q1", "mean_q2", "mean_q3", "mean_q4", "mean_q5"]):
                vals = np.array([h[key] for h in hist])
                ax_q.plot(episodes, vals, color=head_colors[k], linewidth=lw, alpha=alpha,
                          label=head_labels[k] if r_idx == 0 else None)
            # TD target reference (y1)
            y1_vals = np.array([h["mean_y1"] for h in hist])
            ax_q.plot(episodes, y1_vals, color=ref_color, linewidth=lw, alpha=alpha,
                      linestyle="--", label="TD target $y_1$ (ref.)" if r_idx == 0 else None)
        ax_q.set_ylabel("Mean Q-value at chosen action")
        ax_q.set_title(r"Option A (mh\_a): per-head mean Q-value vs. TD target")
        ax_q.legend(loc="upper left", fontsize=7)
        ax_q.grid(True, alpha=0.3)
        # Episodic return subplot (using mh_a runs, episode-indexed)
        for r_idx, r in enumerate(mh_a_runs):
            ep = np.arange(1, len(r["episode_returns"]) + 1)
            lw = 1.5 if r_idx == 0 else 0.6
            alpha = 0.9 if r_idx == 0 else 0.4
            ax_r.plot(ep, r["episode_returns"], color=head_colors[0],
                      linewidth=lw, alpha=alpha * 0.35)
            # Smoothed mean
            w_sm = min(50, len(r["episode_returns"]) // 4)
            if w_sm >= 1:
                smooth = np.convolve(r["episode_returns"], np.ones(w_sm) / w_sm, mode="valid")
                ax_r.plot(ep[w_sm - 1:], smooth, color=head_colors[0],
                          linewidth=lw + 0.5, alpha=alpha,
                          label=f"mh_a seed={r['seed']} (smoothed)" if r_idx == 0 else None)
        ax_r.set_xlabel("Episode")
        ax_r.set_ylabel("Episodic return")
        ax_r.set_title(r"Option A (mh\_a): episodic return — policy head $Q_1$")
        ax_r.legend(loc="lower right", fontsize=8)
        ax_r.grid(True, alpha=0.3)
        fig.tight_layout()
        qv_name = f"{fig_prefix}mha_head_qvals.pdf"
        fig.savefig(output_dir / qv_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / qv_name}")

    # Option C vs DQN comparison
    option_c_subset = [r for r in all_results if r["algo"] in ("dqn", "mh_c")]
    if option_c_subset and any(r["algo"] == "mh_c" for r in option_c_subset):
        # Learning curves
        fig, ax = plt.subplots(figsize=(7, 4))
        for algo in ["dqn", "mh_c"]:
            runs = [r for r in option_c_subset if r["algo"] == algo]
            if not runs:
                continue
            color = colors.get(algo, "#333")
            label = labels.get(algo, algo)
            for r in runs:
                ep = np.arange(1, len(r["episode_returns"]) + 1)
                ax.plot(ep, r["episode_returns"], color=color, alpha=0.25, linewidth=0.7)
            max_len = max(len(r["episode_returns"]) for r in runs)
            arr = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                arr[i, : len(r["episode_returns"])] = r["episode_returns"]
            mean_ret = np.nanmean(arr, axis=0)
            ep_mean = np.arange(1, max_len + 1)
            w_sm = min(50, max_len // 4)
            if w_sm >= 1:
                smooth = np.convolve(mean_ret, np.ones(w_sm) / w_sm, mode="valid")
                ax.plot(ep_mean[w_sm - 1 :], smooth, color=color, linewidth=2, label=label)
            else:
                ax.plot(ep_mean, mean_ret, color=color, linewidth=2, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episodic return")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title(f"{labels.get('dqn','DQN')} vs {labels.get('mh_c','Option C')} — episodic return")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        oc_name = f"{fig_prefix}option_c_learning_curves.pdf"
        fig.savefig(output_dir / oc_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / oc_name}")
        # Loss curves
        fig, ax = plt.subplots(figsize=(7, 4))
        for algo in ["dqn", "mh_c"]:
            runs = [r for r in option_c_subset if r["algo"] == algo]
            if not runs:
                continue
            color = colors.get(algo, "#333")
            label = labels.get(algo, algo)
            for r in runs:
                ep_idx = np.arange(1, len(r["episode_losses"]) + 1)
                ax.plot(ep_idx, r["episode_losses"], color=color, alpha=0.25, linewidth=0.7)
            max_len = max(len(r["episode_losses"]) for r in runs)
            arr = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                arr[i, : len(r["episode_losses"])] = r["episode_losses"]
            mean_loss = np.nanmean(arr, axis=0)
            ep_idx = np.arange(1, len(mean_loss) + 1)
            w_sm = min(20, max_len // 4)
            if w_sm >= 1:
                smooth = np.convolve(mean_loss, np.ones(w_sm) / w_sm, mode="valid")
                ax.plot(ep_idx[w_sm - 1 :], smooth, color=color, linewidth=2, label=label)
            else:
                ax.plot(ep_idx, mean_loss, color=color, linewidth=2, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean TD loss (per episode)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"Training loss: {labels.get('dqn','DQN')} vs {labels.get('mh_c','Option C')} (episode-averaged)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        oc_loss_name = f"{fig_prefix}option_c_loss.pdf"
        fig.savefig(output_dir / oc_loss_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {output_dir / oc_loss_name}")

    # Multihead softmax combination weights w_k over training steps (first multihead run for clarity)
    mh_runs = [r for r in all_results if r["algo"] == "multihead"]
    if mh_runs:
        r = mh_runs[0]
        pe = r.get("paper_extra") or {}
        hist = pe.get("multihead_w_history") or []
        if hist:
            fig, ax = plt.subplots(figsize=(7, 4))
            head_names = [r"$w_1$ full", r"$w_2$ upr.", r"$w_3$ short", r"$w_4$ cart", r"$w_5$ pole"]
            linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
            episodes = np.array([h.get("episode", h["step"]) for h in hist])
            w_arr = np.array([h["w"] for h in hist])
            for k in range(min(5, w_arr.shape[1])):
                ax.plot(
                    episodes,
                    w_arr[:, k],
                    linestyle=linestyles[k % len(linestyles)],
                    linewidth=1.8,
                    label=head_names[k],
                )
            ax.set_xlabel("Episode")
            ax.set_ylabel(r"$w_k$ (softmax of logits)")
            ax.set_title(
                f"Multihead (5 critics + combined Q): learned mixture weights $w_k$ — seed {r['seed']}"
            )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            for r2 in mh_runs[1:]:
                h2 = (r2.get("paper_extra") or {}).get("multihead_w_history") or []
                if not h2:
                    continue
                ep2 = np.array([h.get("episode", h["step"]) for h in h2])
                w2 = np.array([h["w"] for h in h2])
                for k in range(min(5, w2.shape[1])):
                    ax.plot(ep2, w2[:, k], color="gray", alpha=0.35, linewidth=0.8)
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            sw_name = f"{fig_prefix}multihead_softmax_weights.pdf"
            fig.savefig(output_dir / sw_name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log(f"[plot] Saved {output_dir / sw_name}")

    # ---- New analysis plots (tasks 1, 3, 4, 5, 7, 8) ----
    _plot_qvalue_comparison(all_results, output_dir, fig_prefix, colors, labels)
    _plot_grad_wz_norm(all_results, output_dir, fig_prefix, colors, labels)
    _plot_q1_seed_analysis(all_results, output_dir, fig_prefix)
    _plot_returns_dual_panel(all_results, output_dir, fig_prefix, colors, labels, algos_seen)
    _plot_step_loss_epsilon(all_results, output_dir, fig_prefix, colors, labels)
    _plot_confidence_gate_analysis(all_results, output_dir, fig_prefix, colors, labels)
    _plot_architecture_diagram(output_dir, fig_prefix)

    # Example: gradient×hidden attribution (first DQN run with contribution_sample)
    for r in all_results:
        if r["algo"] != "dqn":
            continue
        pe = r.get("paper_extra") or {}
        cs = pe.get("contribution_sample")
        if not cs or "contrib_h_dim120" not in cs:
            continue
        c = np.array(cs["contrib_h_dim120"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(np.arange(len(c)), c, width=1.0, color="#1f77b4", alpha=0.85)
        ax.set_xlabel(r"Feature index $i$ (first ReLU hidden, dim 120)")
        ax.set_ylabel(r"$h_i \,\partial Q_a / \partial h_i$")
        ax.set_title(
            f"DQN attribution (greedy action $a={cs.get('action_index', '?')}$), seed {r['seed']}"
        )
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        out_c = output_dir / f"{fig_prefix}contrib_dqn_seed{r['seed']}.pdf"
        fig.savefig(out_c, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"[plot] Saved {out_c}")


def _plot_qvalue_comparison(all_results, output_dir, fig_prefix, colors, labels):
    """Task 3 — mean Q at chosen action: DQN vs mixed vs mh_a Q1."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    qval_algos = {
        a: [r for r in all_results if r["algo"] == a and (r.get("paper_extra") or {}).get("qval_history")]
        for a in ("dqn", "mixed", "mixed3f", "multihead", "mh_b", "mh_c")
    }
    mha_qval_runs = [
        r for r in all_results
        if r["algo"] == "mh_a" and (r.get("paper_extra") or {}).get("head_qval_history")
    ]
    if not any(qval_algos.values()) and not mha_qval_runs:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ref_color = "#7f7f7f"
    y_plotted = False
    for algo, runs in qval_algos.items():
        if not runs:
            continue
        color = colors.get(algo, "#333")
        for r_idx, r in enumerate(runs):
            hist = r["paper_extra"]["qval_history"]
            episodes = [h.get("episode", h["step"]) for h in hist]
            qvals = [h["mean_q"] for h in hist]
            lw, alpha = (1.8, 1.0) if r_idx == 0 else (0.7, 0.4)
            ax.plot(episodes, qvals, color=color, linewidth=lw, alpha=alpha,
                    label=labels.get(algo, algo) if r_idx == 0 else None)
        if not y_plotted and runs[0]["paper_extra"]["qval_history"]:
            hist0 = runs[0]["paper_extra"]["qval_history"]
            ax.plot([h.get("episode", h["step"]) for h in hist0], [h["mean_y"] for h in hist0],
                    color=ref_color, linewidth=1.1, linestyle="--",
                    label=r"TD target $y_1$ (first-plotted algo ref.)")
            y_plotted = True
    for r_idx, r in enumerate(mha_qval_runs):
        hist = r["paper_extra"]["head_qval_history"]
        episodes = [h.get("episode", h["step"]) for h in hist]
        q1 = [h["mean_q1"] for h in hist]
        lw, alpha = (1.8, 1.0) if r_idx == 0 else (0.7, 0.4)
        ax.plot(episodes, q1, color="#9467bd", linewidth=lw, alpha=alpha,
                label=r"$Q_1$ — Option A (mh\_a)" if r_idx == 0 else None)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Q-value at chosen action")
    ax.set_title(r"Mean Q-value by episode: all models")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    name = f"{fig_prefix}qvalue_comparison.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def _plot_grad_wz_norm(all_results, output_dir, fig_prefix, colors, labels):
    """Task 7 — W_z gradient norm raw vs clipped for all mixed-type models."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    norm_runs: Dict[str, List] = {}
    for r in all_results:
        algo = r["algo"]
        pe = r.get("paper_extra") or {}
        hist = []
        if algo == "mh_a":
            hist = pe.get("head_qval_history", [])
        else:
            hist = pe.get("qval_history", [])
        if hist and "grad_wz_norm_raw" in hist[0]:
            norm_runs.setdefault(algo, []).append((r, hist))
    if not norm_runs:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for algo, run_list in norm_runs.items():
        color = colors.get(algo, "#333")
        for r_idx, (r, hist) in enumerate(run_list):
            episodes = np.array([h.get("episode", h["step"]) for h in hist])
            raw = np.array([h.get("grad_wz_norm_raw", np.nan) for h in hist])
            clipped = np.array([h.get("grad_wz_norm_clipped", np.nan) for h in hist])
            lw, alpha = (1.8, 1.0) if r_idx == 0 else (0.6, 0.35)
            ax.plot(episodes, raw, color=color, linewidth=lw, alpha=alpha, linestyle="-",
                    label=f"{labels.get(algo, algo)} raw" if r_idx == 0 else None)
            ax.plot(episodes, clipped, color=color, linewidth=lw * 0.7, alpha=alpha, linestyle=":",
                    label=f"{labels.get(algo, algo)} clipped" if r_idx == 0 else None)

    ax.axhline(FEEDBACK_GRAD_CLIP, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"clip threshold = {FEEDBACK_GRAD_CLIP}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\|\nabla_{W_z} L\|_2$")
    ax.set_title(r"$W_z$ gradient norm by episode: raw vs clipped ($Q_1$ decay / normalisation analysis)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    name = f"{fig_prefix}grad_wz_norm.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def _plot_q1_seed_analysis(all_results, output_dir, fig_prefix):
    """Task 4 — Q1 values across all seeds for mh_a (consistency check)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        return

    mha_runs = [
        r for r in all_results
        if r["algo"] == "mh_a" and (r.get("paper_extra") or {}).get("head_qval_history")
    ]
    if not mha_runs:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    palette = cm.tab10(np.linspace(0, 1, max(10, len(mha_runs))))
    all_q1: List[Tuple] = []
    for r_idx, r in enumerate(mha_runs):
        hist = r["paper_extra"]["head_qval_history"]
        episodes = np.array([h.get("episode", h["step"]) for h in hist])
        q1 = np.array([h["mean_q1"] for h in hist])
        all_q1.append((episodes, q1))
        ax.plot(episodes, q1, color=palette[r_idx], linewidth=0.9, alpha=0.75,
                label=f"seed {r['seed']}")

    if len(all_q1) > 1:
        max_len = max(len(q) for _, q in all_q1)
        q1_arr = np.full((len(all_q1), max_len), np.nan)
        for i, (_, q) in enumerate(all_q1):
            q1_arr[i, : len(q)] = q
        mean_q1 = np.nanmean(q1_arr, axis=0)
        ref_eps = all_q1[0][0]
        ax.plot(ref_eps[: len(mean_q1)], mean_q1, color="black", linewidth=2.4,
                label="cross-seed mean", zorder=5)

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Mean $Q_1$ at chosen action")
    ax.set_title(r"$Q_1$ consistency across seeds — Option A (mh\_a)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    name = f"{fig_prefix}q1_seed_analysis.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def _plot_returns_dual_panel(all_results, output_dir, fig_prefix, colors, labels, algos_seen):
    """Task 5 — episodic return vs training steps (left) and vs episode index (right)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 4))
    for algo in algos_seen:
        runs = [r for r in all_results if r["algo"] == algo]
        color = colors.get(algo, "#333")
        label = labels.get(algo, algo)
        for r in runs:
            ep_steps = np.array(r["episode_steps"])
            returns = np.array(r["episode_returns"])
            ep_idx = np.arange(1, len(returns) + 1)
            ax_s.plot(ep_steps, returns, color=color, alpha=0.18, linewidth=0.7)
            ax_e.plot(ep_idx, returns, color=color, alpha=0.18, linewidth=0.7)
        if not runs:
            continue
        max_len = max(len(r["episode_returns"]) for r in runs)
        ret_arr = np.full((len(runs), max_len), np.nan)
        step_arr = np.full((len(runs), max_len), np.nan)
        for i, r in enumerate(runs):
            ret_arr[i, : len(r["episode_returns"])] = r["episode_returns"]
            step_arr[i, : len(r["episode_steps"])] = r["episode_steps"]
        mean_ret = np.nanmean(ret_arr, axis=0)
        mean_steps = np.nanmean(step_arr, axis=0)
        ep_mean = np.arange(1, max_len + 1)
        w = min(50, max_len // 4)
        if w >= 1:
            smooth = np.convolve(mean_ret, np.ones(w) / w, mode="valid")
            ax_s.plot(mean_steps[w - 1 :], smooth, color=color, linewidth=2.0, label=label)
            ax_e.plot(ep_mean[w - 1 :], smooth, color=color, linewidth=2.0, label=label)
        else:
            ax_s.plot(mean_steps, mean_ret, color=color, linewidth=2.0, label=label)
            ax_e.plot(ep_mean, mean_ret, color=color, linewidth=2.0, label=label)

    # Mark episode-boundary transitions every ~10% of training in the step-axis panel
    ref_run = next(
        (r for r in all_results if r["algo"] == "dqn" and r["episode_steps"]),
        next((r for r in all_results if r["episode_steps"]), None),
    )
    if ref_run:
        ref_steps = np.array(ref_run["episode_steps"])
        if len(ref_steps) >= 10:
            interval = max(1, len(ref_steps) // 10)
            for i in range(0, len(ref_steps), interval):
                ax_s.axvline(ref_steps[i], color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    ax_s.set_xlabel("Environment step")
    ax_s.set_ylabel("Episodic return")
    ax_s.set_title("Return vs. training steps\n(dotted lines = episode boundaries every 10%)")
    ax_s.legend(loc="lower right", fontsize=7)
    ax_s.grid(True, alpha=0.3)

    ax_e.set_xlabel("Episode")
    ax_e.set_ylabel("Episodic return")
    ax_e.set_title("Return vs. episode index")
    ax_e.legend(loc="lower right", fontsize=7)
    ax_e.grid(True, alpha=0.3)

    fig.tight_layout()
    name = f"{fig_prefix}returns_steps_vs_episodes.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def _plot_step_loss_epsilon(all_results, output_dir, fig_prefix, colors, labels):
    """Task 8 — step-level TD loss with epsilon-greedy schedule overlay."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    runs_with_data = [
        r for r in all_results if (r.get("paper_extra") or {}).get("step_loss_history")
    ]
    if not runs_with_data:
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    plotted: set = set()
    eps_drawn = False
    for r in runs_with_data:
        algo = r["algo"]
        hist = r["paper_extra"]["step_loss_history"]
        episodes = np.array([h.get("episode", h["step"]) for h in hist])
        losses = np.array([h["loss"] for h in hist])
        eps = np.array([h["epsilon"] for h in hist])
        color = colors.get(algo, "#333")
        lw, alpha = (1.6, 0.9) if algo not in plotted else (0.6, 0.35)
        ax1.plot(episodes, losses, color=color, linewidth=lw, alpha=alpha,
                 label=labels.get(algo, algo) if algo not in plotted else None)
        if not eps_drawn:
            ax2.plot(episodes, eps, color="#7f7f7f", linewidth=1.2, linestyle="--",
                     alpha=0.55, label=r"$\epsilon$ (exploration rate)")
            eps_drawn = True
        plotted.add(algo)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("TD loss (MSE snapshot)")
    ax2.set_ylabel(r"$\epsilon$-greedy exploration rate", color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")
    ax2.set_ylim(0, 1.05)
    ax1.set_title(
        r"TD loss snapshots vs. $\epsilon$-greedy schedule (by episode)"
        "\n(correlate loss peaks with exploration phase)"
    )
    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    name = f"{fig_prefix}step_loss_epsilon.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def _plot_architecture_diagram(output_dir, fig_prefix):
    """Task 1 — text diagram of each model variant with detach() points annotated."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _variants = [
        (
            "DQN  (no detach — full backprop)",
            "#1f77b4",
            [
                "obs ──► Linear(obs_dim → 120) ──► ReLU ──► Linear(120 → 84) ──► ReLU ──► Linear(84 → A)",
                "Gradient flows through every layer.  W_z (first layer) updated by Adam.",
            ],
        ),
        (
            "Mixed  (detach z before trunk)",
            "#d62728",
            [
                "obs ──► [W_z: Linear(obs_dim → 120) = z] ──DETACH(z)──► ReLU ──► Linear(120→84) ──► ReLU ──► Linear(84→A)",
                "Trunk updated by backprop from Q-loss.  W_z updated ONLY via manual B-feedback: ΔW_z = B·e.",
                "B ∈ ℝ^{F×A}  fixed sparse;  update rule bypasses Adam (direct .data.add_).",
            ],
        ),
        (
            "mh_a / Option A  (multi-detach: z and per-head h)",
            "#9467bd",
            [
                "obs ────────────► linear_feature(z) ──DETACH(z)──► trunk ──► head1(Q1)  ← backprop from Q1 loss",
                "                                                    trunk ──DETACH(h)──► head2, head3  (aux. heads)",
                "obs_cart (pole masked) ── linear_feature(z_cart) ──DETACH──► trunk ──DETACH──► head4(Q4)",
                "obs_pole (cart masked) ── linear_feature(z_pole) ──DETACH──► trunk ──DETACH──► head5(Q5)",
                "W_z updated by B-feedback combining all 5 head errors:  ΔW_z = B · [e1, e2, e3, e4, e5].",
            ],
        ),
        (
            "mh_c / Option C  (no detach + B-feedback bonus)",
            "#17becf",
            [
                "obs ──► linear_feature(z) ──► trunk ──► head1(Q1)",
                "Full backprop: W_z updated by Adam (Q1 gradient flows through z).",
                "ADDITIONALLY W_z.data receives additive B-feedback nudge (bypasses Adam).",
            ],
        ),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    y = 0.97
    ax.text(0.0, y, "Architecture & DETACH() point reference — model variants",
            fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)
    y -= 0.08
    for title, color, lines in _variants:
        ax.text(0.0, y, title, fontsize=9, fontweight="bold", color=color,
                va="top", transform=ax.transAxes)
        y -= 0.05
        for line in lines:
            ax.text(0.015, y, line, fontsize=7.8, va="top", transform=ax.transAxes,
                    fontfamily="monospace")
            y -= 0.042
        y -= 0.025

    fig.tight_layout()
    name = f"{fig_prefix}architecture_detach_diagram.pdf"
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] Saved {output_dir / name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CleanRL DQN vs Mixed backprop, comparable setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--algos",
        type=str,
        default=",".join(DEFAULT_ALGOS),
        help="dqn,mixed,mixed3f,multihead,mh_a,mh_3a,mh_b,mh_c",
    )
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--total-timesteps", type=int, default=None, help="If set, run for this many steps")
    p.add_argument("--n-episodes", type=int, default=None, help="If set, run until this many episodes")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Save final model checkpoints here (e.g. results/checkpoints)")
    p.add_argument(
        "--paper-dir",
        type=str,
        default=None,
        help="If set, save paper_extra JSON (multihead w, attribution) under this directory (relative to repo root)",
    )
    p.add_argument(
        "--confidence-gating",
        action="store_true",
        help=(
            "Use partial-head action-gap margins to gate W_z cart/pole compartments for "
            "Q_full/Q1 path (multihead, mh_a, mh_b, mh_c, mh_3a; CartPole-v1 only)."
        ),
    )
    p.add_argument(
        "--multihead-action",
        type=str,
        default="combined",
        choices=("combined", "head1"),
        help="multihead only: greedy action from Q_combined (default) or Q1 only",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="Write outputs under paper/experiment_<timestamp>/ (figures, summary, paper JSON)",
    )
    p.add_argument(
        "--fig-prefix",
        type=str,
        default="",
        help="Prefix for saved figure filenames (e.g. 'v2_') to avoid overwriting previous results",
    )
    p.add_argument("--save-figures", action="store_true", default=True)
    p.add_argument("--no-save-figures", action="store_false", dest="save_figures")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    if args.paper:
        output_dir = root / "paper" / f"experiment_{ts_run}"
        output_dir.mkdir(parents=True, exist_ok=True)
        paper_dir_resolved: Optional[Path] = output_dir
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        paper_dir_resolved = None
        if args.paper_dir:
            paper_dir_resolved = Path(args.paper_dir)
            if not paper_dir_resolved.is_absolute():
                paper_dir_resolved = root / paper_dir_resolved
            paper_dir_resolved.mkdir(parents=True, exist_ok=True)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    algos = [
        a
        for a in algos
        if a in ("dqn", "mixed", "mixed3f", "multihead", "mh_a", "mh_3a", "mh_b", "mh_c")
    ] or list(DEFAULT_ALGOS)
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
    if paper_dir_resolved:
        log(f"  Paper extras: {paper_dir_resolved}")
    if checkpoint_dir:
        log(f"  Checkpoints: {checkpoint_dir}")
    log("=" * 60)

    all_results: List[Dict] = []
    for algo in algos:
        for seed in seeds:
            log(f"\n[run] {algo} seed={seed} ...")
            t0 = time.perf_counter()
            paper_slug = (
                f"{algo}_cg"
                if args.confidence_gating
                and algo in MULTIHEAD_CONFIDENCE_GATING_ALGOS
                else None
            )
            episode_returns, episode_steps, episode_losses, paper_extra = run_training(
                algo=algo,
                env_id=args.env,
                seed=seed,
                device=device,
                total_timesteps=args.total_timesteps,
                n_episodes=args.n_episodes,
                checkpoint_dir=checkpoint_dir,
                paper_dir=paper_dir_resolved,
                multihead_action_mode=args.multihead_action,
                confidence_gating=args.confidence_gating,
                paper_save_algo=paper_slug,
            )
            display_algo = f"{algo}_cg" if paper_extra.get("confidence_gating") else algo
            elapsed = time.perf_counter() - t0
            all_results.append({
                "algo": display_algo,
                "seed": seed,
                "episode_returns": episode_returns,
                "episode_steps": episode_steps,
                "episode_losses": episode_losses,
                "paper_extra": paper_extra,
                "training_time_sec": elapsed,
                "n_episodes": len(episode_returns),
                "final_return": episode_returns[-1] if episode_returns else None,
                "mean_last_100": (
                    np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else None
                ),
            })
            log(f"  Done in {elapsed:.1f}s | episodes={len(episode_returns)} | final_return={all_results[-1]['final_return']}")

    ts = ts_run if args.paper else time.strftime("%Y%m%d_%H%M%S")
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

    fig_prefix = getattr(args, "fig_prefix", "")
    if args.save_figures:
        log("Saving figures ...")
        plot_and_save(all_results, output_dir, fig_prefix=fig_prefix)

    if args.paper:
        fig_dir = root / "paper" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for stem in (
            "cleanrl_vs_mixed_learning_curves",
            "cleanrl_vs_mixed_loss",
            "dqn_vs_multihead_learning_curves",
            "multihead_softmax_weights",
            "options_ab_learning_curves",
            "options_ab_loss",
            "option_c_learning_curves",
            "option_c_loss",
            "mha_head_losses",
            "mha_head_qvals",
            # New analysis figures
            "qvalue_comparison",
            "grad_wz_norm",
            "q1_seed_analysis",
            "returns_steps_vs_episodes",
            "step_loss_epsilon",
            "architecture_detach_diagram",
            "confidence_gate_k_trace",
        ):
            src = output_dir / f"{fig_prefix}{stem}.pdf"
            if src.exists():
                shutil.copy(src, fig_dir / f"{fig_prefix}{stem}.pdf")
        for src in sorted(output_dir.glob(f"{fig_prefix}contrib_dqn_seed*.pdf")):
            shutil.copy(src, fig_dir / f"{fig_prefix}contrib_dqn_sample.pdf")
            break
        log(f"Copied key figures to {fig_dir}")

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
