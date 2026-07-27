"""
Four-way CartPole DQN comparison runner.

Trains and compares four modalities:
  1. baseline    – standard DQN, no gate           (from confgate_analytical)
  2. analytical  – Z-norm margin gate, fixed range (from confgate_analytical)
  3. learnable   – 12 nn.Parameter gate scalars    (from confgate_learnable)
  4. rnn         – 20-unit episodic GRU outputs LR mult, ε floor, k_cart, k_pole

Usage:
    # All four modalities, 10 seeds, 500k steps
    python code/confgate_fourway.py

    # Only baseline and RNN, 3 seeds, fast schedule
    python code/confgate_fourway.py --modalities baseline,rnn --seeds 1,2,3 \\
        --fast --epsilon-decay-episodes 800 --total-timesteps 200000

    # Plot only (skip training, reload checkpoints)
    python code/confgate_fourway.py --plot-only

Checkpoints: paper/_tmp_b_feedb_cg/ckpt_fourway_{decay_tag}[_fast]/
Figures:     paper/figures/conf_fourway/
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Paths — add code/ dir to sys.path so sibling modules are importable
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import confgate_analytical as cg_analytical  # noqa: E402
import confgate_learnable  as cg_learnable   # noqa: E402

FIG_DIR = ROOT / "paper" / "figures" / "fourway_gate"

# ---------------------------------------------------------------------------
# Hyper-parameters (must match sibling modules)
# ---------------------------------------------------------------------------
LEARNING_RATE          = 2.5e-4
GAMMA                  = 0.99
BUFFER_SIZE            = 10_000
BATCH_SIZE             = 128
START_E                = 1.0
END_E                  = 0.05
EPSILON_DECAY_EPISODES = 3_500
LEARNING_STARTS        = 10_000
TRAIN_FREQUENCY        = 10
TARGET_NETWORK_FREQ    = 500
LOG_EVERY              = 5_000

OBS_TO_120 = 120
HIDDEN_84  = 84
RNN_HIDDEN = 20

DEFAULT_SEEDS       = list(range(1, 11))
DEFAULT_TOTAL_STEPS = 500_000

ALL_MODALITIES = [
    "baseline", "analytical", "learnable",
    "rnn", "rnn_reinforce", "gru_burnin",
    "lstm", "lstm_burnin",
]

# Burn-in hyper-parameters (shared by GRU and LSTM burn-in modalities)
LSTM_SEQ_LEN      = 20    # total sequence length stored per entry
LSTM_BURN_IN      = 10    # steps used to warm up hidden state, no gradient
LSTM_LEARN_LEN    = LSTM_SEQ_LEN - LSTM_BURN_IN   # steps that produce loss (10)
LSTM_SEQ_BATCH    = 32    # sequences per training step (≈ 320 transition-equivalents)
LSTM_SEQ_BUFFER   = 2_000 # max sequences in the burn-in replay buffer

# Plotting colours
COLOR_BASE      = "#2ca02c"   # green      – baseline
COLOR_GATE      = "#9467bd"   # purple     – analytical
COLOR_LEARN     = "#d62728"   # red        – learnable
COLOR_RNN       = "#ff7f0e"   # orange     – GRU controller (direct gradient)
COLOR_GRU_BI    = "#bcbd22"   # olive      – GRU controller with burn-in
COLOR_REINFORCE = "#1f77b4"   # blue       – RNN controller with REINFORCE
COLOR_LSTM      = "#e377c2"   # pink       – LSTM controller (single-step)
COLOR_LSTM_BI   = "#17becf"   # teal       – LSTM controller with burn-in


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _linear_schedule(
    start: float, end: float, duration_ep: int, n_ep_done: int
) -> float:
    frac = min(1.0, n_ep_done / max(1, duration_ep))
    return start + frac * (end - start)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid") if w >= 2 else arr


# ---------------------------------------------------------------------------
# RNN Controller
# ---------------------------------------------------------------------------
class RNNController(nn.Module):
    """
    20-unit GRUCell that produces four per-step training control signals:

      lr_mult   : multiplicative scale on LEARNING_RATE, ∈ (0, ∞),  init ≈ 1.0
      eps_floor : exploration ε floor,                  ∈ [0, END_E], init ≈ END_E
      k_cart    : W_z cart-compartment scale,           ∈ (0, ∞),  init ≈ 1.0
      k_pole    : W_z pole-compartment scale,           ∈ (0, ∞),  init ≈ 1.0

    GRU input (126 dims):
      concat(Q_full.detach, Q_cart.detach, Q_pole.detach, z.detach)
      where z = linear_feature(obs) before the ReLU trunk.

    All GRU inputs are detached so the controller does not interfere with
    the main Q-learning gradient.  k_cart / k_pole are NOT detached when
    used in _gated_z, so loss_full → Q_full_gated → z_gated → k → GRU
    provides a gradient path that trains the controller.
    """

    GRU_INPUT  = 2 + 2 + 2 + OBS_TO_120  # 126
    GRU_HIDDEN = RNN_HIDDEN                # 20

    def __init__(self) -> None:
        super().__init__()
        self.gru_cell    = nn.GRUCell(self.GRU_INPUT, self.GRU_HIDDEN)
        self.output_head = nn.Linear(self.GRU_HIDDEN, 4)
        # Warm-start biases so the model starts as a functional baseline:
        #   softplus(0.5413) ≈ 1.0  → lr_mult ≈ 1,  k_cart ≈ 1,  k_pole ≈ 1
        #   sigmoid(5.0) * END_E ≈ 0.993 * END_E ≈ END_E → eps_floor ≈ END_E
        with torch.no_grad():
            self.output_head.bias.data = torch.tensor(
                [0.5413, 5.0, 0.5413, 0.5413], dtype=torch.float32
            )
            nn.init.zeros_(self.output_head.weight)

    def forward(
        self,
        x: torch.Tensor,   # (B, 126) — caller must detach all inputs
        h: torch.Tensor,   # (B, RNN_HIDDEN)
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns (rnn_out dict, h_next).  All values in rnn_out are (B,)."""
        h_next = self.gru_cell(x, h)           # (B, 20)
        out    = self.output_head(h_next)       # (B, 4)
        return {
            "lr_mult":   F.softplus(out[:, 0]) + 1e-3,          # > 0
            "eps_floor": torch.sigmoid(out[:, 1]) * END_E,      # ∈ [0, END_E]
            "k_cart":    F.softplus(out[:, 2]) + 1e-3,          # > 0
            "k_pole":    F.softplus(out[:, 3]) + 1e-3,          # > 0
        }, h_next


# ---------------------------------------------------------------------------
# RNN Controller — Option 2: REINFORCE on loss improvement
# ---------------------------------------------------------------------------
class RNNController_with_loss_improvement(nn.Module):
    """
    GRU controller trained via REINFORCE using the per-step loss improvement
    as the reward signal.  Outputs only lr_mult and eps_floor — no W_z gating.

    Gradient path (decoupled from Q-learning):
      ctrl_reward = loss_before - loss_after   (detached scalar)
      ctrl_loss   = -ctrl_reward * log(lr_mult.mean())
      opt_ctrl.step()   ← separate optimizer, controller params only

    Because lr_mult is applied to the optimizer externally (.item()) and the
    controller's backward pass is done after opt_main.step(), there is no path
    by which the controller can collapse the Q-learning loss.

    Inputs  (126 dims, all detached):
      concat(Q_full, Q_cart, Q_pole, z_before_trunk)
    Outputs:
      lr_mult   ∈ (0, ∞),    init ≈ 1.0
      eps_floor ∈ [0, END_E], init ≈ END_E
    """

    GRU_INPUT  = 2 + 2 + 2 + OBS_TO_120  # 126
    GRU_HIDDEN = RNN_HIDDEN               # 20

    def __init__(self) -> None:
        super().__init__()
        self.gru_cell    = nn.GRUCell(self.GRU_INPUT, self.GRU_HIDDEN)
        self.output_head = nn.Linear(self.GRU_HIDDEN, 2)
        with torch.no_grad():
            # softplus(0.5413) ≈ 1.0 → lr_mult ≈ 1.0 at init
            # sigmoid(5.0) * END_E ≈ END_E → eps_floor ≈ END_E at init
            self.output_head.bias.data = torch.tensor(
                [0.5413, 5.0], dtype=torch.float32
            )
            nn.init.zeros_(self.output_head.weight)

    def forward(
        self,
        x: torch.Tensor,   # (B, 126) — caller supplies detached inputs
        h: torch.Tensor,   # (B, RNN_HIDDEN)
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns (rnn_out dict, h_next)."""
        h_next = self.gru_cell(x, h)
        out    = self.output_head(h_next)       # (B, 2)
        return {
            "lr_mult":   F.softplus(out[:, 0]) + 1e-3,     # > 0
            "eps_floor": torch.sigmoid(out[:, 1]) * END_E, # ∈ [0, END_E]
        }, h_next


# ---------------------------------------------------------------------------
# LSTM Controller (modality 5)
# ---------------------------------------------------------------------------
class LSTMController(nn.Module):
    """
    20-unit LSTMCell controller — identical inputs and outputs to RNNController
    but uses an LSTM cell instead of a GRU cell.

    LSTM maintains two separate state vectors per step:
      h  (hidden state)  — carries short-term sequential patterns
      c  (cell state)    — carries long-term memory via input/forget gates

    This gives the controller an extra degree of freedom compared to the GRU:
    the cell state can accumulate signals over many steps while the hidden state
    is gated more selectively.  Both states are stored in the replay buffer and
    replayed with each transition.

    Inputs  (126 dims, all detached from the Q-learning graph):
        concat(Q_full.detach, Q_cart.detach, Q_pole.detach, z.detach)
    Outputs (same four scalars as RNNController):
        lr_mult   ∈ (0, ∞),    init ≈ 1.0
        eps_floor ∈ [0, END_E], init ≈ END_E
        k_cart    ∈ (0, ∞),    init ≈ 1.0
        k_pole    ∈ (0, ∞),    init ≈ 1.0
    """

    LSTM_INPUT  = 2 + 2 + 2 + OBS_TO_120  # 126
    LSTM_HIDDEN = RNN_HIDDEN               # 20

    def __init__(self) -> None:
        super().__init__()
        self.lstm_cell   = nn.LSTMCell(self.LSTM_INPUT, self.LSTM_HIDDEN)
        self.output_head = nn.Linear(self.LSTM_HIDDEN, 4)
        with torch.no_grad():
            self.output_head.bias.data = torch.tensor(
                [0.5413, 5.0, 0.5413, 0.5413], dtype=torch.float32
            )
            nn.init.zeros_(self.output_head.weight)

    def forward(
        self,
        x:  torch.Tensor,                           # (B, 126) — caller must detach
        hc: Tuple[torch.Tensor, torch.Tensor],       # (h, c) each (B, LSTM_HIDDEN)
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Returns (lstm_out dict, (h_next, c_next)).  All values in dict are (B,)."""
        h, c = hc
        h_next, c_next = self.lstm_cell(x, (h, c))
        out = self.output_head(h_next)              # (B, 4)
        return {
            "lr_mult":   F.softplus(out[:, 0]) + 1e-3,
            "eps_floor": torch.sigmoid(out[:, 1]) * END_E,
            "k_cart":    F.softplus(out[:, 2]) + 1e-3,
            "k_pole":    F.softplus(out[:, 3]) + 1e-3,
        }, (h_next, c_next)


# ---------------------------------------------------------------------------
# Network — 3-head DQN + RNNController_with_loss_improvement
# ---------------------------------------------------------------------------
class BFeedbackRNNLossImprovNetwork(nn.Module):
    """
    Standard 3-head DQN backbone (no W_z gating) paired with
    RNNController_with_loss_improvement.

    The controller is NOT included in opt_main.  It has its own opt_ctrl and
    is trained via REINFORCE on the loss improvement signal after each
    Q-network update step.

    forward() returns rnn_input alongside the other outputs so the training
    loop can reuse it for the in-graph REINFORCE backward without a full
    second forward pass.
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head_full = nn.Linear(HIDDEN_84, n_actions)
        self.head_cart = nn.Linear(HIDDEN_84, n_actions)
        self.head_pole = nn.Linear(HIDDEN_84, n_actions)
        self.rnn_controller = RNNController_with_loss_improvement()

    @staticmethod
    def _mask_cart(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 2:4] = 0.0; return out

    @staticmethod
    def _mask_pole(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 0:2] = 0.0; return out

    def zero_hidden(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(1, RNN_HIDDEN, device=device, dtype=dtype)

    def forward(
        self,
        obs: torch.Tensor,     # (B, obs_dim)
        h_prev: torch.Tensor,  # (B, RNN_HIDDEN)
    ) -> Tuple[
        torch.Tensor,              # Q_full    (B, n_actions)
        torch.Tensor,              # Q_cart    (B, n_actions)
        torch.Tensor,              # Q_pole    (B, n_actions)
        torch.Tensor,              # h_next    (B, RNN_HIDDEN)
        Dict[str, torch.Tensor],   # rnn_out: lr_mult, eps_floor
        torch.Tensor,              # rnn_input (B, 126) — detached, for REINFORCE reuse
    ]:
        # ---- Aux paths (trunk detached — trains only aux heads) ----
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart.detach()))

        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole.detach()))

        # ---- Full path (standard, no gating) ----
        z      = self.linear_feature(obs)
        Q_full = self.head_full(self.trunk(z))

        # ---- Controller forward with all-detached inputs ----
        rnn_input = torch.cat(
            [Q_full.detach(), Q_cart.detach(), Q_pole.detach(), z.detach()],
            dim=-1,
        )  # (B, 126) — no grad_fn; safe to reuse after loss_full.backward()
        rnn_out, h_next = self.rnn_controller(rnn_input, h_prev)

        return Q_full, Q_cart, Q_pole, h_next, rnn_out, rnn_input

    def forward_q_only(
        self, obs: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Q_full, _, _, h_next, _, _ = self.forward(obs, h_prev)
        return Q_full, h_next


# ---------------------------------------------------------------------------
# Network — 3-head DQN + RNNController (modality 4)
# ---------------------------------------------------------------------------
class BFeedbackRNNControllerNetwork(nn.Module):
    """
    Identical 3-head backbone to confgate_learnable.BFeedbackConfGateNetwork,
    extended with an RNNController that adapts lr_mult, eps_floor, k_cart,
    k_pole on every training step.

    Two-pass forward:
      Pass 1 — ungated: computes Q_full_ungated, Q_cart, Q_pole, z.
      RNN    — receives detached (Q_full, Q_cart, Q_pole, z); produces
               lr_mult, eps_floor, k_cart, k_pole (and h_next).
      Pass 2 — gated:  applies k_cart / k_pole (NOT detached) via _gated_z
               to produce Q_full_gated for the main TD loss.

    Gradient path for the RNN controller:
      loss_full → Q_full_gated → z_gated → k_cart / k_pole
               → output_head → GRU cell weights
    No retain_graph needed because all GRU inputs were detached.
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head_full = nn.Linear(HIDDEN_84, n_actions)
        self.head_cart = nn.Linear(HIDDEN_84, n_actions)
        self.head_pole = nn.Linear(HIDDEN_84, n_actions)
        self.rnn_controller = RNNController()

    # ------------------------------------------------------------------
    def _gating_active(self) -> bool:
        return self.obs_dim == 4 and self.n_actions == 2

    @staticmethod
    def _mask_cart(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 2:4] = 0.0; return out

    @staticmethod
    def _mask_pole(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 0:2] = 0.0; return out

    def _gated_z(
        self,
        obs: torch.Tensor,
        k_cart: torch.Tensor,  # (B,) — NOT detached so grad reaches RNN
        k_pole: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        W  = self.linear_feature.weight      # (120, obs_dim)
        b  = self.linear_feature.bias        # (120,)
        kc = k_cart.unsqueeze(-1)            # (B, 1)
        kp = k_pole.unsqueeze(-1)            # (B, 1)
        return (
            kc * (obs[..., 0:2] @ W[:, 0:2].T)
            + kp * (obs[..., 2:4] @ W[:, 2:4].T)
            + b
        )

    def zero_hidden(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Return a zero hidden state (1, RNN_HIDDEN) for rollout initialisation."""
        return torch.zeros(1, RNN_HIDDEN, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,       # (B, obs_dim)
        h_prev: torch.Tensor,    # (B, RNN_HIDDEN)
        detach_k: bool = False,  # if True, detach k_cart/k_pole before z_gated
    ) -> Tuple[
        torch.Tensor,           # Q_full_gated  (B, n_actions)
        torch.Tensor,           # Q_cart        (B, n_actions)
        torch.Tensor,           # Q_pole        (B, n_actions)
        torch.Tensor,           # z_ungated     (B, 120)
        torch.Tensor,           # h_next        (B, RNN_HIDDEN)
        Dict[str, torch.Tensor],  # rnn_out
    ]:
        # ---- Aux paths (trunk detached, trains only aux heads) ----
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart.detach()))

        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole.detach()))

        # ---- Pass 1: ungated full path (feeds RNN input, not main loss) ----
        z              = self.linear_feature(obs)
        Q_full_ungated = self.head_full(self.trunk(z))

        # ---- RNN step: all inputs detached ----
        rnn_input = torch.cat(
            [Q_full_ungated.detach(), Q_cart.detach(), Q_pole.detach(), z.detach()],
            dim=-1,
        )  # (B, 126)
        rnn_out, h_next = self.rnn_controller(rnn_input, h_prev)

        # ---- Pass 2: gated path ----
        # detach_k=True: k_cart/k_pole are detached before z_gated so the
        # multi-step BPTT gradient does NOT flow through the gate path.
        # This prevents gradient explosion when training with burn-in.
        if self._gating_active():
            k_c = rnn_out["k_cart"].detach() if detach_k else rnn_out["k_cart"]
            k_p = rnn_out["k_pole"].detach() if detach_k else rnn_out["k_pole"]
            z_gated      = self._gated_z(obs, k_c, k_p)
            Q_full_gated = self.head_full(self.trunk(z_gated))
        else:
            Q_full_gated = Q_full_ungated

        return Q_full_gated, Q_cart, Q_pole, z, h_next, rnn_out

    def forward_q_only(
        self, obs: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Q_full_gated, _, _, _, h_next, _ = self.forward(obs, h_prev)
        return Q_full_gated, h_next


# ---------------------------------------------------------------------------
# Network — 3-head DQN + LSTMController (modality 5)
# ---------------------------------------------------------------------------
class BFeedbackLSTMControllerNetwork(nn.Module):
    """
    Identical 3-head backbone to BFeedbackRNNControllerNetwork, but the
    meta-controller is an LSTMController rather than a GRUCell.

    Because the LSTM has two state vectors (h, c), the network accepts and
    returns (h_prev, c_prev) / (h_next, c_next) pairs rather than a single
    hidden state.  The gradient path for the controller is the same as for
    the GRU modality:
        loss_full → Q_full_gated → z_gated → k_cart / k_pole
                  → output_head → LSTMCell weights
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        self.linear_feature   = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head_full = nn.Linear(HIDDEN_84, n_actions)
        self.head_cart = nn.Linear(HIDDEN_84, n_actions)
        self.head_pole = nn.Linear(HIDDEN_84, n_actions)
        self.lstm_controller = LSTMController()

    def _gating_active(self) -> bool:
        return self.obs_dim == 4 and self.n_actions == 2

    @staticmethod
    def _mask_cart(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 2:4] = 0.0; return out

    @staticmethod
    def _mask_pole(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 0:2] = 0.0; return out

    def _gated_z(
        self,
        obs:    torch.Tensor,   # (B, obs_dim)
        k_cart: torch.Tensor,   # (B,) — NOT detached so grad reaches LSTM
        k_pole: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        W  = self.linear_feature.weight
        b  = self.linear_feature.bias
        kc = k_cart.unsqueeze(-1)
        kp = k_pole.unsqueeze(-1)
        return (
            kc * (obs[..., 0:2] @ W[:, 0:2].T)
            + kp * (obs[..., 2:4] @ W[:, 2:4].T)
            + b
        )

    def zero_hidden(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h, c) zero states, each (1, LSTM_HIDDEN), for rollout init."""
        z = torch.zeros(1, RNN_HIDDEN, device=device, dtype=dtype)
        return z, z.clone()

    def forward(
        self,
        obs:    torch.Tensor,                          # (B, obs_dim)
        h_prev: torch.Tensor,                          # (B, RNN_HIDDEN)
        c_prev: torch.Tensor,                          # (B, RNN_HIDDEN)
    ) -> Tuple[
        torch.Tensor,                                  # Q_full_gated  (B, n_actions)
        torch.Tensor,                                  # Q_cart        (B, n_actions)
        torch.Tensor,                                  # Q_pole        (B, n_actions)
        torch.Tensor,                                  # z_ungated     (B, 120)
        torch.Tensor,                                  # h_next        (B, RNN_HIDDEN)
        torch.Tensor,                                  # c_next        (B, RNN_HIDDEN)
        Dict[str, torch.Tensor],                       # lstm_out
    ]:
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart.detach()))

        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole.detach()))

        z              = self.linear_feature(obs)
        Q_full_ungated = self.head_full(self.trunk(z))

        lstm_input = torch.cat(
            [Q_full_ungated.detach(), Q_cart.detach(), Q_pole.detach(), z.detach()],
            dim=-1,
        )  # (B, 126)
        lstm_out, (h_next, c_next) = self.lstm_controller(
            lstm_input, (h_prev, c_prev)
        )

        if self._gating_active():
            z_gated      = self._gated_z(obs, lstm_out["k_cart"], lstm_out["k_pole"])
            Q_full_gated = self.head_full(self.trunk(z_gated))
        else:
            Q_full_gated = Q_full_ungated

        return Q_full_gated, Q_cart, Q_pole, z, h_next, c_next, lstm_out

    def forward_q_only(
        self,
        obs:    torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q_full_gated, _, _, _, h_next, c_next, _ = self.forward(obs, h_prev, c_prev)
        return Q_full_gated, h_next, c_next


# ---------------------------------------------------------------------------
# Replay buffer with per-transition hidden-state storage
# ---------------------------------------------------------------------------
_RNNSamples = namedtuple(
    "_RNNSamples",
    ["observations", "actions", "next_observations", "dones", "rewards", "gate_h_in"],
)


class RNNReplayBuffer:
    """Standard uniform replay buffer + gate_h_in column (shape [N, RNN_HIDDEN])."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        hidden_size: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.device      = device
        self.pos  = 0
        self.full = False

        self.observations      = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions           = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards           = np.zeros((buffer_size,), dtype=np.float32)
        self.dones             = np.zeros((buffer_size,), dtype=np.float32)
        self.gate_h_in         = np.zeros((buffer_size, hidden_size), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: float,
        h_in: np.ndarray,   # (hidden_size,) — hidden state BEFORE processing obs
    ) -> None:
        self.observations[self.pos]      = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos]   = action
        self.rewards[self.pos]   = reward
        self.dones[self.pos]     = done
        self.gate_h_in[self.pos] = h_in
        self.pos  = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> _RNNSamples:
        idx = np.random.randint(0, len(self), size=batch_size)
        return _RNNSamples(
            observations      = torch.tensor(self.observations[idx],      device=self.device),
            actions           = torch.tensor(self.actions[idx],           device=self.device).unsqueeze(1),
            next_observations = torch.tensor(self.next_observations[idx], device=self.device),
            dones             = torch.tensor(self.dones[idx],             device=self.device),
            rewards           = torch.tensor(self.rewards[idx],           device=self.device),
            gate_h_in         = torch.tensor(self.gate_h_in[idx],        device=self.device),
        )


# ---------------------------------------------------------------------------
# Replay buffer — LSTM (stores h_in and c_in separately)
# ---------------------------------------------------------------------------
_LSTMSamples = namedtuple(
    "_LSTMSamples",
    ["observations", "actions", "next_observations", "dones", "rewards",
     "gate_h_in", "gate_c_in"],
)


class LSTMReplayBuffer:
    """Uniform replay buffer that stores both h_in and c_in per transition."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        hidden_size: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.device      = device
        self.pos  = 0
        self.full = False

        self.observations      = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions           = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards           = np.zeros((buffer_size,), dtype=np.float32)
        self.dones             = np.zeros((buffer_size,), dtype=np.float32)
        self.gate_h_in         = np.zeros((buffer_size, hidden_size), dtype=np.float32)
        self.gate_c_in         = np.zeros((buffer_size, hidden_size), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: float,
        h_in: np.ndarray,
        c_in: np.ndarray,
    ) -> None:
        self.observations[self.pos]      = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos]   = action
        self.rewards[self.pos]   = reward
        self.dones[self.pos]     = done
        self.gate_h_in[self.pos] = h_in
        self.gate_c_in[self.pos] = c_in
        self.pos  = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> _LSTMSamples:
        idx = np.random.randint(0, len(self), size=batch_size)
        return _LSTMSamples(
            observations      = torch.tensor(self.observations[idx],      device=self.device),
            actions           = torch.tensor(self.actions[idx],           device=self.device).unsqueeze(1),
            next_observations = torch.tensor(self.next_observations[idx], device=self.device),
            dones             = torch.tensor(self.dones[idx],             device=self.device),
            rewards           = torch.tensor(self.rewards[idx],           device=self.device),
            gate_h_in         = torch.tensor(self.gate_h_in[idx],        device=self.device),
            gate_c_in         = torch.tensor(self.gate_c_in[idx],        device=self.device),
        )


# ---------------------------------------------------------------------------
# Sequence replay buffer for LSTM burn-in (modality 7)
# ---------------------------------------------------------------------------
_BurnInSamples = namedtuple(
    "_BurnInSamples",
    ["obs_seq", "next_obs_seq", "actions", "rewards", "dones", "h0", "c0"],
)


class LSTMBurnInBuffer:
    """
    Replay buffer that stores fixed-length sequences of transitions together
    with the LSTM state (h0, c0) at the start of each sequence.

    This enables proper burn-in training (R2D2-style):
      • The first LSTM_BURN_IN steps of each sequence are run under torch.no_grad()
        to re-warm the hidden state using the current network weights.
      • Only the remaining LSTM_LEARN_LEN steps contribute to the TD loss,
        and BPTT flows through all of them.

    Sequence storage layout (per entry):
      obs_seq      : (SEQ_LEN, obs_dim)  — observations for each step
      next_obs_seq : (SEQ_LEN, obs_dim)  — corresponding next observations
      actions      : (SEQ_LEN,)
      rewards      : (SEQ_LEN,)
      dones        : (SEQ_LEN,)          — 1.0 where an episode ended
      h0           : (RNN_HIDDEN,)       — hidden state at sequence start
      c0           : (RNN_HIDDEN,)       — cell state at sequence start

    Sequences may span episode boundaries; the training loop zeroes h/c
    wherever dones[t] == 1 before processing the next step.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        hidden_size: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.seq_len     = seq_len
        self.device      = device
        self.pos         = 0
        self.full        = False

        self.obs_seq      = np.zeros((buffer_size, seq_len, *obs_shape), dtype=np.float32)
        self.next_obs_seq = np.zeros((buffer_size, seq_len, *obs_shape), dtype=np.float32)
        self.actions      = np.zeros((buffer_size, seq_len),             dtype=np.int64)
        self.rewards      = np.zeros((buffer_size, seq_len),             dtype=np.float32)
        self.dones        = np.zeros((buffer_size, seq_len),             dtype=np.float32)
        self.h0           = np.zeros((buffer_size, hidden_size),          dtype=np.float32)
        self.c0           = np.zeros((buffer_size, hidden_size),          dtype=np.float32)

    def add(
        self,
        obs_seq:      np.ndarray,   # (seq_len, obs_dim)
        next_obs_seq: np.ndarray,   # (seq_len, obs_dim)
        actions:      np.ndarray,   # (seq_len,) int64
        rewards:      np.ndarray,   # (seq_len,)
        dones:        np.ndarray,   # (seq_len,)
        h0:           np.ndarray,   # (hidden_size,)
        c0:           np.ndarray,   # (hidden_size,)
    ) -> None:
        self.obs_seq[self.pos]      = obs_seq
        self.next_obs_seq[self.pos] = next_obs_seq
        self.actions[self.pos]      = actions
        self.rewards[self.pos]      = rewards
        self.dones[self.pos]        = dones
        self.h0[self.pos]           = h0
        self.c0[self.pos]           = c0
        self.pos  = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> _BurnInSamples:
        idx = np.random.randint(0, len(self), size=batch_size)
        return _BurnInSamples(
            obs_seq      = torch.tensor(self.obs_seq[idx],      device=self.device),
            next_obs_seq = torch.tensor(self.next_obs_seq[idx], device=self.device),
            actions      = torch.tensor(self.actions[idx],      device=self.device),
            rewards      = torch.tensor(self.rewards[idx],      device=self.device),
            dones        = torch.tensor(self.dones[idx],        device=self.device),
            h0           = torch.tensor(self.h0[idx],           device=self.device),
            c0           = torch.tensor(self.c0[idx],           device=self.device),
        )


# ---------------------------------------------------------------------------
# Sequence replay buffer for LSTM burn-in (modality 7)
# ---------------------------------------------------------------------------
_BurnInSamples = namedtuple(
    "_BurnInSamples",
    ["obs_seq", "next_obs_seq", "actions", "rewards", "dones", "h0", "c0"],
)


class LSTMBurnInBuffer:
    """
    Stores fixed-length sequences of transitions + the LSTM state (h0, c0)
    at the start of each sequence.

    This enables R2D2-style burn-in training:
      - Steps 0..BURN_IN-1 are run with torch.no_grad() to re-warm (h, c).
      - Steps BURN_IN..SEQ_LEN-1 accumulate TD loss; BPTT flows through them.
      - Episode boundaries within a sequence are handled by zeroing (h, c)
        wherever dones[t] == 1 before processing step t+1.

    Buffer entry shape:
      obs_seq / next_obs_seq : (SEQ_LEN, obs_dim)
      actions / rewards / dones : (SEQ_LEN,)
      h0 / c0                : (RNN_HIDDEN,)
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape:   Tuple[int, ...],
        hidden_size: int,
        seq_len:     int,
        device:      torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.seq_len     = seq_len
        self.device      = device
        self.pos  = 0
        self.full = False

        self.obs_seq      = np.zeros((buffer_size, seq_len, *obs_shape), dtype=np.float32)
        self.next_obs_seq = np.zeros((buffer_size, seq_len, *obs_shape), dtype=np.float32)
        self.actions      = np.zeros((buffer_size, seq_len),             dtype=np.int64)
        self.rewards      = np.zeros((buffer_size, seq_len),             dtype=np.float32)
        self.dones        = np.zeros((buffer_size, seq_len),             dtype=np.float32)
        self.h0           = np.zeros((buffer_size, hidden_size),          dtype=np.float32)
        self.c0           = np.zeros((buffer_size, hidden_size),          dtype=np.float32)

    def add(
        self,
        obs_seq:      np.ndarray,
        next_obs_seq: np.ndarray,
        actions:      np.ndarray,
        rewards:      np.ndarray,
        dones:        np.ndarray,
        h0:           np.ndarray,
        c0:           np.ndarray,
    ) -> None:
        self.obs_seq[self.pos]      = obs_seq
        self.next_obs_seq[self.pos] = next_obs_seq
        self.actions[self.pos]      = actions
        self.rewards[self.pos]      = rewards
        self.dones[self.pos]        = dones
        self.h0[self.pos]           = h0
        self.c0[self.pos]           = c0
        self.pos  = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> _BurnInSamples:
        idx = np.random.randint(0, len(self), size=batch_size)
        return _BurnInSamples(
            obs_seq      = torch.tensor(self.obs_seq[idx],      device=self.device),
            next_obs_seq = torch.tensor(self.next_obs_seq[idx], device=self.device),
            actions      = torch.tensor(self.actions[idx],      device=self.device),
            rewards      = torch.tensor(self.rewards[idx],      device=self.device),
            dones        = torch.tensor(self.dones[idx],        device=self.device),
            h0           = torch.tensor(self.h0[idx],           device=self.device),
            c0           = torch.tensor(self.c0[idx],           device=self.device),
        )


# ---------------------------------------------------------------------------
# Sequence replay buffer for GRU burn-in
# ---------------------------------------------------------------------------
_GRUBurnInSamples = namedtuple(
    "_GRUBurnInSamples",
    ["obs_seq", "next_obs_seq", "actions", "rewards", "dones", "h0"],
)


class GRUBurnInBuffer:
    """
    Replay buffer that stores fixed-length sequences of transitions together
    with the GRU hidden state h0 at the start of each sequence.

    Identical in spirit to LSTMBurnInBuffer but stores only h0 (no cell state)
    since the GRU uses a single recurrent vector.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape:   Tuple[int, ...],
        hidden_size: int,
        seq_len:     int,
        device:      torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.seq_len     = seq_len
        self.device      = device
        self.pos  = 0
        self.full = False

        self.obs_seq      = np.zeros((buffer_size, seq_len, *obs_shape), dtype=np.float32)
        self.next_obs_seq = np.zeros((buffer_size, seq_len, *obs_shape), dtype=np.float32)
        self.actions      = np.zeros((buffer_size, seq_len),             dtype=np.int64)
        self.rewards      = np.zeros((buffer_size, seq_len),             dtype=np.float32)
        self.dones        = np.zeros((buffer_size, seq_len),             dtype=np.float32)
        self.h0           = np.zeros((buffer_size, hidden_size),          dtype=np.float32)

    def add(
        self,
        obs_seq:      np.ndarray,
        next_obs_seq: np.ndarray,
        actions:      np.ndarray,
        rewards:      np.ndarray,
        dones:        np.ndarray,
        h0:           np.ndarray,
    ) -> None:
        self.obs_seq[self.pos]      = obs_seq
        self.next_obs_seq[self.pos] = next_obs_seq
        self.actions[self.pos]      = actions
        self.rewards[self.pos]      = rewards
        self.dones[self.pos]        = dones
        self.h0[self.pos]           = h0
        self.pos  = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> _GRUBurnInSamples:
        idx = np.random.randint(0, len(self), size=batch_size)
        return _GRUBurnInSamples(
            obs_seq      = torch.tensor(self.obs_seq[idx],      device=self.device),
            next_obs_seq = torch.tensor(self.next_obs_seq[idx], device=self.device),
            actions      = torch.tensor(self.actions[idx],      device=self.device),
            rewards      = torch.tensor(self.rewards[idx],      device=self.device),
            dones        = torch.tensor(self.dones[idx],        device=self.device),
            h0           = torch.tensor(self.h0[idx],           device=self.device),
        )


# ---------------------------------------------------------------------------
# Training loop — modality 4 (RNN controller)
# ---------------------------------------------------------------------------
def run_one_rnn(
    seed: int,
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    target_freq: int = TARGET_NETWORK_FREQ,
    main_grad_clip: float = 0.0,
    learning_starts: int = LEARNING_STARTS,
    train_frequency: int = TRAIN_FREQUENCY,
) -> Dict:
    """Single-seed training for modality 4 (RNN meta-controller).

    The GRU hidden state is maintained episodically during rollout and reset
    on episode boundaries.  Each transition stores the hidden state h_in that
    was active before that observation was processed; batch training reuses
    the stored h_in so the GRU context is approximately preserved.
    """
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    label     = f"seed={seed}"

    q_net = BFeedbackRNNControllerNetwork(obs_dim, n_actions).to(device)
    t_net = BFeedbackRNNControllerNetwork(obs_dim, n_actions).to(device)
    t_net.load_state_dict(q_net.state_dict())

    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters())
        + list(q_net.rnn_controller.parameters()),
        lr=LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )

    rb = RNNReplayBuffer(BUFFER_SIZE, env.observation_space.shape, RNN_HIDDEN, device)

    episode_returns: List[float] = []
    rnn_history: List[Dict]      = []
    loss_list: List[float]       = []

    _eps_end   = END_E
    _eps_decay = float(epsilon_decay_episodes)

    # Episodic hidden state — shape (1, RNN_HIDDEN) for single-env rollout
    h = q_net.zero_hidden(device)

    obs, _ = env.reset(seed=seed)
    t = 0

    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        # Save h_in BEFORE this step's processing (what was active when choosing action)
        h_in_np = h.squeeze(0).cpu().numpy()   # (RNN_HIDDEN,)

        # Always run forward to keep hidden state updated (even for random actions)
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Q_act, h_next = q_net.forward_q_only(obs_t, h)
        h = h_next.detach()

        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(Q_act.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done     = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]

        rb.add(obs, real_nxt, action, float(reward), float(done), h_in_np)

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()
            h = q_net.zero_hidden(device)   # reset hidden state at episode boundary

        # --- Training step ---
        if t > learning_starts and t % train_frequency == 0 and len(rb) >= BATCH_SIZE:
            data = rb.sample(BATCH_SIZE)

            with torch.no_grad():
                B       = data.observations.shape[0]
                h_zeros = torch.zeros(B, RNN_HIDDEN, device=device)
                tQ_full, tQ_cart, tQ_pole, _, _, _ = t_net.forward(
                    data.next_observations, h_zeros
                )
                r      = data.rewards.flatten()
                d      = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            # Forward with stored h_in — k_cart / k_pole NOT detached in _gated_z
            # so gradients flow: loss_full → Q_full_gated → k → GRU weights
            Q_full, Q_cart, Q_pole, _z, _h, rnn_out = q_net.forward(
                data.observations, data.gate_h_in
            )

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            # Modulate optimizer LR and ε floor from batch-averaged RNN outputs
            lr_mult_val   = float(rnn_out["lr_mult"].mean().item())
            eps_floor_val = float(rnn_out["eps_floor"].mean().item())
            opt_main.param_groups[0]["lr"] = LEARNING_RATE * lr_mult_val
            _eps_end = eps_floor_val

            opt_main.zero_grad()
            loss_full.backward()   # no retain_graph needed — k inputs were detached
            if main_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    list(q_net.linear_feature.parameters())
                    + list(q_net.trunk.parameters())
                    + list(q_net.head_full.parameters())
                    + list(q_net.rnn_controller.parameters()),
                    max_norm=main_grad_clip,
                )
            opt_main.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            loss_list.append(loss_full.item())

            if t % LOG_EVERY == 0:
                rnn_history.append({
                    "step":           t,
                    "episode":        len(episode_returns),
                    "lr_mult_mean":   lr_mult_val,
                    "eps_floor_mean": eps_floor_val,
                    "k_cart_mean":    float(rnn_out["k_cart"].mean().item()),
                    "k_pole_mean":    float(rnn_out["k_pole"].mean().item()),
                })

        if t % target_freq == 0:
            t_net.load_state_dict(q_net.state_dict())

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [rnn_ctrl] {label} step={t+1}/{total_timesteps}"
                f"  ep={len(episode_returns)}"
                f"  eps={_linear_schedule(START_E, _eps_end, int(_eps_decay), len(episode_returns)):.3f}",
                flush=True,
            )

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  [rnn_ctrl] WARNING {label}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns)) if episode_returns else 0.0
    )
    print(
        f"  Done [rnn_ctrl] {label}: ep={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"b_feedb_cg_rnn_ctrl_seed{seed}.pt"
        torch.save(
            {
                "algo":            "b_feedb_cg_rnn_ctrl",
                "seed":            seed,
                "episode_returns": episode_returns,
                "rnn_history":     rnn_history,
                "q_network":       q_net.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    return {
        "seed":            seed,
        "episode_returns": episode_returns,
        "rnn_history":     rnn_history,
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Training loop — RNN controller with REINFORCE on loss improvement
# ---------------------------------------------------------------------------
def run_one_rnn_reinforce(
    seed: int,
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    total_episodes: Optional[int] = None,
    epsilon_decay_episodes: int = 800,
    target_freq: int = 1,
    main_grad_clip: float = 10.0,
    ctrl_grad_clip: float = 10.0,
    learning_starts: int = 1_000,
    train_frequency: int = 4,
) -> Dict:
    """Single-seed training for the REINFORCE RNN controller.

    Two separate optimizers:
      opt_main  — Q-network weights only (linear_feature, trunk, head_full)
      opt_ctrl  — RNN controller only (gru_cell, output_head)

    Per training step:
      1.  Forward q_net → Q_full, Q_cart, Q_pole, rnn_out (lr_mult, eps_floor), rnn_input
      2.  Apply lr_mult externally to opt_main LR; apply eps_floor to ε schedule.
      3.  Record loss_before, run opt_main.step().
      4.  Measure loss_after with a no-grad forward on the updated Q-network.
      5.  ctrl_reward = loss_before - loss_after  (positive → improvement).
      6.  Second controller forward (in-graph, reuses rnn_input) → REINFORCE loss.
      7.  opt_ctrl.step().

    The controller cannot minimise the Q-learning loss by collapsing lr_mult
    to zero, because (a) lr_mult is applied via .item() and (b) the controller's
    gradient comes from ctrl_reward, which is zero when nothing improves.
    """
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    label     = f"seed={seed}"

    q_net = BFeedbackRNNLossImprovNetwork(obs_dim, n_actions).to(device)
    t_net = BFeedbackRNNLossImprovNetwork(obs_dim, n_actions).to(device)
    t_net.load_state_dict(q_net.state_dict())

    # Q-network optimizer — controller NOT included
    q_params = (
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters())
    )
    opt_main = optim.Adam(q_params, lr=LEARNING_RATE)
    opt_aux  = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )
    # Separate controller optimizer
    opt_ctrl = optim.Adam(q_net.rnn_controller.parameters(), lr=LEARNING_RATE)

    rb = RNNReplayBuffer(BUFFER_SIZE, env.observation_space.shape, RNN_HIDDEN, device)

    episode_returns: List[float] = []
    ctrl_history: List[Dict]     = []
    loss_list: List[float]       = []

    _eps_end   = END_E
    _eps_decay = float(epsilon_decay_episodes)

    h = q_net.zero_hidden(device)   # episodic hidden state (1, RNN_HIDDEN)

    obs, _ = env.reset(seed=seed)
    t = 0

    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        # Save hidden state BEFORE this step's processing
        h_in_np = h.squeeze(0).cpu().numpy()   # (RNN_HIDDEN,)

        # Run forward to maintain consistent hidden state (even for random actions)
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Q_act, h_next = q_net.forward_q_only(obs_t, h)
        h = h_next.detach()

        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done)
        action = (
            env.action_space.sample()
            if random.random() < eps
            else int(Q_act.argmax(dim=1).item())
        )

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done     = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]

        rb.add(obs, real_nxt, action, float(reward), float(done), h_in_np)

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()
            h = q_net.zero_hidden(device)

        # --- Training step ---
        if t > learning_starts and t % train_frequency == 0 and len(rb) >= BATCH_SIZE:
            data = rb.sample(BATCH_SIZE)

            # Compute targets via frozen target network
            with torch.no_grad():
                B       = data.observations.shape[0]
                h_zeros = torch.zeros(B, RNN_HIDDEN, device=device)
                tQ_full, tQ_cart, tQ_pole, _, _, _ = t_net.forward(
                    data.next_observations, h_zeros
                )
                r      = data.rewards.flatten()
                d      = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            # ── Step 1: Q-network forward ──────────────────────────────────
            Q_full, Q_cart, Q_pole, _h, rnn_out, rnn_input = q_net.forward(
                data.observations, data.gate_h_in
            )
            # rnn_input is built from .detach() tensors — no grad_fn

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            # ── Step 2: apply controller outputs externally (detached) ─────
            lr_mult_val   = float(rnn_out["lr_mult"].mean().item())
            eps_floor_val = float(rnn_out["eps_floor"].mean().item())
            opt_main.param_groups[0]["lr"] = LEARNING_RATE * lr_mult_val
            _eps_end = eps_floor_val

            # ── Step 3: Q-network update ───────────────────────────────────
            loss_before = loss_full.item()

            opt_main.zero_grad()
            loss_full.backward()
            if main_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(q_params, max_norm=main_grad_clip)
            opt_main.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            loss_list.append(loss_before)

            # ── Step 4: measure loss improvement ──────────────────────────
            with torch.no_grad():
                z_fresh  = q_net.linear_feature(data.observations)
                Q_fresh  = q_net.head_full(q_net.trunk(z_fresh))
                loss_after = F.mse_loss(
                    y_full, Q_fresh.gather(1, data.actions).squeeze()
                ).item()
            ctrl_reward = loss_before - loss_after   # + → improvement, - → degradation

            # ── Step 5: REINFORCE update for controller ────────────────────
            # Second controller forward using the same (detached) rnn_input —
            # produces in-graph lr_mult so grad flows to controller weights.
            rnn_out_ctrl, _ = q_net.rnn_controller(rnn_input, data.gate_h_in)
            ctrl_loss = -ctrl_reward * torch.log(rnn_out_ctrl["lr_mult"].mean() + 1e-6)

            opt_ctrl.zero_grad()
            ctrl_loss.backward()
            if ctrl_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    q_net.rnn_controller.parameters(), max_norm=ctrl_grad_clip
                )
            opt_ctrl.step()

            if t % LOG_EVERY == 0:
                ctrl_history.append({
                    "step":           t,
                    "episode":        len(episode_returns),
                    "lr_mult_mean":   lr_mult_val,
                    "eps_floor_mean": eps_floor_val,
                    "ctrl_reward":    ctrl_reward,
                    "ctrl_loss":      float(ctrl_loss.item()),
                })

        if t % target_freq == 0:
            t_net.load_state_dict(q_net.state_dict())

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [rnn_reinforce] {label} step={t+1}/{total_timesteps}"
                f"  ep={len(episode_returns)}"
                f"  eps={_linear_schedule(START_E, _eps_end, int(_eps_decay), len(episode_returns)):.3f}",
                flush=True,
            )

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  [rnn_reinforce] WARNING {label}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns)) if episode_returns else 0.0
    )
    print(
        f"  Done [rnn_reinforce] {label}: ep={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ep_str    = f"_{total_episodes}ep" if total_episodes is not None else ""
        ckpt_path = checkpoint_dir / f"b_feedb_rnn_reinforce{ep_str}_seed{seed}.pt"
        torch.save(
            {
                "algo":            "b_feedb_rnn_reinforce",
                "seed":            seed,
                "episode_returns": episode_returns,
                "ctrl_history":    ctrl_history,
                "q_network":       q_net.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    return {
        "seed":            seed,
        "episode_returns": episode_returns,
        "ctrl_history":    ctrl_history,
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Training loop — modality 5 (LSTM controller)
# ---------------------------------------------------------------------------
def run_one_lstm(
    seed: int,
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    target_freq: int = TARGET_NETWORK_FREQ,
    main_grad_clip: float = 10.0,
    learning_starts: int = LEARNING_STARTS,
    train_frequency: int = TRAIN_FREQUENCY,
) -> Dict:
    """Single-seed training for the LSTM meta-controller (modality 5).

    Mirrors run_one_rnn exactly, but uses BFeedbackLSTMControllerNetwork and
    LSTMReplayBuffer.  The key difference is that both h and c are stored per
    transition and replayed; the gradient path is identical to the GRU modality
    (k_cart / k_pole are not detached, so loss_full → z_gated → k → LSTM weights).
    """
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    label     = f"seed={seed}"

    q_net = BFeedbackLSTMControllerNetwork(obs_dim, n_actions).to(device)
    t_net = BFeedbackLSTMControllerNetwork(obs_dim, n_actions).to(device)
    t_net.load_state_dict(q_net.state_dict())

    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters())
        + list(q_net.lstm_controller.parameters()),
        lr=LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )

    rb = LSTMReplayBuffer(BUFFER_SIZE, env.observation_space.shape, RNN_HIDDEN, device)

    episode_returns: List[float] = []
    lstm_history:   List[Dict]   = []
    loss_list:      List[float]  = []

    _eps_end   = END_E
    _eps_decay = float(epsilon_decay_episodes)

    h, c = q_net.zero_hidden(device)   # (1, RNN_HIDDEN) each

    obs, _ = env.reset(seed=seed)
    t = 0

    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        h_in_np = h.squeeze(0).cpu().numpy()   # (RNN_HIDDEN,)
        c_in_np = c.squeeze(0).cpu().numpy()

        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Q_act, h_next, c_next = q_net.forward_q_only(obs_t, h, c)
        h = h_next.detach()
        c = c_next.detach()

        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done)
        action = (
            env.action_space.sample()
            if random.random() < eps
            else int(Q_act.argmax(dim=1).item())
        )

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done     = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]

        rb.add(obs, real_nxt, action, float(reward), float(done), h_in_np, c_in_np)

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()
            h, c = q_net.zero_hidden(device)

        if t > learning_starts and t % train_frequency == 0 and len(rb) >= BATCH_SIZE:
            data = rb.sample(BATCH_SIZE)

            with torch.no_grad():
                B        = data.observations.shape[0]
                h_zeros  = torch.zeros(B, RNN_HIDDEN, device=device)
                c_zeros  = torch.zeros(B, RNN_HIDDEN, device=device)
                tQ_full, tQ_cart, tQ_pole, _, _, _, _ = t_net.forward(
                    data.next_observations, h_zeros, c_zeros
                )
                r      = data.rewards.flatten()
                d      = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            Q_full, Q_cart, Q_pole, _z, _h, _c, lstm_out = q_net.forward(
                data.observations, data.gate_h_in, data.gate_c_in
            )

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            lr_mult_val   = float(lstm_out["lr_mult"].mean().item())
            eps_floor_val = float(lstm_out["eps_floor"].mean().item())
            opt_main.param_groups[0]["lr"] = LEARNING_RATE * lr_mult_val
            _eps_end = eps_floor_val

            opt_main.zero_grad()
            loss_full.backward()
            if main_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    list(q_net.linear_feature.parameters())
                    + list(q_net.trunk.parameters())
                    + list(q_net.head_full.parameters())
                    + list(q_net.lstm_controller.parameters()),
                    max_norm=main_grad_clip,
                )
            opt_main.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            loss_list.append(loss_full.item())

            if t % LOG_EVERY == 0:
                lstm_history.append({
                    "step":           t,
                    "episode":        len(episode_returns),
                    "lr_mult_mean":   lr_mult_val,
                    "eps_floor_mean": eps_floor_val,
                    "k_cart_mean":    float(lstm_out["k_cart"].mean().item()),
                    "k_pole_mean":    float(lstm_out["k_pole"].mean().item()),
                })

        if t % target_freq == 0:
            t_net.load_state_dict(q_net.state_dict())

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [lstm_ctrl] {label} step={t+1}/{total_timesteps}"
                f"  ep={len(episode_returns)}"
                f"  eps={_linear_schedule(START_E, _eps_end, int(_eps_decay), len(episode_returns)):.3f}",
                flush=True,
            )

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  [lstm_ctrl] WARNING {label}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns)) if episode_returns else 0.0
    )
    print(
        f"  Done [lstm_ctrl] {label}: ep={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"b_feedb_cg_lstm_ctrl_seed{seed}.pt"
        torch.save(
            {
                "algo":            "b_feedb_cg_lstm_ctrl",
                "seed":            seed,
                "episode_returns": episode_returns,
                "lstm_history":    lstm_history,
                "q_network":       q_net.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    return {
        "seed":            seed,
        "episode_returns": episode_returns,
        "lstm_history":    lstm_history,
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers — modality 4
# ---------------------------------------------------------------------------
def _ckpt_path_rnn(ckpt_dir: Path, seed: int) -> Path:
    return ckpt_dir / f"b_feedb_cg_rnn_ctrl_seed{seed}.pt"


def _load_result_rnn(ckpt_dir: Path, seed: int) -> Dict:
    path = _ckpt_path_rnn(ckpt_dir, seed)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep   = ckpt["episode_returns"]
    last100 = (
        float(np.mean(ep[-100:])) if len(ep) >= 100
        else float(np.mean(ep)) if ep else 0.0
    )
    return {
        "seed":            seed,
        "episode_returns": ep,
        "rnn_history":     ckpt.get("rnn_history", []),
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Training loop — modality 7 (LSTM controller with burn-in)
# ---------------------------------------------------------------------------
def run_one_lstm_burnin(
    seed: int,
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    target_freq: int = TARGET_NETWORK_FREQ,
    main_grad_clip: float = 10.0,
    learning_starts: int = LEARNING_STARTS,
    train_frequency: int = TRAIN_FREQUENCY,
) -> Dict:
    """Single-seed training for the LSTM controller with burn-in (modality 7).

    Key differences from run_one_lstm:
      1. Sequences of LSTM_SEQ_LEN transitions are stored together with the
         initial (h0, c0) via LSTMBurnInBuffer.
      2. At each training step, the first LSTM_BURN_IN steps of the sampled
         sequence are used to re-warm (h, c) under torch.no_grad().
      3. The TD loss is computed over the remaining LSTM_LEARN_LEN steps,
         with BPTT flowing through all of them.
      4. Episode boundaries inside a sequence zero out (h, c) before the
         next step (both in burn-in and learning phases).

    This eliminates the stale-hidden-state problem of single-step replay:
    (h, c) used for the gradient computation are always consistent with the
    current network weights.
    """
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    label     = f"seed={seed}"

    q_net = BFeedbackLSTMControllerNetwork(obs_dim, n_actions).to(device)
    t_net = BFeedbackLSTMControllerNetwork(obs_dim, n_actions).to(device)
    t_net.load_state_dict(q_net.state_dict())

    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters())
        + list(q_net.lstm_controller.parameters()),
        lr=LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )

    rb = LSTMBurnInBuffer(
        LSTM_SEQ_BUFFER, env.observation_space.shape, RNN_HIDDEN, LSTM_SEQ_LEN, device
    )

    episode_returns: List[float] = []
    lstm_history:   List[Dict]   = []

    _eps_end   = END_E
    _eps_decay = float(epsilon_decay_episodes)

    h, c = q_net.zero_hidden(device)

    # Sequence collector state
    pend_obs:  List[np.ndarray] = []
    pend_nxt:  List[np.ndarray] = []
    pend_act:  List[int]        = []
    pend_rew:  List[float]      = []
    pend_done: List[float]      = []
    seq_h0 = h.squeeze(0).cpu().numpy()
    seq_c0 = c.squeeze(0).cpu().numpy()

    def _flush_sequence() -> None:
        """Store the pending sequence if it meets the minimum length."""
        n = len(pend_obs)
        if n < 2:          # too short to be useful
            return
        # Pad to SEQ_LEN with repeats of the last step (marked done)
        obs_arr  = list(pend_obs)
        nxt_arr  = list(pend_nxt)
        act_arr  = list(pend_act)
        rew_arr  = list(pend_rew)
        done_arr = list(pend_done)
        while len(obs_arr) < LSTM_SEQ_LEN:
            obs_arr.append(obs_arr[-1])
            nxt_arr.append(nxt_arr[-1])
            act_arr.append(0)
            rew_arr.append(0.0)
            done_arr.append(1.0)   # padding is treated as done
        rb.add(
            np.array(obs_arr[:LSTM_SEQ_LEN],  dtype=np.float32),
            np.array(nxt_arr[:LSTM_SEQ_LEN],  dtype=np.float32),
            np.array(act_arr[:LSTM_SEQ_LEN],  dtype=np.int64),
            np.array(rew_arr[:LSTM_SEQ_LEN],  dtype=np.float32),
            np.array(done_arr[:LSTM_SEQ_LEN], dtype=np.float32),
            seq_h0, seq_c0,
        )

    obs, _ = env.reset(seed=seed)
    t = 0

    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        # --- Rollout step ---
        h_in_np = h.squeeze(0).cpu().numpy()
        c_in_np = c.squeeze(0).cpu().numpy()

        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Q_act, h_next, c_next = q_net.forward_q_only(obs_t, h, c)
        h = h_next.detach()
        c = c_next.detach()

        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done)
        action = (
            env.action_space.sample()
            if random.random() < eps
            else int(Q_act.argmax(dim=1).item())
        )

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done     = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]

        pend_obs.append(obs.copy())
        pend_nxt.append(real_nxt)
        pend_act.append(action)
        pend_rew.append(float(reward))
        pend_done.append(float(done))

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        # Flush when sequence is full or episode ends
        if len(pend_obs) == LSTM_SEQ_LEN or done:
            _flush_sequence()
            pend_obs.clear(); pend_nxt.clear()
            pend_act.clear(); pend_rew.clear(); pend_done.clear()
            if done:
                h, c   = q_net.zero_hidden(device)
                seq_h0 = np.zeros(RNN_HIDDEN, dtype=np.float32)
                seq_c0 = np.zeros(RNN_HIDDEN, dtype=np.float32)
            else:
                seq_h0 = h.squeeze(0).cpu().numpy()
                seq_c0 = c.squeeze(0).cpu().numpy()
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # --- Training step ---
        if (t > learning_starts
                and t % train_frequency == 0
                and len(rb) >= LSTM_SEQ_BATCH):
            data = rb.sample(LSTM_SEQ_BATCH)
            # data.obs_seq:  (B, SEQ_LEN, obs_dim)
            B = data.obs_seq.shape[0]

            h_b = data.h0.clone()   # (B, RNN_HIDDEN)
            c_b = data.c0.clone()

            # ── Burn-in: re-warm (h, c) with current weights (no gradient) ──
            with torch.no_grad():
                for bt in range(LSTM_BURN_IN):
                    obs_bt = data.obs_seq[:, bt]   # (B, obs_dim)
                    _, _, _, _, h_b, c_b, _ = q_net.forward(obs_bt, h_b, c_b)
                    # Zero out where step bt ended an episode
                    if bt < LSTM_BURN_IN - 1:
                        mask  = data.dones[:, bt].unsqueeze(1)   # (B, 1)
                        h_b = h_b * (1.0 - mask)
                        c_b = c_b * (1.0 - mask)

            h_b = h_b.detach()
            c_b = c_b.detach()

            # ── Learning: compute TD loss over LEARN steps (with BPTT) ──────
            loss_full_sum = torch.zeros(1, device=device)
            loss_cart_sum = torch.zeros(1, device=device)
            loss_pole_sum = torch.zeros(1, device=device)
            lstm_out_last: Dict[str, torch.Tensor] = {}

            for lt in range(LSTM_LEARN_LEN):
                st = LSTM_BURN_IN + lt          # index in full sequence

                # Reset where previous step ended an episode
                if lt > 0:
                    mask  = data.dones[:, st - 1].unsqueeze(1)
                    h_b = h_b * (1.0 - mask)
                    c_b = c_b * (1.0 - mask)

                obs_st      = data.obs_seq[:, st]          # (B, obs_dim)
                next_obs_st = data.next_obs_seq[:, st]
                acts_st     = data.actions[:, st].unsqueeze(1)   # (B, 1)
                rews_st     = data.rewards[:, st]
                done_st     = data.dones[:, st]

                # Frozen target (use zeroed h for simplicity, like other modalities)
                with torch.no_grad():
                    h_tgt = torch.zeros_like(h_b)
                    c_tgt = torch.zeros_like(c_b)
                    tQ_full, tQ_cart, tQ_pole, _, _, _, _ = t_net.forward(
                        next_obs_st, h_tgt, c_tgt
                    )
                    y_full = rews_st + GAMMA * (1.0 - done_st) * tQ_full.max(dim=1).values
                    y_cart = rews_st + GAMMA * (1.0 - done_st) * tQ_cart.max(dim=1).values
                    y_pole = rews_st + GAMMA * (1.0 - done_st) * tQ_pole.max(dim=1).values

                Q_full, Q_cart, Q_pole, _, h_b, c_b, lstm_out_last = q_net.forward(
                    obs_st, h_b, c_b
                )

                qf_sa = Q_full.gather(1, acts_st).squeeze()
                qc_sa = Q_cart.gather(1, acts_st).squeeze()
                qp_sa = Q_pole.gather(1, acts_st).squeeze()

                loss_full_sum = loss_full_sum + F.mse_loss(y_full, qf_sa)
                loss_cart_sum = loss_cart_sum + F.mse_loss(y_cart, qc_sa)
                loss_pole_sum = loss_pole_sum + F.mse_loss(y_pole, qp_sa)

            loss_full = loss_full_sum / LSTM_LEARN_LEN
            loss_cart = loss_cart_sum / LSTM_LEARN_LEN
            loss_pole = loss_pole_sum / LSTM_LEARN_LEN

            lr_mult_val   = float(lstm_out_last["lr_mult"].mean().item())   if lstm_out_last else 1.0
            eps_floor_val = float(lstm_out_last["eps_floor"].mean().item()) if lstm_out_last else END_E
            opt_main.param_groups[0]["lr"] = LEARNING_RATE * lr_mult_val
            _eps_end = eps_floor_val

            opt_main.zero_grad()
            loss_full.backward()
            if main_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    list(q_net.linear_feature.parameters())
                    + list(q_net.trunk.parameters())
                    + list(q_net.head_full.parameters())
                    + list(q_net.lstm_controller.parameters()),
                    max_norm=main_grad_clip,
                )
            opt_main.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            if t % LOG_EVERY == 0 and lstm_out_last:
                lstm_history.append({
                    "step":           t,
                    "episode":        len(episode_returns),
                    "lr_mult_mean":   lr_mult_val,
                    "eps_floor_mean": eps_floor_val,
                    "k_cart_mean":    float(lstm_out_last["k_cart"].mean().item()),
                    "k_pole_mean":    float(lstm_out_last["k_pole"].mean().item()),
                })

        if t % target_freq == 0:
            t_net.load_state_dict(q_net.state_dict())

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [lstm_bi] {label} step={t+1}/{total_timesteps}"
                f"  ep={len(episode_returns)}"
                f"  seqs={len(rb)}"
                f"  eps={_linear_schedule(START_E, _eps_end, int(_eps_decay), len(episode_returns)):.3f}",
                flush=True,
            )

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  [lstm_bi] WARNING {label}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns)) if episode_returns else 0.0
    )
    print(
        f"  Done [lstm_bi] {label}: ep={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"b_feedb_cg_lstm_bi_seed{seed}.pt"
        torch.save(
            {
                "algo":            "b_feedb_cg_lstm_bi",
                "seed":            seed,
                "episode_returns": episode_returns,
                "lstm_history":    lstm_history,
                "q_network":       q_net.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    return {
        "seed":            seed,
        "episode_returns": episode_returns,
        "lstm_history":    lstm_history,
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Training loop — GRU controller with burn-in
# ---------------------------------------------------------------------------
def run_one_gru_burnin(
    seed: int,
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    target_freq: int = TARGET_NETWORK_FREQ,
    main_grad_clip: float = 10.0,
    learning_starts: int = LEARNING_STARTS,
    train_frequency: int = TRAIN_FREQUENCY,
) -> Dict:
    """Single-seed training for the GRU controller with burn-in.

    Uses the same BFeedbackRNNControllerNetwork (GRUCell backbone) as
    run_one_rnn, but replaces single-transition replay with sequence replay:

      • GRUBurnInBuffer stores sequences of LSTM_SEQ_LEN (20) consecutive
        transitions together with the GRU hidden state h0 at the start of
        each sequence.
      • At each training step, the first LSTM_BURN_IN (10) steps re-warm h
        under torch.no_grad() using the current weights (fixing the
        stale-hidden-state problem of the standard GRU modality).
      • The remaining LSTM_LEARN_LEN (10) steps accumulate the TD loss;
        BPTT flows through all of them into the GRU controller.
      • Episode boundaries inside a sequence zero h before the next step.
    """
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    label     = f"seed={seed}"

    q_net = BFeedbackRNNControllerNetwork(obs_dim, n_actions).to(device)
    t_net = BFeedbackRNNControllerNetwork(obs_dim, n_actions).to(device)
    t_net.load_state_dict(q_net.state_dict())

    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters())
        + list(q_net.rnn_controller.parameters()),
        lr=LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )

    rb = GRUBurnInBuffer(
        LSTM_SEQ_BUFFER, env.observation_space.shape, RNN_HIDDEN, LSTM_SEQ_LEN, device
    )

    episode_returns: List[float] = []
    rnn_history:    List[Dict]   = []

    _eps_end   = END_E
    _eps_decay = float(epsilon_decay_episodes)

    h = q_net.zero_hidden(device)   # (1, RNN_HIDDEN)

    # Sequence collector
    pend_obs:  List[np.ndarray] = []
    pend_nxt:  List[np.ndarray] = []
    pend_act:  List[int]        = []
    pend_rew:  List[float]      = []
    pend_done: List[float]      = []
    seq_h0 = h.squeeze(0).cpu().numpy()

    def _flush_sequence() -> None:
        n = len(pend_obs)
        if n < 2:
            return
        obs_arr  = list(pend_obs);  nxt_arr  = list(pend_nxt)
        act_arr  = list(pend_act);  rew_arr  = list(pend_rew)
        done_arr = list(pend_done)
        while len(obs_arr) < LSTM_SEQ_LEN:
            obs_arr.append(obs_arr[-1]);  nxt_arr.append(nxt_arr[-1])
            act_arr.append(0);            rew_arr.append(0.0)
            done_arr.append(1.0)
        rb.add(
            np.array(obs_arr[:LSTM_SEQ_LEN],  dtype=np.float32),
            np.array(nxt_arr[:LSTM_SEQ_LEN],  dtype=np.float32),
            np.array(act_arr[:LSTM_SEQ_LEN],  dtype=np.int64),
            np.array(rew_arr[:LSTM_SEQ_LEN],  dtype=np.float32),
            np.array(done_arr[:LSTM_SEQ_LEN], dtype=np.float32),
            seq_h0,
        )

    obs, _ = env.reset(seed=seed)
    t = 0

    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        h_in_np = h.squeeze(0).cpu().numpy()

        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Q_act, h_next = q_net.forward_q_only(obs_t, h)
        h = h_next.detach()

        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done)
        action = (
            env.action_space.sample()
            if random.random() < eps
            else int(Q_act.argmax(dim=1).item())
        )

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done     = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]

        pend_obs.append(obs.copy());   pend_nxt.append(real_nxt)
        pend_act.append(action);       pend_rew.append(float(reward))
        pend_done.append(float(done))

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        if len(pend_obs) == LSTM_SEQ_LEN or done:
            _flush_sequence()
            pend_obs.clear(); pend_nxt.clear()
            pend_act.clear(); pend_rew.clear(); pend_done.clear()
            if done:
                h      = q_net.zero_hidden(device)
                seq_h0 = np.zeros(RNN_HIDDEN, dtype=np.float32)
            else:
                seq_h0 = h.squeeze(0).cpu().numpy()
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # --- Training step ---
        if (t > learning_starts
                and t % train_frequency == 0
                and len(rb) >= LSTM_SEQ_BATCH):
            data = rb.sample(LSTM_SEQ_BATCH)
            B = data.obs_seq.shape[0]

            h_b = data.h0.clone()   # (B, RNN_HIDDEN)

            # ── Burn-in: re-warm h with current weights (no gradient) ──────
            with torch.no_grad():
                for bt in range(LSTM_BURN_IN):
                    obs_bt = data.obs_seq[:, bt]
                    _, _, _, _, h_b, _ = q_net.forward(obs_bt, h_b, detach_k=True)
                    if bt < LSTM_BURN_IN - 1:
                        mask = data.dones[:, bt].unsqueeze(1)
                        h_b  = h_b * (1.0 - mask)

            h_b = h_b.detach()

            # ── Learning: TD loss over LEARN steps, BPTT through GRU ────────
            # detach_k=True: k_cart/k_pole are detached so the multi-step
            # BPTT gradient does NOT flow through the gate path (prevents
            # gradient explosion from the 10-step BPTT chain).
            loss_full_sum = torch.zeros(1, device=device)
            loss_cart_sum = torch.zeros(1, device=device)
            loss_pole_sum = torch.zeros(1, device=device)
            rnn_out_last: Dict[str, torch.Tensor] = {}

            for lt in range(LSTM_LEARN_LEN):
                st = LSTM_BURN_IN + lt
                if lt > 0:
                    mask = data.dones[:, st - 1].unsqueeze(1)
                    h_b  = h_b * (1.0 - mask)

                obs_st      = data.obs_seq[:, st]
                next_obs_st = data.next_obs_seq[:, st]
                acts_st     = data.actions[:, st].unsqueeze(1)
                rews_st     = data.rewards[:, st]
                done_st     = data.dones[:, st]

                with torch.no_grad():
                    h_tgt = torch.zeros_like(h_b)
                    tQ_full, tQ_cart, tQ_pole, _, _, _ = t_net.forward(
                        next_obs_st, h_tgt, detach_k=True
                    )
                    y_full = rews_st + GAMMA * (1.0 - done_st) * tQ_full.max(dim=1).values
                    y_cart = rews_st + GAMMA * (1.0 - done_st) * tQ_cart.max(dim=1).values
                    y_pole = rews_st + GAMMA * (1.0 - done_st) * tQ_pole.max(dim=1).values

                Q_full, Q_cart, Q_pole, _, h_b, rnn_out_last = q_net.forward(
                    obs_st, h_b, detach_k=True
                )

                qf_sa = Q_full.gather(1, acts_st).squeeze()
                qc_sa = Q_cart.gather(1, acts_st).squeeze()
                qp_sa = Q_pole.gather(1, acts_st).squeeze()

                loss_full_sum = loss_full_sum + F.mse_loss(y_full, qf_sa)
                loss_cart_sum = loss_cart_sum + F.mse_loss(y_cart, qc_sa)
                loss_pole_sum = loss_pole_sum + F.mse_loss(y_pole, qp_sa)

            loss_full = loss_full_sum / LSTM_LEARN_LEN
            loss_cart = loss_cart_sum / LSTM_LEARN_LEN
            loss_pole = loss_pole_sum / LSTM_LEARN_LEN

            lr_mult_val   = float(rnn_out_last["lr_mult"].mean().item())   if rnn_out_last else 1.0
            eps_floor_val = float(rnn_out_last["eps_floor"].mean().item()) if rnn_out_last else END_E
            opt_main.param_groups[0]["lr"] = LEARNING_RATE * lr_mult_val
            _eps_end = eps_floor_val

            opt_main.zero_grad()
            loss_full.backward()
            if main_grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    list(q_net.linear_feature.parameters())
                    + list(q_net.trunk.parameters())
                    + list(q_net.head_full.parameters())
                    + list(q_net.rnn_controller.parameters()),
                    max_norm=main_grad_clip,
                )
            opt_main.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            if t % LOG_EVERY == 0 and rnn_out_last:
                rnn_history.append({
                    "step":           t,
                    "episode":        len(episode_returns),
                    "lr_mult_mean":   lr_mult_val,
                    "eps_floor_mean": eps_floor_val,
                    "k_cart_mean":    float(rnn_out_last["k_cart"].mean().item()),
                    "k_pole_mean":    float(rnn_out_last["k_pole"].mean().item()),
                })

        if t % target_freq == 0:
            t_net.load_state_dict(q_net.state_dict())

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [gru_bi] {label} step={t+1}/{total_timesteps}"
                f"  ep={len(episode_returns)}"
                f"  seqs={len(rb)}"
                f"  eps={_linear_schedule(START_E, _eps_end, int(_eps_decay), len(episode_returns)):.3f}",
                flush=True,
            )

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  [gru_bi] WARNING {label}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns)) if episode_returns else 0.0
    )
    print(
        f"  Done [gru_bi] {label}: ep={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"b_feedb_cg_gru_bi_seed{seed}.pt"
        torch.save(
            {
                "algo":            "b_feedb_cg_gru_bi",
                "seed":            seed,
                "episode_returns": episode_returns,
                "rnn_history":     rnn_history,
                "q_network":       q_net.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    return {
        "seed":            seed,
        "episode_returns": episode_returns,
        "rnn_history":     rnn_history,
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers — GRU burn-in
# ---------------------------------------------------------------------------
def _ckpt_path_gru_burnin(ckpt_dir: Path, seed: int) -> Path:
    return ckpt_dir / f"b_feedb_cg_gru_bi_seed{seed}.pt"


def _load_result_gru_burnin(ckpt_dir: Path, seed: int) -> Dict:
    path = _ckpt_path_gru_burnin(ckpt_dir, seed)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep   = ckpt["episode_returns"]
    last100 = (
        float(np.mean(ep[-100:])) if len(ep) >= 100
        else float(np.mean(ep)) if ep else 0.0
    )
    return {
        "seed":            seed,
        "episode_returns": ep,
        "rnn_history":     ckpt.get("rnn_history", []),
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers — modality 5 (LSTM controller)
# ---------------------------------------------------------------------------
def _ckpt_path_lstm(ckpt_dir: Path, seed: int) -> Path:
    return ckpt_dir / f"b_feedb_cg_lstm_ctrl_seed{seed}.pt"


def _load_result_lstm(ckpt_dir: Path, seed: int) -> Dict:
    path = _ckpt_path_lstm(ckpt_dir, seed)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep   = ckpt["episode_returns"]
    last100 = (
        float(np.mean(ep[-100:])) if len(ep) >= 100
        else float(np.mean(ep)) if ep else 0.0
    )
    return {
        "seed":            seed,
        "episode_returns": ep,
        "lstm_history":    ckpt.get("lstm_history", []),
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers — modality 7 (LSTM burn-in)
# ---------------------------------------------------------------------------
def _ckpt_path_lstm_burnin(ckpt_dir: Path, seed: int) -> Path:
    return ckpt_dir / f"b_feedb_cg_lstm_bi_seed{seed}.pt"


def _load_result_lstm_burnin(ckpt_dir: Path, seed: int) -> Dict:
    path = _ckpt_path_lstm_burnin(ckpt_dir, seed)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep   = ckpt["episode_returns"]
    last100 = (
        float(np.mean(ep[-100:])) if len(ep) >= 100
        else float(np.mean(ep)) if ep else 0.0
    )
    return {
        "seed":            seed,
        "episode_returns": ep,
        "lstm_history":    ckpt.get("lstm_history", []),
        "last100_mean":    last100,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers — REINFORCE controller
# ---------------------------------------------------------------------------
def _ckpt_path_rnn_reinforce(ckpt_dir: Path, seed: int, total_episodes: int) -> Path:
    return ckpt_dir / f"b_feedb_rnn_reinforce_{total_episodes}ep_seed{seed}.pt"


def _load_result_rnn_reinforce(ckpt_dir: Path, seed: int, total_episodes: int) -> Dict:
    path = _ckpt_path_rnn_reinforce(ckpt_dir, seed, total_episodes)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep   = ckpt["episode_returns"]
    last100 = (
        float(np.mean(ep[-100:])) if len(ep) >= 100
        else float(np.mean(ep)) if ep else 0.0
    )
    return {
        "seed":            seed,
        "episode_returns": ep,
        "ctrl_history":    ckpt.get("ctrl_history", []),
        "last100_mean":    last100,
    }


def _load_baseline_hparams_result(tag: str, seed: int, total_episodes: int = 750) -> Dict:
    """Load a result from the baseline_hparams_comparison checkpoint directory."""
    ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_baseline_hparams_{total_episodes}ep"
    path = ckpt_dir / f"baseline_{tag}_seed{seed}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep   = ckpt["episode_returns"]
    last100 = (
        float(np.mean(ep[-100:])) if len(ep) >= 100
        else float(np.mean(ep)) if ep else 0.0
    )
    return {
        "seed":         seed,
        "tag":          tag,
        "episode_returns": ep,
        "last100_mean": last100,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_fourway_comparison(
    base_results:  List[Dict],
    gate_results:  List[Dict],
    learn_results: List[Dict],
    rnn_results:   List[Dict],
    out_jpg: Path,
    n_seeds: int,
    decay_tag: str,
    n_ep: Optional[int] = None,
    gru_bi_results:  Optional[List[Dict]] = None,
    lstm_results:    Optional[List[Dict]] = None,
    lstm_bi_results: Optional[List[Dict]] = None,
) -> None:
    """Multi-curve learning figure with ±1 std shading.  Missing modalities are skipped."""

    def _prep(results: List[Dict]):
        if not results:
            return np.array([]), np.array([]), np.array([])
        arrays = [np.asarray(r["episode_returns"]) for r in results]
        n = min(len(a) for a in arrays)
        if n == 0:
            return np.array([]), np.array([]), np.array([])
        M  = np.stack([a[:n] for a in arrays], axis=0)
        mu = M.mean(0)
        sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
        W  = max(1, n // 200)
        ep = np.arange(1, n + 1)
        return ep[W - 1:], _smooth(mu, W), _smooth(sd, W)

    def _l100(results: Optional[List[Dict]]) -> float:
        return float(np.mean([r["last100_mean"] for r in results])) if results else 0.0

    curves = [
        (_prep(base_results),              COLOR_BASE,    f"baseline              last-100: {_l100(base_results):.1f}"),
        (_prep(gate_results),              COLOR_GATE,    f"analytical gate       last-100: {_l100(gate_results):.1f}"),
        (_prep(learn_results),             COLOR_LEARN,   f"learnable gate        last-100: {_l100(learn_results):.1f}"),
        (_prep(rnn_results),               COLOR_RNN,     f"GRU controller        last-100: {_l100(rnn_results):.1f}"),
        (_prep(gru_bi_results  or []),     COLOR_GRU_BI,  f"GRU + burn-in         last-100: {_l100(gru_bi_results  or []):.1f}"),
        (_prep(lstm_results    or []),     COLOR_LSTM,    f"LSTM controller       last-100: {_l100(lstm_results    or []):.1f}"),
        (_prep(lstm_bi_results or []),     COLOR_LSTM_BI, f"LSTM + burn-in        last-100: {_l100(lstm_bi_results or []):.1f}"),
    ]

    n_active = sum(1 for (ep, _, _), _, _ in curves if len(ep) > 0)
    ep_str   = f"{n_ep:,} ep" if n_ep else decay_tag
    fig, ax  = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    for (ep, sm, sd), col, lbl in curves:
        if len(ep) == 0:
            continue
        ax.fill_between(ep, sm - sd, sm + sd, color=col, alpha=0.15, linewidth=0)
        ax.plot(ep, sm, color=col, lw=2.2, label=lbl)

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"Gate comparison ({n_active}-way, {n_seeds} seed(s), {ep_str}, ±1 std)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_rnn_controller_history(
    rnn_results: List[Dict],
    out_jpg: Path,
) -> None:
    """2×2 panel showing mean controller outputs over training (±1 std).

    Works for both GRU (key: rnn_history) and LSTM (key: lstm_history) results
    by checking both keys.
    """
    histories = [
        r.get("rnn_history") or r.get("lstm_history") or []
        for r in rnn_results
    ]
    histories = [h for h in histories if h]
    if not histories:
        print("  No controller history to plot; skipping.", flush=True)
        return

    ref_steps = [h["step"] for h in histories[0]]
    keys   = ["lr_mult_mean", "eps_floor_mean", "k_cart_mean", "k_pole_mean"]
    labels = ["LR multiplier", "ε floor", "k_cart", "k_pole"]
    init_v = [1.0, END_E, 1.0, 1.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, key, lbl, init, col in zip(axes.flatten(), keys, labels, init_v, colors):
        arrays = []
        for hist in histories:
            step_map = {h["step"]: h[key] for h in hist}
            arrays.append([step_map.get(s, np.nan) for s in ref_steps])
        M  = np.array(arrays)
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(len(mu))
        x  = np.array(ref_steps)
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.2, linewidth=0)
        ax.plot(x, mu, color=col, lw=2.0)
        ax.axhline(init, color="gray", lw=1.0, ls="--", alpha=0.5,
                   label=f"init ({init})")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_title(lbl, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("RNN controller outputs over training (mean ± 1 std)", fontsize=12)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


# ---------------------------------------------------------------------------
# Plot: REINFORCE controller vs baseline_hparams conditions
# ---------------------------------------------------------------------------
def plot_reinforce_vs_baseline(
    baseline_results: Dict[str, Dict],   # {"current": ..., "optimized": ..., "rnn_controller": ...}
    reinforce_result: Dict,
    seed: int,
    total_episodes: int,
    out_jpg: Path,
) -> None:
    """
    Reproduces the style of baseline_hparams_comparison.py but adds the
    REINFORCE RNN controller as a 4th curve.

    baseline_results keys  colour  label
      current              green   baseline (current params)
      optimized            red     baseline (optimized params)
      rnn_controller       orange  RNN meta-ctrl (direct gradient)
    reinforce_result       blue    RNN meta-ctrl (REINFORCE)
    """
    W = max(1, total_episodes // 200)

    def _prep(ep_returns: List[float]):
        arr = np.asarray(ep_returns)
        n   = len(arr)
        sm  = np.convolve(arr, np.ones(W) / W, mode="valid") if W >= 2 else arr
        x   = np.arange(W, n + 1) if W >= 2 else np.arange(1, n + 1)
        return x, sm

    tags_cfg = [
        ("current",        "#2ca02c", "baseline – current params"),
        ("optimized",      "#d62728", "baseline – optimized params"),
        ("rnn_controller", "#ff7f0e", "RNN ctrl – direct gradient"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    for tag, col, lbl in tags_cfg:
        if tag not in baseline_results:
            continue
        ep_r = baseline_results[tag]["episode_returns"]
        if not ep_r:
            continue
        x, sm = _prep(ep_r)
        l100  = baseline_results[tag]["last100_mean"]
        ax.plot(x, sm, color=col, lw=2.2, label=f"{lbl}  last-100: {l100:.1f}")

    # REINFORCE curve
    rf_ep = reinforce_result["episode_returns"]
    if rf_ep:
        x, sm = _prep(rf_ep)
        l100  = reinforce_result["last100_mean"]
        ax.plot(x, sm, color=COLOR_REINFORCE, lw=2.5,
                label=f"RNN ctrl – REINFORCE  last-100: {l100:.1f}")

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"Baseline DQN vs RNN controllers  (seed {seed}, {total_episodes} episodes)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def _agg_print(results: List[Dict], label: str) -> None:
    vals = [r["last100_mean"] for r in results]
    print(
        f"  {label:<32s}  mean={np.mean(vals):6.1f}  std={np.std(vals):5.1f}"
        f"  min={np.min(vals):6.1f}  max={np.max(vals):6.1f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_seeds(s: str) -> List[int]:
    return sorted({int(p.strip()) for p in s.split(",") if p.strip()})


def _parse_modalities(s: str) -> List[str]:
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    unknown = [m for m in parts if m not in ALL_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modality: {unknown}.  Choose from: {ALL_MODALITIES}"
        )
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_STEPS,
                   metavar="N")
    p.add_argument("--epsilon-decay-episodes", type=int,
                   default=EPSILON_DECAY_EPISODES, metavar="N",
                   help="linear ε decay from START_E to END_E over N episodes")
    p.add_argument("--total-episodes", type=int, default=None, metavar="E",
                   help="stop after E episodes (step cap = max(timesteps, 15M))")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--modalities", type=str, default=",".join(ALL_MODALITIES),
                   metavar="LIST",
                   help="comma-separated subset of: "
                        "baseline,analytical,learnable,rnn,rnn_reinforce")
    p.add_argument("--train-missing-only", action="store_true",
                   help="skip seeds whose checkpoint already exists")
    p.add_argument("--plot-only", action="store_true",
                   help="skip training; reload checkpoints and regenerate figures")
    p.add_argument("--fast", action="store_true",
                   help="fast schedule: LEARNING_STARTS=1000, TRAIN_FREQUENCY=4")
    p.add_argument(
        "--compare-baseline", action="store_true",
        help=(
            "Train RNN-REINFORCE (1 seed, 750 ep, optimised hparams) and plot it "
            "alongside the existing baseline_hparams_comparison checkpoints "
            "(current / optimised / rnn_controller).  Overrides other modality flags."
        ),
    )
    args = p.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # --compare-baseline mode: self-contained, returns after plotting     #
    # ------------------------------------------------------------------ #
    if args.compare_baseline:
        COMPARE_SEED  = _parse_seeds(args.seeds)[0]   # first seed (default: 1)
        COMPARE_EP    = 750
        # Step cap matches baseline_hparams_comparison.py formula
        COMPARE_STEPS = COMPARE_EP * 500 + 10_000
        rf_ckpt_dir   = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_rnn_reinforce"
        rf_ckpt_path  = _ckpt_path_rnn_reinforce(rf_ckpt_dir, COMPARE_SEED, COMPARE_EP)

        print("=" * 64, flush=True)
        print("confgate_fourway  —  --compare-baseline mode", flush=True)
        print(f"  Seed            : {COMPARE_SEED}", flush=True)
        print(f"  Total episodes  : {COMPARE_EP}", flush=True)
        print(f"  Step cap        : {COMPARE_STEPS:,}", flush=True)
        print(f"  Device          : {device}", flush=True)
        print(f"  Checkpoint      : {rf_ckpt_path}", flush=True)
        print("=" * 64, flush=True)

        if not args.plot_only:
            if not rf_ckpt_path.exists() or not args.train_missing_only:
                print(
                    f"\n>>> [RNN_REINFORCE] Training seed {COMPARE_SEED} ...", flush=True
                )
                t0 = time.perf_counter()
                run_one_rnn_reinforce(
                    seed=COMPARE_SEED,
                    total_timesteps=COMPARE_STEPS,
                    device=device,
                    checkpoint_dir=rf_ckpt_dir,
                    total_episodes=COMPARE_EP,
                    epsilon_decay_episodes=800,
                    target_freq=1,
                    learning_starts=1_000,
                    train_frequency=4,
                    main_grad_clip=10.0,
                    ctrl_grad_clip=10.0,
                )
                print(
                    f">>> [RNN_REINFORCE] Done in {time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
            else:
                print(">>> [RNN_REINFORCE] Checkpoint present, skipping.", flush=True)

        if not rf_ckpt_path.exists():
            raise FileNotFoundError(f"Missing: {rf_ckpt_path}")

        # Load existing baseline_hparams comparison checkpoints
        baseline_results: Dict[str, Dict] = {}
        for tag in ["current", "optimized", "rnn_controller"]:
            try:
                baseline_results[tag] = _load_baseline_hparams_result(
                    tag, COMPARE_SEED, COMPARE_EP
                )
                print(
                    f"  Loaded baseline_hparams  tag={tag}"
                    f"  last100={baseline_results[tag]['last100_mean']:.1f}",
                    flush=True,
                )
            except FileNotFoundError as e:
                print(f"  WARNING: {e} — skipping {tag}", flush=True)

        rf_result = _load_result_rnn_reinforce(rf_ckpt_dir, COMPARE_SEED, COMPARE_EP)
        print(
            f"  Loaded rnn_reinforce  last100={rf_result['last100_mean']:.1f}",
            flush=True,
        )

        out_fig = (
            ROOT / "paper" / "figures" / "baseline_hparams"
            / f"baseline_vs_rnn_reinforce_seed{COMPARE_SEED}_{COMPARE_EP}ep.jpg"
        )
        plot_reinforce_vs_baseline(
            baseline_results, rf_result, COMPARE_SEED, COMPARE_EP, out_fig
        )
        return

    # ------------------------------------------------------------------ #
    # Normal four/five-way comparison mode                                #
    # ------------------------------------------------------------------ #
    seeds      = _parse_seeds(args.seeds)
    modalities = _parse_modalities(args.modalities)
    eps_decay  = args.epsilon_decay_episodes
    n_ep_tgt   = args.total_episodes
    n_step     = max(args.total_timesteps, 15_000_000) if n_ep_tgt else args.total_timesteps

    fast_sfx   = "_fast" if args.fast else ""
    decay_tag  = f"decay{eps_decay}"
    ckpt_dir   = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_fourway_{decay_tag}{fast_sfx}"

    l_starts   = 1_000 if args.fast else LEARNING_STARTS
    t_freq     = 4     if args.fast else TRAIN_FREQUENCY

    print("=" * 64, flush=True)
    print("confgate_fourway  — four-way CartPole DQN comparison", flush=True)
    print(f"  Seeds       : {seeds}", flush=True)
    print(f"  Modalities  : {modalities}", flush=True)
    if n_ep_tgt:
        print(f"  Episodes    : {n_ep_tgt:,}  (step cap {n_step:,})", flush=True)
    else:
        print(f"  Timesteps   : {n_step:,}", flush=True)
    print(f"  Eps decay   : {eps_decay} episodes", flush=True)
    print(f"  Device      : {device}", flush=True)
    if args.fast:
        print(f"  Fast        : LEARNING_STARTS={l_starts}, TRAIN_FREQUENCY={t_freq}",
              flush=True)
    print(f"  Checkpoints : {ckpt_dir}", flush=True)
    print("=" * 64, flush=True)

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    if not args.plot_only:

        # Modality 1 — baseline
        if "baseline" in modalities:
            seeds_run = (
                [s for s in seeds
                 if not cg_analytical._ckpt_path(ckpt_dir, s, False).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [BASELINE] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    cg_analytical.run_one(
                        seed, n_step, False, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [BASELINE] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [BASELINE] All checkpoints present, skipping.", flush=True)

        # Modality 2 — analytical gate
        if "analytical" in modalities:
            seeds_run = (
                [s for s in seeds
                 if not cg_analytical._ckpt_path(ckpt_dir, s, True).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [ANALYTICAL] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    cg_analytical.run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [ANALYTICAL] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [ANALYTICAL] All checkpoints present, skipping.", flush=True)

        # Modality 3 — learnable gate
        if "learnable" in modalities:
            seeds_run = (
                [s for s in seeds
                 if not cg_learnable._ckpt_path_learnable(ckpt_dir, s).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [LEARNABLE] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    cg_learnable.run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learnable_gate=True,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [LEARNABLE] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [LEARNABLE] All checkpoints present, skipping.", flush=True)

        # Modality 4 — RNN controller (direct gradient)
        if "rnn" in modalities:
            seeds_run = (
                [s for s in seeds if not _ckpt_path_rnn(ckpt_dir, s).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [RNN] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    run_one_rnn(
                        seed, n_step, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [RNN] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [RNN] All checkpoints present, skipping.", flush=True)

        # Modality 5 — RNN controller with REINFORCE
        if "rnn_reinforce" in modalities:
            rf_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_rnn_reinforce"
            seeds_run = (
                [s for s in seeds
                 if not _ckpt_path_rnn_reinforce(rf_ckpt_dir, s, n_ep_tgt or 0).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(
                    f"\n>>> [RNN_REINFORCE] Training seeds {seeds_run} ...", flush=True
                )
                t0 = time.perf_counter()
                for seed in seeds_run:
                    run_one_rnn_reinforce(
                        seed, n_step, device,
                        checkpoint_dir=rf_ckpt_dir,
                        total_episodes=n_ep_tgt,
                        epsilon_decay_episodes=eps_decay,
                        target_freq=t_freq,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(
                    f">>> [RNN_REINFORCE] Done in {time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
            else:
                print(">>> [RNN_REINFORCE] All checkpoints present, skipping.",
                      flush=True)

        # Modality 6 — GRU controller with burn-in
        if "gru_burnin" in modalities:
            seeds_run = (
                [s for s in seeds if not _ckpt_path_gru_burnin(ckpt_dir, s).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [GRU_BURNIN] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    run_one_gru_burnin(
                        seed, n_step, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [GRU_BURNIN] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [GRU_BURNIN] All checkpoints present, skipping.", flush=True)

        # Modality 7 — LSTM controller
        if "lstm" in modalities:
            seeds_run = (
                [s for s in seeds if not _ckpt_path_lstm(ckpt_dir, s).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [LSTM] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    run_one_lstm(
                        seed, n_step, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [LSTM] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [LSTM] All checkpoints present, skipping.", flush=True)

        # Modality 7 — LSTM controller with burn-in
        if "lstm_burnin" in modalities:
            seeds_run = (
                [s for s in seeds if not _ckpt_path_lstm_burnin(ckpt_dir, s).exists()]
                if args.train_missing_only else list(seeds)
            )
            if seeds_run:
                print(f"\n>>> [LSTM_BURNIN] Training seeds {seeds_run} ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_run:
                    run_one_lstm_burnin(
                        seed, n_step, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay,
                        total_episodes=n_ep_tgt,
                        learning_starts=l_starts,
                        train_frequency=t_freq,
                    )
                print(f">>> [LSTM_BURNIN] Done in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                print(">>> [LSTM_BURNIN] All checkpoints present, skipping.", flush=True)

    # ------------------------------------------------------------------ #
    # Verify checkpoints present                                         #
    # ------------------------------------------------------------------ #
    rf_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_rnn_reinforce"
    missing = []
    if "baseline"      in modalities:
        missing += [cg_analytical._ckpt_path(ckpt_dir, s, False)
                    for s in seeds
                    if not cg_analytical._ckpt_path(ckpt_dir, s, False).exists()]
    if "analytical"    in modalities:
        missing += [cg_analytical._ckpt_path(ckpt_dir, s, True)
                    for s in seeds
                    if not cg_analytical._ckpt_path(ckpt_dir, s, True).exists()]
    if "learnable"     in modalities:
        missing += [cg_learnable._ckpt_path_learnable(ckpt_dir, s)
                    for s in seeds
                    if not cg_learnable._ckpt_path_learnable(ckpt_dir, s).exists()]
    if "rnn"           in modalities:
        missing += [_ckpt_path_rnn(ckpt_dir, s)
                    for s in seeds
                    if not _ckpt_path_rnn(ckpt_dir, s).exists()]
    if "rnn_reinforce" in modalities:
        missing += [_ckpt_path_rnn_reinforce(rf_ckpt_dir, s, n_ep_tgt or 0)
                    for s in seeds
                    if not _ckpt_path_rnn_reinforce(
                        rf_ckpt_dir, s, n_ep_tgt or 0).exists()]
    if "gru_burnin"    in modalities:
        missing += [_ckpt_path_gru_burnin(ckpt_dir, s)
                    for s in seeds
                    if not _ckpt_path_gru_burnin(ckpt_dir, s).exists()]
    if "lstm"          in modalities:
        missing += [_ckpt_path_lstm(ckpt_dir, s)
                    for s in seeds
                    if not _ckpt_path_lstm(ckpt_dir, s).exists()]
    if "lstm_burnin"   in modalities:
        missing += [_ckpt_path_lstm_burnin(ckpt_dir, s)
                    for s in seeds
                    if not _ckpt_path_lstm_burnin(ckpt_dir, s).exists()]
    if missing:
        for m in missing:
            print(f"  MISSING: {m}", flush=True)
        raise FileNotFoundError(f"{len(missing)} checkpoint(s) missing (see above).")

    # ------------------------------------------------------------------ #
    # Load results                                                        #
    # ------------------------------------------------------------------ #
    base_res    = ([cg_analytical._load_result(ckpt_dir, s, False)    for s in seeds]
                   if "baseline"      in modalities else [])
    gate_res    = ([cg_analytical._load_result(ckpt_dir, s, True)     for s in seeds]
                   if "analytical"    in modalities else [])
    learn_res   = ([cg_learnable._load_result_learnable(ckpt_dir, s)  for s in seeds]
                   if "learnable"     in modalities else [])
    rnn_res     = ([_load_result_rnn(ckpt_dir, s)                     for s in seeds]
                   if "rnn"           in modalities else [])
    rnn_rf_res  = ([_load_result_rnn_reinforce(rf_ckpt_dir, s, n_ep_tgt or 0)
                    for s in seeds]
                   if "rnn_reinforce" in modalities else [])
    gru_bi_res  = ([_load_result_gru_burnin(ckpt_dir, s)  for s in seeds]
                   if "gru_burnin"    in modalities else [])
    lstm_res    = ([_load_result_lstm(ckpt_dir, s)         for s in seeds]
                   if "lstm"          in modalities else [])
    lstm_bi_res = ([_load_result_lstm_burnin(ckpt_dir, s)  for s in seeds]
                   if "lstm_burnin"   in modalities else [])

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n=== Final-100-episode summary ===", flush=True)
    if base_res:    _agg_print(base_res,    "baseline")
    if gate_res:    _agg_print(gate_res,    "analytical gate")
    if learn_res:   _agg_print(learn_res,   "learnable gate")
    if rnn_res:     _agg_print(rnn_res,     "GRU controller (direct grad)")
    if gru_bi_res:  _agg_print(gru_bi_res,  "GRU + burn-in")
    if rnn_rf_res:  _agg_print(rnn_rf_res,  "RNN controller (REINFORCE)")
    if lstm_res:    _agg_print(lstm_res,    "LSTM controller")
    if lstm_bi_res: _agg_print(lstm_bi_res, "LSTM + burn-in")

    # ------------------------------------------------------------------ #
    # Figures                                                             #
    # ------------------------------------------------------------------ #
    n_str    = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
    ep_tag   = f"_ep{n_ep_tgt}" if n_ep_tgt else ""
    mod_tag  = "_".join(m[:3] for m in sorted(modalities))

    plot_fourway_comparison(
        base_res, gate_res, learn_res, rnn_res,
        FIG_DIR / f"fourway_{mod_tag}_{n_str}_{decay_tag}{fast_sfx}{ep_tag}.jpg",
        n_seeds=len(seeds),
        decay_tag=decay_tag,
        n_ep=n_ep_tgt,
        gru_bi_results=gru_bi_res   or None,
        lstm_results=lstm_res       or None,
        lstm_bi_results=lstm_bi_res or None,
    )

    if rnn_res:
        plot_rnn_controller_history(
            rnn_res,
            FIG_DIR / f"rnn_ctrl_history_{n_str}_{decay_tag}{fast_sfx}{ep_tag}.jpg",
        )
    if gru_bi_res:
        plot_rnn_controller_history(
            gru_bi_res,
            FIG_DIR / f"gru_bi_ctrl_history_{n_str}_{decay_tag}{fast_sfx}{ep_tag}.jpg",
        )
    if lstm_res:
        plot_rnn_controller_history(
            lstm_res,
            FIG_DIR / f"lstm_ctrl_history_{n_str}_{decay_tag}{fast_sfx}{ep_tag}.jpg",
        )
    if lstm_bi_res:
        plot_rnn_controller_history(
            lstm_bi_res,
            FIG_DIR / f"lstm_bi_ctrl_history_{n_str}_{decay_tag}{fast_sfx}{ep_tag}.jpg",
        )


if __name__ == "__main__":
    main()