"""
b_feedback_confgate_detached.py
================================
Detached-W_z variant of b_feedback_confgate: W_z is EXCLUDED from Adam and backprop;
updated ONLY by manual B-feedback (full-head TD error).  Trunk + head_full use Adam on
loss_full; aux heads use a separate Adam with detached trunk.

Usage:
  python code/b_feedback_confgate_detached.py --seeds 1
  python code/b_feedback_confgate_detached.py --plot-only
"""

from __future__ import annotations

import argparse
import json
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
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Hyper-parameters (match CleanRL / run_cleanrl_vs_mixed_dqn.py)
# ---------------------------------------------------------------------------
LEARNING_RATE       = 2.5e-4
LR_LINEAR_MIXED     = 1e-4      # B-feedback LR (sole W_z update — no backprop)
GAMMA               = 0.99
BUFFER_SIZE         = 10_000
BATCH_SIZE          = 128
START_E             = 1.0
END_E               = 0.05
EPSILON_DECAY_EPISODES = 3_500   # linear START_E → END_E over this many completed episodes
LEARNING_STARTS     = 10_000
TRAIN_FREQUENCY     = 10
TARGET_NETWORK_FREQ = 500
TAU                 = 1.0
FEEDBACK_GRAD_CLIP  = 1.0       # max-norm clip for manual W_z update (0 = off)
LOG_EVERY           = 5_000     # steps between gate-trace / loss logging

OBS_TO_120 = 120
HIDDEN_84  = 84

DEFAULT_SEEDS         = [1]
DEFAULT_TOTAL_STEPS   = 500_000

COLOR_BASE = "#2ca02c"   # green — baseline (no gate)
COLOR_GATE = "#9467bd"   # purple — confidence-gated

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
ReplayBufferSamples = namedtuple(
    "ReplayBufferSamples",
    ["observations", "actions", "next_observations", "dones", "rewards"],
)


class ReplayBuffer:
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape   = obs_shape
        self.device      = device
        self.pos  = 0
        self.full = False
        self.observations      = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions  = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards  = np.zeros((buffer_size,), dtype=np.float32)
        self.dones    = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action: int, reward: float, done: float) -> None:
        self.observations[self.pos]      = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = done
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos  = 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper = self.buffer_size if self.full else self.pos
        idx   = np.random.randint(0, upper, size=batch_size)
        return ReplayBufferSamples(
            observations      = torch.tensor(self.observations[idx],      device=self.device),
            actions           = torch.tensor(self.actions[idx],           device=self.device).unsqueeze(1),
            next_observations = torch.tensor(self.next_observations[idx], device=self.device),
            dones             = torch.tensor(self.dones[idx],             device=self.device).unsqueeze(1),
            rewards           = torch.tensor(self.rewards[idx],           device=self.device).unsqueeze(1),
        )


# ---------------------------------------------------------------------------
# B matrix
# ---------------------------------------------------------------------------

def make_b_matrix(F: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """B in R^{F x 1}: trivially maps the single full-head TD error to all F features."""
    return torch.ones(F, 1, device=device, dtype=dtype)


def make_b_matrix_gated(
    F: int, k4_mean: float, k5_mean: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """B scaled per compartment: cart block (rows 0..F//2) × k4_mean, pole block × k5_mean."""
    b = torch.ones(F, 1, device=device, dtype=dtype)
    b[: F // 2] *= k4_mean
    b[F // 2 :] *= k5_mean
    return b


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _scaled_obs(obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
    """Jacobian-correct Wz outer-product scaling for gated first layer."""
    if obs.shape[-1] < 4:
        return obs
    gx = obs.clone()
    gx[:, :2] = gx[:, :2] * k4.to(dtype=gx.dtype, device=gx.device)
    gx[:, 2:4] = gx[:, 2:4] * k5.to(dtype=gx.dtype, device=gx.device)
    return gx


class BFeedbackConfGateNetwork(nn.Module):
    """
    Three-head network (full / cart-only / pole-only) where:
      - linear_feature (W_z) is updated ONLY by manual B-feedback (no Adam, no backprop).
      - trunk + head_full share one Adam optimizer on loss_full; z is detached before trunk
        so TD gradients do not reach W_z.
      - head_cart and head_pole use a separate Adam optimizer;
        their trunk forward pass runs on detached z so they don't
        interfere with W_z via backprop.
      - Confidence gating (optional): margins from Q_cart / Q_pole →
        softmax → k in [k_min, k_max] → scale W_z cart/pole compartments
        for the Q_full path. Gate factors are detached from the aux-head paths.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        confidence_gating: bool = False,
        gate_k_min: float = 0.8,
        gate_k_max: float = 1.0,
        gate_tau: float = 1.0,
    ):
        super().__init__()
        self.obs_dim            = obs_dim
        self.n_actions          = n_actions
        self.confidence_gating  = confidence_gating
        self.gate_k_min         = gate_k_min
        self.gate_k_max         = gate_k_max
        self.gate_tau           = gate_tau   # softmax temperature; <1 sharpens gate

        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head_full = nn.Linear(HIDDEN_84, n_actions)
        self.head_cart = nn.Linear(HIDDEN_84, n_actions)
        self.head_pole = nn.Linear(HIDDEN_84, n_actions)
        # Controller-adjustable effective-width modulator (not a learned parameter)
        self.trunk_scale: float = 1.0

    # ------ utilities -------------------------------------------------------

    @staticmethod
    def _mask_cart(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 2:4] = 0.0; return out

    @staticmethod
    def _mask_pole(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 0:2] = 0.0; return out

    def _gating_active(self) -> bool:
        return self.confidence_gating and self.obs_dim == 4 and self.n_actions == 2

    def _gate_k(self, Q_cart: torch.Tensor, Q_pole: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (k4, k5) shaped (B,1), detached from Q_cart/Q_pole."""
        m_c = (Q_cart[:, 1] - Q_cart[:, 0]).abs()
        m_p = (Q_pole[:, 1] - Q_pole[:, 0]).abs()
        # Temperature τ < 1 sharpens the contrast; τ = 1.0 reproduces original behaviour
        C = F.softmax(torch.stack([m_c, m_p], dim=1) / max(self.gate_tau, 1e-6), dim=1).detach()
        km, kx = self.gate_k_min, self.gate_k_max
        k4 = km + (kx - km) * C[:, 0:1]
        k5 = km + (kx - km) * C[:, 1:2]
        return k4, k5

    def _gated_z(self, obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
        W = self.linear_feature.weight
        b = self.linear_feature.bias
        return k4 * (obs[..., 0:2] @ W[:, 0:2].T) + k5 * (obs[..., 2:4] @ W[:, 2:4].T) + b

    # ------ forward ---------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (Q_full, Q_cart, Q_pole, k4, k5).
        Q_full path: z (gated if enabled) -> trunk(detach) -> head_full  [W_z: B only].
        Aux paths:   masked obs -> linear_feature -> trunk(detach) -> head_cart/head_pole.
        """
        # --- aux heads (trunk detached so they don't pull W_z via backprop) ---
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart.detach()))

        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole.detach()))

        # --- gate factors from aux head margins (detached) ---
        B = obs.shape[0]
        k4 = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        k5 = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        if self._gating_active():
            k4, k5 = self._gate_k(Q_cart, Q_pole)

        # --- full-state head (W_z detached — B-feedback only) ---
        if self._gating_active():
            z_full = self._gated_z(obs, k4, k5)
        else:
            z_full = self.linear_feature(obs)
        z_trunk = self.trunk(z_full.detach())
        if self.trunk_scale != 1.0:
            z_trunk = z_trunk * self.trunk_scale
        Q_full = self.head_full(z_trunk)

        return Q_full, Q_cart, Q_pole, k4, k5

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        Q_full, _, _, _, _ = self.forward(obs)
        return Q_full

    @torch.no_grad()
    def gate_k_no_grad(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gate factors without touching W_z gradient (used for manual B update)."""
        if not self._gating_active():
            ones = torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
            return ones, ones.clone()
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart))
        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole))
        return self._gate_k(Q_cart, Q_pole)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    return max(start + (end - start) / max(duration, 1) * t, end)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Gate Controller
# ---------------------------------------------------------------------------

CTRL_WARMUP_EP = 2000   # episodes before LR/trunk throttling kicks in


class GateController:
    """
    Uses the confidence-gate output (k4, k5) to dynamically modulate training knobs.

    Controlled knobs (all True by default except control_B):
      control_B       (default False) – scale B-matrix compartments by gate EMA.
      control_Wz      (default True)  – ramp gate_k_min from 1.0→0.8 over CTRL_WARMUP_EP
                                        episodes so gating activates gradually.
      control_trunk   (default True)  – scale trunk by imbalance EMA after warmup.
      control_epsilon (default True)  – keep floor at END_E, no decay extension.
      control_lr      (default True)  – scale LR by imbalance EMA after warmup.

    Fix 1: _imbalance_ema initialised to 1.0 → full LR/trunk until real signal exists.
    Fix 2: epsilon floor fixed at END_E; decay period unchanged (no extension).
    Fix 3: gate_k_min ramps from 1.0 → 0.8 over first CTRL_WARMUP_EP episodes so the
           gate is disabled early (k4≈k5≈1 → no interference) and only activates once
           the agent has learned.  Avoids the degenerate k_min=k_max=1 fixed point that
           produced zero imbalance signal and locked LR/trunk at their minimum values.
    """

    def __init__(
        self,
        q_net: BFeedbackConfGateNetwork,
        opt_main: optim.Optimizer,
        total_timesteps: int,
        control_B: bool = False,
        control_Wz: bool = True,
        control_trunk: bool = True,
        control_epsilon: bool = True,
        control_lr: bool = True,
        ema_alpha: float = 0.05,
        use_original_formulas: bool = False,
    ):
        self.q_net            = q_net
        self.opt_main         = opt_main
        self.total_timesteps  = total_timesteps
        self.control_B        = control_B
        self.control_Wz       = control_Wz
        self.control_trunk    = control_trunk
        self.control_epsilon  = control_epsilon
        self.control_lr       = control_lr
        self.ema_alpha             = ema_alpha
        self.use_original_formulas = use_original_formulas
        # Fix 1: start at 1.0 → full LR + full trunk before real signal builds up.
        self._imbalance_ema      = 1.0
        self._k4_ema             = 0.9
        self._k5_ema             = 0.9
        self._effective_lr       = LEARNING_RATE
        self._effective_end_e    = END_E
        self._effective_decay_ep = float(EPSILON_DECAY_EPISODES)

    def step(
        self,
        k4: torch.Tensor,
        k5: torch.Tensor,
        t: int,
        ep_done: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Update EMA, apply all enabled controls. Returns B_mat for this step."""
        k4m = float(k4.mean().item())
        k5m = float(k5.mean().item())
        imbalance = abs(k4m - k5m)
        a = self.ema_alpha
        self._imbalance_ema = (1 - a) * self._imbalance_ema + a * imbalance
        self._k4_ema        = (1 - a) * self._k4_ema        + a * k4m
        self._k5_ema        = (1 - a) * self._k5_ema        + a * k5m

        if self.use_original_formulas:
            # Original (pre-fix) formulas used for ablation variants (lr/trunk/eps each alone).
            # Fix 1 (imbalance_ema=1.0 init) still applies so knobs start at full capacity.
            if self.control_Wz:
                # Original: timestep-progress widening — kept at baseline 0.8 here since
                # progress≈0 for short episode runs anyway.
                self.q_net.gate_k_min = 0.8
            if self.control_trunk:
                self.q_net.trunk_scale = 0.6 + 0.4 * self._imbalance_ema
            if self.control_lr:
                lr = LEARNING_RATE * (0.5 + 0.5 * self._imbalance_ema)
                self._effective_lr = lr
                for g in self.opt_main.param_groups:
                    g["lr"] = lr
            if self.control_epsilon:
                self._effective_end_e    = END_E + (1.0 - self._imbalance_ema) * 0.05
                self._effective_decay_ep = EPSILON_DECAY_EPISODES * (
                    1.0 + (1.0 - self._imbalance_ema) * 0.5
                )
        else:
            # Fixed formulas (all three fixes applied).
            # Fix 3: tighten gate range when imbalance HIGH to prevent divergence crashes.
            #   imbalance≈0   → k_min=0.8  (same as baseline)
            #   imbalance≈0.5 → k_min=0.9  (tighter, less divergence)
            if self.control_Wz:
                self.q_net.gate_k_min = min(0.9, 0.8 + 0.2 * self._imbalance_ema)

            # Trunk and LR stay at full capacity (throttling caused catastrophic forgetting).
            if self.control_trunk:
                self.q_net.trunk_scale = 1.0
            if self.control_lr:
                self._effective_lr = LEARNING_RATE
                for g in self.opt_main.param_groups:
                    g["lr"] = LEARNING_RATE

            # Fix 2: epsilon floor fixed; no decay extension.
            if self.control_epsilon:
                self._effective_end_e    = END_E
                self._effective_decay_ep = float(EPSILON_DECAY_EPISODES)

        # --- B matrix (Option A: only if control_B is True) ---
        if self.control_B:
            return make_b_matrix_gated(
                OBS_TO_120, self._k4_ema, self._k5_ema, device, dtype
            )
        return make_b_matrix(OBS_TO_120, device, dtype)

    @property
    def effective_end_e(self) -> float:
        return self._effective_end_e

    @property
    def effective_decay_ep(self) -> float:
        return self._effective_decay_ep

    def log_state(self) -> Dict:
        return {
            "imbalance_ema":      self._imbalance_ema,
            "k4_ema":             self._k4_ema,
            "k5_ema":             self._k5_ema,
            "effective_lr":       self._effective_lr,
            "effective_end_e":    self._effective_end_e,
            "effective_decay_ep": self._effective_decay_ep,
            "trunk_scale":        self.q_net.trunk_scale,
        }


def _agg_print(results: List[Dict], label: str) -> None:
    vals = [r["last100_mean"] for r in results]
    mu   = float(np.mean(vals))
    sd   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    print(f"  {label:35s}: {mu:.2f} ± {sd:.2f}  (n={len(vals)})", flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_one(
    seed: int,
    total_timesteps: int,
    confidence_gating: bool,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    use_controller: bool = False,
    control_B: bool = False,
    ctrl_tag: str = "",
    ctrl_knobs: Optional[Dict] = None,
    # Option-comparison knobs
    gate_k_min: float = 0.8,
    gate_k_max: float = 1.0,
    gate_tau: float = 1.0,
    target_tau: float = TAU,
    target_freq: int = TARGET_NETWORK_FREQ,
    algo_tag_override: Optional[str] = None,
    main_grad_clip: float = 0.0,      # max-norm clip for opt_main gradients (0 = off)
    learning_starts: int = LEARNING_STARTS,   # steps of pure collection before training
    train_frequency: int = TRAIN_FREQUENCY,   # env steps between gradient updates
) -> Dict:
    """
    Train one seed until `total_episodes` completed episodes (if set), else for
    `total_timesteps` env steps (whichever limit is hit first).
    Epsilon decays linearly from START_E to END_E over `epsilon_decay_episodes`
    completed episodes (index = len(episode_returns) at each env step).
    Returns dict with episode_returns, gate_history, final stats.
    """
    _set_seed(seed)
    env    = gym.make("CartPole-v1")
    env    = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim   = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    gate_ok = confidence_gating and obs_dim == 4 and n_actions == 2
    label   = f"seed={seed} gated={gate_ok}"

    q_net  = BFeedbackConfGateNetwork(
        obs_dim, n_actions,
        confidence_gating=gate_ok,
        gate_k_min=gate_k_min,
        gate_k_max=gate_k_max,
        gate_tau=gate_tau,
    ).to(device)
    t_net  = BFeedbackConfGateNetwork(
        obs_dim, n_actions,
        confidence_gating=gate_ok,
        gate_k_min=gate_k_min,
        gate_k_max=gate_k_max,
        gate_tau=gate_tau,
    ).to(device)
    t_net.load_state_dict(q_net.state_dict())

    # trunk + head_full only — W_z excluded (B-feedback only)
    opt_main = optim.Adam(
        list(q_net.trunk.parameters()) + list(q_net.head_full.parameters()),
        lr=LEARNING_RATE,
    )
    # Aux heads separate — trunk detached for them
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )

    # B matrix: single column (full-head error only → no cross-head interference)
    B_mat = make_b_matrix(OBS_TO_120, device, torch.float32)

    # Gate controller (only active when gating is also active)
    controller: Optional[GateController] = None
    if use_controller and gate_ok:
        extra_kwargs = ctrl_knobs if ctrl_knobs is not None else {}
        controller = GateController(
            q_net, opt_main, total_timesteps, control_B=control_B, **extra_kwargs
        )

    rb = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape, device)

    episode_returns: List[float]    = []
    gate_history: List[Dict]        = []
    controller_history: List[Dict]  = []
    loss_list: List[float]          = []

    # Mutable epsilon parameters (controller may update these each training step)
    _eps_end   = END_E
    _eps_decay = float(epsilon_decay_episodes)

    obs, _ = env.reset(seed=seed)
    t = 0
    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q_net.forward_q_only(x).argmax(dim=1).item())

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done     = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]
        rb.add(obs, real_nxt, action, float(reward), float(done))

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()

        # --- training step ---
        if t > learning_starts and t % train_frequency == 0:
            data = rb.sample(BATCH_SIZE)

            # ---- shared TD target (from target network, Q_full path) ----
            with torch.no_grad():
                tQ_full, tQ_cart, tQ_pole, _, _ = t_net.forward(data.next_observations)
                r = data.rewards.flatten()
                d = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            # ---- forward (W_z detached for Q_full path) ----
            Q_full, Q_cart, Q_pole, k4, k5 = q_net.forward(data.observations)

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            # ---- controller step: update B_mat + all enabled knobs ----
            if controller is not None:
                B_mat = controller.step(k4, k5, t, len(episode_returns), device, torch.float32)
                _eps_end   = controller.effective_end_e
                _eps_decay = controller.effective_decay_ep

            # ---- B-feedback on W_z (sole W_z update): full-head error only ----
            e_full = (y_full - qf_sa).detach().unsqueeze(1)  # (B, 1)
            delta_z = e_full @ B_mat.t()                      # (B, F)
            gx = _scaled_obs(data.observations, k4, k5)
            grad_Wz = (delta_z.t() @ gx) / BATCH_SIZE         # (F, obs_dim)
            grad_bz = delta_z.mean(0)                          # (F,)
            if FEEDBACK_GRAD_CLIP > 0:
                n = grad_Wz.norm()
                if n > FEEDBACK_GRAD_CLIP:
                    grad_Wz = grad_Wz * (FEEDBACK_GRAD_CLIP / n.item())
            with torch.no_grad():
                q_net.linear_feature.weight.data.add_(grad_Wz, alpha=LR_LINEAR_MIXED)
                q_net.linear_feature.bias.data.add_(grad_bz,  alpha=LR_LINEAR_MIXED)

            # ---- backprop trunk + head_full only (W_z excluded) ----
            opt_main.zero_grad()
            loss_full.backward()
            if main_grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(q_net.trunk.parameters()) + list(q_net.head_full.parameters()),
                    max_norm=main_grad_clip,
                )
            opt_main.step()

            # ---- aux heads (trunk detached, own optimizer) ----
            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            loss_list.append(loss_full.item())

            # ---- periodic gate + controller logging ----
            if gate_ok and t % LOG_EVERY == 0:
                gate_history.append({
                    "step":    t,
                    "episode": len(episode_returns),
                    "mean_k4": float(k4.mean().item()),
                    "mean_k5": float(k5.mean().item()),
                })
            if controller is not None and t % LOG_EVERY == 0:
                cs = controller.log_state()
                cs.update({"step": t, "episode": len(episode_returns)})
                controller_history.append(cs)

        # ---- target network update (hard copy when target_tau=1.0, soft Polyak otherwise) ----
        if t % target_freq == 0:
            for tp, qp in zip(t_net.parameters(), q_net.parameters()):
                tp.data.copy_(target_tau * qp.data + (1 - target_tau) * tp.data)

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [b_feedb_cg_det] {label} step={t+1}/{total_timesteps}"
                f"  episodes={len(episode_returns)}"
                f"  eps={_linear_schedule(START_E, _eps_end, int(_eps_decay), n_ep_done):.3f}",
                flush=True,
            )

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  [b_feedb_cg] WARNING {label}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    ctrl_active = use_controller and gate_ok
    if algo_tag_override is not None:
        algo_tag = algo_tag_override
    elif ctrl_active:
        algo_tag = f"b_feedb_cg_det_ctrl{ctrl_tag}"
    elif gate_ok:
        algo_tag = "b_feedb_cg_det_gate"
    else:
        algo_tag = "b_feedb_cg_det"

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{algo_tag}_seed{seed}.pt"
        torch.save({
            "algo":               algo_tag,
            "seed":               seed,
            "episode_returns":    episode_returns,
            "gate_history":       gate_history,
            "controller_history": controller_history,
            "q_network":          q_net.state_dict(),
        }, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    last100 = float(np.mean(episode_returns[-100:])) if len(episode_returns) >= 100 else float(np.mean(episode_returns)) if episode_returns else 0.0
    print(
        f"  Done {label}: episodes={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )
    return {
        "seed":               seed,
        "confidence_gating":  gate_ok,
        "use_controller":     ctrl_active,
        "episode_returns":    episode_returns,
        "gate_history":       gate_history,
        "controller_history": controller_history,
        "last100_mean":       last100,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid") if w >= 2 else arr


def plot_learning_curves(
    base_results: List[Dict],
    gate_results: List[Dict],
    out_jpg: Path,
) -> None:
    b_arrays = [np.asarray(r["episode_returns"]) for r in base_results]
    g_arrays = [np.asarray(r["episode_returns"]) for r in gate_results]
    n = min(min(len(a) for a in b_arrays), min(len(a) for a in g_arrays))
    B = np.stack([a[:n] for a in b_arrays], axis=0)
    G = np.stack([a[:n] for a in g_arrays], axis=0)
    ep = np.arange(1, n + 1)

    mu_b, sb = B.mean(0), B.std(0, ddof=1) if B.shape[0] > 1 else np.zeros(n)
    mu_g, sg = G.mean(0), G.std(0, ddof=1) if G.shape[0] > 1 else np.zeros(n)

    W = max(1, n // 200)
    ep_sm = ep[W - 1:]
    sm_b, sm_sb = _smooth(mu_b, W), _smooth(sb, W)
    sm_g, sm_sg = _smooth(mu_g, W), _smooth(sg, W)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.fill_between(ep_sm, sm_b - sm_sb, sm_b + sm_sb, color=COLOR_BASE, alpha=0.18, linewidth=0)
    ax.fill_between(ep_sm, sm_g - sm_sg, sm_g + sm_sg, color=COLOR_GATE, alpha=0.18, linewidth=0)
    ax.plot(ep_sm, sm_b, color=COLOR_BASE, lw=2.2,
            label=f"baseline (no gate)  last-100 mean: {np.mean(B[:,-100:]):.1f}")
    ax.plot(ep_sm, sm_g, color=COLOR_GATE, lw=2.2,
            label=f"confidence-gated   last-100 mean: {np.mean(G[:,-100:]):.1f}")
    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"B-feedback + conf. gate (3-head, W_z detached / B-only): "
        f"baseline vs gated  ({len(base_results)} seeds, shading = ±1 std)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _interp(hist: List[Dict], x_key: str, y_key: str, grid: np.ndarray) -> np.ndarray:
    xs = np.array([float(h[x_key]) for h in hist])
    ys = np.array([float(h[y_key]) for h in hist])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux: list = []; uy: list = []
    for x, y in zip(xs, ys):
        if ux and abs(x - ux[-1]) < 1e-9:
            uy[-1] = float(y)
        else:
            ux.append(float(x)); uy.append(float(y))
    return np.interp(grid, np.asarray(ux), np.asarray(uy), left=float("nan"), right=float("nan"))


def plot_gate_traces(
    gate_results: List[Dict],
    out_jpg: Path,
) -> None:
    hists = [r["gate_history"] for r in gate_results if r.get("gate_history")]
    if not hists:
        print("[gate plot] no gate history — skipping.", flush=True)
        return

    starts = [float(h[0]["episode"]) for h in hists]
    ends   = [float(h[-1]["episode"]) for h in hists]
    lo, hi = max(starts), min(ends)
    if hi <= lo:
        print("[gate plot] insufficient overlap for gate grid — skipping.", flush=True)
        return
    grid = np.linspace(lo, hi, 512)

    C = np.vstack([_interp(h, "episode", "mean_k4", grid) for h in hists])
    P = np.vstack([_interp(h, "episode", "mean_k5", grid) for h in hists])

    fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    tab10 = plt.get_cmap("tab10")
    thin_c = [tab10(i % 10) for i in range(len(hists))]

    def _draw(ax, rows, ylabel, title, mcol):
        if rows.shape[0] > 1:
            for i, row in enumerate(rows):
                ax.plot(grid, row, color=thin_c[i], lw=0.6, alpha=0.45)
            ax.plot(grid, rows.mean(0), color=mcol, lw=2.2, label="seed mean")
        else:
            ax.plot(grid, rows[0], color=mcol, lw=2.2)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.76, 1.04)
        ax.axhline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    _draw(ax_c, C, r"$k_{\mathrm{cart}}$",
          rf"Batch-mean $k_{{\mathrm{{cart}}}}$ — b_feedb_cg gated ({len(hists)} seeds; thin=individual)",
          COLOR_BASE)
    _draw(ax_p, P, r"$k_{\mathrm{pole}}$",
          rf"Batch-mean $k_{{\mathrm{{pole}}}}$ — b_feedb_cg gated ({len(hists)} seeds; thick=mean)",
          COLOR_GATE)
    ax_p.set_xlabel("Episode", fontsize=11)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_controller_comparison(
    base_results: List[Dict],
    gate_results: List[Dict],
    ctrl_results: List[Dict],
    out_jpg: Path,
    extra_series: Optional[List[Tuple[List[Dict], str, str]]] = None,
    color_base: Optional[str] = None,
    color_gate: Optional[str] = None,
    color_ctrl: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Learning curves: baseline vs confidence-gated vs controller (+ optional extra variants).

    extra_series: list of (results, hex_color, label) for additional curves.
    """
    c_base = color_base or COLOR_BASE
    c_gate = color_gate or COLOR_GATE
    c_ctrl = color_ctrl or "#d62728"   # red

    def _prep(results: List[Dict]):
        arrays = [np.asarray(r["episode_returns"]) for r in results]
        n = min(len(a) for a in arrays)
        M = np.stack([a[:n] for a in arrays], axis=0)
        mu = M.mean(0)
        sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
        return np.arange(1, n + 1), mu, sd

    all_results = [base_results, gate_results, ctrl_results] + (
        [r for r, _, _ in extra_series] if extra_series else []
    )
    n = min(min(len(a) for a in [np.asarray(r2["episode_returns"]) for r2 in res]) for res in all_results)
    W = max(1, n // 200)

    def _sm(ep, mu, sd):
        return ep[W - 1:], _smooth(mu, W), _smooth(sd, W)

    def _prep_sm(results):
        ep, mu, sd = _prep(results)
        return _sm(ep[:n], mu[:n], sd[:n])

    ep_b, sm_b, sd_b = _prep_sm(base_results)
    ep_g, sm_g, sd_g = _prep_sm(gate_results)
    ep_c, sm_c, sd_c = _prep_sm(ctrl_results)

    last100_b = float(np.mean([r["last100_mean"] for r in base_results]))
    last100_g = float(np.mean([r["last100_mean"] for r in gate_results]))
    last100_c = float(np.mean([r["last100_mean"] for r in ctrl_results]))

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    series = [
        (ep_b, sm_b, sd_b, c_base, "baseline",         last100_b),
        (ep_g, sm_g, sd_g, c_gate, "confidence-gated", last100_g),
        (ep_c, sm_c, sd_c, c_ctrl, "controller (all)", last100_c),
    ]
    if extra_series:
        for res, col, lbl in extra_series:
            ep_e, sm_e, sd_e = _prep_sm(res)
            l100_e = float(np.mean([r["last100_mean"] for r in res]))
            series.append((ep_e, sm_e, sd_e, col, lbl, l100_e))

    for ep, sm, sd, col, lbl, l100 in series:
        ax.fill_between(ep, sm - sd, sm + sd, color=col, alpha=0.12, linewidth=0)
        ax.plot(ep, sm, color=col, lw=2.0, label=f"{lbl}  last-100: {l100:.1f}")
    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    plot_title = title or (
        f"B-feedback: controller ablation  ({len(base_results)} seed(s), ±1 std)"
    )
    ax.set_title(plot_title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_controller_state(
    ctrl_results: List[Dict],
    out_jpg: Path,
) -> None:
    """Plot controller internal-state traces (LR, trunk_scale, end_ε, imbalance) vs episode."""
    hists = [r.get("controller_history", []) for r in ctrl_results]
    hists = [h for h in hists if h]
    if not hists:
        print("[ctrl state plot] no controller_history — skipping.", flush=True)
        return

    starts = [float(h[0]["episode"]) for h in hists]
    ends   = [float(h[-1]["episode"]) for h in hists]
    lo, hi = max(starts), min(ends)
    if hi <= lo:
        print("[ctrl state plot] insufficient episode overlap — skipping.", flush=True)
        return
    grid = np.linspace(lo, hi, 512)

    keys   = ["effective_lr", "trunk_scale", "effective_end_e", "imbalance_ema"]
    labels = ["Effective LR", "Trunk scale", "Eff. end ε", "Imbalance EMA"]
    tab10  = plt.get_cmap("tab10")
    thin_c = [tab10(i % 10) for i in range(len(hists))]

    fig, axes = plt.subplots(
        len(keys), 1, figsize=(9, 3 * len(keys)), sharex=True, constrained_layout=True
    )
    for ax, key, ylabel in zip(axes, keys, labels):
        rows = np.vstack([_interp(h, "episode", key, grid) for h in hists])
        if rows.shape[0] > 1:
            for i, row in enumerate(rows):
                ax.plot(grid, row, color=thin_c[i], lw=0.6, alpha=0.45)
            ax.plot(grid, np.nanmean(rows, axis=0), lw=2.2, color="#1f77b4", label="seed mean")
        else:
            ax.plot(grid, rows[0], lw=2.2, color="#1f77b4")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Episode", fontsize=11)
    axes[0].set_title(
        f"Controller state traces  ({len(hists)} seed(s))", fontsize=11
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _print_summary(base: List[Dict], gate: List[Dict]) -> None:
    def _agg(results: List[Dict], label: str) -> None:
        vals = [r["last100_mean"] for r in results]
        mu   = float(np.mean(vals))
        sd   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {label:30s}: {mu:.2f} ± {sd:.2f}  (n={len(vals)})", flush=True)
    print("\n=== Final-100-episode summary ===", flush=True)
    _agg(base, "b_feedb_cg_det baseline")
    _agg(gate, "b_feedb_cg_det + conf. gate")


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: Path, seed: int, gated: bool, ctrl: bool = False) -> Path:
    if ctrl:
        tag = "b_feedb_cg_det_ctrl"
    elif gated:
        tag = "b_feedb_cg_det_gate"
    else:
        tag = "b_feedb_cg_det"
    return ckpt_dir / f"{tag}_seed{seed}.pt"


def _load_result(ckpt_dir: Path, seed: int, gated: bool, ctrl: bool = False) -> Dict:
    path = _ckpt_path(ckpt_dir, seed, gated, ctrl)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hist      = ckpt.get("gate_history", [])
    ctrl_hist = ckpt.get("controller_history", [])
    ep        = ckpt["episode_returns"]
    last100   = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep)) if ep else 0.0
    return {
        "seed":               seed,
        "confidence_gating":  gated,
        "use_controller":     ctrl,
        "episode_returns":    ep,
        "gate_history":       hist,
        "controller_history": ctrl_hist,
        "last100_mean":       last100,
    }


def _ckpt_path_v(ckpt_dir: Path, seed: int, ctrl_tag: str) -> Path:
    """Checkpoint path for a named controller variant (e.g. ctrl_tag='_lr')."""
    return ckpt_dir / f"b_feedb_cg_ctrl{ctrl_tag}_seed{seed}.pt"


def _load_result_v(ckpt_dir: Path, seed: int, ctrl_tag: str) -> Dict:
    """Load a named controller variant checkpoint."""
    path = _ckpt_path_v(ckpt_dir, seed, ctrl_tag)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep     = ckpt["episode_returns"]
    last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep)) if ep else 0.0
    return {
        "seed":               seed,
        "episode_returns":    ep,
        "gate_history":       ckpt.get("gate_history", []),
        "controller_history": ckpt.get("controller_history", []),
        "last100_mean":       last100,
    }


# ---------------------------------------------------------------------------
# Option-comparison helpers
# ---------------------------------------------------------------------------

# Five conditions used by --compare-options
#   tag           : string appended to checkpoint filename
#   label         : legend label in the plot
#   color         : hex colour
#   gating        : confidence_gating flag for run_one
#   gate_k_min    : lower gate bound
#   gate_tau      : softmax temperature
#   target_tau    : Polyak coefficient (1.0 = hard copy)
#   target_freq   : how often to update target net (1 = every step)
#   main_grad_clip: max-norm clip on opt_main gradients (0.0 = off)
OPT_CONDITIONS: List[Dict] = [
    dict(tag="baseline", label="Baseline (no gate)",
         color="#2ca02c", gating=False,
         gate_k_min=0.8,  gate_tau=1.0, target_tau=1.0,   target_freq=500, main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="gate_std", label="Gate standard (k_min=0.8, τ=1.0)",
         color="#9467bd", gating=True,
         gate_k_min=0.8,  gate_tau=1.0, target_tau=1.0,   target_freq=500, main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt1",     label="Opt 1: wide+sharp gate (k_min=0.3, τ=0.5)",
         color="#1f77b4", gating=True,
         gate_k_min=0.3,  gate_tau=0.5, target_tau=1.0,   target_freq=500, main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt2",     label="Opt 2: soft target (τ_poly=0.005)",
         color="#d62728", gating=True,
         gate_k_min=0.8,  gate_tau=1.0, target_tau=0.005, target_freq=1,   main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt12",    label="Opt 1+2: wide gate + soft target",
         color="#ff7f0e", gating=True,
         gate_k_min=0.3,  gate_tau=0.5, target_tau=0.005, target_freq=1,   main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt2_gc",  label="Opt 2+GC: soft target + grad clip (max=10)",
         color="#8c564b", gating=True,
         gate_k_min=0.8,  gate_tau=1.0, target_tau=0.005, target_freq=1,   main_grad_clip=10.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
]

# ---------------------------------------------------------------------------
# Fast-convergence conditions (--compare-fast):
#   Change 1: learning_starts 10000 → 1000
#   Change 2: train_frequency  10   → 4
#   Change 3: epsilon decay episodes set via --epsilon-decay-episodes (suggested 800)
# Two conditions for the focused comparison:
#   baseline_fast : no gating, same fast params  (benchmark)
#   opt2_fast     : gating + soft Polyak target + fast params  (our method)
FAST_CONDITIONS: List[Dict] = [
    dict(tag="baseline_fast", label="Baseline fast (no gate)",
         color="#2ca02c", gating=False,
         gate_k_min=0.8,  gate_tau=1.0, target_tau=1.0,   target_freq=500,
         main_grad_clip=0.0, learning_starts=1_000, train_frequency=4),
    dict(tag="opt2_fast",     label="Opt 2 fast (soft target + fast params)",
         color="#d62728", gating=True,
         gate_k_min=0.8,  gate_tau=1.0, target_tau=0.005, target_freq=1,
         main_grad_clip=0.0, learning_starts=1_000, train_frequency=4),
]


def _ckpt_path_opt(ckpt_dir: Path, seed: int, tag: str) -> Path:
    return ckpt_dir / f"b_feedb_cg_{tag}_seed{seed}.pt"


def _load_result_opt(ckpt_dir: Path, seed: int, tag: str) -> Dict:
    path = _ckpt_path_opt(ckpt_dir, seed, tag)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ep = ckpt["episode_returns"]
    last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else (float(np.mean(ep)) if ep else 0.0)
    return {
        "seed":            seed,
        "episode_returns": ep,
        "gate_history":    ckpt.get("gate_history", []),
        "last100_mean":    last100,
    }


def plot_options_comparison(
    conditions: List[Dict],
    results_per_cond: List[List[Dict]],
    out_jpg: Path,
    n_seeds: int,
) -> None:
    """Plot learning curves for all option-comparison conditions."""
    # Find common episode length across all conditions
    all_arrays = []
    for res_list in results_per_cond:
        all_arrays.extend([np.asarray(r["episode_returns"]) for r in res_list])
    n = min(len(a) for a in all_arrays)
    W = max(1, n // 200)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    for cond, res_list in zip(conditions, results_per_cond):
        arrays = [np.asarray(r["episode_returns"])[:n] for r in res_list]
        M   = np.stack(arrays, axis=0)
        mu  = M.mean(0)
        sd  = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
        ep_sm = np.arange(1, n + 1)[W - 1:]
        sm_mu = _smooth(mu, W)
        sm_sd = _smooth(sd, W)
        l100  = float(np.mean([r["last100_mean"] for r in res_list]))
        col   = cond["color"]
        ax.fill_between(ep_sm, sm_mu - sm_sd, sm_mu + sm_sd,
                        color=col, alpha=0.12, linewidth=0)
        ax.plot(ep_sm, sm_mu, color=col, lw=2.2,
                label=f"{cond['label']}  last-100: {l100:.1f}")

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"B-feedback conf-gate — option comparison  ({n_seeds} seed(s), ±1 std)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_seeds(s: str) -> List[int]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=EPSILON_DECAY_EPISODES,
        metavar="N",
        help="linear epsilon decay from START_E to END_E over N completed episodes",
    )
    p.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        metavar="E",
        help="stop after E completed episodes (uses max(--total-timesteps, 15M) as step safety cap)",
    )
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--train-missing-only", action="store_true",
                   help="Skip seeds whose checkpoint exists already")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip training; regenerate figures from existing checkpoints")
    p.add_argument("--controller", action="store_true",
                   help="Run gate-controller experiment: trains baseline+gated+controller "
                        "and saves figures with '_controller' suffix")
    p.add_argument("--control-b", action="store_true",
                   help="(Option A) Allow controller to modulate B matrix; default False")
    p.add_argument("--compare-options", action="store_true",
                   help="Run 6-condition option comparison: baseline / gate-std / opt1 / opt2 / opt1+2 / opt2+gc "
                        "and save the comparison figure to paper/figures")
    p.add_argument("--compare-opt2", action="store_true",
                   help="Run focused 2-condition comparison: baseline vs opt2 (soft target). "
                        "Reuses existing checkpoints from --compare-options runs.")
    p.add_argument("--compare-fast", action="store_true",
                   help="Run fast-convergence experiment: baseline_fast vs opt2_fast. "
                        "Uses LEARNING_STARTS=1000, TRAIN_FREQUENCY=4, plus opt2 soft target. "
                        "Suggested: --epsilon-decay-episodes 800 --total-timesteps 200000")
    p.add_argument("--compare-ctrl-opt2", action="store_true",
                   help="Run controller single-knob ablation on top of Option 2 (soft Polyak "
                        "target τ=0.005, updated every step).  Mirrors --controller but all runs "
                        "use target_tau=0.005 / target_freq=1.")
    args = p.parse_args()

    seeds  = _parse_seeds(args.seeds)
    eps_decay_ep = args.epsilon_decay_episodes
    n_ep_target = args.total_episodes
    if n_ep_target is not None:
        n_step = max(args.total_timesteps, 15_000_000)
    else:
        n_step = args.total_timesteps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg_det" / f"ckpt_decay{eps_decay_ep}"

    cb_tag    = "_ctrlB" if args.control_b else ""
    decay_tag = f"decay{eps_decay_ep}"

    print("=" * 60, flush=True)
    print("b_feedback_confgate_detached  (W_z B-only, detached from backprop)", flush=True)
    print(f"  Seeds      : {seeds}", flush=True)
    if n_ep_target is not None:
        print(f"  Episodes   : {n_ep_target:,} (step cap {n_step:,})", flush=True)
    else:
        print(f"  Timesteps  : {n_step:,}", flush=True)
    print(f"  Eps decay  : {eps_decay_ep} episodes (START_E -> END_E)", flush=True)
    print(f"  Device     : {device}", flush=True)
    if args.controller:
        print(f"  Mode       : CONTROLLER (control_B={args.control_b})", flush=True)
    if args.compare_options:
        print(f"  Mode       : COMPARE-OPTIONS (6 conditions)", flush=True)
    if args.compare_opt2:
        print(f"  Mode       : COMPARE-OPT2 (baseline vs opt2, focused)", flush=True)
    if args.compare_fast:
        print(f"  Mode       : COMPARE-FAST (baseline_fast vs opt2_fast)", flush=True)
    if args.compare_ctrl_opt2:
        print(f"  Mode       : COMPARE-CTRL-OPT2 (controller ablation + soft target)", flush=True)
    print(f"  Checkpoints: {ckpt_dir}", flush=True)
    print("=" * 60, flush=True)

    # -----------------------------------------------------------------------
    # Fast-convergence experiment branch (--compare-fast)
    # Trains baseline_fast and opt2_fast with:
    #   learning_starts=1000, train_frequency=4, eps_decay via CLI (suggest 800)
    # -----------------------------------------------------------------------
    if args.compare_fast:
        fast_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_fast_{decay_tag}"

        if not args.plot_only:
            for cond in FAST_CONDITIONS:
                tag = cond["tag"]
                algo_tag_str = f"b_feedb_cg_{tag}"
                seeds_to_run = seeds
                if args.train_missing_only:
                    seeds_to_run = [
                        s for s in seeds
                        if not _ckpt_path_opt(fast_ckpt_dir, s, tag).exists()
                    ]
                if not seeds_to_run:
                    print(f">>> {tag.upper()}: all checkpoints present, skipping.", flush=True)
                    continue
                if n_ep_target is not None:
                    print(
                        f"\n>>> [{tag.upper()}] Training seeds {seeds_to_run} "
                        f"for up to {n_ep_target:,} episodes (step cap {n_step:,}) ...",
                        flush=True,
                    )
                else:
                    print(
                        f"\n>>> [{tag.upper()}] Training seeds {seeds_to_run} "
                        f"for {n_step:,} steps ...",
                        flush=True,
                    )
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, cond["gating"], device,
                        checkpoint_dir=fast_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        gate_k_min=cond["gate_k_min"],
                        gate_tau=cond["gate_tau"],
                        target_tau=cond["target_tau"],
                        target_freq=cond["target_freq"],
                        main_grad_clip=cond.get("main_grad_clip", 0.0),
                        learning_starts=cond["learning_starts"],
                        train_frequency=cond["train_frequency"],
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{tag.upper()}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        # Verify
        for cond in FAST_CONDITIONS:
            for s in seeds:
                p_ck = _ckpt_path_opt(fast_ckpt_dir, s, cond["tag"])
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")

        fast_res = [
            [_load_result_opt(fast_ckpt_dir, s, cond["tag"]) for s in seeds]
            for cond in FAST_CONDITIONS
        ]

        print("\n=== Final-100-episode summary (fast comparison) ===", flush=True)
        for cond, res_list in zip(FAST_CONDITIONS, fast_res):
            _agg_print(res_list, cond["label"])

        n_str = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        plot_options_comparison(
            FAST_CONDITIONS,
            fast_res,
            FIG_DIR / f"b_feedback_fast_{n_str}_{decay_tag}.jpg",
            n_seeds=len(seeds),
        )
        return

    # -----------------------------------------------------------------------
    # Focused opt2 experiment branch (--compare-opt2)
    # Trains / loads only "baseline" and "opt2" conditions, produces a clean
    # 2-curve figure.  Checkpoint dir shared with --compare-options so that
    # seeds already trained there are automatically reused when
    # --train-missing-only is set.
    # -----------------------------------------------------------------------
    if args.compare_opt2:
        opt2_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_opts_{decay_tag}"
        focused_conds = [c for c in OPT_CONDITIONS if c["tag"] in {"baseline", "opt2"}]

        if not args.plot_only:
            for cond in focused_conds:
                tag = cond["tag"]
                algo_tag_str = f"b_feedb_cg_{tag}"
                seeds_to_run = seeds
                if args.train_missing_only:
                    seeds_to_run = [
                        s for s in seeds
                        if not _ckpt_path_opt(opt2_ckpt_dir, s, tag).exists()
                    ]
                if not seeds_to_run:
                    print(f">>> {tag.upper()}: all checkpoints present, skipping.", flush=True)
                    continue
                if n_ep_target is not None:
                    print(
                        f"\n>>> [{tag.upper()}] Training seeds {seeds_to_run} "
                        f"for up to {n_ep_target:,} episodes (step cap {n_step:,}) ...",
                        flush=True,
                    )
                else:
                    print(
                        f"\n>>> [{tag.upper()}] Training seeds {seeds_to_run} "
                        f"for {n_step:,} steps ...",
                        flush=True,
                    )
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, cond["gating"], device,
                        checkpoint_dir=opt2_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        gate_k_min=cond["gate_k_min"],
                        gate_tau=cond["gate_tau"],
                        target_tau=cond["target_tau"],
                        target_freq=cond["target_freq"],
                        main_grad_clip=cond.get("main_grad_clip", 0.0),
                        learning_starts=cond.get("learning_starts", LEARNING_STARTS),
                        train_frequency=cond.get("train_frequency", TRAIN_FREQUENCY),
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{tag.upper()}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        # Verify
        for cond in focused_conds:
            for s in seeds:
                p_ck = _ckpt_path_opt(opt2_ckpt_dir, s, cond["tag"])
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")

        focused_res = [
            [_load_result_opt(opt2_ckpt_dir, s, cond["tag"]) for s in seeds]
            for cond in focused_conds
        ]

        print("\n=== Final-100-episode summary (baseline vs opt2) ===", flush=True)
        for cond, res_list in zip(focused_conds, focused_res):
            _agg_print(res_list, cond["label"])

        n_str = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        plot_options_comparison(
            focused_conds,
            focused_res,
            FIG_DIR / f"b_feedback_opt2_{n_str}_{decay_tag}.jpg",
            n_seeds=len(seeds),
        )
        return

    # -----------------------------------------------------------------------
    # Compare-options experiment branch (--compare-options)
    # -----------------------------------------------------------------------
    if args.compare_options:
        opt_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_opts_{decay_tag}"

        if not args.plot_only:
            for cond in OPT_CONDITIONS:
                tag = cond["tag"]
                algo_tag_str = f"b_feedb_cg_{tag}"
                seeds_to_run = seeds
                if args.train_missing_only:
                    seeds_to_run = [
                        s for s in seeds
                        if not _ckpt_path_opt(opt_ckpt_dir, s, tag).exists()
                    ]
                if not seeds_to_run:
                    print(f">>> {tag.upper()}: all checkpoints present, skipping.", flush=True)
                    continue

                if n_ep_target is not None:
                    print(
                        f"\n>>> [{tag.upper()}] Training seeds {seeds_to_run} "
                        f"for up to {n_ep_target:,} episodes (step cap {n_step:,}) ...",
                        flush=True,
                    )
                else:
                    print(
                        f"\n>>> [{tag.upper()}] Training seeds {seeds_to_run} "
                        f"for {n_step:,} steps ...",
                        flush=True,
                    )
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, cond["gating"], device,
                        checkpoint_dir=opt_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        gate_k_min=cond["gate_k_min"],
                        gate_tau=cond["gate_tau"],
                        target_tau=cond["target_tau"],
                        target_freq=cond["target_freq"],
                        main_grad_clip=cond.get("main_grad_clip", 0.0),
                        learning_starts=cond.get("learning_starts", LEARNING_STARTS),
                        train_frequency=cond.get("train_frequency", TRAIN_FREQUENCY),
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{tag.upper()}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        # Verify all checkpoints exist
        for cond in OPT_CONDITIONS:
            for s in seeds:
                p_ck = _ckpt_path_opt(opt_ckpt_dir, s, cond["tag"])
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")

        # Load results
        results_per_cond = [
            [_load_result_opt(opt_ckpt_dir, s, cond["tag"]) for s in seeds]
            for cond in OPT_CONDITIONS
        ]

        print("\n=== Final-100-episode summary (option comparison) ===", flush=True)
        for cond, res_list in zip(OPT_CONDITIONS, results_per_cond):
            _agg_print(res_list, cond["label"])

        n_str = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        # Full 6-condition plot
        plot_options_comparison(
            OPT_CONDITIONS,
            results_per_cond,
            FIG_DIR / f"b_feedback_options_{n_str}_{decay_tag}.jpg",
            n_seeds=len(seeds),
        )
        # Focused 3-condition plot: baseline vs opt2 vs opt2+gc
        focused_tags  = {"baseline", "opt2", "opt2_gc"}
        focused_conds = [c for c in OPT_CONDITIONS if c["tag"] in focused_tags]
        focused_res   = [results_per_cond[i] for i, c in enumerate(OPT_CONDITIONS) if c["tag"] in focused_tags]
        if len(focused_conds) == 3:
            plot_options_comparison(
                focused_conds,
                focused_res,
                FIG_DIR / f"b_feedback_opt2gc_focused_{n_str}_{decay_tag}.jpg",
                n_seeds=len(seeds),
            )
        return

    # -----------------------------------------------------------------------
    # Opt2-Controller ablation branch (--compare-ctrl-opt2)
    # Same single-knob controller ablation as --controller but every run uses
    # Option 2's soft Polyak target (target_tau=0.005, target_freq=1).
    # Conditions:
    #   baseline_o2   : no gate, soft target
    #   gated_o2      : confidence gate, soft target, no controller
    #   ctrl_all_o2   : gate + soft target + all controller knobs
    #   + 4 single-knob variants (_wz, _lr, _trunk, _eps) with soft target
    # -----------------------------------------------------------------------
    if args.compare_ctrl_opt2:
        co2_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_ctrl_opt2_{decay_tag}{cb_tag}"
        OPT2_TARGET_TAU  = 0.005
        OPT2_TARGET_FREQ = 1

        def _ckpt_co2_base(seed: int, tag: str) -> Path:
            return co2_ckpt_dir / f"{tag}_seed{seed}.pt"

        def _ckpt_co2_var(seed: int, ctrl_tag: str) -> Path:
            return co2_ckpt_dir / f"b_feedb_cg_ctrl{ctrl_tag}_seed{seed}.pt"

        def _load_co2_base(seed: int, tag: str) -> Dict:
            ckpt = torch.load(_ckpt_co2_base(seed, tag), map_location="cpu", weights_only=False)
            ep = ckpt["episode_returns"]
            return {
                "seed": seed,
                "episode_returns": ep,
                "gate_history": ckpt.get("gate_history", []),
                "controller_history": ckpt.get("controller_history", []),
                "last100_mean": float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep)) if ep else 0.0,
            }

        def _load_co2_var(seed: int, ctrl_tag: str) -> Dict:
            return _load_result_v(co2_ckpt_dir, seed, ctrl_tag)

        # Base conditions: (override_tag, label, gated, use_controller)
        co2_base_conds = [
            ("b_feedb_cg_o2_baseline",   False, False),
            ("b_feedb_cg_o2_gate",       True,  False),
            ("b_feedb_cg_ctrl_o2_all",   True,  True),
        ]

        co2_variant_conds = [
            ("_wz",    "CTRL_WZ_ONLY",    "#17becf",
             dict(control_Wz=True,  control_trunk=False, control_epsilon=False,
                  control_lr=False, use_original_formulas=False)),
            ("_lr",    "CTRL_LR_ONLY",    "#e377c2",
             dict(control_Wz=False, control_trunk=False, control_epsilon=False,
                  control_lr=True, use_original_formulas=True)),
            ("_trunk", "CTRL_TRUNK_ONLY", "#8c564b",
             dict(control_Wz=False, control_trunk=True,  control_epsilon=False,
                  control_lr=False, use_original_formulas=True)),
            ("_eps",   "CTRL_EPS_ONLY",   "#ff7f0e",
             dict(control_Wz=False, control_trunk=False, control_epsilon=True,
                  control_lr=False, use_original_formulas=True)),
        ]

        if not args.plot_only:
            # --- base conditions ---
            for algo_tag_str, gated, use_ctrl in co2_base_conds:
                lbl = algo_tag_str.upper()
                seeds_to_run = [s for s in seeds if not _ckpt_co2_base(s, algo_tag_str).exists()] \
                               if args.train_missing_only else list(seeds)
                if not seeds_to_run:
                    print(f">>> {lbl}: checkpoints present, skipping.", flush=True)
                    continue
                print(f"\n>>> [{lbl}] Training seeds {seeds_to_run} for {n_step:,} steps ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, gated, device,
                        checkpoint_dir=co2_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        use_controller=use_ctrl,
                        control_B=args.control_b,
                        target_tau=OPT2_TARGET_TAU,
                        target_freq=OPT2_TARGET_FREQ,
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

            # --- single-knob variants ---
            for v_tag, v_lbl, _, v_knobs in co2_variant_conds:
                seeds_to_run = [s for s in seeds if not _ckpt_co2_var(s, v_tag).exists()] \
                               if args.train_missing_only else list(seeds)
                if not seeds_to_run:
                    print(f">>> {v_lbl}: checkpoints present, skipping.", flush=True)
                    continue
                print(f"\n>>> [{v_lbl}] Training seeds {seeds_to_run} for {n_step:,} steps ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=co2_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        use_controller=True,
                        control_B=args.control_b,
                        ctrl_tag=v_tag,
                        ctrl_knobs=v_knobs,
                        target_tau=OPT2_TARGET_TAU,
                        target_freq=OPT2_TARGET_FREQ,
                    )
                print(f">>> [{v_lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        # --- verify checkpoints ---
        for algo_tag_str, _, _ in co2_base_conds:
            for seed in seeds:
                p_ck = _ckpt_co2_base(seed, algo_tag_str)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")
        for v_tag, v_lbl, _, _ in co2_variant_conds:
            for seed in seeds:
                p_ck = _ckpt_co2_var(seed, v_tag)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing variant checkpoint: {p_ck}")

        # --- load results ---
        n_str = f"{len(seeds)}seed"
        base_tag, gate_tag, ctrl_all_tag = [t for t, _, _ in co2_base_conds]
        base_res     = [_load_co2_base(s, base_tag)     for s in seeds]
        gate_res     = [_load_co2_base(s, gate_tag)     for s in seeds]
        ctrl_all_res = [_load_co2_base(s, ctrl_all_tag) for s in seeds]
        variant_res  = [
            ([_load_co2_var(s, v_tag) for s in seeds], v_col, f"ctrl ({v_lbl[5:].lower()})")
            for v_tag, v_lbl, v_col, _ in co2_variant_conds
        ]

        print("\n=== Final-100-episode summary (Opt2 controller ablation) ===", flush=True)
        for res_list, lbl in [
            (base_res, "baseline_o2"), (gate_res, "gated_o2"), (ctrl_all_res, "ctrl_all_o2"),
        ] + [(r, lbl) for r, _, lbl in variant_res]:
            means = [r["last100_mean"] for r in res_list]
            print(f"  {lbl:30s}  last100={np.mean(means):.1f} ± {np.std(means):.1f}", flush=True)

        out_single = FIG_DIR / f"b_feedback_ctrl_opt2_{n_str}_{decay_tag}.jpg"
        plot_controller_comparison(
            base_res, gate_res, ctrl_all_res,
            out_single,
            extra_series=variant_res,
            color_base=COLOR_BASE,
            color_gate=COLOR_GATE,
            color_ctrl="#d62728",
            title=f"B-feedback Opt2 (soft target τ=0.005): controller single-knob ablation "
                  f"({len(seeds)} seed(s), ε-decay={eps_decay_ep} ep)",
        )
        return

    # -----------------------------------------------------------------------
    # Controller experiment branch
    # -----------------------------------------------------------------------
    if args.controller:
        ctrl_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_decay{eps_decay_ep}_ctrl{cb_tag}"

        def need_ctrl(gated: bool, ctrl: bool) -> List[int]:
            if args.train_missing_only:
                return [s for s in seeds if not _ckpt_path(ctrl_ckpt_dir, s, gated, ctrl).exists()]
            return list(seeds)

        conditions = [
            (False, False, "BASELINE"),
            (True,  False, "GATED"),
            (True,  True,  "CONTROLLER"),
        ]

        # Ablation variants: one control knob active at a time, original pre-fix formulas.
        # Each uses (ctrl_tag, label, color, knobs_dict)
        variant_conditions = [
            ("_wz",    "CTRL_WZ_ONLY",    "#17becf",  # cyan
             dict(control_Wz=True,  control_trunk=False, control_epsilon=False,
                  control_lr=False, use_original_formulas=False)),
            ("_lr",    "CTRL_LR_ONLY",    "#e377c2",  # pink
             dict(control_Wz=False, control_trunk=False, control_epsilon=False,
                  control_lr=True, use_original_formulas=True)),
            ("_trunk", "CTRL_TRUNK_ONLY", "#8c564b",  # brown
             dict(control_Wz=False, control_trunk=True,  control_epsilon=False,
                  control_lr=False, use_original_formulas=True)),
            ("_eps",   "CTRL_EPS_ONLY",   "#ff7f0e",  # orange
             dict(control_Wz=False, control_trunk=False, control_epsilon=True,
                  control_lr=False, use_original_formulas=True)),
        ]

        # Two-/three-knob combinations (all with original formulas so LR/Eps modulate actively)
        combo_conditions = [
            ("_eps_wz",    "CTRL_EPS_WZ",    "#17becf",  # cyan
             dict(control_Wz=True,  control_trunk=False, control_epsilon=True,
                  control_lr=False, use_original_formulas=True)),
            ("_eps_lr",    "CTRL_EPS_LR",    "#e377c2",  # pink
             dict(control_Wz=False, control_trunk=False, control_epsilon=True,
                  control_lr=True,  use_original_formulas=True)),
            ("_lr_wz",     "CTRL_LR_WZ",     "#bcbd22",  # olive
             dict(control_Wz=True,  control_trunk=False, control_epsilon=False,
                  control_lr=True,  use_original_formulas=True)),
            ("_eps_lr_wz", "CTRL_EPS_LR_WZ", "#984ea3",  # violet
             dict(control_Wz=True,  control_trunk=False, control_epsilon=True,
                  control_lr=True,  use_original_formulas=True)),
        ]

        if not args.plot_only:
            for gated, ctrl, lbl in conditions:
                seeds_to_run = need_ctrl(gated, ctrl)
                if not seeds_to_run:
                    print(f">>> {lbl}: all checkpoints present, skipping.", flush=True)
                    continue
                if n_ep_target is not None:
                    print(
                        f"\n>>> [{lbl}] Training seeds {seeds_to_run} "
                        f"for up to {n_ep_target:,} episodes (step cap {n_step:,}) ...",
                        flush=True,
                    )
                else:
                    print(
                        f"\n>>> [{lbl}] Training seeds {seeds_to_run} for {n_step:,} steps ...",
                        flush=True,
                    )
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, gated, device,
                        checkpoint_dir=ctrl_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        use_controller=ctrl,
                        control_B=args.control_b,
                    )
                print(f">>> [{lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

            # Train ablation variants
            for v_tag, v_lbl, _, v_knobs in variant_conditions:
                seeds_to_run = [s for s in seeds
                                if not _ckpt_path_v(ctrl_ckpt_dir, s, v_tag).exists()] \
                               if args.train_missing_only else list(seeds)
                if not seeds_to_run:
                    print(f">>> {v_lbl}: all checkpoints present, skipping.", flush=True)
                    continue
                if n_ep_target is not None:
                    print(f"\n>>> [{v_lbl}] Training seeds {seeds_to_run} "
                          f"for up to {n_ep_target:,} episodes ...", flush=True)
                else:
                    print(f"\n>>> [{v_lbl}] Training seeds {seeds_to_run} "
                          f"for {n_step:,} steps ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=ctrl_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        use_controller=True,
                        ctrl_tag=v_tag,
                        ctrl_knobs=v_knobs,
                    )
                print(f">>> [{v_lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        if not args.plot_only:
            # Train combination variants
            for v_tag, v_lbl, _, v_knobs in combo_conditions:
                seeds_to_run = [s for s in seeds
                                if not _ckpt_path_v(ctrl_ckpt_dir, s, v_tag).exists()] \
                               if args.train_missing_only else list(seeds)
                if not seeds_to_run:
                    print(f">>> {v_lbl}: all checkpoints present, skipping.", flush=True)
                    continue
                if n_ep_target is not None:
                    print(f"\n>>> [{v_lbl}] Training seeds {seeds_to_run} "
                          f"for up to {n_ep_target:,} episodes ...", flush=True)
                else:
                    print(f"\n>>> [{v_lbl}] Training seeds {seeds_to_run} "
                          f"for {n_step:,} steps ...", flush=True)
                t0 = time.perf_counter()
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=ctrl_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        use_controller=True,
                        ctrl_tag=v_tag,
                        ctrl_knobs=v_knobs,
                    )
                print(f">>> [{v_lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        # verify main conditions
        for seed in seeds:
            for gated, ctrl, _ in conditions:
                p_ck = _ckpt_path(ctrl_ckpt_dir, seed, gated, ctrl)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")
        # verify variants
        for v_tag, v_lbl, _, _ in variant_conditions:
            for seed in seeds:
                p_ck = _ckpt_path_v(ctrl_ckpt_dir, seed, v_tag)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing variant checkpoint: {p_ck}")
        # verify combo variants
        for v_tag, v_lbl, _, _ in combo_conditions:
            for seed in seeds:
                p_ck = _ckpt_path_v(ctrl_ckpt_dir, seed, v_tag)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing combo checkpoint: {p_ck}")

        base_res = [_load_result(ctrl_ckpt_dir, s, False, False) for s in seeds]
        gate_res = [_load_result(ctrl_ckpt_dir, s, True,  False) for s in seeds]
        ctrl_res = [_load_result(ctrl_ckpt_dir, s, True,  True)  for s in seeds]
        variant_res = [
            ([_load_result_v(ctrl_ckpt_dir, s, v_tag) for s in seeds], v_col, f"ctrl ({v_lbl[5:].lower()})")
            for v_tag, v_lbl, v_col, _ in variant_conditions
        ]
        combo_res = [
            ([_load_result_v(ctrl_ckpt_dir, s, v_tag) for s in seeds], v_col, f"ctrl ({v_lbl[5:].lower()})")
            for v_tag, v_lbl, v_col, _ in combo_conditions
        ]

        print("\n=== Final-100-episode summary (controller experiment) ===", flush=True)
        _agg_print(base_res, "b_feedb_cg baseline")
        _agg_print(gate_res, "b_feedb_cg + conf. gate")
        _agg_print(ctrl_res, "b_feedb_cg + controller (all)")
        for res_list, _, v_lbl in variant_res:
            _agg_print(res_list, f"b_feedb_cg + {v_lbl}")
        print("--- combination variants ---", flush=True)
        for res_list, _, v_lbl in combo_res:
            _agg_print(res_list, f"b_feedb_cg + {v_lbl}")

        n_str = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        plot_controller_comparison(
            base_res, gate_res, ctrl_res,
            FIG_DIR / f"b_feedback_{n_str}_{decay_tag}{cb_tag}_controller.jpg",
            extra_series=variant_res,
        )
        # New graph: combination variants (distinct color per line)
        plot_controller_comparison(
            base_res, gate_res, ctrl_res,
            FIG_DIR / f"b_feedback_{n_str}_{decay_tag}{cb_tag}_controller_combos.jpg",
            extra_series=combo_res,
            color_base="#1f77b4",   # blue
            color_gate="#ff7f0e",   # orange
            color_ctrl="#8c564b",   # brown
            title=(
                f"B-feedback: controller combination ablation  "
                f"({len(seeds)} seed(s), ±1 std)"
            ),
        )
        plot_controller_state(
            ctrl_res,
            FIG_DIR / f"b_feedback_controller_state_{n_str}_{decay_tag}{cb_tag}_controller.jpg",
        )
        plot_gate_traces(
            ctrl_res,
            FIG_DIR / f"b_feedback_controller_gate_{n_str}_{decay_tag}{cb_tag}_controller.jpg",
        )
        return

    # -----------------------------------------------------------------------
    # Default (non-controller) experiment
    # -----------------------------------------------------------------------
    def need(gated: bool) -> List[int]:
        if args.train_missing_only:
            return [s for s in seeds if not _ckpt_path(ckpt_dir, s, gated).exists()]
        return list(seeds)

    if not args.plot_only:
        for gated in (False, True):
            label = "GATED" if gated else "BASELINE"
            seeds_to_run = need(gated)
            if not seeds_to_run:
                print(f">>> {label}: all checkpoints present, skipping.", flush=True)
                continue
            if n_ep_target is not None:
                print(
                    f"\n>>> [{label}] Training seeds {seeds_to_run} "
                    f"for up to {n_ep_target:,} episodes (step cap {n_step:,}) ...",
                    flush=True,
                )
            else:
                print(f"\n>>> [{label}] Training seeds {seeds_to_run} for {n_step:,} steps ...", flush=True)
            t0 = time.perf_counter()
            for seed in seeds_to_run:
                run_one(
                    seed,
                    n_step,
                    gated,
                    device,
                    checkpoint_dir=ckpt_dir,
                    epsilon_decay_episodes=eps_decay_ep,
                    total_episodes=n_ep_target,
                )
            print(f">>> [{label}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

    # verify checkpoints
    for seed in seeds:
        for gated in (False, True):
            p_ck = _ckpt_path(ckpt_dir, seed, gated)
            if not p_ck.exists():
                raise FileNotFoundError(f"Missing checkpoint: {p_ck}")

    base_res = [_load_result(ckpt_dir, s, False) for s in seeds]
    gate_res = [_load_result(ckpt_dir, s, True)  for s in seeds]

    _print_summary(base_res, gate_res)

    n_str = f"{len(seeds)}seeds"
    plot_learning_curves(
        base_res,
        gate_res,
        FIG_DIR / f"b_feedback_confgate_detached_{n_str}_{decay_tag}.jpg",
    )
    plot_gate_traces(
        gate_res,
        FIG_DIR / f"b_feedback_confgate_detached_gate_kcart_kpole_{n_str}_{decay_tag}.jpg",
    )


if __name__ == "__main__":
    main()
