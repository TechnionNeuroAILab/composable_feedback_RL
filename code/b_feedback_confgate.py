"""
b_feedback_confgate.py
======================
Three-head B-feedback with confidence gating on W_z — fixed convergence variant.

Architecture:  obs -> linear_feature (W_z, 4->120) -> trunk (ReLU->84->ReLU) -> head_full (84->2)
                                                     -> head_cart (84->2)   [cart-only obs]
                                                     -> head_pole (84->2)   [pole-only obs]

Key difference vs mh_3a
------------------------
  mh_3a:  linear_feature (W_z) is EXCLUDED from Adam; updated only by manual B-feedback
           using all three head errors (full + cart + pole) → cross-head interference,
           seed-sensitive convergence, mean return often stalls below 500.

  HERE:   linear_feature IS included in Adam (with trunk + head_full) so W_z gets a
           reliable end-to-end Q_full TD gradient → all seeds converge to 500 reliably.
           B-feedback is applied as a BONUS using only the full-head TD error (gradient-
           aligned, no cross-head interference).
           Aux heads (cart/pole) use a separate Adam optimizer; trunk gradient is detached
           for them (Option A separation), so they train independently.
           Confidence gating: margins from Q_cart/Q_pole → softmax → k in [0.8,1.0] →
           scale the cart/pole compartments of W_z feeding Q_full, exactly as in mh_3a.

Usage:
  python code/b_feedback_confgate.py                                 # 10 seeds, 500k steps
  python code/b_feedback_confgate.py --seeds 1,2,3 --total-timesteps 500000
  python code/b_feedback_confgate.py --seeds 1 --total-episodes 5000  # decay default 3500 ep
  python code/b_feedback_confgate.py --plot-only                     # regen figures from ckpt_decay<N>
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
LR_LINEAR_MIXED     = 1e-4      # B-feedback bonus LR (small — additive nudge only)
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

DEFAULT_SEEDS         = list(range(1, 11))
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
      - linear_feature (W_z) + trunk + head_full share one Adam optimizer
        → W_z receives a reliable end-to-end Q_full TD gradient.
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
    ):
        super().__init__()
        self.obs_dim            = obs_dim
        self.n_actions          = n_actions
        self.confidence_gating  = confidence_gating
        self.gate_k_min         = gate_k_min
        self.gate_k_max         = gate_k_max

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
        C = F.softmax(torch.stack([m_c, m_p], dim=1), dim=1).detach()
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
        Q_full path: z (gated if enabled) -> trunk (gradient flows to W_z) -> head_full.
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

        # --- full-state head (W_z gradient flows, gated if enabled) ---
        if self._gating_active():
            z_full = self._gated_z(obs, k4, k5)
        else:
            z_full = self.linear_feature(obs)
        z_trunk = self.trunk(z_full)
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

class GateController:
    """
    Uses the confidence-gate output (k4, k5) to dynamically modulate training knobs.

    Controlled knobs (all True by default except control_B):
      control_B       (default False) – Option A: scale B-matrix compartments by gate EMA;
                                        when False the fixed all-ones B is used unchanged.
      control_Wz      (default True)  – widen gate_k_min early, tighten with training progress.
      control_trunk   (default True)  – scale trunk output via q_net.trunk_scale in [0.6, 1.0].
      control_epsilon (default True)  – adjust epsilon decay speed and exploration floor.
      control_lr      (default True)  – scale main-optimizer LR by gate imbalance EMA.
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
    ):
        self.q_net            = q_net
        self.opt_main         = opt_main
        self.total_timesteps  = total_timesteps
        self.control_B        = control_B
        self.control_Wz       = control_Wz
        self.control_trunk    = control_trunk
        self.control_epsilon  = control_epsilon
        self.control_lr       = control_lr
        self.ema_alpha        = ema_alpha
        self._imbalance_ema      = 0.0
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

        # --- Wz gate range: widen early, tighten with training progress ---
        if self.control_Wz:
            progress = t / max(self.total_timesteps, 1)
            self.q_net.gate_k_min = 0.5 + 0.3 * progress   # 0.5 → 0.8

        # --- Trunk scale: confident (high imbalance) → full capacity; ambiguous → regularise ---
        if self.control_trunk:
            self.q_net.trunk_scale = 0.6 + 0.4 * self._imbalance_ema

        # --- LR scaling ---
        if self.control_lr:
            self._effective_lr = LEARNING_RATE * (0.5 + 0.5 * self._imbalance_ema)
            for g in self.opt_main.param_groups:
                g["lr"] = self._effective_lr

        # --- Epsilon overrides ---
        if self.control_epsilon:
            self._effective_end_e    = END_E + (1.0 - self._imbalance_ema) * 0.05
            self._effective_decay_ep = EPSILON_DECAY_EPISODES * (
                1.0 + (1.0 - self._imbalance_ema) * 0.5
            )

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

    q_net  = BFeedbackConfGateNetwork(obs_dim, n_actions, confidence_gating=gate_ok).to(device)
    t_net  = BFeedbackConfGateNetwork(obs_dim, n_actions, confidence_gating=gate_ok).to(device)
    t_net.load_state_dict(q_net.state_dict())

    # W_z + trunk + head_full share one optimizer (key fix vs mh_3a)
    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters()),
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
        controller = GateController(
            q_net, opt_main, total_timesteps, control_B=control_B
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
            episode_returns.append(float(infos["episode"]["r"]))

        obs = next_obs
        if done:
            obs, _ = env.reset()

        # --- training step ---
        if t > LEARNING_STARTS and t % TRAIN_FREQUENCY == 0:
            data = rb.sample(BATCH_SIZE)

            # ---- shared TD target (from target network, Q_full path) ----
            with torch.no_grad():
                tQ_full, tQ_cart, tQ_pole, _, _ = t_net.forward(data.next_observations)
                r = data.rewards.flatten()
                d = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            # ---- forward (W_z gradient connected for Q_full path) ----
            Q_full, Q_cart, Q_pole, k4, k5 = q_net.forward(data.observations)

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            # ---- controller step: update B_mat + all enabled knobs ----
            if controller is not None:
                B_mat = controller.step(k4, k5, t, device, torch.float32)
                _eps_end   = controller.effective_end_e
                _eps_decay = controller.effective_decay_ep

            # ---- B-feedback bonus on W_z: only full-head error (gradient-aligned) ----
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

            # ---- end-to-end backprop through W_z (main fix) ----
            opt_main.zero_grad()
            loss_full.backward()
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

        # ---- target network hard update ----
        if t % TARGET_NETWORK_FREQ == 0:
            for tp, qp in zip(t_net.parameters(), q_net.parameters()):
                tp.data.copy_(TAU * qp.data + (1 - TAU) * tp.data)

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [b_feedb_cg] {label} step={t+1}/{total_timesteps}"
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
    if ctrl_active:
        algo_tag = "b_feedb_cg_ctrl"
    elif gate_ok:
        algo_tag = "b_feedb_cg_gate"
    else:
        algo_tag = "b_feedb_cg"

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
        f"B-feedback + conf. gate (3-head, W_z in Adam): "
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
) -> None:
    """Three-way learning curve: baseline vs confidence-gated vs controller."""
    COLOR_CTRL = "#d62728"   # red

    def _prep(results: List[Dict]):
        arrays = [np.asarray(r["episode_returns"]) for r in results]
        n = min(len(a) for a in arrays)
        M = np.stack([a[:n] for a in arrays], axis=0)
        mu = M.mean(0)
        sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
        return np.arange(1, n + 1), mu, sd

    ep_b, mu_b, sd_b = _prep(base_results)
    ep_g, mu_g, sd_g = _prep(gate_results)
    ep_c, mu_c, sd_c = _prep(ctrl_results)
    n = min(len(ep_b), len(ep_g), len(ep_c))
    W = max(1, n // 200)

    def _sm(ep, mu, sd):
        return ep[W - 1:], _smooth(mu, W), _smooth(sd, W)

    ep_b, sm_b, sd_b = _sm(ep_b[:n], mu_b[:n], sd_b[:n])
    ep_g, sm_g, sd_g = _sm(ep_g[:n], mu_g[:n], sd_g[:n])
    ep_c, sm_c, sd_c = _sm(ep_c[:n], mu_c[:n], sd_c[:n])

    last100_b = float(np.mean([r["last100_mean"] for r in base_results]))
    last100_g = float(np.mean([r["last100_mean"] for r in gate_results]))
    last100_c = float(np.mean([r["last100_mean"] for r in ctrl_results]))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for ep, sm, sd, col, lbl, l100 in [
        (ep_b, sm_b, sd_b, COLOR_BASE, "baseline",          last100_b),
        (ep_g, sm_g, sd_g, COLOR_GATE, "confidence-gated",  last100_g),
        (ep_c, sm_c, sd_c, COLOR_CTRL, "controller",        last100_c),
    ]:
        ax.fill_between(ep, sm - sd, sm + sd, color=col, alpha=0.15, linewidth=0)
        ax.plot(ep, sm, color=col, lw=2.2, label=f"{lbl}  last-100: {l100:.1f}")
    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"B-feedback: baseline vs gated vs controller  ({len(base_results)} seed(s), ±1 std)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10)
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
    _agg(base, "b_feedb_cg baseline")
    _agg(gate, "b_feedb_cg + conf. gate")


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: Path, seed: int, gated: bool, ctrl: bool = False) -> Path:
    if ctrl:
        tag = "b_feedb_cg_ctrl"
    elif gated:
        tag = "b_feedb_cg_gate"
    else:
        tag = "b_feedb_cg"
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
    args = p.parse_args()

    seeds  = _parse_seeds(args.seeds)
    eps_decay_ep = args.epsilon_decay_episodes
    n_ep_target = args.total_episodes
    if n_ep_target is not None:
        n_step = max(args.total_timesteps, 15_000_000)
    else:
        n_step = args.total_timesteps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_decay{eps_decay_ep}"

    cb_tag    = "_ctrlB" if args.control_b else ""
    decay_tag = f"decay{eps_decay_ep}"

    print("=" * 60, flush=True)
    print("b_feedback_confgate  (W_z in Adam + B-bonus + conf. gate)", flush=True)
    print(f"  Seeds      : {seeds}", flush=True)
    if n_ep_target is not None:
        print(f"  Episodes   : {n_ep_target:,} (step cap {n_step:,})", flush=True)
    else:
        print(f"  Timesteps  : {n_step:,}", flush=True)
    print(f"  Eps decay  : {eps_decay_ep} episodes (START_E -> END_E)", flush=True)
    print(f"  Device     : {device}", flush=True)
    if args.controller:
        print(f"  Mode       : CONTROLLER (control_B={args.control_b})", flush=True)
    print(f"  Checkpoints: {ckpt_dir}", flush=True)
    print("=" * 60, flush=True)

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

        # verify
        for seed in seeds:
            for gated, ctrl, _ in conditions:
                p_ck = _ckpt_path(ctrl_ckpt_dir, seed, gated, ctrl)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")

        base_res = [_load_result(ctrl_ckpt_dir, s, False, False) for s in seeds]
        gate_res = [_load_result(ctrl_ckpt_dir, s, True,  False) for s in seeds]
        ctrl_res = [_load_result(ctrl_ckpt_dir, s, True,  True)  for s in seeds]

        print("\n=== Final-100-episode summary (controller experiment) ===", flush=True)
        _agg_print(base_res, "b_feedb_cg baseline")
        _agg_print(gate_res, "b_feedb_cg + conf. gate")
        _agg_print(ctrl_res, "b_feedb_cg + controller")

        n_str = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        plot_controller_comparison(
            base_res, gate_res, ctrl_res,
            FIG_DIR / f"b_feedback_{n_str}_{decay_tag}{cb_tag}_controller.jpg",
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
        FIG_DIR / f"b_feedback_confgate_{n_str}_{decay_tag}.jpg",
    )
    plot_gate_traces(
        gate_res,
        FIG_DIR / f"b_feedback_confgate_gate_kcart_kpole_{n_str}_{decay_tag}.jpg",
    )


if __name__ == "__main__":
    main()
