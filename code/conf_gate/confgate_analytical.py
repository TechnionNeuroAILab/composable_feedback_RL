"""
Three-head confidence gating on W_z — full backprop variant.

Architecture:  obs -> linear_feature (W_z, 4->120) -> trunk (ReLU->84->ReLU) -> head_full (84->2)
                                                     -> head_cart (84->2)   [cart-only obs]
                                                     -> head_pole (84->2)   [pole-only obs]

 HERE:    linear_feature is included in Adam (with trunk + head_full) so W_z gets a
           reliable end-to-end Q_full TD gradient → all seeds converge to 500 reliably.
           Aux heads (cart/pole) use a separate Adam optimizer; trunk gradient is detached
           for them (Option A separation), so they train independently.
           Confidence gating: Z-normalized margins from Q_cart/Q_pole → softmax → k in [0.8,1.0] →
           scale the cart/pole compartments of W_z feeding Q_full.
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
FIG_DIR = ROOT / "paper" / "figures" / "z_normed_m_fullbackprop"

# ---------------------------------------------------------------------------
# Hyper-parameters (match CleanRL / run_cleanrl_vs_mixed_dqn.py)
# ---------------------------------------------------------------------------
LEARNING_RATE       = 2.5e-4
GAMMA               = 0.99
BUFFER_SIZE         = 10_000
BATCH_SIZE          = 128
START_E             = 1.0
END_E               = 0.05
EPSILON_DECAY_EPISODES = 3_500   # linear START_E → END_E over this many completed episodes
LEARNING_STARTS     = 10_000
TRAIN_FREQUENCY     = 10
TARGET_NETWORK_FREQ = 500
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
# Network
# ---------------------------------------------------------------------------

class BFeedbackConfGateNetwork(nn.Module):
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
        self.trunk_scale: float = 1.0

    @staticmethod
    def _mask_cart(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 2:4] = 0.0; return out

    @staticmethod
    def _mask_pole(obs: torch.Tensor) -> torch.Tensor:
        out = obs.clone(); out[..., 0:2] = 0.0; return out

    def _gating_active(self) -> bool:
        return self.confidence_gating and self.obs_dim == 4 and self.n_actions == 2

    def _gate_k(self, Q_cart: torch.Tensor, Q_pole: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Returns (k4, k5) shaped (B,1) and a dict of margin stats, detached."""
        # Raw Margins
        m_c_raw = (Q_cart[:, 1] - Q_cart[:, 0]).abs()
        m_p_raw = (Q_pole[:, 1] - Q_pole[:, 0]).abs()

        # Batch Mean and Std for Z-Normalization and logging
        mc_mean, mc_std = m_c_raw.mean(), m_c_raw.std(unbiased=False)
        mp_mean, mp_std = m_p_raw.mean(), m_p_raw.std(unbiased=False)

        # Z-Normalization
        eps = 1e-8
        m_c_norm = (m_c_raw - mc_mean) / (mc_std + eps)
        m_p_norm = (m_p_raw - mp_mean) / (mp_std + eps)

        # Softmax Routing on Normalized Margins
        C = F.softmax(torch.stack([m_c_norm, m_p_norm], dim=1), dim=1).detach()
        km, kx = self.gate_k_min, self.gate_k_max
        k4 = km + (kx - km) * C[:, 0:1]
        k5 = km + (kx - km) * C[:, 1:2]

        stats = {
            "mc_mean": float(mc_mean.item()), "mc_std": float(mc_std.item()),
            "mp_mean": float(mp_mean.item()), "mp_std": float(mp_std.item()),
            "norm_sum_mean": float((m_c_norm + m_p_norm).mean().item()),
            "norm_sum_abs_mean": float((m_c_norm + m_p_norm).abs().mean().item()),
        }
        return k4, k5, stats

    def _gated_z(self, obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
        W = self.linear_feature.weight
        b = self.linear_feature.bias
        return k4 * (obs[..., 0:2] @ W[:, 0:2].T) + k5 * (obs[..., 2:4] @ W[:, 2:4].T) + b

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict]]:
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart.detach()))

        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole.detach()))

        B = obs.shape[0]
        k4 = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        k5 = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        m_stats = None

        if self._gating_active():
            k4, k5, m_stats = self._gate_k(Q_cart, Q_pole)

        if self._gating_active():
            z_full = self._gated_z(obs, k4, k5)
        else:
            z_full = self.linear_feature(obs)
            
        z_trunk = self.trunk(z_full)
        if self.trunk_scale != 1.0:
            z_trunk = z_trunk * self.trunk_scale
        Q_full = self.head_full(z_trunk)

        return Q_full, Q_cart, Q_pole, k4, k5, m_stats

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        Q_full, _, _, _, _, _ = self.forward(obs)
        return Q_full

    @torch.no_grad()
    def gate_k_no_grad(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        if not self._gating_active():
            ones = torch.ones(obs.shape, 1, device=obs.device, dtype=obs.dtype)
            return ones, ones.clone(), None
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
    Uses the confidence-gate output and margin statistics to dynamically modulate training knobs.
    m is computed directly each step as the batch-mean absolute sum of Z-normalized margins:
    m = mean_i|m̃_cart_i + m̃_pole_i| / 2  (no clipping).
    """

    def __init__(
        self,
        q_net: BFeedbackConfGateNetwork,
        opt_main: optim.Optimizer,
        total_timesteps: int,
        control_Wz: bool = True,
        control_trunk: bool = True,
        control_epsilon: bool = True,
        control_lr: bool = True,
        use_original_formulas: bool = False,
    ):
        self.q_net             = q_net
        self.opt_main          = opt_main
        self.total_timesteps   = total_timesteps
        self.control_Wz        = control_Wz
        self.control_trunk     = control_trunk
        self.control_epsilon   = control_epsilon
        self.control_lr        = control_lr
        self.use_original_formulas = use_original_formulas

        self._k4_mean          = 1.0
        self._k5_mean          = 1.0
        self._m                = 0.0  # Current control signal

        self._effective_lr       = LEARNING_RATE
        self._effective_end_e    = END_E
        self._effective_decay_ep = float(EPSILON_DECAY_EPISODES)

    def step(
        self,
        k4: torch.Tensor,
        k5: torch.Tensor,
        m_stats: Optional[Dict[str, float]],
        t: int,
        ep_done: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        k4m = float(k4.mean().item())
        k5m = float(k5.mean().item())
        self._k4_mean = k4m
        self._k5_mean = k5m

        # Derive control signal m directly from Z-normalized margins (no EMA, no clip)
        if m_stats is not None:
            raw = float(m_stats.get("norm_sum_abs_mean", 0.0))
            self._m = raw / 2.0

        m = self._m

        # Apply Smart Controls using the stabilized `m` metric
        if self.control_lr:
            self._effective_lr = LEARNING_RATE * (0.5 + 0.5 * m)
            for g in self.opt_main.param_groups:
                g["lr"] = self._effective_lr

        if self.control_epsilon:
            self._effective_end_e = END_E + (1.0 - m) * 0.05
            self._effective_decay_ep = float(EPSILON_DECAY_EPISODES)
            
        if self.control_Wz:
            self.q_net.gate_k_min = min(0.9, 0.8 + 0.2 * m)
            
        if self.control_trunk:
            self.q_net.trunk_scale = 1.0

    @property
    def effective_end_e(self) -> float:
        return self._effective_end_e

    @property
    def effective_decay_ep(self) -> float:
        return self._effective_decay_ep

    def log_state(self) -> Dict:
        return {
            "m":                  self._m,
            "k4_mean":            self._k4_mean,
            "k5_mean":            self._k5_mean,
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
    ctrl_tag: str = "",
    ctrl_knobs: Optional[Dict] = None,
    # Option-comparison knobs
    gate_k_min: float = 0.8,
    gate_k_max: float = 1.0,
    target_freq: int = TARGET_NETWORK_FREQ,
    algo_tag_override: Optional[str] = None,
    main_grad_clip: float = 0.0,      # max-norm clip for opt_main gradients (0 = off)
    learning_starts: int = LEARNING_STARTS,   # steps of pure collection before training
    train_frequency: int = TRAIN_FREQUENCY,   # env steps between gradient updates
) -> Dict:
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
    ).to(device)
    t_net  = BFeedbackConfGateNetwork(
        obs_dim, n_actions,
        confidence_gating=gate_ok,
        gate_k_min=gate_k_min,
        gate_k_max=gate_k_max,
    ).to(device)
    t_net.load_state_dict(q_net.state_dict())

    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters()),
        lr=LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )

    controller: Optional[GateController] = None
    if use_controller and gate_ok:
        extra_kwargs = ctrl_knobs if ctrl_knobs is not None else {}
        controller = GateController(
            q_net, opt_main, total_timesteps, **extra_kwargs
        )

    rb = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape, device)

    episode_returns: List[float]    = []
    gate_history: List[Dict]        = []
    controller_history: List[Dict]  = []
    episode_diagnostics: List[Dict] = []
    loss_list: List[float]          = []

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
            ep_idx = len(episode_returns)
            eps_val = _linear_schedule(START_E, _eps_end, int(_eps_decay), ep_idx - 1)
            lr_val = float(opt_main.param_groups[0]["lr"])
            wz = q_net.linear_feature.weight.data
            ep_entry: Dict = {
                "episode": ep_idx,
                "epsilon": eps_val,
                "lr": lr_val,
                "wz_mean": float(wz.abs().mean().item()),
            }
            if gate_ok and len(rb) >= 1:
                with torch.no_grad():
                    bs = min(BATCH_SIZE, len(rb))
                    diag_data = rb.sample(bs)
                    _, _, _, k4_d, k5_d, m_stats_d = q_net.forward(diag_data.observations)
                if m_stats_d is not None:
                    ep_entry.update({
                        "mean_k4": float(k4_d.mean().item()),
                        "mean_k5": float(k5_d.mean().item()),
                        "m": float(m_stats_d["norm_sum_abs_mean"] / 2.0),
                        "m1": float(m_stats_d["mc_mean"]),
                        "m2": float(m_stats_d["mp_mean"]),
                    })
            episode_diagnostics.append(ep_entry)

        obs = next_obs
        if done:
            obs, _ = env.reset()

        # --- training step ---
        if t > learning_starts and t % train_frequency == 0:
            data = rb.sample(BATCH_SIZE)

            with torch.no_grad():
                tQ_full, tQ_cart, tQ_pole, _, _, _ = t_net.forward(data.next_observations)
                r = data.rewards.flatten()
                d = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            Q_full, Q_cart, Q_pole, k4, k5, m_stats = q_net.forward(data.observations)

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            if controller is not None:
                controller.step(k4, k5, m_stats, t, len(episode_returns), device, torch.float32)
                _eps_end   = controller.effective_end_e
                _eps_decay = controller.effective_decay_ep

            opt_main.zero_grad()
            loss_full.backward()
            if main_grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(q_net.linear_feature.parameters())
                    + list(q_net.trunk.parameters())
                    + list(q_net.head_full.parameters()),
                    max_norm=main_grad_clip,
                )
            opt_main.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            loss_list.append(loss_full.item())

            if gate_ok and t % LOG_EVERY == 0:
                m_val = float(m_stats["norm_sum_abs_mean"] / 2.0) if m_stats else 0.0
                gate_history.append({
                    "step":    t,
                    "episode": len(episode_returns),
                    "mean_k4": float(k4.mean().item()),
                    "mean_k5": float(k5.mean().item()),
                    "m":       m_val,
                    "mc_mean": m_stats["mc_mean"] if m_stats else 0.0,
                    "mc_std":  m_stats["mc_std"] if m_stats else 0.0,
                    "mp_mean": m_stats["mp_mean"] if m_stats else 0.0,
                    "mp_std":  m_stats["mp_std"] if m_stats else 0.0,
                    "lr":      float(opt_main.param_groups[0]["lr"]),
                    "epsilon": _linear_schedule(
                        START_E, _eps_end, int(_eps_decay), len(episode_returns)
                    ),
                    "wz_mean": float(q_net.linear_feature.weight.data.abs().mean().item()),
                })
            if controller is not None and t % LOG_EVERY == 0:
                cs = controller.log_state()
                cs.update({"step": t, "episode": len(episode_returns)})
                controller_history.append(cs)

        if t % target_freq == 0:
            t_net.load_state_dict(q_net.state_dict())

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
    if algo_tag_override is not None:
        algo_tag = algo_tag_override
    elif ctrl_active:
        algo_tag = f"b_feedb_cg_ctrl{ctrl_tag}"
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
            "episode_diagnostics": episode_diagnostics,
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
        "episode_diagnostics": episode_diagnostics,
        "last100_mean":       last100,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid") if w >= 2 else arr


def plot_baseline_vs_controller(
    base_results: List[Dict],
    ctrl_results: List[Dict],
    out_jpg: Path,
    decay_tag: str,
    n_ep: Optional[int] = None,
) -> None:
    """Baseline (no gate) vs controller (unclipped m)."""
    def _prep_sm(results: List[Dict]):
        arrays = [np.asarray(r["episode_returns"]) for r in results]
        n = min(len(a) for a in arrays)
        M = np.stack([a[:n] for a in arrays], axis=0)
        mu = M.mean(0)
        ep = np.arange(1, n + 1)
        W = max(1, n // 200)
        return ep[W - 1:n], _smooth(mu[:n], W)

    ep_b, sm_b = _prep_sm(base_results)
    ep_c, sm_c = _prep_sm(ctrl_results)

    l100_b = float(np.mean([r["last100_mean"] for r in base_results]))
    l100_c = float(np.mean([r["last100_mean"] for r in ctrl_results]))

    ep_str = f"{n_ep:,} ep" if n_ep else decay_tag
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(ep_b, sm_b, color=COLOR_BASE, lw=2.2,
            label=f"baseline (no gate)  last-100: {l100_b:.1f}")
    ax.plot(ep_c, sm_c, color=COLOR_GATE, lw=2.2,
            label=f"controller (unclipped m)  last-100: {l100_c:.1f}")
    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"Conf. gate (full backprop): baseline vs controller ({len(base_results)} seed(s), {ep_str}, ±1 std)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


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
        f"Conf. gate (3-head, W_z in Adam): "
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


def plot_margin_stats(gate_results: List[Dict], out_jpg: Path) -> None:
    """Plots the raw M stats (mean and std) to sanity check Z-normalization."""
    hists = [r["gate_history"] for r in gate_results if r.get("gate_history")]
    if not hists:
        return

    starts = [float(h[0]["episode"]) for h in hists]
    ends   = [float(h[-1]["episode"]) for h in hists]
    grid = np.linspace(max(starts), min(ends), 512)

    MC_mu = np.vstack([_interp(h, "episode", "mc_mean", grid) for h in hists])
    MC_sd = np.vstack([_interp(h, "episode", "mc_std", grid) for h in hists])
    MP_mu = np.vstack([_interp(h, "episode", "mp_mean", grid) for h in hists])
    MP_sd = np.vstack([_interp(h, "episode", "mp_std", grid) for h in hists])

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        fig.patch.set_facecolor('#121212')
        
        c_cart = "#00ffcc" # Cyan
        c_pole = "#ff00ff" # Magenta
        
        for ax in axes:
            ax.set_facecolor('#121212')
            ax.grid(True, color='#333333', alpha=0.6, linestyle='--')

        axes[0].plot(grid, np.nanmean(MC_mu, axis=0), color=c_cart, lw=2.5, label=r"Cart M Mean ($\mu$)")
        axes[0].fill_between(grid, np.nanmean(MC_mu - MC_sd, axis=0), np.nanmean(MC_mu + MC_sd, axis=0), color=c_cart, alpha=0.15)
        axes[0].set_title("Cart Head: Pre-Normalized Margins (M)", color='white')
        axes[0].legend(loc="upper left")

        axes[1].plot(grid, np.nanmean(MP_mu, axis=0), color=c_pole, lw=2.5, label=r"Pole M Mean ($\mu$)")
        axes[1].fill_between(grid, np.nanmean(MP_mu - MP_sd, axis=0), np.nanmean(MP_mu + MP_sd, axis=0), color=c_pole, alpha=0.15)
        axes[1].set_title("Pole Head: Pre-Normalized Margins (M)", color='white')
        axes[1].set_xlabel("Episode")
        axes[1].legend(loc="upper left")

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
    c_base = color_base or COLOR_BASE
    c_gate = color_gate or COLOR_GATE
    c_ctrl = color_ctrl or "#d62728"   

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
        f"Conf. gate: controller ablation  ({len(base_results)} seed(s), ±1 std)"
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
    """Plot controller internal-state traces (LR, trunk_scale, end_ε, m) vs episode."""
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

    keys   = ["effective_lr", "trunk_scale", "effective_end_e", "m"]
    labels = ["Effective LR", "Trunk scale", "Eff. end ε", "Margin control signal m"]
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
            ax.plot(grid, rows, lw=2.2, color="#1f77b4")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Episode", fontsize=11)
    axes.set_title(
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


def _diag_tag(n_seeds: int, decay_tag: str) -> str:
    return f"{n_seeds}seed{'s' if n_seeds > 1 else ''}_{decay_tag}"


def _episode_eps_schedule(n_ep: int, eps_decay: int = EPSILON_DECAY_EPISODES) -> np.ndarray:
    return np.array([
        _linear_schedule(START_E, END_E, eps_decay, ep - 1) for ep in range(1, n_ep + 1)
    ])


def _theta_vertical_deg(theta_rad: float) -> float:
    return abs(abs(np.rad2deg(theta_rad)) - 90.0)


@torch.no_grad()
def _collect_theta_margin_rollout(
    ckpt_path: Path,
    device: torch.device,
    n_episodes: int = 150,
    seed: int = 1,
) -> List[Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    q_net = BFeedbackConfGateNetwork(4, 2, confidence_gating=True).to(device)
    q_net.load_state_dict(ckpt["q_network"])
    q_net.eval()

    env = gym.make("CartPole-v1")
    records: List[Dict] = []
    obs, _ = env.reset(seed=seed)
    for ep in range(1, n_episodes + 1):
        terminated = truncated = False
        while not (terminated or truncated):
            theta_t = float(obs[2])
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, Q_cart, Q_pole, _, _, _ = q_net.forward(x)
            m_cart = float((Q_cart[0, 1] - Q_cart[0, 0]).abs().item())
            m_pole = float((Q_pole[0, 1] - Q_pole[0, 0]).abs().item())
            action = int(q_net.forward_q_only(x).argmax(dim=1).item())
            next_obs, _, terminated, truncated, _ = env.step(action)
            theta_next = float(next_obs[2])
            delta_vert = _theta_vertical_deg(theta_next) - _theta_vertical_deg(theta_t)
            records.append({
                "episode": ep,
                "m_cart": m_cart,
                "m_pole": m_pole,
                "delta_vert": delta_vert,
            })
            obs = next_obs
        obs, _ = env.reset()
    env.close()
    return records


def plot_diagnostics_all(
    base_result: Dict,
    gate_result: Dict,
    ckpt_dir: Path,
    seed: int,
    decay_tag: str,
    device: torch.device,
    eps_decay: int = EPSILON_DECAY_EPISODES,
) -> None:
    """Generate the seven diagnostic figures for one seed."""
    tag = _diag_tag(1, decay_tag)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_diag = base_result.get("episode_diagnostics") or []
    gate_diag = gate_result.get("episode_diagnostics") or []
    if not gate_diag:
        print("[diagnostics] gated episode_diagnostics missing — skipping diagnostic plots.", flush=True)
        return

    # --- 1) k_cart, k_pole, m vs episodes (gated) ---
    g_ep = np.array([d["episode"] for d in gate_diag if "mean_k4" in d], dtype=float)
    if len(g_ep):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(g_ep, [d["mean_k4"] for d in gate_diag if "mean_k4" in d],
                color="#1f77b4", lw=1.8, label=r"$k_{\mathrm{cart}}$")
        ax.plot(g_ep, [d["mean_k5"] for d in gate_diag if "mean_k5" in d],
                color="#ff7f0e", lw=1.8, label=r"$k_{\mathrm{pole}}$")
        ax.plot(g_ep, [d["m"] for d in gate_diag if "mean_k4" in d],
                color=COLOR_GATE, lw=1.8, label=r"$m$")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Gate weight / control signal", fontsize=12)
        ax.set_title(
            rf"$k_{{\mathrm{{cart}}}}$, $k_{{\mathrm{{pole}}}}$, and $m$ vs episode "
            rf"(seed {seed}, {decay_tag})",
            fontsize=11,
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        out1 = FIG_DIR / f"confgate_k_m_vs_episodes_{tag}.jpg"
        fig.savefig(out1, dpi=150, bbox_inches="tight", format="jpeg")
        plt.close(fig)
        print(f"Wrote {out1}", flush=True)

    # --- 2) epsilon baseline vs gated ---
    n_ep = max(len(base_diag), len(gate_diag))
    ep_axis = np.arange(1, n_ep + 1)
    eps_base = (
        np.array([d["epsilon"] for d in base_diag], dtype=float)
        if base_diag else _episode_eps_schedule(n_ep, eps_decay)
    )
    eps_gate = (
        np.array([d["epsilon"] for d in gate_diag], dtype=float)
        if gate_diag else _episode_eps_schedule(n_ep, eps_decay)
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(ep_axis[: len(eps_base)], eps_base, color=COLOR_BASE, lw=2.0,
            label="baseline (no gate)")
    ax.plot(ep_axis[: len(eps_gate)], eps_gate, color=COLOR_GATE, lw=2.0,
            label="confidence-gated")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(r"$\varepsilon$", fontsize=12)
    ax.set_title(rf"$\varepsilon$-greedy schedule (seed {seed}, {decay_tag})", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    out2 = FIG_DIR / f"confgate_epsilon_baseline_vs_gated_{tag}.jpg"
    fig.savefig(out2, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out2}", flush=True)

    # --- 3) lr (m-correlated) vs episodes ---
    m_arr = np.array([d["m"] for d in gate_diag if "m" in d], dtype=float)
    ep_m = np.array([d["episode"] for d in gate_diag if "m" in d], dtype=float)
    lr_corr = LEARNING_RATE * (0.5 + 0.5 * m_arr)
    lr_act = np.array([d["lr"] for d in gate_diag if "m" in d], dtype=float)
    if len(ep_m):
        fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax1.plot(ep_m, m_arr, color=COLOR_GATE, lw=1.8, label=r"$m$")
        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel(r"Control signal $m$", fontsize=12, color=COLOR_GATE)
        ax1.tick_params(axis="y", labelcolor=COLOR_GATE)
        ax2 = ax1.twinx()
        ax2.plot(ep_m, lr_corr, color="#d62728", lw=1.8,
                 label=r"$\eta(m)=\eta_0(0.5+0.5m)$")
        ax2.plot(ep_m, lr_act, color="gray", lw=1.0, ls="--", alpha=0.7,
                 label=r"Adam LR (actual)")
        ax2.set_ylabel(r"Learning rate $\eta$", fontsize=12, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        if len(m_arr) > 2:
            r = float(np.corrcoef(m_arr, lr_corr)[0, 1])
            ax1.set_title(
                rf"$m$ and $\eta(m)$ vs episode — Pearson $r={r:.3f}$ "
                rf"(seed {seed}, {decay_tag})",
                fontsize=11,
            )
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        out3 = FIG_DIR / f"confgate_lr_m_correlation_{tag}.jpg"
        fig.savefig(out3, dpi=150, bbox_inches="tight", format="jpeg")
        plt.close(fig)
        print(f"Wrote {out3}", flush=True)

    # --- 4) average |W_z| vs episodes ---
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if base_diag:
        ax.plot([d["episode"] for d in base_diag],
                [d["wz_mean"] for d in base_diag],
                color=COLOR_BASE, lw=1.8, label="baseline (no gate)")
    ax.plot([d["episode"] for d in gate_diag],
            [d["wz_mean"] for d in gate_diag],
            color=COLOR_GATE, lw=1.8, label="confidence-gated")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(r"Mean $|\,W_z\,|$", fontsize=12)
    ax.set_title(rf"Average $|W_z|$ vs episode (seed {seed}, {decay_tag})", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    out4 = FIG_DIR / f"confgate_wz_mean_vs_episodes_{tag}.jpg"
    fig.savefig(out4, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out4}", flush=True)

    # --- 5) m1, m2 early episodes ---
    early = [d for d in gate_diag if d.get("episode", 0) <= 50 and "m1" in d]
    if early:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot([d["episode"] for d in early], [d["m1"] for d in early],
                color="#1f77b4", lw=2.0, marker="o", ms=3,
                label=r"$M_{\mathrm{cart}}$ (raw $|Q_1-Q_0|$ mean)")
        ax.plot([d["episode"] for d in early], [d["m2"] for d in early],
                color="#ff7f0e", lw=2.0, marker="s", ms=3,
                label=r"$M_{\mathrm{pole}}$ (raw $|Q_1-Q_0|$ mean)")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Raw margin (batch mean)", fontsize=12)
        ax.set_title(rf"Raw margins $M_{{\mathrm{{cart}}}}$, $M_{{\mathrm{{pole}}}}$ "
                     rf"(episodes 1–50, seed {seed})", fontsize=11)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        out5 = FIG_DIR / f"confgate_m1_m2_early_{tag}.jpg"
        fig.savefig(out5, dpi=150, bbox_inches="tight", format="jpeg")
        plt.close(fig)
        print(f"Wrote {out5}", flush=True)

    # --- 6) lr early episodes ---
    early_lr = [d for d in gate_diag if d.get("episode", 0) <= 50]
    if early_lr:
        ep_e = np.array([d["episode"] for d in early_lr], dtype=float)
        m_e = np.array([d.get("m", 0.0) for d in early_lr], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(ep_e, [d["lr"] for d in early_lr], color=COLOR_BASE, lw=2.0,
                marker="o", ms=3, label="Adam LR (actual)")
        ax.plot(ep_e, LEARNING_RATE * (0.5 + 0.5 * m_e), color=COLOR_GATE, lw=2.0,
                marker="s", ms=3, label=r"$\eta(m)=\eta_0(0.5+0.5m)$")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel(r"Learning rate $\eta$", fontsize=12)
        ax.set_title(rf"Learning rate vs episode (episodes 1–50, seed {seed})", fontsize=11)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        out6 = FIG_DIR / f"confgate_lr_early_{tag}.jpg"
        fig.savefig(out6, dpi=150, bbox_inches="tight", format="jpeg")
        plt.close(fig)
        print(f"Wrote {out6}", flush=True)

    # --- 7) margins vs |theta|-to-90 delta ---
    ckpt_path = _ckpt_path(ckpt_dir, seed, gated=True)
    rollout = _collect_theta_margin_rollout(ckpt_path, device, n_episodes=150, seed=seed)
    if rollout:
        m_cart_r = np.array([r["m_cart"] for r in rollout])
        m_pole_r = np.array([r["m_pole"] for r in rollout])
        d_vert = np.array([r["delta_vert"] for r in rollout])
        ep_r = np.array([r["episode"] for r in rollout])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, margins, name, col in [
            (axes[0], m_cart_r, r"$M_{\mathrm{cart}}$", "#1f77b4"),
            (axes[1], m_pole_r, r"$M_{\mathrm{pole}}$", "#ff7f0e"),
        ]:
            sc = ax.scatter(d_vert, margins, c=ep_r, cmap="viridis", s=8, alpha=0.55)
            if len(d_vert) > 2:
                r_val = float(np.corrcoef(d_vert, margins)[0, 1])
                z = np.polyfit(d_vert, margins, 1)
                xs = np.linspace(d_vert.min(), d_vert.max(), 100)
                ax.plot(xs, np.poly1d(z)(xs), color=col, lw=1.5, ls="--",
                        label=rf"Pearson $r={r_val:.3f}$")
            ax.set_xlabel(
                r"$|\theta_{t+1}^{\circ}-90| - |\theta_t^{\circ}-90|$ (deg)", fontsize=11
            )
            ax.set_ylabel(name, fontsize=11)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=axes, label="Episode", shrink=0.85)
        fig.suptitle(
            rf"Q-value margins vs pole vertical-distance change "
            rf"(greedy rollout, seed {seed}, {decay_tag})",
            fontsize=11,
        )
        out7 = FIG_DIR / f"confgate_margin_vs_theta_delta_{tag}.jpg"
        fig.savefig(out7, dpi=150, bbox_inches="tight", format="jpeg")
        plt.close(fig)
        print(f"Wrote {out7}", flush=True)


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
        "episode_diagnostics": ckpt.get("episode_diagnostics", []),
        "last100_mean":       last100,
    }


def _ckpt_path_v(ckpt_dir: Path, seed: int, ctrl_tag: str) -> Path:
    return ckpt_dir / f"b_feedb_cg_ctrl{ctrl_tag}_seed{seed}.pt"


def _load_result_v(ckpt_dir: Path, seed: int, ctrl_tag: str) -> Dict:
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

OPT_CONDITIONS: List[Dict] = [
    dict(tag="baseline", label="Baseline (no gate)",
         color="#2ca02c", gating=False,
         gate_k_min=0.8,  target_freq=500, main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="gate_std", label="Gate standard (k_min=0.8)",
         color="#9467bd", gating=True,
         gate_k_min=0.8,  target_freq=500, main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt1",     label="Opt 1: wide gate (k_min=0.3)",
         color="#1f77b4", gating=True,
         gate_k_min=0.3,  target_freq=500, main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt2",     label="Opt 2: frequent target update",
         color="#d62728", gating=True,
         gate_k_min=0.8,  target_freq=1,   main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt12",    label="Opt 1+2: wide gate + frequent target update",
         color="#ff7f0e", gating=True,
         gate_k_min=0.3,  target_freq=1,   main_grad_clip=0.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
    dict(tag="opt2_gc",  label="Opt 2+GC: frequent target update + grad clip (max=10)",
         color="#8c564b", gating=True,
         gate_k_min=0.8,  target_freq=1,   main_grad_clip=10.0,
         learning_starts=LEARNING_STARTS, train_frequency=TRAIN_FREQUENCY),
]

FAST_CONDITIONS: List[Dict] = [
    dict(tag="baseline_fast", label="Baseline fast (no gate)",
         color="#2ca02c", gating=False,
         gate_k_min=0.8,  target_freq=500,
         main_grad_clip=0.0, learning_starts=1_000, train_frequency=4),
    dict(tag="opt2_fast",     label="Opt 2 fast (frequent target update + fast params)",
         color="#d62728", gating=True,
         gate_k_min=0.8,  target_freq=1,
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
        f"Conf. gate — option comparison  ({n_seeds} seed(s), ±1 std)",
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
    p.add_argument("--compare-options", action="store_true",
                   help="Run 6-condition option comparison: baseline / gate-std / opt1 / opt2 / opt1+2 / opt2+gc "
                        "and save the comparison figure to paper/figures")
    p.add_argument("--compare-opt2", action="store_true",
                   help="Run focused 2-condition comparison: baseline vs opt2 (frequent target update). "
                        "Reuses existing checkpoints from --compare-options runs.")
    p.add_argument("--compare-fast", action="store_true",
                   help="Run fast-convergence experiment: baseline_fast vs opt2_fast. "
                        "Uses LEARNING_STARTS=1000, TRAIN_FREQUENCY=4, plus opt2 frequent target update. "
                        "Suggested: --epsilon-decay-episodes 800 --total-timesteps 200000")
    p.add_argument("--compare-ctrl-opt2", action="store_true",
                   help="Run controller single-knob ablation on top of Option 2 (frequent "
                        "target update every step).  Mirrors --controller but all runs "
                        "use target_freq=1.")
    p.add_argument("--compare-ctrl-clip", action="store_true",
                   help="Compare baseline vs controller (unclipped m). "
                        "Saves figure to paper/figures/z_normed_m_fullbackprop/.")
    p.add_argument("--plot-diagnostics", action="store_true",
                   help="Generate diagnostic plots (k/m, epsilon, lr, Wz, early margins, "
                        "theta-margin correlation) for baseline vs gated runs.")
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

    decay_tag = f"decay{eps_decay_ep}"

    print("=" * 60, flush=True)
    print("confgate_analytical  (W_z in Adam + conf. gate, full backprop)", flush=True)
    print(f"  Seeds      : {seeds}", flush=True)
    if n_ep_target is not None:
        print(f"  Episodes   : {n_ep_target:,} (step cap {n_step:,})", flush=True)
    else:
        print(f"  Timesteps  : {n_step:,}", flush=True)
    print(f"  Eps decay  : {eps_decay_ep} episodes (START_E -> END_E)", flush=True)
    print(f"  Device     : {device}", flush=True)
    if args.controller:
        print(f"  Mode       : CONTROLLER", flush=True)
    if args.compare_options:
        print(f"  Mode       : COMPARE-OPTIONS (6 conditions)", flush=True)
    if args.compare_opt2:
        print(f"  Mode       : COMPARE-OPT2 (baseline vs opt2, focused)", flush=True)
    if args.compare_fast:
        print(f"  Mode       : COMPARE-FAST (baseline_fast vs opt2_fast)", flush=True)
    if args.compare_ctrl_opt2:
        print(f"  Mode       : COMPARE-CTRL-OPT2 (controller ablation + frequent target update)", flush=True)
    if args.compare_ctrl_clip:
        print(f"  Mode       : COMPARE-CTRL (baseline vs controller, unclipped m)", flush=True)
    print(f"  Checkpoints: {ckpt_dir}", flush=True)
    print("=" * 60, flush=True)

    if args.plot_diagnostics:
        if args.plot_only:
            for seed in seeds:
                for gated in (False, True):
                    if not _ckpt_path(ckpt_dir, seed, gated).exists():
                        raise FileNotFoundError(f"Missing checkpoint: {_ckpt_path(ckpt_dir, seed, gated)}")
                base_r = _load_result(ckpt_dir, seed, False)
                gate_r = _load_result(ckpt_dir, seed, True)
                if not gate_r.get("episode_diagnostics"):
                    print(
                        f"[diagnostics] seed {seed}: retraining gated "
                        "(checkpoint lacks episode_diagnostics) ...",
                        flush=True,
                    )
                    run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                    )
                    gate_r = _load_result(ckpt_dir, seed, True)
                if not base_r.get("episode_diagnostics"):
                    print(
                        f"[diagnostics] seed {seed}: retraining baseline "
                        "(checkpoint lacks episode_diagnostics) ...",
                        flush=True,
                    )
                    run_one(
                        seed, n_step, False, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                    )
                    base_r = _load_result(ckpt_dir, seed, False)
                plot_diagnostics_all(
                    base_r, gate_r, ckpt_dir, seed, decay_tag, device, eps_decay_ep,
                )
        else:
            for gated in (False, True):
                label = "GATED" if gated else "BASELINE"
                seeds_to_run = list(seeds)
                if args.train_missing_only:
                    seeds_to_run = [
                        s for s in seeds
                        if not _ckpt_path(ckpt_dir, s, gated).exists()
                    ]
                if not seeds_to_run:
                    print(f">>> {label}: checkpoints present, skipping train.", flush=True)
                    continue
                print(f"\n>>> [{label}] Training seeds {seeds_to_run} for diagnostics ...", flush=True)
                for seed in seeds_to_run:
                    run_one(
                        seed, n_step, gated, device,
                        checkpoint_dir=ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                    )
            for seed in seeds:
                base_r = _load_result(ckpt_dir, seed, False)
                gate_r = _load_result(ckpt_dir, seed, True)
                if not gate_r.get("episode_diagnostics") or not base_r.get("episode_diagnostics"):
                    raise RuntimeError(
                        f"seed {seed}: episode_diagnostics missing after training"
                    )
                plot_diagnostics_all(
                    base_r, gate_r, ckpt_dir, seed, decay_tag, device, eps_decay_ep,
                )
        return

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
                        target_freq=cond["target_freq"],
                        main_grad_clip=cond.get("main_grad_clip", 0.0),
                        learning_starts=cond["learning_starts"],
                        train_frequency=cond["train_frequency"],
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{tag.upper()}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

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
                        target_freq=cond["target_freq"],
                        main_grad_clip=cond.get("main_grad_clip", 0.0),
                        learning_starts=cond.get("learning_starts", LEARNING_STARTS),
                        train_frequency=cond.get("train_frequency", TRAIN_FREQUENCY),
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{tag.upper()}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

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
                        target_freq=cond["target_freq"],
                        main_grad_clip=cond.get("main_grad_clip", 0.0),
                        learning_starts=cond.get("learning_starts", LEARNING_STARTS),
                        train_frequency=cond.get("train_frequency", TRAIN_FREQUENCY),
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{tag.upper()}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

        for cond in OPT_CONDITIONS:
            for s in seeds:
                p_ck = _ckpt_path_opt(opt_ckpt_dir, s, cond["tag"])
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")

        results_per_cond = [
            [_load_result_opt(opt_ckpt_dir, s, cond["tag"]) for s in seeds]
            for cond in OPT_CONDITIONS
        ]

        print("\n=== Final-100-episode summary (option comparison) ===", flush=True)
        for cond, res_list in zip(OPT_CONDITIONS, results_per_cond):
            _agg_print(res_list, cond["label"])

        n_str = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        plot_options_comparison(
            OPT_CONDITIONS,
            results_per_cond,
            FIG_DIR / f"b_feedback_options_{n_str}_{decay_tag}.jpg",
            n_seeds=len(seeds),
        )
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

    if args.compare_ctrl_opt2:
        co2_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_ctrl_opt2_{decay_tag}"
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
                        target_freq=OPT2_TARGET_FREQ,
                        algo_tag_override=algo_tag_str,
                    )
                print(f">>> [{lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

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
                        ctrl_tag=v_tag,
                        ctrl_knobs=v_knobs,
                        target_freq=OPT2_TARGET_FREQ,
                    )
                print(f">>> [{v_lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

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
            title=f"Conf. gate Opt2 (frequent target update): controller single-knob ablation "
                  f"({len(seeds)} seed(s), ε-decay={eps_decay_ep} ep)",
        )
        return

    if args.compare_ctrl_clip:
        ep_tag = f"_ep{n_ep_target}" if n_ep_target is not None else ""
        ctrl_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_ctrl_{decay_tag}{ep_tag}"

        def _ckpt_ctrl_base(seed: int) -> Path:
            return ctrl_ckpt_dir / f"b_feedb_cg_seed{seed}.pt"

        def _ckpt_ctrl(seed: int) -> Path:
            return ctrl_ckpt_dir / f"b_feedb_cg_ctrl_seed{seed}.pt"

        def _load_ctrl_run(seed: int) -> Dict:
            ckpt = torch.load(_ckpt_ctrl(seed), map_location="cpu", weights_only=False)
            ep = ckpt["episode_returns"]
            last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep)) if ep else 0.0
            return {
                "seed": seed,
                "episode_returns": ep,
                "gate_history": ckpt.get("gate_history", []),
                "controller_history": ckpt.get("controller_history", []),
                "last100_mean": last100,
            }

        if not args.plot_only:
            base_seeds_to_run = list(seeds)
            if args.train_missing_only:
                base_seeds_to_run = [s for s in seeds if not _ckpt_ctrl_base(s).exists()]
            if base_seeds_to_run:
                ep_str = f"up to {n_ep_target:,} episodes" if n_ep_target else f"{n_step:,} steps"
                print(f"\n>>> [BASELINE] Training seeds {base_seeds_to_run} for {ep_str} ...", flush=True)
                t0 = time.perf_counter()
                for seed in base_seeds_to_run:
                    run_one(
                        seed, n_step, False, device,
                        checkpoint_dir=ctrl_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                    )
                print(f">>> [BASELINE] Done in {time.perf_counter() - t0:.1f}s", flush=True)
            else:
                print(">>> BASELINE: all checkpoints present, skipping.", flush=True)

            ctrl_seeds_to_run = list(seeds)
            if args.train_missing_only:
                ctrl_seeds_to_run = [s for s in seeds if not _ckpt_ctrl(s).exists()]
            if ctrl_seeds_to_run:
                ep_str = f"up to {n_ep_target:,} episodes" if n_ep_target else f"{n_step:,} steps"
                print(f"\n>>> [CONTROLLER] Training seeds {ctrl_seeds_to_run} for {ep_str} ...", flush=True)
                t0 = time.perf_counter()
                for seed in ctrl_seeds_to_run:
                    run_one(
                        seed, n_step, True, device,
                        checkpoint_dir=ctrl_ckpt_dir,
                        epsilon_decay_episodes=eps_decay_ep,
                        total_episodes=n_ep_target,
                        use_controller=True,
                        ctrl_tag="_ctrl",
                    )
                print(f">>> [CONTROLLER] Done in {time.perf_counter() - t0:.1f}s", flush=True)
            else:
                print(">>> CONTROLLER: all checkpoints present, skipping.", flush=True)

        for seed in seeds:
            if not _ckpt_ctrl_base(seed).exists():
                raise FileNotFoundError(f"Missing baseline checkpoint: {_ckpt_ctrl_base(seed)}")
            if not _ckpt_ctrl(seed).exists():
                raise FileNotFoundError(f"Missing controller checkpoint: {_ckpt_ctrl(seed)}")

        base_res  = [_load_result(ctrl_ckpt_dir, s, False) for s in seeds]
        ctrl_res  = [_load_ctrl_run(s) for s in seeds]

        print("\n=== Final-100-episode summary (baseline vs controller) ===", flush=True)
        _agg_print(base_res, "baseline (no gate)")
        _agg_print(ctrl_res, "controller (unclipped m)")

        n_str  = f"{len(seeds)}seed{'s' if len(seeds) > 1 else ''}"
        ep_tag2 = f"_ep{n_ep_target}" if n_ep_target is not None else ""
        plot_baseline_vs_controller(
            base_res,
            ctrl_res,
            FIG_DIR / f"b_feedback_ctrl_{n_str}_{decay_tag}{ep_tag2}.jpg",
            decay_tag,
            n_ep=n_ep_target,
        )
        return

    if args.controller:
        ctrl_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_decay{eps_decay_ep}_ctrl"

        def need_ctrl(gated: bool, ctrl: bool) -> List[int]:
            if args.train_missing_only:
                return [s for s in seeds if not _ckpt_path(ctrl_ckpt_dir, s, gated, ctrl).exists()]
            return list(seeds)

        conditions = [
            (False, False, "BASELINE"),
            (True,  False, "GATED"),
            (True,  True,  "CONTROLLER"),
        ]

        variant_conditions = [
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

        combo_conditions = [
            ("_eps_wz",    "CTRL_EPS_WZ",    "#17becf",  
             dict(control_Wz=True,  control_trunk=False, control_epsilon=True,
                  control_lr=False, use_original_formulas=True)),
            ("_eps_lr",    "CTRL_EPS_LR",    "#e377c2",  
             dict(control_Wz=False, control_trunk=False, control_epsilon=True,
                  control_lr=True,  use_original_formulas=True)),
            ("_lr_wz",     "CTRL_LR_WZ",     "#bcbd22",  
             dict(control_Wz=True,  control_trunk=False, control_epsilon=False,
                  control_lr=True,  use_original_formulas=True)),
            ("_eps_lr_wz", "CTRL_EPS_LR_WZ", "#984ea3",  
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
                    )
                print(f">>> [{lbl}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

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

        for seed in seeds:
            for gated, ctrl, _ in conditions:
                p_ck = _ckpt_path(ctrl_ckpt_dir, seed, gated, ctrl)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {p_ck}")
        for v_tag, v_lbl, _, _ in variant_conditions:
            for seed in seeds:
                p_ck = _ckpt_path_v(ctrl_ckpt_dir, seed, v_tag)
                if not p_ck.exists():
                    raise FileNotFoundError(f"Missing variant checkpoint: {p_ck}")
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
            FIG_DIR / f"b_feedback_{n_str}_{decay_tag}_controller.jpg",
            extra_series=variant_res,
        )
        plot_controller_comparison(
            base_res, gate_res, ctrl_res,
            FIG_DIR / f"b_feedback_{n_str}_{decay_tag}_controller_combos.jpg",
            extra_series=combo_res,
            color_base="#1f77b4",   
            color_gate="#ff7f0e",   
            color_ctrl="#8c564b",   
            title=(
                f"Conf. gate: controller combination ablation  "
                f"({len(seeds)} seed(s), ±1 std)"
            ),
        )
        plot_controller_state(
            ctrl_res,
            FIG_DIR / f"b_feedback_controller_state_{n_str}_{decay_tag}_controller.jpg",
        )
        plot_gate_traces(
            ctrl_res,
            FIG_DIR / f"b_feedback_controller_gate_{n_str}_{decay_tag}_controller.jpg",
        )
        plot_margin_stats(
            ctrl_res,
            FIG_DIR / f"b_feedback_controller_margin_stats_{n_str}_{decay_tag}_controller.jpg",
        )
        return

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
    plot_margin_stats(
        gate_res,
        FIG_DIR / f"b_feedback_confgate_margin_stats_{n_str}_{decay_tag}.jpg",
    )

if __name__ == "__main__":
    main()