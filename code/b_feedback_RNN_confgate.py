"""
b_feedback_RNN_confgate.py
==========================
Three-head B-feedback with GRU confidence gating and W_z fully detached from backprop.

  - W_z: updated only by manual B-feedback (full-head TD error).
  - trunk + head_full: Adam on loss_full (z uses detached W in gated path).
  - head_cart / head_pole: separate Adam, detached trunk.
  - GRU gate: aux Q features -> k4, k5 in [k_min, k_max]; trained via loss_full
    (gradients through k4/k5 only, not W_z). Online rollout keeps GRU hidden state;
    replay training uses zero hidden per sample (shuffled buffer).

Usage:
  python code/b_feedback_RNN_confgate.py --seeds 1
  python code/b_feedback_RNN_confgate.py --plot-only
"""

from __future__ import annotations

import argparse
import random
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

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"

LEARNING_RATE = 2.5e-4
GATE_LEARNING_RATE = 1e-4
LR_LINEAR_MIXED = 1e-4
GAMMA = 0.99
BUFFER_SIZE = 10_000
BATCH_SIZE = 128
START_E = 1.0
END_E = 0.05
EPSILON_DECAY_EPISODES = 3_500
LEARNING_STARTS = 10_000
TRAIN_FREQUENCY = 10
TARGET_NETWORK_FREQ = 500
TAU = 1.0
FEEDBACK_GRAD_CLIP = 1.0
LOG_EVERY = 5_000

OBS_TO_120 = 120
HIDDEN_84 = 84
GATE_HIDDEN = 16
GATE_K_MIN = 0.8
GATE_K_MAX = 1.0

DEFAULT_SEEDS = [1]
DEFAULT_TOTAL_STEPS = 500_000

COLOR_BASE = "#2ca02c"
COLOR_RNN = "#9467bd"

ReplayBufferSamples = namedtuple(
    "ReplayBufferSamples",
    ["observations", "actions", "next_observations", "dones", "rewards"],
)


class ReplayBuffer:
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action: int, reward: float, done: float) -> None:
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


def make_b_matrix(F: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones(F, 1, device=device, dtype=dtype)


def _scaled_obs(obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] < 4:
        return obs
    gx = obs.clone()
    gx[:, :2] = gx[:, :2] * k4.to(dtype=gx.dtype, device=gx.device)
    gx[:, 2:4] = gx[:, 2:4] * k5.to(dtype=gx.dtype, device=gx.device)
    return gx


def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    return max(start + (end - start) / max(duration, 1) * t, end)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid") if w >= 2 else arr


# ---------------------------------------------------------------------------
# GRU gate (Option C — replay uses hidden=None; online keeps state on q_net)
# ---------------------------------------------------------------------------

class GRUGate(nn.Module):
    """Maps detached aux Q features to (k4, k5) with optional recurrent state."""

    INPUT_DIM = 6

    def __init__(
        self,
        hidden_dim: int = GATE_HIDDEN,
        k_min: float = GATE_K_MIN,
        k_max: float = GATE_K_MAX,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_min = k_min
        self.k_max = k_max
        self.cell = nn.GRUCell(self.INPUT_DIM, hidden_dim)
        self.k_head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def q_features(Q_cart: torch.Tensor, Q_pole: torch.Tensor) -> torch.Tensor:
        m_c = (Q_cart[:, 1] - Q_cart[:, 0]).abs()
        m_p = (Q_pole[:, 1] - Q_pole[:, 0]).abs()
        return torch.stack(
            [m_c, m_p, Q_cart[:, 0], Q_cart[:, 1], Q_pole[:, 0], Q_pole[:, 1]],
            dim=1,
        ).detach()

    def forward(
        self,
        Q_cart: torch.Tensor,
        Q_pole: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.q_features(Q_cart, Q_pole)
        B = x.shape[0]
        if hidden is None:
            hidden = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        h = self.cell(x, hidden)
        raw = torch.sigmoid(self.k_head(h))
        span = self.k_max - self.k_min
        k4 = self.k_min + span * raw[:, 0:1]
        k5 = self.k_min + span * raw[:, 1:2]
        return k4, k5, h


# ---------------------------------------------------------------------------
# Network (W_z detached from backprop; GRU gate when enabled)
# ---------------------------------------------------------------------------

class BFeedbackRNNConfGateNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        confidence_gating: bool = False,
        gate_k_min: float = GATE_K_MIN,
        gate_k_max: float = GATE_K_MAX,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.confidence_gating = confidence_gating
        self.gate_k_min = gate_k_min
        self.gate_k_max = gate_k_max
        self.trunk_scale: float = 1.0

        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head_full = nn.Linear(HIDDEN_84, n_actions)
        self.head_cart = nn.Linear(HIDDEN_84, n_actions)
        self.head_pole = nn.Linear(HIDDEN_84, n_actions)
        self.gate_rnn = GRUGate(GATE_HIDDEN, gate_k_min, gate_k_max)
        self._gate_hidden: Optional[torch.Tensor] = None

    def reset_gate_hidden(self) -> None:
        self._gate_hidden = None

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

    def _gating_active(self) -> bool:
        return self.confidence_gating and self.obs_dim == 4 and self.n_actions == 2

    def _gated_z(self, obs: torch.Tensor, k4: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
        """Gated z with W_z detached so loss_full grads reach k4/k5 only, not W_z."""
        W = self.linear_feature.weight.detach()
        b = self.linear_feature.bias.detach()
        return (
            k4 * (obs[..., 0:2] @ W[:, 0:2].T)
            + k5 * (obs[..., 2:4] @ W[:, 2:4].T)
            + b
        )

    def _gate_k(
        self,
        Q_cart: torch.Tensor,
        Q_pole: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k4, k5, _ = self.gate_rnn(Q_cart, Q_pole, hidden=hidden)
        return k4, k5

    def forward(
        self,
        obs: torch.Tensor,
        gate_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart.detach()))

        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole.detach()))

        B = obs.shape[0]
        k4 = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        k5 = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        if self._gating_active():
            k4, k5 = self._gate_k(Q_cart, Q_pole, hidden=gate_hidden)

        if self._gating_active():
            z_full = self._gated_z(obs, k4, k5)
        else:
            z_full = self.linear_feature(obs).detach()

        z_trunk = self.trunk(z_full)
        if self.trunk_scale != 1.0:
            z_trunk = z_trunk * self.trunk_scale
        Q_full = self.head_full(z_trunk)
        return Q_full, Q_cart, Q_pole, k4, k5

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self._gate_hidden
        Q_full, _, _, _, _ = self.forward(obs, gate_hidden=hidden)
        return Q_full

    @torch.no_grad()
    def update_gate_hidden_from_obs(self, obs: torch.Tensor) -> None:
        """Advance GRU state during env interaction (batch size 1)."""
        if not self._gating_active():
            return
        z_cart = self.linear_feature(self._mask_cart(obs))
        Q_cart = self.head_cart(self.trunk(z_cart))
        z_pole = self.linear_feature(self._mask_pole(obs))
        Q_pole = self.head_pole(self.trunk(z_pole))
        _, _, h = self.gate_rnn(Q_cart, Q_pole, hidden=self._gate_hidden)
        self._gate_hidden = h


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_one(
    seed: int,
    total_timesteps: int,
    confidence_gating: bool,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    algo_tag_override: Optional[str] = None,
) -> Dict:
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    gate_ok = confidence_gating and obs_dim == 4 and n_actions == 2
    label = f"seed={seed} rnn_gated={gate_ok}"

    q_net = BFeedbackRNNConfGateNetwork(
        obs_dim, n_actions, confidence_gating=gate_ok
    ).to(device)
    t_net = BFeedbackRNNConfGateNetwork(
        obs_dim, n_actions, confidence_gating=gate_ok
    ).to(device)
    t_net.load_state_dict(q_net.state_dict())

    opt_main = optim.Adam(
        list(q_net.trunk.parameters()) + list(q_net.head_full.parameters()),
        lr=LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=LEARNING_RATE,
    )
    opt_gate = optim.Adam(q_net.gate_rnn.parameters(), lr=GATE_LEARNING_RATE)

    B_mat = make_b_matrix(OBS_TO_120, device, torch.float32)
    rb = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape, device)

    episode_returns: List[float] = []
    gate_history: List[Dict] = []
    loss_list: List[float] = []

    obs, _ = env.reset(seed=seed)
    q_net.reset_gate_hidden()
    t = 0

    while t < total_timesteps and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        n_ep_done = len(episode_returns)
        eps = _linear_schedule(START_E, END_E, epsilon_decay_episodes, n_ep_done)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q_net.forward_q_only(x).argmax(dim=1).item())
                q_net.update_gate_hidden_from_obs(x)

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]
        rb.add(obs, real_nxt, action, float(reward), float(done))

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()
            q_net.reset_gate_hidden()

        if t > LEARNING_STARTS and t % TRAIN_FREQUENCY == 0:
            data = rb.sample(BATCH_SIZE)

            with torch.no_grad():
                tQ_full, _, _, _, _ = t_net.forward(data.next_observations, gate_hidden=None)
                r = data.rewards.flatten()
                d = data.dones.flatten()
                y_full = r + GAMMA * (1 - d) * tQ_full.max(dim=1).values
                _, tQ_cart, tQ_pole, _, _ = t_net.forward(data.next_observations, gate_hidden=None)
                y_cart = r + GAMMA * (1 - d) * tQ_cart.max(dim=1).values
                y_pole = r + GAMMA * (1 - d) * tQ_pole.max(dim=1).values

            Q_full, Q_cart, Q_pole, k4, k5 = q_net.forward(
                data.observations, gate_hidden=None
            )

            qf_sa = Q_full.gather(1, data.actions).squeeze()
            qc_sa = Q_cart.gather(1, data.actions).squeeze()
            qp_sa = Q_pole.gather(1, data.actions).squeeze()

            loss_full = F.mse_loss(y_full, qf_sa)
            loss_cart = F.mse_loss(y_cart, qc_sa)
            loss_pole = F.mse_loss(y_pole, qp_sa)

            e_full = (y_full - qf_sa).detach().unsqueeze(1)
            delta_z = e_full @ B_mat.t()
            gx = _scaled_obs(data.observations, k4.detach(), k5.detach())
            grad_Wz = (delta_z.t() @ gx) / BATCH_SIZE
            grad_bz = delta_z.mean(0)
            if FEEDBACK_GRAD_CLIP > 0:
                n = grad_Wz.norm()
                if n > FEEDBACK_GRAD_CLIP:
                    grad_Wz = grad_Wz * (FEEDBACK_GRAD_CLIP / n.item())
            with torch.no_grad():
                q_net.linear_feature.weight.data.add_(grad_Wz, alpha=LR_LINEAR_MIXED)
                q_net.linear_feature.bias.data.add_(grad_bz, alpha=LR_LINEAR_MIXED)

            opt_main.zero_grad()
            opt_gate.zero_grad()
            loss_full.backward()
            opt_main.step()
            if gate_ok:
                opt_gate.step()

            opt_aux.zero_grad()
            (loss_cart + loss_pole).backward()
            opt_aux.step()

            loss_list.append(loss_full.item())

            if gate_ok and t % LOG_EVERY == 0:
                gate_history.append({
                    "step": t,
                    "episode": len(episode_returns),
                    "mean_k4": float(k4.mean().item()),
                    "mean_k5": float(k5.mean().item()),
                })

        if t % TARGET_NETWORK_FREQ == 0:
            for tp, qp in zip(t_net.parameters(), q_net.parameters()):
                tp.data.copy_(TAU * qp.data + (1 - TAU) * tp.data)

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [b_feedb_rnn] {label} step={t+1}/{total_timesteps}"
                f"  episodes={len(episode_returns)}"
                f"  eps={eps:.3f}",
                flush=True,
            )
        t += 1

    env.close()

    if algo_tag_override is not None:
        algo_tag = algo_tag_override
    elif gate_ok:
        algo_tag = "b_feedb_rnn_gate"
    else:
        algo_tag = "b_feedb_rnn_base"

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{algo_tag}_seed{seed}.pt"
        torch.save(
            {
                "algo": algo_tag,
                "seed": seed,
                "episode_returns": episode_returns,
                "gate_history": gate_history,
                "q_network": q_net.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns))
        if episode_returns
        else 0.0
    )
    print(
        f"  Done {label}: episodes={len(episode_returns)}"
        f"  last100={last100:.1f}",
        flush=True,
    )
    return {
        "seed": seed,
        "confidence_gating": gate_ok,
        "episode_returns": episode_returns,
        "gate_history": gate_history,
        "last100_mean": last100,
    }


# ---------------------------------------------------------------------------
# Plotting / checkpoint I/O
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: Path, seed: int, gated: bool) -> Path:
    tag = "b_feedb_rnn_gate" if gated else "b_feedb_rnn_base"
    return ckpt_dir / f"{tag}_seed{seed}.pt"


def _load_result(ckpt_dir: Path, seed: int, gated: bool) -> Dict:
    data = torch.load(_ckpt_path(ckpt_dir, seed, gated), map_location="cpu", weights_only=False)
    return {
        "seed": seed,
        "episode_returns": data["episode_returns"],
        "gate_history": data.get("gate_history", []),
        "last100_mean": float(np.mean(data["episode_returns"][-100:]))
        if len(data["episode_returns"]) >= 100
        else float(np.mean(data["episode_returns"])),
    }


def plot_learning_curves(
    base_results: List[Dict],
    rnn_results: List[Dict],
    out_jpg: Path,
    decay_tag: str,
) -> None:
    b_arrays = [np.asarray(r["episode_returns"]) for r in base_results]
    r_arrays = [np.asarray(r["episode_returns"]) for r in rnn_results]
    n = min(min(len(a) for a in b_arrays), min(len(a) for a in r_arrays))
    B = np.stack([a[:n] for a in b_arrays], axis=0)
    R = np.stack([a[:n] for a in r_arrays], axis=0)
    ep = np.arange(1, n + 1)

    mu_b = B.mean(0)
    mu_r = R.mean(0)
    sd_b = B.std(0, ddof=1) if B.shape[0] > 1 else np.zeros(n)
    sd_r = R.std(0, ddof=1) if R.shape[0] > 1 else np.zeros(n)

    W = max(1, n // 200)
    ep_sm = ep[W - 1:]
    sm_b, sm_r = _smooth(mu_b, W), _smooth(mu_r, W)
    sm_sb, sm_sr = _smooth(sd_b, W), _smooth(sd_r, W)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if B.shape[0] > 1:
        ax.fill_between(ep_sm, sm_b - sm_sb, sm_b + sm_sb, color=COLOR_BASE, alpha=0.18, linewidth=0)
        ax.fill_between(ep_sm, sm_r - sm_sr, sm_r + sm_sr, color=COLOR_RNN, alpha=0.18, linewidth=0)
    ax.plot(ep_sm, sm_b, color=COLOR_BASE, lw=2.2,
            label=f"baseline (no gate)  last-100: {np.mean(B[:, -100:]):.1f}")
    ax.plot(ep_sm, sm_r, color=COLOR_RNN, lw=2.2,
            label=f"GRU gate (W_z detached)  last-100: {np.mean(R[:, -100:]):.1f}")
    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"B-feedback RNN conf. gate (W_z detached): baseline vs GRU gate  "
        f"({len(base_results)} seed(s), {decay_tag})",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_gate_traces(rnn_results: List[Dict], out_jpg: Path) -> None:
    hists = [r["gate_history"] for r in rnn_results if r.get("gate_history")]
    if not hists:
        print("[gate plot] no gate history — skipping.", flush=True)
        return

    starts = [float(h[0]["episode"]) for h in hists]
    ends = [float(h[-1]["episode"]) for h in hists]
    lo, hi = max(starts), min(ends)
    if hi <= lo:
        return
    grid = np.linspace(lo, hi, 512)

    def _interp(hist, y_key):
        xs = np.array([float(h["episode"]) for h in hist])
        ys = np.array([float(h[y_key]) for h in hist])
        order = np.argsort(xs)
        return np.interp(grid, xs[order], ys[order], left=np.nan, right=np.nan)

    C = np.vstack([_interp(h, "mean_k4") for h in hists])
    P = np.vstack([_interp(h, "mean_k5") for h in hists])

    fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    ax_c.plot(grid, C.mean(0), color=COLOR_RNN, lw=2.2, label=r"$k_{\mathrm{cart}}$")
    ax_p.plot(grid, P.mean(0), color=COLOR_BASE, lw=2.2, label=r"$k_{\mathrm{pole}}$")
    for ax in (ax_c, ax_p):
        ax.set_ylim(0.76, 1.04)
        ax.axhline(GATE_K_MIN, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(GATE_K_MAX, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    ax_c.set_ylabel(r"$k_{\mathrm{cart}}$", fontsize=11)
    ax_p.set_ylabel(r"$k_{\mathrm{pole}}$", fontsize=11)
    ax_p.set_xlabel("Episode", fontsize=11)
    ax_c.set_title("GRU gate batch-mean traces (b_feedback_RNN_confgate)", fontsize=11)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _parse_seeds(s: str) -> List[int]:
    return sorted({int(p.strip()) for p in s.split(",") if p.strip()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--epsilon-decay-episodes", type=int, default=EPSILON_DECAY_EPISODES)
    p.add_argument("--total-episodes", type=int, default=None)
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--train-missing-only", action="store_true")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    eps_decay = args.epsilon_decay_episodes
    n_step = max(args.total_timesteps, 15_000_000) if args.total_episodes else args.total_timesteps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decay_tag = f"decay{eps_decay}"
    ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_rnn" / f"ckpt_{decay_tag}"
    n_str = f"{len(seeds)}seed{'s' if len(seeds) != 1 else ''}"

    print("=" * 60, flush=True)
    print("b_feedback_RNN_confgate  (W_z detached + GRU gate)", flush=True)
    print(f"  Seeds: {seeds}  Steps cap: {n_step:,}  Device: {device}", flush=True)
    print(f"  Checkpoints: {ckpt_dir}", flush=True)
    print("=" * 60, flush=True)

    def need(gated: bool) -> List[int]:
        if args.train_missing_only:
            return [s for s in seeds if not _ckpt_path(ckpt_dir, s, gated).exists()]
        return list(seeds)

    if not args.plot_only:
        for gated in (False, True):
            lbl = "RNN-GATE" if gated else "BASELINE"
            seeds_run = need(gated)
            if not seeds_run:
                print(f">>> {lbl}: checkpoints present, skip.", flush=True)
                continue
            print(f"\n>>> [{lbl}] seeds {seeds_run} ...", flush=True)
            t0 = time.perf_counter()
            for seed in seeds_run:
                run_one(
                    seed,
                    n_step,
                    gated,
                    device,
                    checkpoint_dir=ckpt_dir,
                    epsilon_decay_episodes=eps_decay,
                    total_episodes=args.total_episodes,
                )
            print(f">>> [{lbl}] done in {time.perf_counter() - t0:.1f}s", flush=True)

    for seed in seeds:
        for gated in (False, True):
            if not _ckpt_path(ckpt_dir, seed, gated).exists():
                raise FileNotFoundError(f"Missing: {_ckpt_path(ckpt_dir, seed, gated)}")

    base_res = [_load_result(ckpt_dir, s, False) for s in seeds]
    rnn_res = [_load_result(ckpt_dir, s, True) for s in seeds]

    print("\n=== Summary ===", flush=True)
    for label, res in [("baseline", base_res), ("GRU gate", rnn_res)]:
        vals = [r["last100_mean"] for r in res]
        print(f"  {label:12s}: {np.mean(vals):.2f}", flush=True)

    plot_learning_curves(
        base_res,
        rnn_res,
        FIG_DIR / f"b_feedback_RNN_confgate_{n_str}_{decay_tag}.jpg",
        decay_tag,
    )
    plot_gate_traces(
        rnn_res,
        FIG_DIR / f"b_feedback_RNN_confgate_gate_{n_str}_{decay_tag}.jpg",
    )


if __name__ == "__main__":
    main()
