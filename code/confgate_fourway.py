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

FIG_DIR = ROOT / "paper" / "figures" / "conf_fourway"

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

ALL_MODALITIES = ["baseline", "analytical", "learnable", "rnn"]

# Plotting colours
COLOR_BASE  = "#2ca02c"   # green  – baseline
COLOR_GATE  = "#9467bd"   # purple – analytical
COLOR_LEARN = "#d62728"   # red    – learnable
COLOR_RNN   = "#ff7f0e"   # orange – RNN controller


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
        obs: torch.Tensor,     # (B, obs_dim)
        h_prev: torch.Tensor,  # (B, RNN_HIDDEN)
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

        # ---- Pass 2: gated path (k NOT detached → grad reaches RNN) ----
        if self._gating_active():
            z_gated      = self._gated_z(obs, rnn_out["k_cart"], rnn_out["k_pole"])
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
) -> None:
    """Four-curve learning figure with ±1 std shading.  Missing modalities are skipped."""

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

    def _l100(results: List[Dict]) -> float:
        return float(np.mean([r["last100_mean"] for r in results])) if results else 0.0

    curves = [
        (_prep(base_results),  COLOR_BASE,  f"baseline           last-100: {_l100(base_results):.1f}"),
        (_prep(gate_results),  COLOR_GATE,  f"analytical gate    last-100: {_l100(gate_results):.1f}"),
        (_prep(learn_results), COLOR_LEARN, f"learnable gate     last-100: {_l100(learn_results):.1f}"),
        (_prep(rnn_results),   COLOR_RNN,   f"RNN controller     last-100: {_l100(rnn_results):.1f}"),
    ]

    ep_str = f"{n_ep:,} ep" if n_ep else decay_tag
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    for (ep, sm, sd), col, lbl in curves:
        if len(ep) == 0:
            continue
        ax.fill_between(ep, sm - sd, sm + sd, color=col, alpha=0.15, linewidth=0)
        ax.plot(ep, sm, color=col, lw=2.2, label=lbl)

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"Four-way gate comparison  ({n_seeds} seed(s), {ep_str}, ±1 std)",
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
    """2×2 panel showing mean RNN controller outputs over training (±1 std)."""
    histories = [r.get("rnn_history", []) for r in rnn_results]
    histories = [h for h in histories if h]
    if not histories:
        print("  No rnn_history to plot; skipping.", flush=True)
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
                   help="comma-separated subset of: baseline,analytical,learnable,rnn")
    p.add_argument("--train-missing-only", action="store_true",
                   help="skip seeds whose checkpoint already exists")
    p.add_argument("--plot-only", action="store_true",
                   help="skip training; reload checkpoints and regenerate figures")
    p.add_argument("--fast", action="store_true",
                   help="fast schedule: LEARNING_STARTS=1000, TRAIN_FREQUENCY=4")
    args = p.parse_args()

    seeds      = _parse_seeds(args.seeds)
    modalities = _parse_modalities(args.modalities)
    eps_decay  = args.epsilon_decay_episodes
    n_ep_tgt   = args.total_episodes
    n_step     = max(args.total_timesteps, 15_000_000) if n_ep_tgt else args.total_timesteps
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Modality 4 — RNN controller
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

    # ------------------------------------------------------------------ #
    # Verify checkpoints present                                         #
    # ------------------------------------------------------------------ #
    missing = []
    if "baseline"   in modalities:
        missing += [cg_analytical._ckpt_path(ckpt_dir, s, False)
                    for s in seeds
                    if not cg_analytical._ckpt_path(ckpt_dir, s, False).exists()]
    if "analytical" in modalities:
        missing += [cg_analytical._ckpt_path(ckpt_dir, s, True)
                    for s in seeds
                    if not cg_analytical._ckpt_path(ckpt_dir, s, True).exists()]
    if "learnable"  in modalities:
        missing += [cg_learnable._ckpt_path_learnable(ckpt_dir, s)
                    for s in seeds
                    if not cg_learnable._ckpt_path_learnable(ckpt_dir, s).exists()]
    if "rnn"        in modalities:
        missing += [_ckpt_path_rnn(ckpt_dir, s)
                    for s in seeds
                    if not _ckpt_path_rnn(ckpt_dir, s).exists()]
    if missing:
        for m in missing:
            print(f"  MISSING: {m}", flush=True)
        raise FileNotFoundError(f"{len(missing)} checkpoint(s) missing (see above).")

    # ------------------------------------------------------------------ #
    # Load results                                                        #
    # ------------------------------------------------------------------ #
    base_res  = ([cg_analytical._load_result(ckpt_dir, s, False)    for s in seeds]
                 if "baseline"   in modalities else [])
    gate_res  = ([cg_analytical._load_result(ckpt_dir, s, True)     for s in seeds]
                 if "analytical" in modalities else [])
    learn_res = ([cg_learnable._load_result_learnable(ckpt_dir, s)  for s in seeds]
                 if "learnable"  in modalities else [])
    rnn_res   = ([_load_result_rnn(ckpt_dir, s)                     for s in seeds]
                 if "rnn"        in modalities else [])

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n=== Final-100-episode summary ===", flush=True)
    if base_res:  _agg_print(base_res,  "baseline")
    if gate_res:  _agg_print(gate_res,  "analytical gate")
    if learn_res: _agg_print(learn_res, "learnable gate")
    if rnn_res:   _agg_print(rnn_res,   "RNN controller")

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
    )

    if rnn_res:
        plot_rnn_controller_history(
            rnn_res,
            FIG_DIR / f"rnn_ctrl_history_{n_str}_{decay_tag}{fast_sfx}{ep_tag}.jpg",
        )


if __name__ == "__main__":
    main()
