"""
dqn_controller.py
=================
Regular single-head DQN (full observation only) with optional LR controller
driven by cart/pole Q-value margins from masked probe forward passes.

Architecture matches the B-feedback trunk: obs -> W_z (120) -> ReLU -> 84 -> ReLU -> Q.

Conditions (--compare):
  baseline   : standard DQN, soft Polyak target (τ=0.005)
  ctrl_lr    : LR controller (lr_only, original formulas) + soft target

Usage:
  python code/dqn_controller.py --compare --total-episodes 5000 \\
      --epsilon-decay-episodes 3500 --seeds 1,2,3,4,5,6,7,8,9,10
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
LOG_EVERY = 5_000

OBS_TO_120 = 120
HIDDEN_84 = 84

DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_TOTAL_STEPS = 500_000

COLOR_DQN_BASE = "#1f77b4"   # blue
COLOR_DQN_CTRL = "#ff7f0e"   # orange
COLOR_BF_BASE  = "#2ca02c"   # green
COLOR_BF_CTRL  = "#e377c2"   # pink

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


class RegularDQNNetwork(nn.Module):
    """Single-head DQN: obs -> W_z -> trunk -> Q."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.linear_feature = nn.Linear(obs_dim, OBS_TO_120)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN_84, n_actions)
        self.trunk_scale: float = 1.0

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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.linear_feature(obs)
        z = self.trunk(z)
        if self.trunk_scale != 1.0:
            z = z * self.trunk_scale
        return self.head(z)

    def forward_q_only(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    @torch.no_grad()
    def probe_confidence(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cart/pole Q margins via masked probe passes (controller signal only)."""
        if self.obs_dim != 4 or self.n_actions != 2:
            ones = torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
            return ones, ones.clone()
        Q_cart = self.forward(self._mask_cart(obs))
        Q_pole = self.forward(self._mask_pole(obs))
        m_c = (Q_cart[:, 1] - Q_cart[:, 0]).abs()
        m_p = (Q_pole[:, 1] - Q_pole[:, 0]).abs()
        C = F.softmax(torch.stack([m_c, m_p], dim=1), dim=1)
        k4 = 0.8 + 0.2 * C[:, 0:1]
        k5 = 0.8 + 0.2 * C[:, 1:2]
        return k4, k5


class DQNController:
    """LR / trunk / epsilon controller driven by probe confidence (k4, k5)."""

    def __init__(
        self,
        q_net: RegularDQNNetwork,
        optimizer: optim.Optimizer,
        control_lr: bool = False,
        control_trunk: bool = False,
        control_epsilon: bool = False,
        ema_alpha: float = 0.05,
        use_original_formulas: bool = True,
        epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    ):
        self.q_net = q_net
        self.optimizer = optimizer
        self.control_lr = control_lr
        self.control_trunk = control_trunk
        self.control_epsilon = control_epsilon
        self.ema_alpha = ema_alpha
        self.use_original_formulas = use_original_formulas
        self._epsilon_decay_episodes = float(epsilon_decay_episodes)
        self._imbalance_ema = 1.0
        self._k4_ema = 0.9
        self._k5_ema = 0.9
        self._effective_lr = LEARNING_RATE
        self._effective_end_e = END_E
        self._effective_decay_ep = self._epsilon_decay_episodes

    def step(self, k4: torch.Tensor, k5: torch.Tensor) -> None:
        k4m = float(k4.mean().item())
        k5m = float(k5.mean().item())
        imbalance = abs(k4m - k5m)
        a = self.ema_alpha
        self._imbalance_ema = (1 - a) * self._imbalance_ema + a * imbalance
        self._k4_ema = (1 - a) * self._k4_ema + a * k4m
        self._k5_ema = (1 - a) * self._k5_ema + a * k5m

        if self.use_original_formulas:
            if self.control_trunk:
                self.q_net.trunk_scale = 0.6 + 0.4 * self._imbalance_ema
            else:
                self.q_net.trunk_scale = 1.0
            if self.control_lr:
                lr = LEARNING_RATE * (0.5 + 0.5 * self._imbalance_ema)
                self._effective_lr = lr
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
            else:
                self._effective_lr = LEARNING_RATE
                for g in self.optimizer.param_groups:
                    g["lr"] = LEARNING_RATE
            if self.control_epsilon:
                self._effective_end_e = END_E + (1.0 - self._imbalance_ema) * 0.05
                self._effective_decay_ep = self._epsilon_decay_episodes * (
                    1.0 + (1.0 - self._imbalance_ema) * 0.5
                )
            else:
                self._effective_end_e = END_E
                self._effective_decay_ep = self._epsilon_decay_episodes
        else:
            self.q_net.trunk_scale = 1.0
            self._effective_lr = LEARNING_RATE
            for g in self.optimizer.param_groups:
                g["lr"] = LEARNING_RATE
            self._effective_end_e = END_E
            self._effective_decay_ep = self._epsilon_decay_episodes

    @property
    def effective_end_e(self) -> float:
        return self._effective_end_e

    @property
    def effective_decay_ep(self) -> float:
        return self._effective_decay_ep

    def log_state(self) -> Dict:
        return {
            "imbalance_ema": self._imbalance_ema,
            "k4_ema": self._k4_ema,
            "k5_ema": self._k5_ema,
            "effective_lr": self._effective_lr,
            "effective_end_e": self._effective_end_e,
            "effective_decay_ep": self._effective_decay_ep,
            "trunk_scale": self.q_net.trunk_scale,
        }


def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    return max(start + (end - start) / max(duration, 1) * t, end)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def run_one(
    seed: int,
    total_timesteps: int,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    total_episodes: Optional[int] = None,
    use_controller: bool = False,
    ctrl_knobs: Optional[Dict] = None,
    target_tau: float = 0.005,
    target_freq: int = 1,
    algo_tag: str = "dqn_base_soft",
) -> Dict:
    _set_seed(seed)
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    q_net = RegularDQNNetwork(obs_dim, n_actions).to(device)
    t_net = RegularDQNNetwork(obs_dim, n_actions).to(device)
    t_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    controller: Optional[DQNController] = None
    if use_controller:
        knobs = ctrl_knobs or {}
        controller = DQNController(
            q_net, optimizer,
            epsilon_decay_episodes=epsilon_decay_episodes,
            **knobs,
        )

    rb = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape, device)
    episode_returns: List[float] = []
    controller_history: List[Dict] = []

    _eps_end = END_E
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
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q_net.forward_q_only(x).argmax(dim=1).item())

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

        if t > LEARNING_STARTS and t % TRAIN_FREQUENCY == 0:
            data = rb.sample(BATCH_SIZE)

            with torch.no_grad():
                tQ = t_net.forward(data.next_observations)
                r = data.rewards.flatten()
                d = data.dones.flatten()
                y = r + GAMMA * (1 - d) * tQ.max(dim=1).values

            Q = q_net.forward(data.observations)
            q_sa = Q.gather(1, data.actions).squeeze()
            loss = F.mse_loss(y, q_sa)

            if controller is not None:
                with torch.no_grad():
                    k4, k5 = q_net.probe_confidence(data.observations)
                controller.step(k4, k5)
                _eps_end = controller.effective_end_e
                _eps_decay = controller.effective_decay_ep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if controller is not None and t % LOG_EVERY == 0:
                cs = controller.log_state()
                cs.update({"step": t, "episode": len(episode_returns)})
                controller_history.append(cs)

        if t % target_freq == 0:
            for tp, qp in zip(t_net.parameters(), q_net.parameters()):
                tp.data.copy_(target_tau * qp.data + (1 - target_tau) * tp.data)

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [{algo_tag}] seed={seed} step={t+1}/{total_timesteps}"
                f"  episodes={len(episode_returns)}"
                f"  eps={eps:.3f}",
                flush=True,
            )
        t += 1

    env.close()
    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns)) if episode_returns else 0.0
    )
    print(
        f"  Done [{algo_tag}] seed={seed}: episodes={len(episode_returns)}"
        f"  last100={last100:.1f}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{algo_tag}_seed{seed}.pt"
        torch.save({
            "algo": algo_tag,
            "seed": seed,
            "episode_returns": episode_returns,
            "controller_history": controller_history,
            "q_network": q_net.state_dict(),
        }, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}", flush=True)

    return {
        "seed": seed,
        "episode_returns": episode_returns,
        "controller_history": controller_history,
        "last100_mean": last100,
    }


def _load_ep(path: Path) -> np.ndarray:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return np.array(ck["episode_returns"], dtype=float)


def _prep_sm(results: List[Dict], n: int, W: int):
    arrays = [np.asarray(r["episode_returns"])[:n] for r in results]
    M = np.stack(arrays)
    mu = M.mean(0)
    sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
    ep = np.arange(W, n + 1)
    return ep, _smooth(mu, W), _smooth(sd, W)


def plot_comparison(
    dqn_base: List[Dict],
    dqn_ctrl: List[Dict],
    bf_base: List[Dict],
    bf_ctrl: List[Dict],
    out_jpg: Path,
    eps_decay: int,
) -> None:
    all_res = [dqn_base, dqn_ctrl, bf_base, bf_ctrl]
    n = min(min(len(r["episode_returns"]) for r in res) for res in all_res)
    W = max(1, n // 200)

    ep_db, mu_db, sd_db = _prep_sm(dqn_base, n, W)
    ep_dc, mu_dc, sd_dc = _prep_sm(dqn_ctrl, n, W)
    ep_bb, mu_bb, sd_bb = _prep_sm(bf_base, n, W)
    ep_bc, mu_bc, sd_bc = _prep_sm(bf_ctrl, n, W)

    l100_db = float(np.mean([r["last100_mean"] for r in dqn_base]))
    l100_dc = float(np.mean([r["last100_mean"] for r in dqn_ctrl]))
    l100_bb = float(np.mean([r["last100_mean"] for r in bf_base]))
    l100_bc = float(np.mean([r["last100_mean"] for r in bf_ctrl]))

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    series = [
        (ep_bb, mu_bb, sd_bb, COLOR_BF_BASE,  "Baseline (soft τ=0.005)",          l100_bb),
        (ep_bc, mu_bc, sd_bc, COLOR_BF_CTRL,  "B-feedback ctrl lr_only + soft",   l100_bc),
        (ep_dc, mu_dc, sd_dc, COLOR_DQN_CTRL, "Regular DQN ctrl lr_only + soft",  l100_dc),
    ]
    for ep, mu, sd, col, lbl, l100 in series:
        ax.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.10, linewidth=0)
        ax.plot(ep, mu, color=col, lw=2.0, label=f"{lbl}  last-100: {l100:.1f}")

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max (500)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episodic return", fontsize=12)
    ax.set_title(
        f"Regular DQN vs B-feedback: ctrl lr_only + soft target "
        f"({len(dqn_base)} seeds, ±1 std, ε-decay={eps_decay}, 5000 ep)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 520)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Regular DQN with LR controller")
    p.add_argument("--compare", action="store_true",
                   help="Train baseline + ctrl_lr (soft target) and plot vs B-feedback")
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--total-episodes", type=int, default=5_000)
    p.add_argument("--epsilon-decay-episodes", type=int, default=3500)
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--train-missing-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    eps_decay = args.epsilon_decay_episodes
    decay_tag = f"decay{eps_decay}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = ROOT / "paper" / "_tmp_dqn_ctrl" / f"ckpt_{decay_tag}"
    bf_ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_ctrl_lr_opt2_10s_decay3500"

    LR_KNOBS = dict(
        control_lr=True,
        control_trunk=False,
        control_epsilon=False,
        use_original_formulas=True,
    )

    conditions = [
        ("dqn_base_soft", False, None),
        ("dqn_ctrl_lr_soft", True, LR_KNOBS),
    ]

    print("=" * 60, flush=True)
    print("dqn_controller  (regular single-head DQN + optional LR ctrl)", flush=True)
    print(f"  Seeds      : {seeds}", flush=True)
    print(f"  Episodes   : {args.total_episodes:,} (step cap {args.total_timesteps:,})", flush=True)
    print(f"  Eps decay  : {eps_decay}", flush=True)
    print(f"  Soft target: τ=0.005, update every step", flush=True)
    print(f"  Device     : {device}", flush=True)
    print("=" * 60, flush=True)

    if not args.plot_only:
        for tag, use_ctrl, knobs in conditions:
            seeds_to_run = seeds
            if args.train_missing_only:
                seeds_to_run = [
                    s for s in seeds
                    if not (ckpt_dir / f"{tag}_seed{s}.pt").exists()
                ]
            if not seeds_to_run:
                print(f">>> {tag}: all checkpoints present, skipping.", flush=True)
                continue
            print(f"\n>>> [{tag}] Training seeds {seeds_to_run} ...", flush=True)
            t0 = time.perf_counter()
            for seed in seeds_to_run:
                run_one(
                    seed, args.total_timesteps, device,
                    checkpoint_dir=ckpt_dir,
                    epsilon_decay_episodes=eps_decay,
                    total_episodes=args.total_episodes,
                    use_controller=use_ctrl,
                    ctrl_knobs=knobs,
                    target_tau=0.005,
                    target_freq=1,
                    algo_tag=tag,
                )
            print(f">>> [{tag}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Load DQN results
    dqn_base_res = []
    dqn_ctrl_res = []
    for s in seeds:
        p_base = ckpt_dir / f"dqn_base_soft_seed{s}.pt"
        p_ctrl = ckpt_dir / f"dqn_ctrl_lr_soft_seed{s}.pt"
        if not p_base.exists() or not p_ctrl.exists():
            raise FileNotFoundError(f"Missing DQN checkpoint for seed {s}")
        ep = _load_ep(p_base)
        last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep))
        dqn_base_res.append({"seed": s, "episode_returns": ep, "last100_mean": last100})
        ep = _load_ep(p_ctrl)
        last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep))
        dqn_ctrl_res.append({"seed": s, "episode_returns": ep, "last100_mean": last100})

    # Load B-feedback results (same seeds, same decay=3500 setup)
    bf_base_res = []
    bf_ctrl_res = []
    for s in seeds:
        p_base = bf_ckpt_dir / f"base_seed{s}.pt"
        p_ctrl = bf_ckpt_dir / f"ctrllr_seed{s}.pt"
        if not p_base.exists() or not p_ctrl.exists():
            raise FileNotFoundError(f"Missing B-feedback checkpoint for seed {s}: {p_base} / {p_ctrl}")
        ep = _load_ep(p_base)
        last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep))
        bf_base_res.append({"seed": s, "episode_returns": ep, "last100_mean": last100})
        ep = _load_ep(p_ctrl)
        last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep))
        bf_ctrl_res.append({"seed": s, "episode_returns": ep, "last100_mean": last100})

    print("\n=== Final-100-episode summary ===", flush=True)
    for res, lbl in [
        (dqn_base_res, "Regular DQN baseline"),
        (dqn_ctrl_res, "Regular DQN ctrl lr_only"),
        (bf_base_res,  "B-feedback baseline"),
        (bf_ctrl_res,  "B-feedback ctrl lr_only"),
    ]:
        vals = [r["last100_mean"] for r in res]
        print(f"  {lbl:30s}  {np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f}", flush=True)

    n_str = f"{len(seeds)}seeds"
    plot_comparison(
        dqn_base_res, dqn_ctrl_res, bf_base_res, bf_ctrl_res,
        FIG_DIR / f"dqn_vs_bfeedback_ctrl_lr_{n_str}_{decay_tag}.jpg",
        eps_decay=eps_decay,
    )


if __name__ == "__main__":
    main()
