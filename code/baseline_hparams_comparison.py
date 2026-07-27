"""
Baseline-only CartPole comparison using the implementation from confgate_learnable.py.

Runs three no-gate DQN baselines:
  1. current: the hyperparameters currently used in confgate_learnable.py
  2. optimized: faster-start training schedule from the analysis
  3. rnn_controller: optimized baseline plus a small biologically motivated
     recurrent controller for plasticity, exploration, and replay intensity

Default run:
    python code/baseline_hparams_comparison.py
"""

from __future__ import annotations

import argparse
import random
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import confgate_learnable as cg


ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "paper" / "_tmp_b_feedb_cg"
FIG_DIR = ROOT / "paper" / "figures" / "baseline_hparams"


CONDITIONS: List[Dict] = [
    {
        "tag": "current",
        "label": "Baseline current params",
        "color": "#2ca02c",
        "epsilon_decay_episodes": cg.EPSILON_DECAY_EPISODES,
        "learning_starts": cg.LEARNING_STARTS,
        "train_frequency": cg.TRAIN_FREQUENCY,
        "target_freq": cg.TARGET_NETWORK_FREQ,
        "main_grad_clip": 0.0,
    },
    {
        "tag": "optimized",
        "label": "Baseline optimized params",
        "color": "#d62728",
        "epsilon_decay_episodes": 800,
        "learning_starts": 1_000,
        "train_frequency": 4,
        "target_freq": 1,
        "main_grad_clip": 0.0,
    },
    {
        "tag": "rnn_controller",
        "label": "Baseline RNN controller",
        "color": "#ff7f0e",
        "epsilon_decay_episodes": 800,
        "learning_starts": 1_000,
        "train_frequency": 4,
        "target_freq": 1,
        "main_grad_clip": 10.0,
    },
]


def _ckpt_dir(total_episodes: int) -> Path:
    return CKPT_ROOT / f"ckpt_baseline_hparams_{total_episodes}ep"


def _ckpt_path(seed: int, tag: str, total_episodes: int) -> Path:
    return _ckpt_dir(total_episodes) / f"baseline_{tag}_seed{seed}.pt"


def _load_result(seed: int, tag: str, total_episodes: int) -> Dict:
    ckpt = torch.load(_ckpt_path(seed, tag, total_episodes), map_location="cpu", weights_only=False)
    ep = ckpt["episode_returns"]
    last100 = float(np.mean(ep[-100:])) if len(ep) >= 100 else float(np.mean(ep)) if ep else 0.0
    return {
        "seed": seed,
        "episode_returns": ep,
        "last100_mean": last100,
        "controller_history": ckpt.get("controller_history", []),
        "algo": ckpt.get("algo", tag),
    }


ControlledReplayBufferSamples = namedtuple(
    "ControlledReplayBufferSamples",
    ["observations", "actions", "next_observations", "dones", "rewards", "hiddens"],
)


class ControlledReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        hidden_size: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.hidden_size = hidden_size
        self.device = device
        self.pos = 0
        self.full = False
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.hiddens = np.zeros((buffer_size, hidden_size), dtype=np.float32)

    def add(self, obs, next_obs, action: int, reward: float, done: float, hidden: np.ndarray) -> None:
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.hiddens[self.pos] = hidden
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> ControlledReplayBufferSamples:
        upper = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        return ControlledReplayBufferSamples(
            observations=torch.tensor(self.observations[idx], device=self.device),
            actions=torch.tensor(self.actions[idx], device=self.device).unsqueeze(1),
            next_observations=torch.tensor(self.next_observations[idx], device=self.device),
            dones=torch.tensor(self.dones[idx], device=self.device).unsqueeze(1),
            rewards=torch.tensor(self.rewards[idx], device=self.device).unsqueeze(1),
            hiddens=torch.tensor(self.hiddens[idx], device=self.device),
        )


class BiologicalRNNController(nn.Module):
    """Small GRU controller for adaptive plasticity, exploration, and replay."""

    def __init__(self, input_dim: int, hidden_size: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 4)
        with torch.no_grad():
            self.out.weight.zero_()
            # Starts near optimized baseline: plasticity=1, lr=1, eps floor=0.05, replay=1.
            self.out.bias.copy_(torch.tensor([4.0, 0.0, -4.0, -4.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        hidden_next = self.cell(x, hidden)
        raw = self.out(hidden_next)
        plasticity_gate = torch.sigmoid(raw[:, 0:1])
        lr_scale = 0.25 + 1.75 * torch.sigmoid(raw[:, 1:2])
        epsilon_floor = cg.END_E + 0.15 * torch.sigmoid(raw[:, 2:3])
        replay_intensity = 1.0 + 3.0 * torch.sigmoid(raw[:, 3:4])
        controls = {
            "plasticity_gate": plasticity_gate,
            "lr_scale": lr_scale,
            "epsilon_floor": epsilon_floor,
            "replay_intensity": replay_intensity,
        }
        return controls, hidden_next


def _controller_input(
    obs_tensor: torch.Tensor,
    q_values: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    episode_progress: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([obs_tensor, q_values.detach(), reward, done, episode_progress], dim=1)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_comparison(results_by_tag: Dict[str, Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for cond in CONDITIONS:
        result = results_by_tag[cond["tag"]]
        returns = np.asarray(result["episode_returns"], dtype=np.float32)
        window = max(1, len(returns) // 200)
        episodes = np.arange(1, len(returns) + 1)
        episodes_sm = episodes[window - 1 :]
        returns_sm = _smooth(returns, window)
        ax.plot(
            episodes_sm,
            returns_sm,
            color=cond["color"],
            lw=2.2,
            label=f"{cond['label']} (last-100: {result['last100_mean']:.1f})",
        )

    ax.axhline(500, color="gray", lw=1.0, ls="--", alpha=0.6, label="max return (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title("Baseline DQN: current vs optimized vs RNN controller")
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def run_rnn_controller(
    cond: Dict,
    seed: int,
    total_episodes: int,
    total_timesteps: int,
    checkpoint_dir: Path,
) -> Dict:
    cg._set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    q_net = cg.BFeedbackConfGateNetwork(obs_dim, n_actions, confidence_gating=False).to(device)
    t_net = cg.BFeedbackConfGateNetwork(obs_dim, n_actions, confidence_gating=False).to(device)
    t_net.load_state_dict(q_net.state_dict())

    controller = BiologicalRNNController(input_dim=obs_dim + n_actions + 3).to(device)
    opt_main = optim.Adam(
        list(q_net.linear_feature.parameters())
        + list(q_net.trunk.parameters())
        + list(q_net.head_full.parameters())
        + list(controller.parameters()),
        lr=cg.LEARNING_RATE,
    )
    opt_aux = optim.Adam(
        list(q_net.head_cart.parameters()) + list(q_net.head_pole.parameters()),
        lr=cg.LEARNING_RATE,
    )

    rb = ControlledReplayBuffer(cg.BUFFER_SIZE, env.observation_space.shape, controller.hidden_size, device)
    episode_returns: List[float] = []
    controller_history: List[Dict] = []
    loss_list: List[float] = []

    obs, _ = env.reset(seed=seed)
    hidden = torch.zeros(1, controller.hidden_size, device=device)
    last_reward = torch.zeros(1, 1, device=device)
    last_done = torch.zeros(1, 1, device=device)
    t = 0

    def one_update(step: int) -> Tuple[float, Dict[str, float]]:
        data = rb.sample(cg.BATCH_SIZE)

        with torch.no_grad():
            tQ_full, tQ_cart, tQ_pole, _, _, _ = t_net.forward(data.next_observations)
            r = data.rewards.flatten()
            d = data.dones.flatten()
            y_full = r + cg.GAMMA * (1 - d) * tQ_full.max(dim=1).values
            y_cart = r + cg.GAMMA * (1 - d) * tQ_cart.max(dim=1).values
            y_pole = r + cg.GAMMA * (1 - d) * tQ_pole.max(dim=1).values

        Q_full, Q_cart, Q_pole, _, _, _ = q_net.forward(data.observations)
        qf_sa = Q_full.gather(1, data.actions).squeeze()
        qc_sa = Q_cart.gather(1, data.actions).squeeze()
        qp_sa = Q_pole.gather(1, data.actions).squeeze()

        loss_full = F.mse_loss(y_full, qf_sa)
        loss_cart = F.mse_loss(y_cart, qc_sa)
        loss_pole = F.mse_loss(y_pole, qp_sa)

        progress = torch.full(
            (cg.BATCH_SIZE, 1),
            min(1.0, len(episode_returns) / max(1, total_episodes)),
            device=device,
        )
        ctrl_in = _controller_input(data.observations, Q_full, data.rewards, data.dones, progress)
        controls, _ = controller(ctrl_in, data.hiddens)
        plasticity = controls["plasticity_gate"].mean()
        lr_scale = controls["lr_scale"].mean()
        effective_loss = loss_full * plasticity * lr_scale

        opt_main.zero_grad()
        effective_loss.backward()
        if cond["main_grad_clip"] > 0:
            nn.utils.clip_grad_norm_(
                list(q_net.linear_feature.parameters())
                + list(q_net.trunk.parameters())
                + list(q_net.head_full.parameters())
                + list(controller.parameters()),
                max_norm=cond["main_grad_clip"],
            )
        opt_main.step()

        opt_aux.zero_grad()
        (loss_cart + loss_pole).backward()
        opt_aux.step()

        loss_list.append(float(loss_full.item()))
        control_log = {
            "step": step,
            "episode": len(episode_returns),
            "plasticity_gate": float(plasticity.item()),
            "lr_scale": float(lr_scale.item()),
            "epsilon_floor": float(controls["epsilon_floor"].mean().item()),
            "replay_intensity": float(controls["replay_intensity"].mean().item()),
        }
        return control_log["replay_intensity"], control_log

    while t < total_timesteps and len(episode_returns) < total_episodes:
        n_ep_done = len(episode_returns)
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net.forward_q_only(x)
            progress = torch.tensor(
                [[min(1.0, n_ep_done / max(1, total_episodes))]],
                dtype=torch.float32,
                device=device,
            )
            ctrl_in = _controller_input(x, q_values, last_reward, last_done, progress)
            controls, hidden_next = controller(ctrl_in, hidden)
            eps_end = float(controls["epsilon_floor"].item())
            eps = cg._linear_schedule(cg.START_E, eps_end, cond["epsilon_decay_episodes"], n_ep_done)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(q_values.argmax(dim=1).item())

        hidden_before = hidden.detach().cpu().numpy().squeeze(0)
        next_obs, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]
        rb.add(obs, real_nxt, action, float(reward), float(done), hidden_before)

        if "episode" in infos:
            episode_returns.append(float(np.asarray(infos["episode"]["r"]).item()))

        obs = next_obs
        hidden = hidden_next.detach()
        last_reward = torch.tensor([[float(reward)]], dtype=torch.float32, device=device)
        last_done = torch.tensor([[float(done)]], dtype=torch.float32, device=device)
        if done:
            obs, _ = env.reset()
            hidden = torch.zeros(1, controller.hidden_size, device=device)
            last_reward.zero_()
            last_done.fill_(1.0)

        if t > cond["learning_starts"] and t % cond["train_frequency"] == 0:
            replay_intensity, control_log = one_update(t)
            extra_updates = max(0, min(3, int(round(replay_intensity)) - 1))
            for _ in range(extra_updates):
                replay_intensity, control_log = one_update(t)
            if t % cg.LOG_EVERY == 0:
                controller_history.append(control_log)

        if t % cond["target_freq"] == 0:
            t_net.load_state_dict(q_net.state_dict())

        if (t + 1) % 100_000 == 0 or t == 0:
            print(
                f"  [rnn_controller] seed={seed} step={t+1}/{total_timesteps}"
                f"  episodes={len(episode_returns)}  eps={eps:.3f}",
                flush=True,
            )
        t += 1

    env.close()

    if len(episode_returns) < total_episodes:
        print(
            f"  [rnn_controller] WARNING seed={seed}: hit step cap {total_timesteps} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"baseline_{cond['tag']}_seed{seed}.pt"
    torch.save(
        {
            "algo": f"baseline_{cond['tag']}",
            "seed": seed,
            "episode_returns": episode_returns,
            "controller_history": controller_history,
            "losses": loss_list,
            "q_network": q_net.state_dict(),
            "controller": controller.state_dict(),
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
        f"  Done seed={seed} rnn_controller: episodes={len(episode_returns)}"
        f"  final={episode_returns[-1] if episode_returns else 0:.0f}"
        f"  last100={last100:.1f}",
        flush=True,
    )
    return {
        "seed": seed,
        "episode_returns": episode_returns,
        "controller_history": controller_history,
        "last100_mean": last100,
    }


def run_condition(cond: Dict, seed: int, total_episodes: int, total_timesteps: int, force: bool) -> Dict:
    ckpt_dir = _ckpt_dir(total_episodes)
    ckpt_path = _ckpt_path(seed, cond["tag"], total_episodes)
    if ckpt_path.exists():
        if force:
            ckpt_path.unlink()
        else:
            print(f">>> [{cond['tag']}] checkpoint exists, loading {ckpt_path}", flush=True)
            return _load_result(seed, cond["tag"], total_episodes)

    print(
        f"\n>>> [{cond['tag']}] running seed={seed} for {total_episodes} episodes "
        f"(step cap {total_timesteps:,})",
        flush=True,
    )
    t0 = time.perf_counter()
    if cond["tag"] == "rnn_controller":
        result = run_rnn_controller(cond, seed, total_episodes, total_timesteps, ckpt_dir)
    else:
        result = cg.run_one(
            seed=seed,
            total_timesteps=total_timesteps,
            confidence_gating=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_dir=ckpt_dir,
            epsilon_decay_episodes=cond["epsilon_decay_episodes"],
            total_episodes=total_episodes,
            target_freq=cond["target_freq"],
            algo_tag_override=f"baseline_{cond['tag']}",
            main_grad_clip=cond["main_grad_clip"],
            learning_starts=cond["learning_starts"],
            train_frequency=cond["train_frequency"],
        )
    print(f">>> [{cond['tag']}] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=750)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Step safety cap. Default is enough for all episodes at CartPole max length.",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoints already exist")
    args = parser.parse_args()

    total_timesteps = args.total_timesteps
    if total_timesteps is None:
        total_timesteps = args.total_episodes * 500 + 10_000

    print("=" * 72, flush=True)
    print("Baseline-only hyperparameter comparison", flush=True)
    print(f"  Seed        : {args.seed}", flush=True)
    print(f"  Episodes    : {args.total_episodes}", flush=True)
    print(f"  Step cap    : {total_timesteps:,}", flush=True)
    print(f"  Checkpoints : {_ckpt_dir(args.total_episodes)}", flush=True)
    print(f"  Figures     : {FIG_DIR}", flush=True)
    print("=" * 72, flush=True)

    results_by_tag: Dict[str, Dict] = {}
    for cond in CONDITIONS:
        if args.plot_only:
            results_by_tag[cond["tag"]] = _load_result(args.seed, cond["tag"], args.total_episodes)
        else:
            results_by_tag[cond["tag"]] = run_condition(
                cond,
                seed=args.seed,
                total_episodes=args.total_episodes,
                total_timesteps=total_timesteps,
                force=args.force,
            )

    print("\n=== Final summary ===", flush=True)
    for cond in CONDITIONS:
        result = results_by_tag[cond["tag"]]
        returns = result["episode_returns"]
        print(
            f"{cond['label']}: episodes={len(returns)} "
            f"final={returns[-1] if returns else 0:.0f} "
            f"last100={result['last100_mean']:.1f}",
            flush=True,
        )

    plot_comparison(
        results_by_tag,
        FIG_DIR / f"baseline_current_vs_optimized_vs_rnn_seed{args.seed}_{args.total_episodes}ep.jpg",
    )


if __name__ == "__main__":
    main()
