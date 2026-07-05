#!/usr/bin/env python3
"""
Bike Rider DQN — Multi-Lane Training Script

Trains a DQN agent on BikeEnvAdvanced for one or more n_lanes values,
then generates diagnostic plots for each run.

Usage:
    python bike_dqn_multi_lanes.py                         # default: 2 3 5 10 20 lanes
    python bike_dqn_multi_lanes.py --lanes 3 5 10          # custom lane counts
    python bike_dqn_multi_lanes.py --timesteps 500000      # fewer training steps
"""

import argparse
import math
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor, load_results

# ── Paths relative to this file ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_DEFAULT_OUTPUT = _HERE / "training_results"
_DEFAULT_PLOT_DIR = _HERE.parent / "plotting" / "5_lanes_example_new"

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class BikeEnvAdvanced(gym.Env):
    """
    Bike riding environment with configurable number of lanes.

    Observation: [x_pos, velocity, hole_dist_lane_0, ..., hole_dist_lane_(n-1)]
    Actions:     0=Left, 1=Stay, 2=Right, 3=Accelerate, 4=Brake
    Reward:      current velocity (per non-fall step)
    """

    def __init__(self, n_lanes: int = 10, *, reset_holes_on_fall: bool = False):
        super().__init__()
        self.n_lanes = n_lanes
        self.reset_holes_on_fall = reset_holes_on_fall
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(2 + n_lanes,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.max_speed = 12.0
        self.visibility_range = 25.0
        self.reset()

    def _sigmoid(self, val: float, threshold: float, steepness: float = 3.0) -> float:
        return 1 / (1 + math.exp(-steepness * (val - threshold)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x_pos = float(self.n_lanes // 2)
        self.velocity = 0.0
        self.angle = 0.0
        self.time_left = 1000
        self.holes_passed = 0
        # self.grace_period_steps = 0
        self.holes = [random.uniform(40, 60) + i * 25 for i in range(self.n_lanes)]
        return self._get_obs(), {}

    def step(self, action):
        self.time_left -= 1
        reward, reason, fell = 0.0, "None", False
        passed_this_step = False

        if action == 0:
            self.angle = -0.45
        elif action == 2:
            self.angle = 0.45
        else:
            self.angle = 0.0

        if action == 3:
            self.velocity = min(self.max_speed, self.velocity + 1.0)
        elif action == 4:
            self.velocity = max(0.0, self.velocity * 0.9)

        self.x_pos += math.sin(self.angle) * self.velocity * 0.1
        self.x_pos = float(np.clip(self.x_pos, 0, self.n_lanes - 1))

        # p_slow = (
            0.0
            if self.grace_period_steps > 0
            else (self._sigmoid(1.5 - self.velocity, 0.5) if self.velocity < 1.5 else 0.0)
        )
        p_turn = self._sigmoid(self.velocity, 8.5) if self.angle != 0 else 0.0
        p_fast = self._sigmoid(self.velocity, 9.0, steepness=5.0)

        current_lane = int(round(self.x_pos))

        if random.random() < p_turn:
            fell, reason = True, "Turn Slip"
        elif random.random() < p_fast:
            fell, reason = True, "Speed Wobble"
        elif abs(self.holes[current_lane]) < 0.8:
            fell, reason = True, "Hole Collision"
        elif random.random() < p_slow:
            fell, reason = True, "Slow Unstable"

        # if self.grace_period_steps > 0:
        #     self.grace_period_steps -= 1

        if fell:
            self.velocity = 0.0
            self.angle = 0.0
            self.time_left -= 50
            #self.grace_period_steps = 5
            if self.reset_holes_on_fall:
                self.holes = [random.uniform(40, 60) + i * 25 for i in range(self.n_lanes)]
        else:
            reward = self.velocity
            for i in range(self.n_lanes):
                old_h = self.holes[i]
                new_h = old_h - self.velocity * 0.1
                if old_h >= 0 and new_h < 0:
                    passed_this_step = True
                    self.holes_passed += 1

        for i in range(self.n_lanes):
            self.holes[i] -= self.velocity * 0.1
            if self.holes[i] < -5:
                self.holes[i] = random.uniform(40, 60)

        terminated = self.time_left <= 0
        info = {"fell": fell, "reason": reason, "passed": passed_this_step}
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        h_obs = [h if h <= self.visibility_range else 100.0 for h in self.holes]
        return np.array([self.x_pos, self.velocity, *h_obs], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_lanes: int,
    base_output: Path,
    total_timesteps: int = 1_000_000,
    seed: int = 1,
    *,
    exploration_fraction: float = 0.25,
    exploration_final_eps: float = 0.025,
    learning_starts: int = 5000,
    batch_size: int = 128,
    reset_holes_on_fall: bool = False,
):
    """Train a DQN agent for a given n_lanes value. Returns (model, log_dir)."""
    log_dir = base_output / f"lanes_{n_lanes}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        BikeEnvAdvanced(n_lanes=n_lanes, reset_holes_on_fall=reset_holes_on_fall),
        str(log_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(log_dir),
        name_prefix=f"bike_model_{n_lanes}lanes",
    )

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        learning_rate=1e-4,
        learning_starts=learning_starts,
        batch_size=batch_size,
        seed=seed,
        verbose=0,
    )

    print(f"  [n_lanes={n_lanes}] Training {total_timesteps:,} steps …")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    save_path = log_dir / f"bike_model_final_{n_lanes}lanes"
    model.save(str(save_path))
    env.close()
    print(f"  [n_lanes={n_lanes}] Model saved → {save_path}")
    return model, log_dir


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_NAMES = {0: "Left", 1: "Stay", 2: "Right", 3: "Accelerate", 4: "Brake"}
_ACTION_COLORS = {"Left": "orange", "Stay": "gray", "Right": "orange",
                  "Accelerate": "green", "Brake": "red"}


def _save(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


def plot_learning_curves(df: pd.DataFrame, n_lanes: int, out_dir: Path):
    """Episode rewards and episode length with moving averages."""
    window = 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Learning Curves — {n_lanes} Lanes", fontsize=14, fontweight="bold")

    # Reward
    axes[0].plot(df.index, df["r"], alpha=0.2, color="steelblue", linewidth=0.6)
    rm = df["r"].rolling(window=window, min_periods=1)
    axes[0].plot(df.index, rm.mean(), color="red", linewidth=2, label=f"{window}-ep MA")
    axes[0].fill_between(df.index, rm.quantile(0.25), rm.quantile(0.75),
                         alpha=0.2, color="red", label="IQR")
    axes[0].set(xlabel="Episode", ylabel="Total Reward", title="Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode length
    axes[1].plot(df.index, df["l"], alpha=0.2, color="seagreen", linewidth=0.6)
    rl = df["l"].rolling(window=window, min_periods=1).mean()
    axes[1].plot(df.index, rl, color="darkgreen", linewidth=2, label=f"{window}-ep MA")
    axes[1].set(xlabel="Episode", ylabel="Episode Length (steps)",
                title="Episode Length Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / f"lanes_{n_lanes}_01_learning_curves.png")


def plot_moving_averages(df: pd.DataFrame, n_lanes: int, out_dir: Path):
    """Multi-scale moving averages of episode reward."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Multi-Scale Moving Averages — {n_lanes} Lanes",
                 fontsize=14, fontweight="bold")
    for window in [50, 100, 200, 500]:
        if len(df) >= window:
            ax.plot(df.index,
                    df["r"].rolling(window=window, min_periods=1).mean(),
                    linewidth=2, label=f"{window}-ep window")
    ax.set(xlabel="Episode", ylabel="Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / f"lanes_{n_lanes}_02_moving_averages.png")


def plot_fall_analysis(model, n_lanes: int, out_dir: Path,
                       n_eval_episodes: int = 100,
                       reset_holes_on_fall: bool = False):
    """Fall reasons, timing, conditions, and falls-per-episode histogram."""
    fall_rows, surv_rows = [], []

    for ep in range(n_eval_episodes):
        env = BikeEnvAdvanced(n_lanes=n_lanes, reset_holes_on_fall=reset_holes_on_fall)
        obs, _ = env.reset()
        done, step = False, 0
        ep_falls = []

        while not done and step < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if info["fell"]:
                fall_rows.append({
                    "episode": ep, "step": step,
                    "reason": info["reason"],
                    "velocity": env.velocity,
                    "position": env.x_pos,
                })
                ep_falls.append(step)
            step += 1
            done = term or trunc

        surv_rows.append({"episode": ep, "final_step": step,
                          "num_falls": len(ep_falls), "fall_steps": ep_falls})
        env.close()

    fall_df = pd.DataFrame(fall_rows)
    surv_df = pd.DataFrame(surv_rows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Fall Analysis — {n_lanes} Lanes", fontsize=14, fontweight="bold")

    if len(fall_df) > 0:
        # Fall reasons bar
        fc = fall_df["reason"].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(fc)))
        axes[0, 0].bar(fc.index, fc.values, color=colors, edgecolor="black", alpha=0.8)
        for i, (r, c) in enumerate(fc.items()):
            axes[0, 0].text(i, c, f"{c / len(fall_df) * 100:.1f}%",
                            ha="center", va="bottom", fontweight="bold")
        axes[0, 0].set(ylabel="Number of Falls", title="Fall Reasons")
        axes[0, 0].tick_params(axis="x", rotation=30)
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # Falls per episode histogram
        axes[0, 1].hist(surv_df["num_falls"],
                        bins=range(0, int(surv_df["num_falls"].max()) + 2),
                        color="coral", edgecolor="black", alpha=0.7)
        axes[0, 1].set(xlabel="Number of Falls", ylabel="Number of Episodes",
                       title=f"Falls per Episode  (mean={surv_df['num_falls'].mean():.2f})")
        axes[0, 1].grid(True, alpha=0.3)

        # Fall conditions scatter (position vs velocity)
        codes = fall_df["reason"].astype("category").cat.codes
        axes[1, 0].scatter(fall_df["position"], fall_df["velocity"],
                           c=codes, cmap="tab10", alpha=0.6, s=80,
                           edgecolors="black", linewidth=0.4)
        for i, r in enumerate(fall_df["reason"].unique()):
            axes[1, 0].scatter([], [], c=f"C{i}", label=r, s=80, edgecolors="black")
        axes[1, 0].set(xlabel="Lane Position", ylabel="Velocity at Fall",
                       title="Fall Conditions (Position vs Velocity)")
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        # Fall timing histogram
        axes[1, 1].hist(fall_df["step"], bins=50,
                        color="indianred", edgecolor="black", alpha=0.7)
        axes[1, 1].set(xlabel="Step Number", ylabel="Number of Falls",
                       title="When Falls Occur During Episodes")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No falls recorded!", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout()
    _save(fig, out_dir / f"lanes_{n_lanes}_03_fall_analysis.png")


def plot_action_frequency(model, n_lanes: int, out_dir: Path,
                          n_steps: int = 500,
                          reset_holes_on_fall: bool = False):
    """Bar chart of how often each action is chosen."""
    env = BikeEnvAdvanced(n_lanes=n_lanes, reset_holes_on_fall=reset_holes_on_fall)
    obs, _ = env.reset()
    actions_taken = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=False)
        a_int = action.item() if hasattr(action, "item") else int(action)
        obs, _, term, trunc, _ = env.step(action)
        actions_taken.append(a_int)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    counts = pd.Series(actions_taken).value_counts().sort_index()
    counts.index = [_ACTION_NAMES.get(i, f"Action {i}") for i in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [_ACTION_COLORS.get(name, "blue") for name in counts.index]
    counts.plot(kind="bar", color=bar_colors, edgecolor="black", alpha=0.8, ax=ax)
    ax.set(title=f"Action Frequency — {n_lanes} Lanes",
           ylabel="Times Selected", xlabel="Action")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, out_dir / f"lanes_{n_lanes}_04_action_frequency.png")


def plot_trajectory_heatmap(model, n_lanes: int, out_dir: Path,
                             n_episodes: int = 50,
                             reset_holes_on_fall: bool = False):
    """2-D density heatmaps: lane usage over episode progress + speed vs lane."""
    positions, velocities, step_pcts = [], [], []

    for _ in range(n_episodes):
        env = BikeEnvAdvanced(n_lanes=n_lanes, reset_holes_on_fall=reset_holes_on_fall)
        obs, _ = env.reset()
        done, step, max_steps = False, 0, 1000

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            positions.append(env.x_pos)
            velocities.append(env.velocity)
            step_pcts.append(step / max_steps * 100)
            step += 1
            done = term or trunc
        env.close()

    traj_df = pd.DataFrame({"position": positions, "velocity": velocities,
                             "step_pct": step_pcts})

    lane_bins = max(10, n_lanes * 3)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Trajectory Heatmaps — {n_lanes} Lanes", fontsize=14, fontweight="bold")

    # Lane usage over episode progress
    h1 = axes[0].hist2d(traj_df["step_pct"], traj_df["position"],
                        bins=[50, lane_bins], cmap="YlOrRd", cmin=1)
    plt.colorbar(h1[3], ax=axes[0]).set_label("Frequency", rotation=270, labelpad=15)
    for lane in range(n_lanes):
        axes[0].axhline(y=lane, color="white", linestyle="--", alpha=0.35, linewidth=0.8)
    axes[0].set(xlabel="Progress Through Episode (%)", ylabel="Lane Position",
                title="Lane Usage Over Episode Progress")

    # Speed vs lane position
    h2 = axes[1].hist2d(traj_df["position"], traj_df["velocity"],
                        bins=[lane_bins, 30], cmap="viridis", cmin=1)
    plt.colorbar(h2[3], ax=axes[1]).set_label("Frequency", rotation=270, labelpad=15)
    for lane in range(n_lanes):
        axes[1].axvline(x=lane, color="white", linestyle="--", alpha=0.35, linewidth=0.8)
    axes[1].set(xlabel="Lane Position", ylabel="Velocity",
                title="Speed-Position Heatmap")

    plt.tight_layout()
    _save(fig, out_dir / f"lanes_{n_lanes}_05_trajectory_heatmap.png")


def plot_all_lanes_comparison(all_data: dict, out_dir: Path):
    """Overlay moving-average reward curves for all trained lane counts."""
    window = 100
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Learning Curves — All Lane Counts (100-ep MA)",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    for (n_lanes, df), color in zip(all_data.items(), colors):
        ma = df["r"].rolling(window=window, min_periods=1).mean()
        ax.plot(df.index, ma, linewidth=2, color=color, label=f"{n_lanes} lanes")

    ax.set(xlabel="Episode", ylabel="Mean Reward (100-ep MA)")
    ax.legend(title="n_lanes")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "all_lanes_00_comparison.png")


def generate_all_plots(model, log_dir: Path, n_lanes: int, out_dir: Path,
                       reset_holes_on_fall: bool = False):
    """Load training data and generate all per-run diagnostic plots."""
    print(f"  [n_lanes={n_lanes}] Generating plots …")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(str(log_dir))
    print(f"    Loaded {len(df):,} episodes")

    plot_learning_curves(df, n_lanes, out_dir)
    plot_moving_averages(df, n_lanes, out_dir)
    plot_fall_analysis(model, n_lanes, out_dir, reset_holes_on_fall=reset_holes_on_fall)
    plot_action_frequency(model, n_lanes, out_dir, reset_holes_on_fall=reset_holes_on_fall)
    plot_trajectory_heatmap(model, n_lanes, out_dir, reset_holes_on_fall=reset_holes_on_fall)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bike Rider DQN — multi-lane training and plotting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lanes", nargs="+", type=int, default=[2, 3, 5, 10, 20],
                        metavar="N", help="Lane counts to train sequentially")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Training timesteps per run")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT,
                        help="Root directory for training logs/models")
    parser.add_argument("--plot-dir", type=Path, default=_DEFAULT_PLOT_DIR,
                        help="Directory for output plots")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for DQN and environment")
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-final-eps", type=float, default=0.025)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--reset-holes-on-fall",
        action="store_true",
        help="Randomly reinitialize all hole positions after a fall",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = {}

    for n_lanes in args.lanes:
        print(f"\n{'='*60}")
        print(f"  N_LANES = {n_lanes}")
        print(f"{'='*60}")
        model, log_dir = train(
            n_lanes,
            args.output_dir,
            args.timesteps,
            seed=args.seed,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            reset_holes_on_fall=args.reset_holes_on_fall,
        )
        df = generate_all_plots(
            model, log_dir, n_lanes, args.plot_dir,
            reset_holes_on_fall=args.reset_holes_on_fall,
        )
        all_dfs[n_lanes] = df

    # Comparison plot across all lane counts
    if len(all_dfs) > 1:
        print("\nGenerating cross-lane comparison plot …")
        plot_all_lanes_comparison(all_dfs, args.plot_dir)

    print(f"\nAll runs complete. Plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
