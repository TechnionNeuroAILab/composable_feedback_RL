"""Additional helper plots from the notebook (cells 4–9)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import load_results as load_results_ts
from stable_baselines3.common.results_plotter import ts2xy

from ..env import BikeEnvAdvanced

PathLike = Union[str, Path]


def plot_comprehensive_performance(log_path: PathLike, show: bool = True) -> None:
    """Three-panel view from raw ``monitor.csv`` (skips SB3 header row)."""
    log_path = Path(log_path)
    csv_file = log_path / "monitor.csv"
    if not csv_file.exists():
        print(f"Error: Could not find {csv_file}")
        return
    data = pd.read_csv(csv_file, skiprows=1)
    window = 50
    _, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(data["r"], alpha=0.2, color="gray")
    axs[0].plot(data["r"].rolling(window=window).mean(), color="red", linewidth=2)
    axs[0].set_title("1. Learning Curve (Reward)")
    axs[0].set_ylabel("Total Reward")
    axs[0].set_xlabel("Episode")

    axs[1].plot(data["l"], alpha=0.2, color="gray")
    axs[1].plot(data["l"].rolling(window=window).mean(), color="blue", linewidth=2)
    axs[1].set_title("2. Survival Time (Steps)")
    axs[1].set_ylabel("Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].scatter(data.index, data["r"], c=data["l"], cmap="viridis", alpha=0.5, s=10)
    axs[2].set_title("3. Efficiency Map (Reward vs Length)")
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Episode")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_results_timesteps(
    log_folder: PathLike,
    title: str = "DQN Bike Training",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """Learning curve vs timesteps + episode lengths (uses ``ts2xy``)."""
    results = load_results_ts(str(log_folder))

    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        weights = np.ones(window) / window
        return np.convolve(values, weights, "valid")

    x, y = ts2xy(results, "timesteps")
    y_moving = moving_average(y, window=50)
    x_moving = x[len(x) - len(y_moving) :]

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x, y, alpha=0.2, color="blue", label="Raw Episode Reward")
    plt.plot(x_moving, y_moving, color="blue", linewidth=2, label="Moving Average (50 eps)")
    plt.title(f"{title} - Learning Curve")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(results["l"], color="green", alpha=0.6)
    plt.title("Episode Length (Steps until Terminated/Fell)")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def visualize_behavior(
    model: DQN,
    env: BikeEnvAdvanced,
    num_steps: int = 300,
    save_path: Optional[PathLike] = "agent_behavior.png",
    show: bool = True,
) -> None:
    """Single-episode lane and velocity (raw env, not Monitor)."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    history = {"lane": [], "velocity": [], "steps": []}
    for i in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _info = env.step(int(action))
        history["lane"].append(env.x_pos)
        history["velocity"].append(env.velocity)
        history["steps"].append(i)
        if done:
            break
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(history["steps"], history["lane"], "g-", linewidth=2)
    ax1.set_yticks(list(range(int(env.n_lanes))))
    ax1.set_yticklabels([f"Lane {j}" for j in range(int(env.n_lanes))])
    ax1.set_ylabel("Lane Position")
    ax1.set_title("Agent Trajectory (Lane Changes)")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.plot(history["steps"], history["velocity"], "b-")
    ax2.set_ylabel("Velocity")
    ax2.set_xlabel("Steps")
    ax2.set_title("Velocity Profile")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_behavioral_profile(model: DQN, env, show: bool = True) -> None:
    """Velocity with fall annotations; ``env`` may be Monitor-wrapped."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    steps, vels, rewards, falls = [], [], [], []
    for t in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        steps.append(t)
        vels.append(float(obs[1]))
        rewards.append(reward)
        if info.get("fell"):
            falls.append((t, float(obs[1]), info.get("reason")))
        if done:
            break
    _, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(steps, vels, color="dodgerblue", label="Velocity", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Velocity", color="dodgerblue")
    for step, vel, reason in falls:
        ax1.scatter(step, vel, color="red", s=100, zorder=5)
        ax1.annotate(str(reason), (step, vel), xytext=(5, 5), textcoords="offset points", color="red", weight="bold")
    plt.title("Behavioral Profile: Velocity and Fall Events")
    plt.grid(True, alpha=0.2)
    if show:
        plt.show()
    else:
        plt.close()


def plot_action_distribution(model: DQN, env, show: bool = True) -> None:
    """Stochastic policy rollout histogram (~500 steps)."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    actions_taken: list[int] = []
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=False)
        action_int = int(action.item()) if hasattr(action, "item") else int(action)
        obs, _r, done, _info = env.step(action_int)
        actions_taken.append(action_int)
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    labels_map = {0: "Left", 1: "Stay", 2: "Right", 3: "Fast", 4: "Slow"}
    counts = pd.Series(actions_taken).value_counts().sort_index()
    counts.index = [labels_map.get(i, f"Action {i}") for i in counts.index]
    plt.figure(figsize=(8, 4))
    colors = ["orange", "gray", "orange", "green", "red"]
    counts.plot(kind="bar", color=colors[: len(counts)], edgecolor="black", alpha=0.8)
    plt.title("Action Frequency (What is the agent choosing most?)")
    plt.ylabel("Number of times selected")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    if show:
        plt.show()
    else:
        plt.close()


def compare_model_velocity_traces(
    checkpoint_paths: Sequence[PathLike],
    env,
    steps: int = 200,
    show: bool = True,
) -> None:
    """Overlay velocity trajectories from multiple checkpoints (notebook cell 8)."""
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    plt.figure(figsize=(12, 6))
    for i, path in enumerate(checkpoint_paths):
        temp_model = DQN.load(str(path))
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        vels = []
        for _t in range(steps):
            action, _ = temp_model.predict(obs, deterministic=True)
            obs, _r, done, _info = env.step(int(action))
            vels.append(float(obs[1]))
            if done:
                break
        label = Path(path).stem
        plt.plot(vels, label=label, color=colors[i % len(colors)], linewidth=2)
    plt.title("Evolution of Velocity Control across Checkpoints")
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.close()


def plot_velocity_and_episode_length_rolling(
    log_dir: PathLike,
    window: int = 100,
    show: bool = True,
) -> None:
    """Rolling velocity proxy (reward) and episode length from ``monitor.csv``."""
    log_dir = Path(log_dir)
    results_file = log_dir / "monitor.csv"
    df = pd.read_csv(results_file, skiprows=1)
    df["rolling_velocity"] = df["r"].rolling(window=window).mean()
    df["episode"] = df.index
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["episode"], df["rolling_velocity"], color="blue", label="Avg Velocity (proxy)")
    plt.title(f"Velocity Over Episodes (Rolling {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward / Velocity proxy")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(df["episode"], df["l"].rolling(window=window).mean(), color="green")
    plt.title("Avg Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps Survived")
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()
