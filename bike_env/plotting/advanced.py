"""Evaluation-time plots: falls, survival, behavior, heatmaps, GIF (notebook cell 3)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from stable_baselines3 import DQN

from ..env import BikeEnvAdvanced

PathLike = Union[str, Path]

ACTION_NAMES = ["Left", "Stay", "Right", "Accelerate", "Brake"]


def _setup() -> None:
    if _HAS_SNS:
        sns.set_style("whitegrid")


def collect_fall_and_survival_data(
    model: DQN,
    n_lanes: int = 3,
    n_eval_episodes: int = 100,
    max_steps: int = 1000,
):
    fall_data: List[dict] = []
    survival_data: List[dict] = []

    for episode in range(n_eval_episodes):
        env = BikeEnvAdvanced(n_lanes=n_lanes)
        obs = env.reset()
        done = False
        step = 0
        episode_falls: List[int] = []

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, info = env.step(int(action))
            if info.get("fell", False):
                fall_data.append(
                    {
                        "episode": episode,
                        "step": step,
                        "reason": info.get("reason", "Unknown"),
                        "velocity": env.velocity,
                        "position": env.x_pos,
                        "angle": env.angle,
                    }
                )
                episode_falls.append(step)
            step += 1

        survival_data.append(
            {
                "episode": episode,
                "final_step": step,
                "num_falls": len(episode_falls),
                "fall_steps": episode_falls,
            }
        )

    return pd.DataFrame(fall_data), pd.DataFrame(survival_data), survival_data


def plot_fall_probability_analysis(
    fall_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    output_dir: PathLike,
    n_eval_episodes: int,
    filename: str = "fall_probability_analysis.png",
    show: bool = False,
) -> Path:
    _setup()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Fall Analysis", fontsize=16, fontweight="bold")

    if len(fall_df) > 0:
        fall_counts = fall_df["reason"].value_counts()
        colors = plt.cm.Set3(range(len(fall_counts)))
        axes[0, 0].bar(fall_counts.index, fall_counts.values, color=colors, edgecolor="black", alpha=0.8)
        axes[0, 0].set_ylabel("Number of Falls", fontsize=12)
        axes[0, 0].set_title("Fall Reasons Distribution", fontsize=13, fontweight="bold")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis="y")
        total_falls = len(fall_df)
        for i, (_reason, count) in enumerate(fall_counts.items()):
            pct = (count / total_falls) * 100
            axes[0, 0].text(i, count, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")

        falls_per_ep = survival_df["num_falls"]
        max_falls = int(falls_per_ep.max()) if len(falls_per_ep) else 0
        axes[0, 1].hist(
            falls_per_ep,
            bins=range(0, max_falls + 2),
            color="coral",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Number of Falls", fontsize=12)
        axes[0, 1].set_ylabel("Number of Episodes", fontsize=12)
        mean_falls = falls_per_ep.mean() if len(falls_per_ep) else 0.0
        axes[0, 1].set_title(f"Falls per Episode (Mean: {mean_falls:.2f})", fontsize=13, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(
            fall_df["position"],
            fall_df["velocity"],
            c=fall_df["reason"].astype("category").cat.codes,
            cmap="tab10",
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[1, 0].set_xlabel("Lane Position", fontsize=12)
        axes[1, 0].set_ylabel("Velocity at Fall", fontsize=12)
        axes[1, 0].set_title("Fall Conditions (Position vs Velocity)", fontsize=13, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        unique_reasons = fall_df["reason"].unique()
        for i, reason in enumerate(unique_reasons):
            axes[1, 0].scatter([], [], c=f"C{i}", label=reason, s=100, edgecolors="black")
        axes[1, 0].legend(loc="best", fontsize=9)

        axes[1, 1].hist(fall_df["step"], bins=50, color="indianred", edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("Step Number", fontsize=12)
        axes[1, 1].set_ylabel("Number of Falls", fontsize=12)
        axes[1, 1].set_title("When Falls Occur During Episodes", fontsize=13, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No falls recorded!", ha="center", va="center", fontsize=14)
            ax.axis("off")

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_survival_analysis(
    survival_df: pd.DataFrame,
    survival_data: list,
    output_dir: PathLike,
    n_eval_episodes: int,
    filename: str = "survival_analysis.png",
    show: bool = False,
) -> Path:
    _setup()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Survival Analysis", fontsize=16, fontweight="bold")

    if len(survival_df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        path = output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    max_steps = int(max(survival_df["final_step"]))
    survival_curve = np.zeros(max_steps + 1)
    survival_curve[0] = n_eval_episodes
    for episode_data in survival_data:
        final_step = episode_data["final_step"]
        survival_curve[1 : final_step + 1] += 1

    survival_pct = (survival_curve / n_eval_episodes) * 100
    steps_pct = np.arange(len(survival_curve)) / max(max_steps, 1) * 100

    axes[0].plot(steps_pct, survival_pct, linewidth=3, color="darkblue", label="Survival Rate")
    axes[0].fill_between(steps_pct, 0, survival_pct, alpha=0.3, color="blue")
    axes[0].axhline(y=50, color="red", linestyle="--", alpha=0.5, linewidth=2, label="50% Survival")
    axes[0].set_xlabel("Progress Through Episode (%)", fontsize=12)
    axes[0].set_ylabel("Survival Rate (%)", fontsize=12)
    axes[0].set_title("Episode Survival Curve", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    median_survival_idx = np.where(survival_pct <= 50)[0]
    median_pct: Optional[float] = None
    if len(median_survival_idx) > 0:
        median_pct = float(steps_pct[median_survival_idx[0]])
        axes[0].axvline(x=median_pct, color="orange", linestyle=":", linewidth=2, alpha=0.7)
        axes[0].text(median_pct + 2, 55, f"Median: {median_pct:.1f}%", fontsize=10, color="orange", fontweight="bold")

    fall_steps_all: List[int] = []
    for ep_data in survival_data:
        fall_steps_all.extend(ep_data["fall_steps"])

    if len(fall_steps_all) > 0:
        fall_steps_pct = np.array(fall_steps_all) / max(max_steps, 1) * 100
        bins = 50
        hist, bin_edges = np.histogram(fall_steps_pct, bins=bins, range=(0, 100))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axes[1].bar(bin_centers, hist, width=100 / bins, color="crimson", edgecolor="darkred", alpha=0.7, label="Falls per bin")
        axes[1].set_xlabel("Progress Through Episode (%)", fontsize=12)
        axes[1].set_ylabel("Number of Falls", fontsize=12)
        axes[1].set_title("Fall Distribution Over Episode Progress", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].legend(fontsize=11)
    else:
        axes[1].text(50, 0.5, "No falls recorded", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_behavioral_single_episode(
    model: DQN,
    n_lanes: int,
    output_dir: PathLike,
    max_steps: int = 500,
    filename: str = "behavioral_analysis.png",
    show: bool = False,
) -> Path:
    _setup()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = BikeEnvAdvanced(n_lanes=n_lanes)
    obs = env.reset()
    done = False
    step = 0

    behavior_log = {
        "step": [],
        "position": [],
        "velocity": [],
        "action": [],
        "angle": [],
        "reward": [],
        "holes": [[] for _ in range(n_lanes)],
        "turned": [],
        "passed_hole": [],
        "fell": [],
    }

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        behavior_log["step"].append(step)
        behavior_log["position"].append(env.x_pos)
        behavior_log["velocity"].append(env.velocity)
        behavior_log["action"].append(int(action))
        behavior_log["angle"].append(env.angle)
        behavior_log["reward"].append(reward)
        for i in range(n_lanes):
            behavior_log["holes"][i].append(env.holes[i])
        behavior_log["turned"].append(abs(env.angle) > 0.1)
        behavior_log["passed_hole"].append(info.get("passed", False))
        behavior_log["fell"].append(info.get("fell", False))
        step += 1

    steps = behavior_log["step"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle("Detailed Behavioral Analysis (Single Episode)", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(steps, behavior_log["position"], linewidth=2.5, color="blue", label="Bike Position", zorder=3)
    points = ax1.scatter(
        steps,
        behavior_log["position"],
        c=behavior_log["velocity"],
        cmap="RdYlGn",
        s=30,
        zorder=4,
        edgecolors="black",
        linewidth=0.3,
    )
    plt.colorbar(points, ax=ax1).set_label("Velocity", rotation=270, labelpad=15)

    for lane in range(n_lanes):
        hole_positions = behavior_log["holes"][lane]
        lane_holes_x = []
        lane_holes_y = []
        for s, h in zip(steps, hole_positions):
            if abs(h) < 0.8:
                lane_holes_x.append(s)
                lane_holes_y.append(lane)
        if lane_holes_x:
            ax1.scatter(
                lane_holes_x,
                lane_holes_y,
                marker="x",
                s=200,
                color="red",
                linewidth=3,
                label="Hole" if lane == 0 else "",
                zorder=5,
            )

    turn_indices = [i for i, turned in enumerate(behavior_log["turned"]) if turned]
    if turn_indices:
        turn_steps = [steps[i] for i in turn_indices[::5]]
        turn_positions = [behavior_log["position"][i] for i in turn_indices[::5]]
        ax1.scatter(
            turn_steps,
            turn_positions,
            marker="^",
            s=100,
            color="yellow",
            edgecolors="black",
            linewidth=1,
            label="Turning",
            zorder=6,
            alpha=0.7,
        )

    pass_indices = [i for i, passed in enumerate(behavior_log["passed_hole"]) if passed]
    if pass_indices:
        pass_steps = [steps[i] for i in pass_indices]
        pass_positions = [behavior_log["position"][i] for i in pass_indices]
        ax1.scatter(
            pass_steps,
            pass_positions,
            marker="*",
            s=300,
            color="gold",
            edgecolors="black",
            linewidth=1.5,
            label="Passed Hole",
            zorder=7,
        )

    fall_indices = [i for i, fell in enumerate(behavior_log["fell"]) if fell]
    if fall_indices:
        fall_steps = [steps[i] for i in fall_indices]
        fall_positions = [behavior_log["position"][i] for i in fall_indices]
        ax1.scatter(
            fall_steps,
            fall_positions,
            marker="X",
            s=400,
            color="red",
            edgecolors="black",
            linewidth=2,
            label="Crash",
            zorder=8,
        )

    for lane in range(n_lanes):
        ax1.axhline(y=lane, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Lane Position", fontsize=12)
    ax1.set_title("Track View with Events", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.5, n_lanes - 0.5)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.plot(steps, behavior_log["velocity"], linewidth=2, color="green", label="Velocity")
    ax2.fill_between(steps, 0, behavior_log["velocity"], alpha=0.3, color="green")
    if turn_indices:
        turn_steps_all = [steps[i] for i in turn_indices]
        turn_velocities = [behavior_log["velocity"][i] for i in turn_indices]
        ax2.scatter(turn_steps_all, turn_velocities, marker="^", s=50, color="yellow", edgecolors="black", alpha=0.5, zorder=5)
    if fall_indices:
        fall_velocities = [behavior_log["velocity"][i] for i in fall_indices]
        fs = [steps[i] for i in fall_indices]
        ax2.scatter(fs, fall_velocities, marker="X", s=200, color="red", edgecolors="black", linewidth=2, zorder=6)
    ax2.axhline(y=env.max_speed, color="red", linestyle="--", alpha=0.5, label=f"Max Speed ({env.max_speed})")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Velocity", fontsize=12)
    ax2.set_title("Velocity Over Time", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    action_colors = {"Left": "blue", "Stay": "gray", "Right": "red", "Accelerate": "green", "Brake": "orange"}
    for action_idx, action_name in enumerate(ACTION_NAMES):
        action_steps = [s for s, a in zip(steps, behavior_log["action"]) if a == action_idx]
        if action_steps:
            ax3.scatter(
                action_steps,
                [action_idx] * len(action_steps),
                marker="s",
                s=20,
                color=action_colors[action_name],
                label=action_name,
                alpha=0.6,
            )
    ax3.set_xlabel("Step", fontsize=12)
    ax3.set_ylabel("Action", fontsize=12)
    ax3.set_yticks(range(5))
    ax3.set_yticklabels(ACTION_NAMES)
    ax3.set_title("Action Timeline", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10, ncol=5)
    ax3.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_trajectory_heatmaps(
    model: DQN,
    n_lanes: int,
    output_dir: PathLike,
    n_heatmap_episodes: int = 50,
    max_steps: int = 1000,
    filename: str = "trajectory_heatmap.png",
    show: bool = False,
) -> Path:
    _setup()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_data = {"position": [], "velocity": [], "step_pct": []}
    for _episode in range(n_heatmap_episodes):
        env = BikeEnvAdvanced(n_lanes=n_lanes)
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, done, _i = env.step(int(action))
            trajectory_data["position"].append(env.x_pos)
            trajectory_data["velocity"].append(env.velocity)
            trajectory_data["step_pct"].append((step / max_steps) * 100)
            step += 1

    traj_df = pd.DataFrame(trajectory_data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Trajectory Heatmaps", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    h1 = ax1.hist2d(traj_df["step_pct"], traj_df["position"], bins=[50, 30], cmap="YlOrRd", cmin=1)
    plt.colorbar(h1[3], ax=ax1).set_label("Frequency", rotation=270, labelpad=15)
    ax1.set_xlabel("Progress Through Episode (%)", fontsize=12)
    ax1.set_ylabel("Lane Position", fontsize=12)
    ax1.set_title("Lane Usage Heatmap", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.5, n_lanes - 0.5)
    for lane in range(n_lanes):
        ax1.axhline(y=lane, color="white", linestyle="--", alpha=0.5, linewidth=1)

    ax2 = axes[1]
    h2 = ax2.hist2d(traj_df["position"], traj_df["velocity"], bins=[30, 30], cmap="viridis", cmin=1)
    plt.colorbar(h2[3], ax=ax2).set_label("Frequency", rotation=270, labelpad=15)
    ax2.set_xlabel("Lane Position", fontsize=12)
    ax2.set_ylabel("Velocity", fontsize=12)
    ax2.set_title("Speed-Position Heatmap", fontsize=13, fontweight="bold")
    ax2.set_xlim(-0.5, n_lanes - 0.5)
    for lane in range(n_lanes):
        ax2.axvline(x=lane, color="white", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def generate_bike_animation_gif(
    model: DQN,
    n_lanes: int,
    output_dir: PathLike,
    max_steps: int = 300,
    fps: int = 20,
    filename: str = "bike_agent_animation.gif",
    show: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = BikeEnvAdvanced(n_lanes=n_lanes)
    obs = env.reset()
    done = False
    step = 0
    frames_data: List[dict] = []

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        frame = {
            "step": step,
            "position": env.x_pos,
            "velocity": env.velocity,
            "angle": env.angle,
            "holes": env.holes.copy(),
            "action": action,
            "action_name": ACTION_NAMES[action],
        }
        frames_data.append(frame)
        obs, _r, done, _info = env.step(action)
        step += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Bike Agent Behavior", fontsize=14, fontweight="bold")

    def animate(frame_idx: int):
        ax1.clear()
        ax2.clear()
        frame = frames_data[frame_idx]
        ax1.set_xlim(-2, 30)
        ax1.set_ylim(-0.5, n_lanes - 0.5)
        ax1.set_xlabel("Distance Ahead", fontsize=11)
        ax1.set_ylabel("Lane", fontsize=11)
        ax1.set_title(
            f"Step {frame['step']} | Action: {frame['action_name']} | Velocity: {frame['velocity']:.1f}",
            fontsize=12,
            fontweight="bold",
        )
        for lane in range(n_lanes):
            ax1.axhline(y=lane, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
            ax1.fill_between([-2, 30], lane - 0.4, lane + 0.4, color="lightgray", alpha=0.2)
        for lane, hole_dist in enumerate(frame["holes"]):
            if 0 <= hole_dist <= 25:
                ax1.add_patch(Circle((hole_dist, lane), 0.3, color="black", zorder=3))
                if hole_dist < 5:
                    ax1.add_patch(Circle((hole_dist, lane), 0.5, color="red", alpha=0.3, zorder=2))
        bike_x = 0
        bike_y = frame["position"]
        angle_deg = np.degrees(frame["angle"])
        ax1.add_patch(
            Rectangle(
                (bike_x - 0.5, bike_y - 0.15),
                1.0,
                0.3,
                angle=angle_deg,
                color="blue",
                zorder=5,
                alpha=0.8,
            )
        )
        direction_x = bike_x + np.cos(frame["angle"]) * 0.7
        direction_y = bike_y + np.sin(frame["angle"]) * 0.7
        ax1.plot([bike_x, direction_x], [bike_y, direction_y], "r-", linewidth=3, zorder=6)
        speed_bar_length = frame["velocity"] / env.max_speed * 2
        ax1.arrow(
            bike_x,
            bike_y + 0.5,
            speed_bar_length,
            0,
            head_width=0.15,
            head_length=0.3,
            fc="green",
            ec="green",
            alpha=0.7,
            zorder=4,
        )
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        history_length = min(50, frame_idx + 1)
        start_idx = max(0, frame_idx - history_length + 1)
        history_frames = frames_data[start_idx : frame_idx + 1]
        history_steps = [f["step"] for f in history_frames]
        history_velocities = [f["velocity"] for f in history_frames]
        ax2.plot(history_steps, history_velocities, "g-", linewidth=2)
        ax2.scatter([frame["step"]], [frame["velocity"]], color="red", s=100, zorder=5, edgecolors="black")
        ax2.axhline(y=env.max_speed, color="red", linestyle="--", alpha=0.5, label=f"Max Speed ({env.max_speed})")
        ax2.set_xlabel("Step", fontsize=11)
        ax2.set_ylabel("Velocity", fontsize=11)
        ax2.set_title("Velocity History", fontsize=12)
        ax2.set_ylim(-1, env.max_speed + 2)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

    anim = FuncAnimation(fig, animate, frames=len(frames_data), interval=50, repeat=True)
    gif_path = output_dir / filename
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))
    plt.close()
    return gif_path


def run_all_advanced_visualizations(
    model_path: PathLike,
    n_lanes: int = 3,
    output_dir: Optional[PathLike] = None,
    n_eval_episodes: int = 100,
    n_heatmap_episodes: int = 50,
    gif_max_steps: int = 300,
    show: bool = False,
) -> dict:
    """
    Load a saved DQN and generate fall/survival/behavior/heatmap/GIF figures.

    ``output_dir`` defaults to ``<parent of model>/visualizations_advanced``.
    """
    model_path = Path(model_path)
    model = DQN.load(str(model_path))
    if output_dir is None:
        output_dir = model_path.parent / "visualizations_advanced"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fall_df, survival_df, survival_data = collect_fall_and_survival_data(
        model, n_lanes=n_lanes, n_eval_episodes=n_eval_episodes
    )
    paths = {
        "falls": plot_fall_probability_analysis(
            fall_df, survival_df, output_dir, n_eval_episodes=n_eval_episodes, show=show
        ),
        "survival": plot_survival_analysis(
            survival_df, survival_data, output_dir, n_eval_episodes=n_eval_episodes, show=show
        ),
        "behavior": plot_behavioral_single_episode(model, n_lanes, output_dir, show=show),
        "heatmap": plot_trajectory_heatmaps(
            model, n_lanes, output_dir, n_heatmap_episodes=n_heatmap_episodes, show=show
        ),
        "gif": generate_bike_animation_gif(model, n_lanes, output_dir, max_steps=gif_max_steps, show=show),
    }
    return paths
