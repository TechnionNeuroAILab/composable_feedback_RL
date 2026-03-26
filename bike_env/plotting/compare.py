"""Compare multiple training runs / scenarios on the same axes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.monitor import load_results

PathLike = Union[str, Path]

DEFAULT_LOSS_TAG = "train/loss"


def _tensorboard_event_dir(log_dir: PathLike) -> Optional[Path]:
    """Return a subdirectory of ``log_dir/tensorboard`` that contains event files, if any."""
    root = Path(log_dir) / "tensorboard"
    if not root.is_dir():
        return None
    candidates: List[Path] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and any(sub.glob("events.out*")):
            candidates.append(sub)
    return candidates[0] if candidates else None


def load_train_loss_from_tensorboard(
    log_dir: PathLike,
    tag: str = DEFAULT_LOSS_TAG,
) -> pd.DataFrame:
    """
    Load DQN ``train/loss`` scalars from TensorBoard logs under ``<log_dir>/tensorboard``.

    Returns a DataFrame with columns ``step``, ``loss`` (empty if not found).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_dir = _tensorboard_event_dir(log_dir)
    if event_dir is None:
        return pd.DataFrame(columns=["step", "loss"])

    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        return pd.DataFrame(columns=["step", "loss"])

    events = ea.Scalars(tag)
    return pd.DataFrame(
        {"step": [e.step for e in events], "loss": [e.value for e in events]}
    )


def _line_colors(n: int) -> List[str]:
    """Distinct default cycle colors (C0, C1, …) so each lane run is easy to tell apart."""
    return [f"C{i % 10}" for i in range(n)]


def plot_lane_sweep_comparison(
    log_dirs: Sequence[PathLike],
    labels: Sequence[str],
    output_dir: PathLike,
    *,
    reward_window: int = 50,
    falls_window: int = 50,
    loss_smooth_window: int = 50,
    title_prefix: str = "",
    show: bool = False,
) -> dict:
    """
    Write three comparison figures: rolling episode reward, training loss (TensorBoard), rolling falls per episode.

    ``monitor.csv`` must include ``episode_falls`` (runs trained with ``FallCounterWrapper`` + ``info_keywords``).
    Older logs without that column still produce reward and loss plots; falls are skipped with a note in the title.
    """
    if len(log_dirs) != len(labels):
        raise ValueError("log_dirs and labels must have the same length")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _line_colors(len(labels))
    prefix = f"{title_prefix} — " if title_prefix else ""

    paths: dict = {}

    # --- Reward ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        df = load_results(str(log_dir))
        curve = df["r"].rolling(window=reward_window, min_periods=1).mean()
        ax.plot(curve.index, curve.values, linewidth=2, label=label, color=colors[i])
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode return (rolling mean, window={reward_window})")
    ax.set_title(f"{prefix}Reward")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_reward = output_dir / "compare_reward.png"
    fig.savefig(p_reward, dpi=200, bbox_inches="tight")
    paths["reward"] = p_reward
    if show:
        plt.show()
    else:
        plt.close()

    # --- Loss (global training step) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    any_loss = False
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        ldf = load_train_loss_from_tensorboard(log_dir)
        if ldf.empty:
            continue
        any_loss = True
        y = ldf["loss"].values
        x = ldf["step"].values
        if loss_smooth_window > 1 and len(y) >= loss_smooth_window:
            s = pd.Series(y).rolling(window=loss_smooth_window, min_periods=1).mean()
            y = s.values
        ax.plot(x, y, linewidth=1.8, label=label, color=colors[i], alpha=0.95)
    ax.set_xlabel("Environment step (SB3 global step)")
    ax.set_ylabel("DQN loss (Huber / smooth L1)")
    if not any_loss:
        ax.text(
            0.5,
            0.5,
            "No TensorBoard loss data found under log_dir/tensorboard",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    else:
        ax.legend(loc="best", framealpha=0.9)
    ax.set_title(f"{prefix}Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_loss = output_dir / "compare_loss.png"
    fig.savefig(p_loss, dpi=200, bbox_inches="tight")
    paths["loss"] = p_loss
    if show:
        plt.show()
    else:
        plt.close()

    # --- Falls per episode ---
    fig, ax = plt.subplots(figsize=(12, 6))
    any_falls = False
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        df = load_results(str(log_dir))
        if "episode_falls" not in df.columns:
            continue
        any_falls = True
        curve = df["episode_falls"].rolling(window=falls_window, min_periods=1).mean()
        ax.plot(curve.index, curve.values, linewidth=2, label=label, color=colors[i])
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Falls per episode (rolling mean, window={falls_window})")
    if not any_falls:
        ax.text(
            0.5,
            0.5,
            "No episode_falls in monitor.csv — re-train with the current bike_env training stack",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
    else:
        ax.legend(loc="best", framealpha=0.9)
    ax.set_title(f"{prefix}Falls per episode")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_falls = output_dir / "compare_falls.png"
    fig.savefig(p_falls, dpi=200, bbox_inches="tight")
    paths["falls"] = p_falls
    if show:
        plt.show()
    else:
        plt.close()

    return paths


def plot_episode_rewards_comparison(
    log_dirs: Sequence[PathLike],
    labels: Sequence[str],
    output_path: Optional[PathLike] = None,
    rolling_window: int = 50,
    title: str = "Episode reward (rolling mean) — multiple runs",
    show: bool = False,
) -> None:
    """
    Overlay rolling-mean episode rewards for several Monitor log directories.

    ``log_dirs`` and ``labels`` must have the same length.
    """
    if len(log_dirs) != len(labels):
        raise ValueError("log_dirs and labels must have the same length")
    plt.figure(figsize=(12, 6))
    for log_dir, label in zip(log_dirs, labels):
        df = load_results(str(log_dir))
        curve = df["r"].rolling(window=rolling_window, min_periods=1).mean()
        plt.plot(curve.index, curve.values, linewidth=2, label=label)
    plt.xlabel("Episode index")
    plt.ylabel(f"Reward (rolling {rolling_window})")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def collect_run_summary_rows(log_dirs: Iterable[PathLike], scenario_labels: Sequence[str]) -> pd.DataFrame:
    """Build a small table of mean/std reward and timesteps per run."""
    rows: List[dict] = []
    for log_dir, name in zip(log_dirs, scenario_labels):
        df = load_results(str(log_dir))
        rows.append(
            {
                "run": name,
                "episodes": len(df),
                "total_timesteps": int(df["l"].sum()),
                "mean_r": float(df["r"].mean()),
                "std_r": float(df["r"].std()),
                "last_100_mean_r": float(df["r"].tail(100).mean()),
            }
        )
    return pd.DataFrame(rows)
