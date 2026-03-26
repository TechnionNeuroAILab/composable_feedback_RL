"""Plots from Monitor / ``load_results`` training logs (notebook cell 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import load_results

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

PathLike = Union[str, Path]


def setup_plot_style() -> None:
    if _HAS_SNS:
        sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (15, 10)


def load_monitor_dataframe(log_dir: PathLike) -> pd.DataFrame:
    return load_results(str(log_dir))


def plot_episode_rewards_overview(
    df: pd.DataFrame,
    output_dir: PathLike,
    window: int = 100,
    filename: str = "01_episode_rewards.png",
    show: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Progress Overview", fontsize=16, fontweight="bold")

    axes[0, 0].plot(df.index, df["r"], alpha=0.3, color="blue", linewidth=0.5)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("Episode Rewards (Raw)")
    axes[0, 0].grid(True, alpha=0.3)

    rolling_mean = df["r"].rolling(window=window, min_periods=1).mean()
    axes[0, 1].plot(df.index, rolling_mean, color="red", linewidth=2)
    axes[0, 1].fill_between(
        df.index,
        df["r"].rolling(window=window, min_periods=1).quantile(0.25),
        df["r"].rolling(window=window, min_periods=1).quantile(0.75),
        alpha=0.3,
        color="red",
    )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward (Rolling Mean)")
    axes[0, 1].set_title(f"Episode Rewards ({window}-Episode Moving Average)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df.index, df["l"], alpha=0.3, color="green", linewidth=0.5)
    rolling_length = df["l"].rolling(window=window, min_periods=1).mean()
    axes[1, 0].plot(df.index, rolling_length, color="darkgreen", linewidth=2, label=f"{window}-ep avg")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Episode Length (timesteps)")
    axes[1, 0].set_title("Episode Length Over Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df.index, df["r"].cumsum(), color="purple", linewidth=2)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Cumulative Reward")
    axes[1, 1].set_title("Cumulative Reward Over Training")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_performance_metrics(
    df: pd.DataFrame,
    output_dir: PathLike,
    window: int = 100,
    filename: str = "02_performance_metrics.png",
    show: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Detailed Performance Metrics", fontsize=16, fontweight="bold")

    n_bins = 5
    bin_size = max(len(df) // n_bins, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, n_bins))

    axes[0, 0].set_title("Reward Distribution by Training Phase")
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(df)
        phase_data = df["r"].iloc[start_idx:end_idx]
        axes[0, 0].hist(phase_data, bins=50, alpha=0.5, label=f"Episodes {start_idx}-{end_idx}", color=colors[i])
    axes[0, 0].set_xlabel("Reward")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    rolling_mean = df["r"].rolling(window=window, min_periods=1).mean()
    rolling_std = df["r"].rolling(window=window, min_periods=1).std()
    axes[0, 1].plot(df.index, rolling_mean, color="red", linewidth=2, label="Mean")
    axes[0, 1].fill_between(
        df.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.3,
        color="red",
        label="±1 STD",
    )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_title(f"Learning Curve with Variance ({window}-ep window)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    sample_size = min(5000, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    axes[0, 2].scatter(df["l"].iloc[sample_indices], df["r"].iloc[sample_indices], alpha=0.3, s=10)
    axes[0, 2].set_xlabel("Episode Length")
    axes[0, 2].set_ylabel("Episode Reward")
    axes[0, 2].set_title("Reward vs Episode Length")
    axes[0, 2].grid(True, alpha=0.3)

    segment_size = max(len(df) // 50, 1)
    segments = []
    mean_rewards = []
    for i in range(0, len(df), segment_size):
        end_idx = min(i + segment_size, len(df))
        segments.append(i + segment_size // 2)
        mean_rewards.append(df["r"].iloc[i:end_idx].mean())

    axes[1, 0].plot(segments, mean_rewards, marker="o", linewidth=2, markersize=3)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Average Reward")
    axes[1, 0].set_title("Performance Improvement (Segmented Averages)")
    axes[1, 0].grid(True, alpha=0.3)

    cummax = df["r"].cummax()
    axes[1, 1].plot(df.index, cummax, color="gold", linewidth=2)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Best Reward So Far")
    axes[1, 1].set_title("Best Performance Evolution")
    axes[1, 1].grid(True, alpha=0.3)

    threshold = df["r"].quantile(0.75)
    success_rate = df["r"].rolling(window=window, min_periods=1).apply(
        lambda x: (x > threshold).sum() / len(x)
    )
    axes[1, 2].plot(df.index, success_rate * 100, color="green", linewidth=2)
    axes[1, 2].axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% baseline")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Success Rate (%)")
    axes[1, 2].set_title(f"Success Rate (reward > {threshold:.1f}, {window}-ep window)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_training_summary_board(
    df: pd.DataFrame,
    output_dir: PathLike,
    window: int = 100,
    filename: str = "03_training_summary.png",
    show: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.axis("off")

    total_episodes = len(df)
    total_timesteps = df["l"].sum()
    mean_reward = df["r"].mean()
    std_reward = df["r"].std()
    max_reward = df["r"].max()
    min_reward = df["r"].min()
    mean_length = df["l"].mean()

    final_100_mean = df["r"].tail(100).mean()
    final_100_std = df["r"].tail(100).std()
    first_100_mean = df["r"].head(100).mean()
    denom = abs(first_100_mean) if first_100_mean != 0 else 1.0
    improvement = ((final_100_mean - first_100_mean) / denom) * 100

    stats_text = f"""
TRAINING SUMMARY
{'=' * 63}
Total Episodes: {total_episodes:,} | Total Timesteps: {total_timesteps:,}

OVERALL PERFORMANCE:
Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}
Max Reward: {max_reward:.2f} | Min Reward: {min_reward:.2f}
Mean Episode Length: {mean_length:.1f} timesteps

LEARNING PROGRESS:
First 100 Episodes: {first_100_mean:.2f}
Final 100 Episodes: {final_100_mean:.2f} ± {final_100_std:.2f}
Improvement: {improvement:+.1f}%
"""

    ax_stats.text(
        0.5,
        0.5,
        stats_text,
        fontsize=12,
        fontfamily="monospace",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(df["r"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(mean_reward, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_reward:.2f}")
    ax1.axvline(df["r"].median(), color="orange", linestyle="--", linewidth=2, label=f"Median: {df['r'].median():.2f}")
    ax1.set_xlabel("Reward")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Reward Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist(df["l"], bins=50, color="green", edgecolor="black", alpha=0.7)
    ax2.axvline(mean_length, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_length:.1f}")
    ax2.set_xlabel("Episode Length")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Episode Length Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 2])
    quarter = max(len(df) // 4, 1)
    data_to_plot = [
        df["r"].iloc[:quarter],
        df["r"].iloc[quarter : 2 * quarter],
        df["r"].iloc[2 * quarter : 3 * quarter],
        df["r"].iloc[3 * quarter :],
    ]
    try:
        bp = ax3.boxplot(data_to_plot, tick_labels=["Q1", "Q2", "Q3", "Q4"], patch_artist=True)
    except TypeError:
        bp = ax3.boxplot(data_to_plot, labels=["Q1", "Q2", "Q3", "Q4"], patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0, 1, 4))):
        patch.set_facecolor(color)
    ax3.set_ylabel("Reward")
    ax3.set_title("Performance by Training Quarter")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :])
    rolling_mean = df["r"].rolling(window=window, min_periods=1).mean()
    ax4.plot(df.index, df["r"], alpha=0.2, color="blue", linewidth=0.5, label="Raw")
    ax4.plot(df.index, rolling_mean, color="red", linewidth=2, label=f"{window}-ep Moving Avg")
    z = np.polyfit(df.index.astype(float), df["r"].astype(float), 3)
    p = np.poly1d(z)
    ax4.plot(df.index, p(df.index), "--", color="green", linewidth=2, label="Trend (poly-3)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Reward")
    ax4.set_title("Training Progress with Trend")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_time_series_analysis(
    df: pd.DataFrame,
    output_dir: PathLike,
    window: int = 100,
    filename: str = "04_time_series_analysis.png",
    show: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Time Series Analysis", fontsize=16, fontweight="bold")

    for w in (50, 100, 200, 500):
        rolling_mean = df["r"].rolling(window=w, min_periods=1).mean()
        axes[0, 0].plot(df.index, rolling_mean, linewidth=2, label=f"{w}-ep window")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title("Multi-Scale Moving Averages")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    rolling_std = df["r"].rolling(window=window, min_periods=1).std()
    axes[0, 1].plot(df.index, rolling_std, color="purple", linewidth=2)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward Std Dev")
    axes[0, 1].set_title(f"Performance Volatility ({window}-ep window)")
    axes[0, 1].grid(True, alpha=0.3)

    rolling_range = df["r"].rolling(window=window, min_periods=1).apply(lambda x: x.max() - x.min())
    axes[1, 0].plot(df.index, rolling_range, color="orange", linewidth=2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Reward Range")
    axes[1, 0].set_title(f"Performance Range ({window}-ep window)")
    axes[1, 0].grid(True, alpha=0.3)

    reward_diff = df["r"].diff().abs()
    rolling_diff = reward_diff.rolling(window=window, min_periods=1).mean()
    axes[1, 1].plot(df.index, rolling_diff, color="brown", linewidth=2)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Avg Absolute Change")
    axes[1, 1].set_title(f"Episode-to-Episode Variability ({window}-ep window)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def write_training_report(df: pd.DataFrame, report_path: PathLike) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BIKE ENVIRONMENT TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("GENERAL STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Episodes: {len(df):,}\n")
        f.write(f"Total Timesteps: {df['l'].sum():,}\n")
        f.write(f"Training Duration: {df['t'].iloc[-1]:.2f} seconds\n\n")
        f.write("REWARD STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Reward: {df['r'].mean():.2f}\n")
        f.write(f"Std Reward: {df['r'].std():.2f}\n")
        f.write(f"Median Reward: {df['r'].median():.2f}\n")
        f.write(f"Max Reward: {df['r'].max():.2f}\n")
        f.write(f"Min Reward: {df['r'].min():.2f}\n")
        f.write(f"25th Percentile: {df['r'].quantile(0.25):.2f}\n")
        f.write(f"75th Percentile: {df['r'].quantile(0.75):.2f}\n\n")
        f.write("EPISODE LENGTH STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Length: {df['l'].mean():.2f} timesteps\n")
        f.write(f"Std Length: {df['l'].std():.2f} timesteps\n")
        f.write(f"Median Length: {df['l'].median():.2f} timesteps\n")
        f.write(f"Max Length: {df['l'].max()} timesteps\n")
        f.write(f"Min Length: {df['l'].min()} timesteps\n\n")
        f.write("LEARNING PROGRESS:\n")
        f.write("-" * 70 + "\n")
        first_100 = df["r"].head(100).mean()
        last_100 = df["r"].tail(100).mean()
        denom = abs(first_100) if first_100 != 0 else 1.0
        improvement = ((last_100 - first_100) / denom) * 100
        f.write(f"First 100 Episodes Mean: {first_100:.2f}\n")
        f.write(f"Last 100 Episodes Mean: {last_100:.2f}\n")
        f.write(f"Improvement: {improvement:+.2f}%\n\n")
        f.write(f"Best Episode: {df['r'].idxmax()} (Reward: {df['r'].max():.2f})\n")
        f.write(f"Worst Episode: {df['r'].idxmin()} (Reward: {df['r'].min():.2f})\n\n")
        f.write("PERFORMANCE BY TRAINING PHASE:\n")
        f.write("-" * 70 + "\n")
        quarter = max(len(df) // 4, 1)
        for i in range(4):
            start_idx = i * quarter
            end_idx = (i + 1) * quarter if i < 3 else len(df)
            phase_mean = df["r"].iloc[start_idx:end_idx].mean()
            phase_std = df["r"].iloc[start_idx:end_idx].std()
            f.write(
                f"Quarter {i + 1} (Episodes {start_idx}-{end_idx}): "
                f"{phase_mean:.2f} ± {phase_std:.2f}\n"
            )
    return report_path


def generate_all_monitor_figures(
    log_dir: PathLike,
    output_dir: Optional[PathLike] = None,
    window: int = 100,
    show: bool = False,
) -> dict:
    """
    Load ``monitor.csv`` via ``load_results`` and write all standard training figures + report.

    If ``output_dir`` is None, uses ``<log_dir>/visualizations``.
    """
    log_dir = Path(log_dir)
    if output_dir is None:
        output_dir = log_dir / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_monitor_dataframe(log_dir)
    paths = {
        "01_rewards": plot_episode_rewards_overview(df, output_dir, window=window, show=show),
        "02_metrics": plot_performance_metrics(df, output_dir, window=window, show=show),
        "03_summary": plot_training_summary_board(df, output_dir, window=window, show=show),
        "04_time_series": plot_time_series_analysis(df, output_dir, window=window, show=show),
        "report": write_training_report(df, output_dir / "training_report.txt"),
    }
    return paths
