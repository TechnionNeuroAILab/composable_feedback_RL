"""Plot learning curve and per-channel local-controller diagnostics for a run.

Reads the ``config.json`` / ``windows.json`` / ``episode_returns.json``
written by ``train.py --output-dir ...`` and writes a set of PNGs to a
destination directory (typically under ``paper/figures/``).

    python code/ppo_local_channel_control/plot_results.py \
        --results-dir results/ppo_local_channel_control/HalfCheetah-v4__local_control__seed1_5000ep \
        --output-dir paper/figures/ppo_local_channel_control_halfcheetah
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    config = json.loads((results_dir / "config.json").read_text(encoding="utf-8"))
    windows = json.loads((results_dir / "windows.json").read_text(encoding="utf-8"))
    episode_returns = json.loads((results_dir / "episode_returns.json").read_text(encoding="utf-8"))
    return config, windows, episode_returns


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size < width:
        return values
    return np.convolve(values, np.ones(width) / width, mode="valid")


def plot_learning_curve(
    episode_returns: list[float], env_id: str, destination: Path, smooth_width: int = 50
) -> None:
    returns = np.asarray(episode_returns, dtype=float)
    episodes = np.arange(1, returns.size + 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(episodes, returns, alpha=0.25, color="tab:blue", label="episode return")
    width = min(smooth_width, returns.size)
    if width > 1:
        smooth = _smooth(returns, width)
        axis.plot(
            episodes[width - 1 :], smooth, color="tab:blue", linewidth=2, label=f"{width}-episode mean"
        )
    axis.set(xlabel="Episode", ylabel="Episode return", title=f"{env_id}: learning curve ({returns.size} episodes)")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _channel_matrix(windows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row["controllers"][key] for row in windows], dtype=float)


def plot_channel_series(
    windows: list[dict[str, Any]],
    keys: list[str],
    titles: list[str],
    ylabels: list[str],
    env_id: str,
    destination: Path,
    log_scale: list[bool] | None = None,
) -> None:
    log_scale = log_scale or [False] * len(keys)
    steps = np.asarray([row["total_steps"] for row in windows], dtype=float)
    num_panels = len(keys)
    figure, axes = plt.subplots(1, num_panels, figsize=(5.5 * num_panels, 4.5), squeeze=False)
    axes = axes[0]
    num_channels = _channel_matrix(windows, keys[0]).shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_channels, 1)))
    for panel_idx, (key, title, ylabel, log) in enumerate(zip(keys, titles, ylabels, log_scale)):
        axis = axes[panel_idx]
        matrix = _channel_matrix(windows, key)
        for k in range(num_channels):
            axis.plot(steps, matrix[:, k], color=colors[k], label=f"channel {k}")
        if log:
            axis.set_yscale("log")
        axis.set(xlabel="Environment steps", ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        if panel_idx == num_panels - 1:
            axis.legend(fontsize=8, loc="best")
    figure.suptitle(f"{env_id}: per-channel local-controller settings")
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def plot_losses(windows: list[dict[str, Any]], env_id: str, destination: Path) -> None:
    steps = np.asarray([row["total_steps"] for row in windows], dtype=float)
    policy_loss = np.asarray([row["policy_loss"] for row in windows], dtype=float)
    value_loss = np.asarray([row["value_loss"] for row in windows], dtype=float)
    entropy = np.asarray([row["entropy"] for row in windows], dtype=float)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(steps, policy_loss, color="tab:blue")
    axes[0].set(xlabel="Environment steps", ylabel="Clipped surrogate loss", title="Policy loss")
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, value_loss, color="tab:red")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Environment steps", ylabel="Summed critic MSE (log)", title="Value loss")
    axes[1].grid(alpha=0.2)

    axes[2].plot(steps, entropy, color="tab:green")
    axes[2].set(xlabel="Environment steps", ylabel="Mean policy entropy", title="Entropy")
    axes[2].grid(alpha=0.2)

    figure.suptitle(f"{env_id}: PPO training diagnostics")
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def plot_routing_pressure(windows: list[dict[str, Any]], env_id: str, destination: Path) -> None:
    plot_channel_series(
        windows,
        keys=["sparsity_weight", "overlap_weight"],
        titles=["Sparsity pressure", "Overlap pressure"],
        ylabels=["sparsity_weight", "overlap_weight"],
        env_id=env_id,
        destination=destination,
        log_scale=[True, True],
    )


def write_summary(
    config: dict[str, Any],
    windows: list[dict[str, Any]],
    episode_returns: list[float],
    destination: Path,
) -> None:
    returns = np.asarray(episode_returns, dtype=float)
    last_n = min(100, returns.size)
    summary = {
        "env_id": config.get("env_id"),
        "seed": config.get("seed"),
        "routing_mode": config.get("routing_mode", "sparse_b"),
        "num_channels": config.get("num_channels"),
        "num_modules": config.get("num_modules"),
        "total_env_steps": windows[-1]["total_steps"] if windows else 0,
        "num_episodes": int(returns.size),
        "num_windows": len(windows),
        "final_mean_return_last_100": float(returns[-last_n:].mean()) if last_n else None,
        "best_mean_return_last_100": float(
            max(
                (returns[max(0, i - 100) : i].mean() for i in range(last_n, returns.size + 1)),
                default=float("nan"),
            )
        )
        if returns.size
        else None,
        "final_gate": windows[-1]["controllers"]["gate"] if windows else None,
        "final_gae_lambda": windows[-1]["controllers"]["gae_lambda"] if windows else None,
        "final_gamma": windows[-1]["controllers"]["gamma"] if windows else None,
        "final_alpha_specific": windows[-1]["controllers"]["alpha_specific"] if windows else None,
        "final_critic_lr": windows[-1]["controllers"]["critic_lr"] if windows else None,
    }
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_all_plots(results_dir: Path, output_dir: Path) -> None:
    config, windows, episode_returns = load_run(results_dir)
    env_id = config.get("env_id", "unknown-env")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(episode_returns, env_id, output_dir / "01_learning_curve.png")
    if windows:
        plot_channel_series(
            windows,
            keys=["gate"],
            titles=["Channel activation gates g_k"],
            ylabels=["g_k"],
            env_id=env_id,
            destination=output_dir / "02_channel_gates.png",
        )
        plot_channel_series(
            windows,
            keys=["critic_lr", "gae_lambda", "gamma", "alpha_specific"],
            titles=["Critic learning rate", "GAE lambda_k", "Discount gamma_k", "Shared-vs-specific alpha_k"],
            ylabels=["learning rate", "lambda_k", "gamma_k", "alpha_k"],
            env_id=env_id,
            destination=output_dir / "03_channel_hyperparameters.png",
            log_scale=[True, False, False, False],
        )
        plot_losses(windows, env_id, output_dir / "04_training_losses.png")
        if config.get("routing_mode", "sparse_b") == "sparse_b":
            plot_routing_pressure(windows, env_id, output_dir / "05_routing_pressure.png")
        write_summary(config, windows, episode_returns, output_dir / "summary.json")
    print(f"Wrote figures to {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    make_all_plots(args.results_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
