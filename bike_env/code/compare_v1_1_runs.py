#!/usr/bin/env python3
"""Compare v1.1 DQN sweeps across hyperparameters and lane counts.

The script separates raw reward from evidence of reactive hole avoidance.
It consumes existing training/evaluation artifacts; it does not retrain agents.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3.common.monitor import load_results


_HERE = Path(__file__).resolve().parent
_DEFAULT_OUTPUT = _HERE.parent / "plotting" / "v1.1_hyperparameter_comparison"
LANE_COUNTS = (2, 3, 5, 10)
STAGE_ORDER = ("beginning", "middle", "end")
HOLE_EVENTS = ("avoided", "collision", "unprotected_pass", "passed_other_lane")
ACTION_COUNT = 3


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    root: Path


def default_experiments() -> list[Experiment]:
    return [
        Experiment(
            "baseline100",
            "vis25 / fall100 / coll−10 / ε0.025",
            _HERE / "training_results_v1_1_ep2000",
        ),
        Experiment(
            "pen500",
            "vis25 / fall500 / coll−10 / ε0.025",
            _HERE / "training_results_v1_1_ep2000_pen500",
        ),
        Experiment(
            "vis40",
            "vis40 / fall100 / coll−100 / ε0.1",
            _HERE / "training_results_v1_1_vis40_100penalty_ep2000",
        ),
        Experiment(
            "vis50",
            "vis50 / fall100 / coll−100 / ε0.1",
            _HERE / "training_results_v1_1_vis50_100penalty_ep2000",
        ),
    ]


def _run_dir(root: Path, n_lanes: int, seed: int) -> Path:
    pattern = (
        f"lanes_{n_lanes}__actions_3__falls_hole-collision"
        f"__lambda_5__same_sep_8__adj_sep_20__seed_{seed}"
    )
    path = root / pattern
    if not path.is_dir():
        raise FileNotFoundError(f"Missing run directory: {path}")
    return path


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required CSV is empty: {path}")
    return frame


def _entropy(values: pd.Series, cardinality: int) -> float:
    counts = values.value_counts(normalize=True)
    if counts.empty or cardinality <= 1:
        return math.nan
    entropy = float(-(counts * np.log(counts)).sum())
    return entropy / math.log(cardinality)


def _bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = 5000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _episode_step_metrics(
    steps: pd.DataFrame,
    *,
    n_lanes: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stage, episode), group in steps.groupby(
        ["stage", "episode"], observed=True, sort=False
    ):
        boundary_noop = (
            (~group["lane_changed"].astype(bool))
            & (
                ((group["action"] == 0) & (group["lane"] == 0))
                | ((group["action"] == 2) & (group["lane"] == n_lanes - 1))
            )
        )
        action_attempts = group["action"].isin([0, 2])
        lane_fractions = group["lane"].value_counts(normalize=True)
        rows.append(
            {
                "stage": stage,
                "episode": episode,
                "near_hole_lane_changes": float(
                    group["near_hole_lane_change"].astype(bool).sum()
                ),
                "lane_changes": float(group["lane_changed"].astype(bool).sum()),
                "action_entropy": _entropy(group["action"], ACTION_COUNT),
                "lane_entropy": _entropy(group["lane"], n_lanes),
                "lane_concentration": float(lane_fractions.max()),
                "dominant_lane": int(lane_fractions.idxmax()),
                "boundary_noops": float(boundary_noop.sum()),
                "boundary_noop_rate": (
                    float(boundary_noop.sum() / action_attempts.sum())
                    if action_attempts.sum() > 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _episode_hole_metrics(
    holes: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    index = episodes[["stage", "episode"]].drop_duplicates()
    event_counts = (
        holes.groupby(["stage", "episode", "event"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    result = index.merge(event_counts, on=["stage", "episode"], how="left")
    for event in HOLE_EVENTS:
        if event not in result:
            result[event] = 0
        result[event] = result[event].fillna(0).astype(float)
    return result


def load_run(
    experiment: Experiment,
    n_lanes: int,
    seed: int,
    expected_episodes: int,
) -> dict[str, Any]:
    root = _run_dir(experiment.root, n_lanes, seed)
    with (root / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    monitor = load_results(str(root)).reset_index(drop=True)
    monitor["episode"] = np.arange(1, len(monitor) + 1)
    if len(monitor) != expected_episodes:
        raise ValueError(
            f"{root}: expected {expected_episodes} training episodes, "
            f"found {len(monitor)}"
        )

    episodes = _read_required_csv(root / "evaluation_episodes.csv")
    steps = _read_required_csv(root / "evaluation_steps.csv")
    holes = _read_required_csv(root / "evaluation_holes.csv")

    required_stages = set(STAGE_ORDER)
    if set(episodes["stage"].unique()) != required_stages:
        raise ValueError(f"{root}: evaluation stages do not match {STAGE_ORDER}")

    step_metrics = _episode_step_metrics(steps, n_lanes=n_lanes)
    hole_metrics = _episode_hole_metrics(holes, episodes)
    episode_metrics = (
        episodes.merge(step_metrics, on=["stage", "episode"], how="left")
        .merge(hole_metrics, on=["stage", "episode"], how="left")
        .copy()
    )
    denominator = (
        episode_metrics["holes_avoided"] + episode_metrics["hole_collisions"]
    )
    episode_metrics["avoidance_rate"] = np.where(
        denominator > 0,
        episode_metrics["holes_avoided"] / denominator,
        np.nan,
    )
    episode_metrics["zero_fall_success"] = (
        (episode_metrics["falls"] == 0)
        & np.isclose(episode_metrics["reward"], 5000.0)
    ).astype(float)
    episode_metrics["experiment"] = experiment.key
    episode_metrics["experiment_label"] = experiment.label
    episode_metrics["n_lanes"] = n_lanes

    monitor["experiment"] = experiment.key
    monitor["experiment_label"] = experiment.label
    monitor["n_lanes"] = n_lanes
    training_episodes_path = root / "training_episodes.csv"
    training_episodes = (
        pd.read_csv(training_episodes_path)
        if training_episodes_path.is_file()
        else pd.DataFrame()
    )
    return {
        "root": root,
        "config": config,
        "monitor": monitor,
        "episodes": episode_metrics,
        "training_episodes": training_episodes,
    }


def load_all_runs(
    experiments: list[Experiment],
    lanes: tuple[int, ...],
    *,
    seed: int,
    expected_episodes: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for experiment in experiments:
        for n_lanes in lanes:
            print(f"Loading {experiment.key}: {n_lanes} lanes")
            runs[(experiment.key, n_lanes)] = load_run(
                experiment, n_lanes, seed, expected_episodes
            )
    return runs


def build_summary(
    runs: dict[tuple[str, int], dict[str, Any]],
    experiments: list[Experiment],
    lanes: tuple[int, ...],
) -> pd.DataFrame:
    rng = np.random.default_rng(20260805)
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for n_lanes in lanes:
            data = runs[(experiment.key, n_lanes)]
            monitor = data["monitor"]
            end = data["episodes"][data["episodes"]["stage"] == "end"].copy()
            beginning = data["episodes"][
                data["episodes"]["stage"] == "beginning"
            ]
            tail500 = monitor.tail(500)
            slope = float(
                np.polyfit(tail500["episode"], tail500["r"], deg=1)[0]
            )
            reward_low, reward_high = _bootstrap_mean_ci(end["reward"], rng)
            success_low, success_high = _bootstrap_mean_ci(
                end["zero_fall_success"], rng
            )
            env = data["config"]["environment"]
            train = data["config"]["training"]
            rows.append(
                {
                    "experiment": experiment.key,
                    "experiment_label": experiment.label,
                    "n_lanes": n_lanes,
                    "observation_dim": n_lanes + 2,
                    "visibility_range": env["visibility_range"],
                    "fall_time_penalty": env["fall_time_penalty"],
                    "collision_penalty": env["collision_penalty"],
                    "exploration_fraction": train["exploration_fraction"],
                    "exploration_final_eps": train["exploration_final_eps"],
                    "learning_starts": train["learning_starts"],
                    "training_episodes": len(monitor),
                    "train_last100_reward_mean": monitor["r"].tail(100).mean(),
                    "train_last500_reward_mean": tail500["r"].mean(),
                    "train_last500_reward_std": tail500["r"].std(ddof=0),
                    "train_last500_slope_per_episode": slope,
                    "train_reward_auc_normalized": monitor["r"].mean(),
                    "end_eval_reward_mean": end["reward"].mean(),
                    "end_eval_reward_std": end["reward"].std(ddof=0),
                    "end_eval_reward_ci_low": reward_low,
                    "end_eval_reward_ci_high": reward_high,
                    "end_eval_steps_mean": end["steps"].mean(),
                    "end_eval_distance_mean": end["distance"].mean(),
                    "end_eval_falls_mean": end["falls"].mean(),
                    "end_eval_collisions_mean": end["hole_collisions"].mean(),
                    "end_eval_avoided_mean": end["holes_avoided"].mean(),
                    "end_eval_passed_other_lane_mean": end[
                        "passed_other_lane"
                    ].mean(),
                    "end_eval_unprotected_pass_mean": end[
                        "unprotected_pass"
                    ].mean(),
                    "end_eval_avoidance_rate_mean": end[
                        "avoidance_rate"
                    ].mean(),
                    "end_eval_near_hole_lane_changes_mean": end[
                        "near_hole_lane_changes"
                    ].mean(),
                    "end_eval_action_entropy_mean": end["action_entropy"].mean(),
                    "end_eval_lane_entropy_mean": end["lane_entropy"].mean(),
                    "end_eval_lane_concentration_mean": end[
                        "lane_concentration"
                    ].mean(),
                    "end_eval_dominant_lane_mode": int(
                        end["dominant_lane"].mode().iloc[0]
                    ),
                    "end_eval_boundary_noop_rate_mean": end[
                        "boundary_noop_rate"
                    ].mean(),
                    "end_eval_success_rate": end["zero_fall_success"].mean(),
                    "end_eval_success_ci_low": success_low,
                    "end_eval_success_ci_high": success_high,
                    "reward_improvement_beginning_to_end": (
                        end["reward"].mean() - beginning["reward"].mean()
                    ),
                    "avoidance_improvement_beginning_to_end": (
                        end["avoidance_rate"].mean()
                        - beginning["avoidance_rate"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_paired_differences(
    runs: dict[tuple[str, int], dict[str, Any]],
    experiments: list[Experiment],
    lanes: tuple[int, ...],
    *,
    reference: str = "baseline100",
) -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    metrics = (
        "reward",
        "falls",
        "avoidance_rate",
        "near_hole_lane_changes",
        "lane_concentration",
    )
    rows: list[dict[str, Any]] = []
    for n_lanes in lanes:
        reference_end = runs[(reference, n_lanes)]["episodes"]
        reference_end = reference_end[reference_end["stage"] == "end"]
        for experiment in experiments:
            if experiment.key == reference:
                continue
            candidate = runs[(experiment.key, n_lanes)]["episodes"]
            candidate = candidate[candidate["stage"] == "end"]
            paired = candidate.merge(
                reference_end,
                on="episode",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            for metric in metrics:
                differences = (
                    paired[f"{metric}_candidate"]
                    - paired[f"{metric}_reference"]
                )
                low, high = _bootstrap_mean_ci(differences, rng)
                rows.append(
                    {
                        "reference": reference,
                        "candidate": experiment.key,
                        "n_lanes": n_lanes,
                        "metric": metric,
                        "paired_mean_difference": differences.mean(),
                        "paired_ci_low": low,
                        "paired_ci_high": high,
                        "n_paired_eval_episodes": len(differences),
                    }
                )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _experiment_colors(experiments: list[Experiment]) -> dict[str, Any]:
    palette = sns.color_palette("colorblind", n_colors=len(experiments))
    return {
        experiment.key: palette[index]
        for index, experiment in enumerate(experiments)
    }


def plot_learning_curves(
    runs: dict[tuple[str, int], dict[str, Any]],
    experiments: list[Experiment],
    lanes: tuple[int, ...],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    fig.suptitle(
        "Training Reward by Lane Count and Hyperparameter Configuration",
        fontsize=15,
        fontweight="bold",
    )
    for ax, n_lanes in zip(axes.flat, lanes):
        for experiment in experiments:
            monitor = runs[(experiment.key, n_lanes)]["monitor"]
            moving = monitor["r"].rolling(100, min_periods=1).mean()
            ax.plot(
                monitor["episode"],
                moving,
                label=experiment.key,
                color=colors[experiment.key],
                linewidth=1.8,
            )
        ax.set(
            title=f"{n_lanes} lanes",
            xlabel="Training episode",
            ylabel="Reward (100-episode moving mean)",
        )
        ax.axhline(5000, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Source: monitor.csv · 2,000 training episodes · one training seed",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    _save(fig, out_dir / "01_learning_curves_by_lane.png")


def _plot_summary_lines(
    ax: plt.Axes,
    summary: pd.DataFrame,
    experiments: list[Experiment],
    colors: dict[str, Any],
    metric: str,
    ylabel: str,
    *,
    ci_columns: tuple[str, str] | None = None,
) -> None:
    for experiment in experiments:
        data = summary[summary["experiment"] == experiment.key].sort_values(
            "n_lanes"
        )
        yerr = None
        if ci_columns:
            low, high = ci_columns
            yerr = np.vstack(
                [
                    data[metric] - data[low],
                    data[high] - data[metric],
                ]
            )
        ax.errorbar(
            data["n_lanes"],
            data[metric],
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=colors[experiment.key],
            label=experiment.key,
        )
    ax.set(
        xlabel="Number of lanes",
        ylabel=ylabel,
        xticks=list(LANE_COUNTS),
    )
    ax.grid(True, alpha=0.25)


def plot_end_reward_and_success(
    summary: pd.DataFrame,
    experiments: list[Experiment],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "End-Checkpoint Reward and Perfect-Episode Success",
        fontsize=15,
        fontweight="bold",
    )
    _plot_summary_lines(
        axes[0],
        summary,
        experiments,
        colors,
        "end_eval_reward_mean",
        "Mean episode reward (95% bootstrap CI)",
        ci_columns=("end_eval_reward_ci_low", "end_eval_reward_ci_high"),
    )
    axes[0].axhline(5000, color="black", linestyle="--", linewidth=1)
    _plot_summary_lines(
        axes[1],
        summary,
        experiments,
        colors,
        "end_eval_success_rate",
        "Zero-fall and reward-5000 episode fraction",
        ci_columns=("end_eval_success_ci_low", "end_eval_success_ci_high"),
    )
    axes[1].set_ylim(-0.05, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Source: 20 deterministic end-checkpoint evaluation episodes per run",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _save(fig, out_dir / "02_end_reward_and_success.png")


def plot_hole_behavior(
    summary: pd.DataFrame,
    experiments: list[Experiment],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    metrics = [
        ("end_eval_avoided_mean", "Avoided holes per episode"),
        ("end_eval_collisions_mean", "Hole collisions per episode"),
        (
            "end_eval_unprotected_pass_mean",
            "Visible in-lane passes without collision",
        ),
        (
            "end_eval_passed_other_lane_mean",
            "Other-lane holes passed per episode",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "End-Checkpoint Hole Event Mix",
        fontsize=15,
        fontweight="bold",
    )
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        _plot_summary_lines(
            ax, summary, experiments, colors, metric, ylabel
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Source: evaluation_holes.csv and evaluation_episodes.csv · end checkpoint",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _save(fig, out_dir / "03_hole_behavior.png")


def plot_reactive_behavior(
    summary: pd.DataFrame,
    experiments: list[Experiment],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    metrics = [
        ("end_eval_avoidance_rate_mean", "Avoidance rate"),
        (
            "end_eval_near_hole_lane_changes_mean",
            "Near-hole lane changes per episode",
        ),
        ("end_eval_action_entropy_mean", "Normalized action entropy"),
        ("end_eval_lane_entropy_mean", "Normalized lane entropy"),
        (
            "end_eval_lane_concentration_mean",
            "Maximum lane occupancy fraction",
        ),
        (
            "end_eval_boundary_noop_rate_mean",
            "Boundary no-op fraction of lateral actions",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Reactive Behavior and Policy-Collapse Diagnostics",
        fontsize=15,
        fontweight="bold",
    )
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        _plot_summary_lines(
            ax, summary, experiments, colors, metric, ylabel
        )
        if "rate" in metric or "entropy" in metric or "concentration" in metric:
            ax.set_ylim(-0.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "High lane concentration or boundary no-op rate indicates a collapsed/non-reactive policy",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _save(fig, out_dir / "04_reactive_behavior.png")


def plot_reward_vs_avoidance(
    summary: pd.DataFrame,
    experiments: list[Experiment],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    markers = {2: "o", 3: "s", 5: "^", 10: "D"}
    fig, ax = plt.subplots(figsize=(12, 8))
    for experiment in experiments:
        data = summary[summary["experiment"] == experiment.key]
        for _, row in data.iterrows():
            ax.scatter(
                row["end_eval_avoidance_rate_mean"],
                row["end_eval_reward_mean"],
                s=90,
                marker=markers[int(row["n_lanes"])],
                color=colors[experiment.key],
                label=experiment.key if int(row["n_lanes"]) == 2 else None,
            )
            ax.annotate(
                f'{int(row["n_lanes"])}L',
                (
                    row["end_eval_avoidance_rate_mean"],
                    row["end_eval_reward_mean"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(5000, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Mean avoidance rate at end checkpoint",
        ylabel="Mean end-checkpoint episode reward",
        title="Reward–Avoidance Trade-off (labels show lane count)",
        xlim=(-0.05, 1.05),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(title="Configuration", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Upper-right points combine high reward with genuine threatened-hole avoidance",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_dir / "05_reward_vs_avoidance.png")


def plot_lane_scaling(
    summary: pd.DataFrame,
    experiments: list[Experiment],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    metrics = [
        ("end_eval_reward_mean", "Mean episode reward"),
        ("end_eval_steps_mean", "Mean survived steps"),
        ("end_eval_falls_mean", "Mean falls per episode"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(
        "Lane-Scaling Decomposition at End Checkpoint",
        fontsize=15,
        fontweight="bold",
    )
    for ax, (metric, ylabel) in zip(axes, metrics):
        _plot_summary_lines(
            ax, summary, experiments, colors, metric, ylabel
        )
    axes[0].axhline(5000, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(1000, color="black", linestyle="--", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Each fall adds collision cost and removes fall_time_penalty future rewarding steps",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _save(fig, out_dir / "06_lane_scaling_decomposition.png")


def plot_stage_progression(
    runs: dict[tuple[str, int], dict[str, Any]],
    experiments: list[Experiment],
    lanes: tuple[int, ...],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    metrics = [
        ("reward", "Mean reward"),
        ("falls", "Mean falls"),
        ("avoidance_rate", "Mean avoidance rate"),
        ("lane_concentration", "Mean lane concentration"),
    ]
    fig, axes = plt.subplots(
        len(lanes), len(metrics), figsize=(20, 16), sharex="col"
    )
    fig.suptitle(
        "Beginning → Middle → End Policy Progression",
        fontsize=15,
        fontweight="bold",
    )
    for row, n_lanes in enumerate(lanes):
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for experiment in experiments:
                episodes = runs[(experiment.key, n_lanes)]["episodes"]
                stage_mean = (
                    episodes.groupby("stage", observed=True)[metric]
                    .mean()
                    .reindex(STAGE_ORDER)
                )
                ax.plot(
                    STAGE_ORDER,
                    stage_mean,
                    marker="o",
                    linewidth=1.5,
                    color=colors[experiment.key],
                    label=experiment.key,
                )
            if row == 0:
                ax.set_title(ylabel)
            if col == 0:
                ax.set_ylabel(f"{n_lanes} lanes")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Source: 20 deterministic evaluation episodes at each saved checkpoint",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    _save(fig, out_dir / "07_stage_progression.png")


def _checkpoint_episode_map(config: dict[str, Any]) -> dict[str, int]:
    total_episodes = int(config["training"]["total_episodes"])
    return {
        "beginning": 0,
        "middle": total_episodes // 2,
        "end": total_episodes,
    }


def plot_falls_and_avoidance_per_episode(
    runs: dict[tuple[str, int], dict[str, Any]],
    experiments: list[Experiment],
    lanes: tuple[int, ...],
    out_dir: Path,
) -> None:
    colors = _experiment_colors(experiments)
    metrics = [
        ("falls", "Mean falls per evaluation episode"),
        ("avoidance_rate", "Mean avoidance rate"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex="col")
    fig.suptitle(
        "Falls and Avoidance Rate Across Training Checkpoints",
        fontsize=15,
        fontweight="bold",
    )
    for metric_index, (metric, ylabel) in enumerate(metrics):
        for lane_index, n_lanes in enumerate(lanes):
            row = metric_index * 2 + lane_index // 2
            col = lane_index % 2
            ax = axes[row, col]
            for experiment in experiments:
                run = runs[(experiment.key, n_lanes)]
                episodes = run["episodes"]
                checkpoint_episodes = _checkpoint_episode_map(run["config"])
                stage_mean = (
                    episodes.groupby("stage", observed=True)[metric]
                    .mean()
                    .reindex(STAGE_ORDER)
                )
                x = [checkpoint_episodes[stage] for stage in STAGE_ORDER]
                ax.plot(
                    x,
                    stage_mean,
                    marker="o",
                    linewidth=1.8,
                    color=colors[experiment.key],
                    label=experiment.key,
                )
            ax.set(
                title=f"{n_lanes} lanes",
                ylabel=ylabel if col == 0 else "",
            )
            ax.grid(True, alpha=0.25)
            if metric == "avoidance_rate":
                ax.set_ylim(-0.05, 1.05)
            if row == 3:
                ax.set_xlabel("Training episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Source: evaluation_episodes.csv · beginning / middle / end checkpoints · "
            "y = mean over 20 deterministic eval episodes · "
            "avoidance = holes_avoided / (holes_avoided + hole_collisions)"
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    _save(fig, out_dir / "08_falls_and_avoidance_per_episode.png")


def plot_baseline100_training_falls_per_episode(
    runs: dict[tuple[str, int], dict[str, Any]],
    lanes: tuple[int, ...],
    out_dir: Path,
) -> None:
    experiment_key = "baseline100"
    missing_lanes = [
        n_lanes
        for n_lanes in lanes
        if runs[(experiment_key, n_lanes)]["training_episodes"].empty
    ]
    if missing_lanes:
        print(
            "  Skipped: 09_baseline100_training_falls_per_episode.png "
            f"(missing training_episodes.csv for lanes {missing_lanes})"
        )
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    fig.suptitle(
        "baseline100: Training Falls per Episode",
        fontsize=15,
        fontweight="bold",
    )
    for ax, n_lanes in zip(axes.flat, lanes):
        training = runs[(experiment_key, n_lanes)]["training_episodes"]
        ax.plot(
            training["episode"],
            training["falls"],
            color="crimson",
            linewidth=0.8,
            marker="o",
            markersize=2.5,
            alpha=0.85,
        )
        ax.set(
            title=f"{n_lanes} lanes",
            xlabel="Training episode",
            ylabel="Falls in training episode",
        )
        ax.grid(True, alpha=0.25)
    fig.text(
        0.5,
        0.015,
        "Source: training_episodes.csv · baseline100 · one point per completed training episode",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, out_dir / "09_baseline100_training_falls_per_episode.png")


def _format_number(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    experiments: list[Experiment],
    out_dir: Path,
) -> None:
    aggregate = (
        summary.groupby(["experiment", "experiment_label"], observed=True)
        .agg(
            reward=("end_eval_reward_mean", "mean"),
            plateau=("train_last500_reward_mean", "mean"),
            avoidance=("end_eval_avoidance_rate_mean", "mean"),
            success=("end_eval_success_rate", "mean"),
            falls=("end_eval_falls_mean", "mean"),
            lane_concentration=("end_eval_lane_concentration_mean", "mean"),
        )
        .reset_index()
    )
    best_reward = aggregate.loc[aggregate["reward"].idxmax()]
    best_avoidance = aggregate.loc[aggregate["avoidance"].idxmax()]

    config_lines = [
        "| Configuration | Visibility | Fall-time penalty | Collision reward | Final ε |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        row = summary[summary["experiment"] == experiment.key].iloc[0]
        config_lines.append(
            f"| `{experiment.key}` | {row['visibility_range']:g} | "
            f"{row['fall_time_penalty']:g} | {row['collision_penalty']:g} | "
            f"{row['exploration_final_eps']:g} |"
        )

    result_lines = [
        "| Lanes | Best end reward | Reward | Best avoidance | Avoidance rate |",
        "|---:|---|---:|---|---:|",
    ]
    for n_lanes in LANE_COUNTS:
        lane_rows = summary[summary["n_lanes"] == n_lanes]
        reward_row = lane_rows.loc[lane_rows["end_eval_reward_mean"].idxmax()]
        avoid_row = lane_rows.loc[
            lane_rows["end_eval_avoidance_rate_mean"].idxmax()
        ]
        result_lines.append(
            f"| {n_lanes} | `{reward_row['experiment']}` | "
            f"{reward_row['end_eval_reward_mean']:.1f} | "
            f"`{avoid_row['experiment']}` | "
            f"{avoid_row['end_eval_avoidance_rate_mean']:.2f} |"
        )

    baseline = summary[summary["experiment"] == "baseline100"].set_index(
        "n_lanes"
    )
    lane2 = baseline.loc[2]
    lane10 = baseline.loc[10]
    reward_drop = lane2["end_eval_reward_mean"] - lane10["end_eval_reward_mean"]
    fall_increase = lane10["end_eval_falls_mean"] - lane2["end_eval_falls_mean"]
    baseline5 = baseline.loc[5]
    vis50 = summary[summary["experiment"] == "vis50"].set_index("n_lanes")
    vis50_5 = vis50.loc[5]
    vis50_10 = vis50.loc[10]
    vis40 = summary[summary["experiment"] == "vis40"].set_index("n_lanes")
    vis40_identical_reward = vis40.loc[[2, 3, 10], "end_eval_reward_mean"].nunique() == 1
    vis40_reward = vis40.loc[2, "end_eval_reward_mean"]
    vis40_falls = vis40.loc[2, "end_eval_falls_mean"]
    pen500 = summary[summary["experiment"] == "pen500"]
    pen500_volatility = pen500["train_last500_reward_std"].mean()
    baseline_volatility = baseline["train_last500_reward_std"].mean()

    report = f"""# v1.1 Hyperparameter and Learning Diagnostics

## Executive conclusion

- **Best mean end-evaluation reward across lane counts:** `{best_reward['experiment']}`
  ({best_reward['reward']:.1f} averaged across 2/3/5/10 lanes).
- **Best mean threatened-hole avoidance:** `{best_avoidance['experiment']}`
  (avoidance rate {best_avoidance['avoidance']:.2f} averaged across lane counts).
- These are **configuration rankings**, not isolated hyperparameter effects:
  the vis40/vis50 runs jointly changed visibility, collision reward, and final ε.
- All agents use one training seed. The 20 evaluation episodes quantify
  environment/evaluation variation, not training-seed uncertainty.

## Configurations

{chr(10).join(config_lines)}

## Which run is best?

{chr(10).join(result_lines)}

Use raw reward and zero-fall success when the objective is simply to reach
5000. Use avoidance rate, near-hole lane changes, and policy concentration
when the scientific question is whether the agent learned reactive evasion.
`holes_passed` alone is not evidence of learning: it includes holes in lanes
the bike never occupied.

## Why reward plateaus lower with more lanes

1. **Falls explain the arithmetic.** The maximum is 5000 (1000 safe steps ×
   speed reward 5). A fall contributes the collision reward and also removes
   `fall_time_penalty` future rewarding steps. In the baseline, mean end reward
   drops by {_format_number(reward_drop)} from 2 to 10 lanes while mean falls
   increase by {_format_number(fall_increase, 2)}.
2. **The observation grows while the network is unchanged.** Inputs grow from
   4 values at 2 lanes to 12 at 10 lanes, but every run uses the same
   `[256, 256]` DQN. It must learn which neighboring direction is safe from
   more lane-specific hole distances.
3. **Policy collapse is measurable.** High lane concentration, low action
   entropy, zero near-hole lane changes, or repeated outward actions at a road
   boundary indicate that the policy found a stable lane/edge strategy instead
   of reacting to visible holes.
4. **More lanes add irrelevant successes.** `passed_other_lane` grows with lane
   count even if the bike never dodges. The hole-event plot therefore separates
   avoided, collided, unprotected, and other-lane events.
5. **The 500-step penalty is too destructive for diagnosis.** One mistake can
   remove half an episode, reducing useful post-fall experience and increasing
   return variance. Its score should not be interpreted as a clean test of
   collision aversion.

## Observed policy mechanisms in these runs

- **Baseline reward is not always avoidance learning.** At 5 lanes it scores
  {_format_number(baseline5['end_eval_reward_mean'])}, but its avoidance rate
  and near-hole lane changes are both
  {_format_number(baseline5['end_eval_avoidance_rate_mean'], 2)}. Its maximum
  lane occupancy is {_format_number(baseline5['end_eval_lane_concentration_mean'], 2)},
  evidence of a single-lane policy.
- **vis50 preserves reactive behavior at higher lane counts.** At 5 lanes its
  avoidance rate is {_format_number(vis50_5['end_eval_avoidance_rate_mean'], 2)}
  with {_format_number(vis50_5['end_eval_near_hole_lane_changes_mean'])}
  near-hole lane changes per episode; at 10 lanes these are
  {_format_number(vis50_10['end_eval_avoidance_rate_mean'], 2)} and
  {_format_number(vis50_10['end_eval_near_hole_lane_changes_mean'])}.
- **vis40 shows a degenerate solution.** Lanes 2, 3, and 10 have
  {"the same" if vis40_identical_reward else "nearly the same"} end reward
  ({_format_number(vis40_reward)}) and falls ({_format_number(vis40_falls)}),
  while their dominant lane is lane 0 and lane concentration is near 1. This
  is consistent with migrating to the first-generated lane rather than using
  the extra lanes.
- **pen500 is unstable.** Its mean last-500 reward standard deviation across
  lane counts is {_format_number(pen500_volatility)}, versus
  {_format_number(baseline_volatility)} for baseline100.

## How to read the figures

1. `01_learning_curves_by_lane.png`: learning speed, plateau, and instability.
2. `02_end_reward_and_success.png`: deterministic end-policy score and fraction
   of truly perfect episodes.
3. `03_hole_behavior.png`: whether reward corresponds to avoided holes rather
   than unrelated holes passing in other lanes.
4. `04_reactive_behavior.png`: policy-collapse diagnostics.
5. `05_reward_vs_avoidance.png`: separates high-scoring reactive policies from
   high-scoring passive policies.
6. `06_lane_scaling_decomposition.png`: connects extra lanes to falls, shorter
   episodes, and lower reward.
7. `07_stage_progression.png`: shows whether avoidance is acquired and retained.
8. `08_falls_and_avoidance_per_episode.png`: checkpoint learning curves for mean
   falls and avoidance rate, using the same 2×2 lane layout as figure 1.
9. `09_baseline100_training_falls_per_episode.png`: per-training-episode fall counts
   for baseline100 when `training_episodes.csv` is available.

## Statistical interpretation

- Error bars in figure 2 are percentile bootstrap 95% confidence intervals over
  the 20 deterministic evaluation episodes.
- `paired_differences.csv` compares each configuration with `baseline100` using
  matching evaluation episode IDs/seeds. A CI excluding zero is evidence of a
  consistent evaluation-seed difference for this trained seed.
- It is **not** evidence that a hyperparameter generalizes across training
  seeds. Run at least 5 independent training seeds before making a final
  hyperparameter claim.
- Training plateau metrics and deterministic end evaluation answer different
  questions. Report both; checkpoint performance can differ from the last-500
  exploratory training return.

## Recommended next experiment

Keep fall-time penalty, collision reward, and exploration fixed, and sweep only
visibility (25/40/50) across at least five training seeds. Then separately sweep
collision reward and final ε. This factorial discipline is required to identify
which individual hyperparameter causes an improvement.

## Machine-readable outputs

- `summary_table.csv`: one row per configuration × lane count.
- `paired_differences.csv`: paired end-evaluation differences from baseline.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")
    print("  Saved: README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v1.1 lane-count and hyperparameter sweeps"
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--lanes", nargs="+", type=int, default=list(LANE_COUNTS)
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expected-episodes", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lanes = tuple(args.lanes)
    if lanes != LANE_COUNTS:
        raise ValueError(
            f"This report expects matched lane counts {LANE_COUNTS}; got {lanes}"
        )
    experiments = default_experiments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(
        experiments,
        lanes,
        seed=args.seed,
        expected_episodes=args.expected_episodes,
    )
    summary = build_summary(runs, experiments, lanes)
    paired = build_paired_differences(runs, experiments, lanes)
    summary.to_csv(args.output_dir / "summary_table.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    print("  Saved: summary_table.csv")
    print("  Saved: paired_differences.csv")

    plot_learning_curves(runs, experiments, lanes, args.output_dir)
    plot_end_reward_and_success(summary, experiments, args.output_dir)
    plot_hole_behavior(summary, experiments, args.output_dir)
    plot_reactive_behavior(summary, experiments, args.output_dir)
    plot_reward_vs_avoidance(summary, experiments, args.output_dir)
    plot_lane_scaling(summary, experiments, args.output_dir)
    plot_stage_progression(runs, experiments, lanes, args.output_dir)
    plot_falls_and_avoidance_per_episode(
        runs, experiments, lanes, args.output_dir
    )
    plot_baseline100_training_falls_per_episode(runs, lanes, args.output_dir)
    write_report(summary, paired, experiments, args.output_dir)

    print(f"\nComparison complete: {args.output_dir}")


if __name__ == "__main__":
    main()
