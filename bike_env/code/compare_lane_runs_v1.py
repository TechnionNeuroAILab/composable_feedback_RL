#!/usr/bin/env python3
"""Cross-lane comparison for matched bike_dqn_multi_lanes_v1 runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3.common.monitor import load_results

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _HERE / "training_results_v1"
_DEFAULT_OUTPUT = _HERE.parent / "plotting" / "v1_plots" / "lane_comparison"
LANE_COUNTS = (2, 3, 5, 10, 20)
STAGE_ORDER = ("beginning", "middle", "end")
HOLE_EVENTS = ("avoided", "collision", "unprotected_pass")


def run_dir(results_root: Path, n_lanes: int, seed: int = 1) -> Path:
    name = (
        f"lanes_{n_lanes}__actions_3__falls_hole-collision"
        f"__lambda_5__seed_{seed}"
    )
    path = results_root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing run directory: {path}")
    return path


def load_run_data(results_root: Path, n_lanes: int, seed: int = 1) -> dict[str, pd.DataFrame]:
    root = run_dir(results_root, n_lanes, seed)
    monitor = load_results(str(root)).reset_index(drop=True)
    monitor["episode"] = np.arange(1, len(monitor) + 1)
    monitor["n_lanes"] = n_lanes

    episodes = pd.read_csv(root / "evaluation_episodes.csv")
    episodes["n_lanes"] = n_lanes
    episodes["avoidance_rate"] = np.where(
        (episodes["holes_avoided"] + episodes["hole_collisions"]) > 0,
        episodes["holes_avoided"]
        / (episodes["holes_avoided"] + episodes["hole_collisions"]),
        np.nan,
    )
    episodes["reward_per_distance"] = episodes["reward"] / episodes["distance"].replace(
        0, np.nan
    )

    steps = pd.read_csv(root / "evaluation_steps.csv")
    steps["n_lanes"] = n_lanes

    holes = pd.read_csv(root / "evaluation_holes.csv")
    holes["n_lanes"] = n_lanes

    return {
        "monitor": monitor,
        "episodes": episodes,
        "steps": steps,
        "holes": holes,
    }


def build_summary_table(all_runs: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict] = []
    for n_lanes, data in sorted(all_runs.items()):
        monitor = data["monitor"]
        episodes = data["episodes"]
        end = episodes[episodes["stage"] == "end"]
        beginning = episodes[episodes["stage"] == "beginning"]

        rows.append(
            {
                "n_lanes": n_lanes,
                "train_final_100ep_reward_mean": monitor["r"].tail(100).mean(),
                "train_final_100ep_reward_std": monitor["r"].tail(100).std(),
                "end_eval_reward_mean": end["reward"].mean(),
                "end_eval_reward_std": end["reward"].std(),
                "end_falls_mean": end["falls"].mean(),
                "end_falls_std": end["falls"].std(),
                "end_collisions_mean": end["hole_collisions"].mean(),
                "end_collisions_std": end["hole_collisions"].std(),
                "end_avoided_mean": end["holes_avoided"].mean(),
                "end_avoided_std": end["holes_avoided"].std(),
                "end_avoidance_rate_mean": end["avoidance_rate"].mean(),
                "end_avoidance_rate_std": end["avoidance_rate"].std(),
                "end_distance_mean": end["distance"].mean(),
                "end_reward_per_distance_mean": end["reward_per_distance"].mean(),
                "learning_improvement_reward": end["reward"].mean()
                - beginning["reward"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_learning_overlay(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for n_lanes, data in sorted(all_runs.items()):
        monitor = data["monitor"]
        moving = monitor["r"].rolling(100, min_periods=1).mean()
        ax.plot(monitor["episode"], moving, linewidth=2, label=f"{n_lanes} lanes")
    ax.set(
        xlabel="Training Episode",
        ylabel="Mean Reward (100-episode MA)",
        title="Learning Curves Across Lane Counts",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Lanes")
    fig.tight_layout()
    _save(fig, out_dir / "01_learning_curves_overlay.png")


def plot_end_metrics_scaling(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    episodes = pd.concat([d["episodes"] for d in all_runs.values()], ignore_index=True)
    end = episodes[episodes["stage"] == "end"].copy()

    metrics = [
        ("reward", "Episode Reward"),
        ("avoidance_rate", "Avoidance Rate"),
        ("hole_collisions", "Hole Collisions per Episode"),
        ("holes_avoided", "Holes Avoided per Episode"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("End-Stage Performance vs Lane Count", fontsize=14, fontweight="bold")

    for ax, (col, label) in zip(axes.flat, metrics):
        summary = (
            end.groupby("n_lanes", observed=True)[col]
            .agg(["mean", "std"])
            .reindex(LANE_COUNTS)
        )
        ax.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            fmt="o-",
            capsize=4,
            linewidth=2,
            markersize=8,
        )
        ax.set(xlabel="Number of Lanes", ylabel=label, xticks=list(LANE_COUNTS))
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "02_end_metrics_vs_lanes.png")


def plot_end_distributions(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    episodes = pd.concat([d["episodes"] for d in all_runs.values()], ignore_index=True)
    end = episodes[episodes["stage"] == "end"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("End-Stage Eval Episode Distributions", fontsize=14, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["reward", "avoidance_rate", "falls"],
        ["Episode Reward", "Avoidance Rate", "Falls per Episode"],
    ):
        sns.boxplot(data=end, x="n_lanes", y=col, order=list(LANE_COUNTS), ax=ax)
        sns.stripplot(
            data=end,
            x="n_lanes",
            y=col,
            order=list(LANE_COUNTS),
            color="black",
            alpha=0.35,
            size=3,
            ax=ax,
        )
        ax.set(xlabel="Number of Lanes", ylabel=title, title=title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "03_end_eval_boxplots.png")


def plot_stage_progression(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    episodes = pd.concat([d["episodes"] for d in all_runs.values()], ignore_index=True)
    summary = (
        episodes.groupby(["n_lanes", "stage"], observed=True)
        .agg(
            reward=("reward", "mean"),
            avoidance_rate=("avoidance_rate", "mean"),
            falls=("falls", "mean"),
        )
        .reset_index()
    )
    summary["stage"] = pd.Categorical(summary["stage"], categories=STAGE_ORDER, ordered=True)
    summary = summary.sort_values(["n_lanes", "stage"])

    metrics = [
        ("reward", "Mean Episode Reward"),
        ("avoidance_rate", "Mean Avoidance Rate"),
        ("falls", "Mean Falls per Episode"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Learning Progression by Stage and Lane Count", fontsize=14, fontweight="bold")

    for ax, (col, title) in zip(axes, metrics):
        sns.lineplot(
            data=summary,
            x="stage",
            y=col,
            hue="n_lanes",
            hue_order=list(LANE_COUNTS),
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set(xlabel="Checkpoint Stage", ylabel=title, title=title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Lanes", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "04_stage_progression.png")


def plot_hole_behavior(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    rows: list[dict] = []
    for n_lanes, data in all_runs.items():
        holes = data["holes"]
        episodes = data["episodes"]
        end_episodes = sorted(episodes[episodes["stage"] == "end"]["episode"].unique())
        end_holes = holes[
            (holes["stage"] == "end") & (holes["event"].isin(HOLE_EVENTS))
        ]
        for episode in end_episodes:
            ep_events = end_holes[end_holes["episode"] == episode]["event"].value_counts()
            for event in HOLE_EVENTS:
                rows.append(
                    {
                        "n_lanes": n_lanes,
                        "episode": episode,
                        "event": event,
                        "count": int(ep_events.get(event, 0)),
                    }
                )
    filled = pd.DataFrame(rows)
    summary = (
        filled.groupby(["n_lanes", "event"], observed=True)["count"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=summary,
        x="n_lanes",
        y="mean",
        hue="event",
        hue_order=list(HOLE_EVENTS),
        order=list(LANE_COUNTS),
        ax=ax,
    )
    ax.set(
        xlabel="Number of Lanes",
        ylabel="Mean Events per Eval Episode",
        title="Hole Outcomes at End Checkpoint (avoided / collision / unprotected pass)",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Event")
    fig.tight_layout()
    _save(fig, out_dir / "05_hole_behavior_breakdown.png")


def plot_lane_usage(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    n_runs = len(all_runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(4 * n_runs, 4), sharey=False)
    if n_runs == 1:
        axes = [axes]
    fig.suptitle("Lane Usage at End Checkpoint (mean over eval episodes)", fontsize=14, fontweight="bold")

    for ax, (n_lanes, data) in zip(axes, sorted(all_runs.items())):
        steps = data["steps"]
        end_steps = steps[steps["stage"] == "end"]
        fractions: list[dict] = []
        for episode, group in end_steps.groupby("episode", observed=True):
            total = len(group)
            if total == 0:
                continue
            for lane, count in group["lane"].value_counts().items():
                fractions.append(
                    {"episode": episode, "lane": lane, "fraction": count / total}
                )
        frac_df = pd.DataFrame(fractions)
        lane_summary = (
            frac_df.groupby("lane", observed=True)["fraction"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("lane")
        )
        ax.bar(
            lane_summary["lane"],
            lane_summary["mean"],
            yerr=lane_summary["std"].fillna(0),
            capsize=3,
            color="steelblue",
            alpha=0.85,
        )
        ax.set(
            xlabel="Lane",
            ylabel="Fraction of Steps",
            title=f"{n_lanes} lanes",
            xticks=range(n_lanes),
        )
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "06_lane_usage_by_lane_count.png")


def plot_near_hole_maneuvers(all_runs: dict[int, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    step_frames = []
    for n_lanes, data in all_runs.items():
        steps = data["steps"]
        end_steps = steps[steps["stage"] == "end"]
        per_episode = (
            end_steps.groupby(["n_lanes", "episode"], observed=True)["near_hole_lane_change"]
            .sum()
            .reset_index(name="near_hole_lane_changes")
        )
        step_frames.append(per_episode)
    combined = pd.concat(step_frames, ignore_index=True)
    summary = (
        combined.groupby("n_lanes", observed=True)["near_hole_lane_changes"]
        .agg(["mean", "std"])
        .reindex(LANE_COUNTS)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=4,
        color="purple",
        alpha=0.75,
    )
    ax.set(
        xlabel="Number of Lanes",
        ylabel="Mean Near-Hole Lane Changes per Episode",
        title="Near-Hole Maneuvers at End Checkpoint",
        xticks=list(LANE_COUNTS),
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "07_near_hole_lane_changes.png")


def write_methodology(out_dir: Path) -> None:
    text = """# Cross-Lane Comparison Methodology

## Runs included
Matched v1 runs: 2, 3, 5, 10, 20 lanes; 3 actions; hole-only falls;
lambda=5 mean hole gap; fixed speed 5; collision penalty -10; seed 1; 8000 training episodes.

## Uncertainty
Each lane count uses **20 evaluation episodes** at the end checkpoint.
Error bars and boxplots reflect **std / distribution across those 20 episodes**, not across training seeds.

## Key metrics
- **Avoidance rate** (per eval episode): `holes_avoided / (holes_avoided + hole_collisions)`
- **Learning improvement**: end-stage mean reward minus beginning-stage mean reward
- **Near-hole lane changes**: count of steps where the agent changed lane while a visible hole was in the previous lane

## Figures
1. Overlaid 100-episode moving-average training rewards
2. End-stage scalars vs lane count (error bars = eval episode std)
3. End-stage boxplots with individual eval episodes
4. Beginning / middle / end progression by lane count
5. Hole outcome breakdown (avoided, collision, unprotected_pass)
6. Lane occupancy fractions at end checkpoint
7. Near-hole lane-change rate vs lane count
"""
    (out_dir / "00_comparison_methodology.md").write_text(text, encoding="utf-8")
    print("  Saved: 00_comparison_methodology.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matched lane-count v1 runs")
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--lanes", nargs="+", type=int, default=list(LANE_COUNTS))
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_runs: dict[int, dict[str, pd.DataFrame]] = {}
    for n_lanes in args.lanes:
        print(f"Loading {n_lanes} lanes ...")
        all_runs[n_lanes] = load_run_data(args.results_dir, n_lanes, args.seed)

    summary = build_summary_table(all_runs)
    summary.to_csv(args.output_dir / "summary_table.csv", index=False)
    print(f"  Saved: summary_table.csv")

    write_methodology(args.output_dir)
    plot_learning_overlay(all_runs, args.output_dir)
    plot_end_metrics_scaling(all_runs, args.output_dir)
    plot_end_distributions(all_runs, args.output_dir)
    plot_stage_progression(all_runs, args.output_dir)
    plot_hole_behavior(all_runs, args.output_dir)
    plot_lane_usage(all_runs, args.output_dir)
    plot_near_hole_maneuvers(all_runs, args.output_dir)

    print(f"\nComparison complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
