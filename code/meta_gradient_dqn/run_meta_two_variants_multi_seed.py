"""
Run meta_gradient_dqn.py and meta_gradient_confidence_dqn.py across seeds;
plot mean ± std reward and hyperparameter trajectories per variant.

Example:
    python code/run_meta_two_variants_multi_seed.py --seeds 1,2,3,4,5 --total-episodes 5000
    python code/run_meta_two_variants_multi_seed.py --plot-only --seeds 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
FIG_DIR = ROOT / "paper" / "figures" / "meta_gradient_confidence"
RESULTS_DIR = FIG_DIR / "results"

VARIANTS = (
    {
        "tag": "meta_gradient_dqn",
        "module": "meta_gradient_dqn",
        "label": "Vanilla meta-gradient DQN",
        "result_name": "meta_gradient_dqn_seed{seed}_{budget}",
        "color": "#1f77b4",
    },
    {
        "tag": "meta_gradient_confidence_dqn",
        "module": "meta_gradient_confidence_dqn",
        "label": "Confidence meta-gradient DQN",
        "result_name": "meta_seed{seed}_{budget}",
        "color": "#ff7f0e",
    },
)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def _episode_axis_for_steps(
    steps: np.ndarray, episode_returns: list[float], global_steps: int
) -> np.ndarray:
    if len(episode_returns) == 0:
        return steps.astype(np.float64)
    episode_end_steps = np.cumsum(np.asarray(episode_returns, dtype=np.float64))
    if global_steps > episode_end_steps[-1]:
        episode_end_steps = np.concatenate(
            [
                episode_end_steps,
                np.array([float(global_steps)], dtype=np.float64),
            ]
        )
        episode_axis = np.arange(1, len(episode_end_steps) + 1, dtype=np.float64)
    else:
        episode_axis = np.arange(1, len(episode_end_steps) + 1, dtype=np.float64)
    return np.interp(steps, episode_end_steps, episode_axis)


def _result_path(variant: dict, seed: int, budget_tag: str) -> Path:
    name = variant["result_name"].format(seed=seed, budget=budget_tag)
    return RESULTS_DIR / f"{name}.json"


def _run_variant(variant: dict, seed: int, budget_tag: str, output_path: Path) -> dict:
    module = __import__(variant["module"])
    kwargs = {
        "env_id": "CartPole-v1",
        "seed": seed,
        "total_episodes": int(budget_tag.replace("ep", "")),
        "total_timesteps": 500_000,
        "meta_gradient": True,
        "cuda": False,
        "tensorboard": False,
        "log_interval": 1_000,
    }
    if variant["module"] == "meta_gradient_dqn":
        kwargs["returns_output"] = str(output_path)
    else:
        kwargs["output_json"] = str(output_path)
    return module.run_training(module.Args(**kwargs))


def _load_or_run(
    variant: dict, seed: int, budget_tag: str, *, plot_only: bool
) -> dict:
    path = _result_path(variant, seed, budget_tag)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if plot_only:
        raise FileNotFoundError(f"Missing results: {path}")
    print(f"Running {variant['label']} seed={seed}...", flush=True)
    result = _run_variant(variant, seed, budget_tag, path)
    returns = result["episode_returns"]
    print(
        f"  finished: {len(returns)} episodes, "
        f"last50={np.mean(returns[-50:]):.1f}, "
        f"final gamma={result.get('final_gamma', float('nan')):.5f}, "
        f"final lr={result.get('final_learning_rate', float('nan')):.2e}",
        flush=True,
    )
    return result


def _reward_mean_stats(
    blobs: list[dict], smooth_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(blob["episode_returns"]) for blob in blobs)
    episode_axis = np.arange(1, max_len + 1)
    smoothed_runs: list[np.ndarray] = []
    for blob in blobs:
        values = np.asarray(blob["episode_returns"], dtype=np.float64)
        if smooth_window > 1 and len(values) >= smooth_window:
            sm = _smooth(values, smooth_window)
            x = np.arange(smooth_window, smooth_window + len(sm))
        else:
            sm = values
            x = np.arange(1, len(sm) + 1)
        padded = np.full(max_len, np.nan, dtype=np.float64)
        padded[x - 1] = sm
        smoothed_runs.append(padded)

    stack = np.stack(smoothed_runs, axis=0)
    valid = np.all(~np.isnan(stack), axis=0)
    x_plot = episode_axis[valid]
    mean = np.mean(stack[:, valid], axis=0)
    std = np.std(stack[:, valid], axis=0)
    return x_plot, mean, std


def _knob_mean_stats(
    blobs: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    gamma_runs: list[tuple[np.ndarray, np.ndarray]] = []
    lr_runs: list[tuple[np.ndarray, np.ndarray]] = []
    max_episode = 0.0

    for blob in blobs:
        history = blob.get("controller_history", [])
        if not history:
            continue
        steps = np.asarray([entry["step"] for entry in history], dtype=np.float64)
        gamma = np.asarray([entry["gamma"] for entry in history], dtype=np.float64)
        lr = np.asarray([entry["learning_rate"] for entry in history], dtype=np.float64)
        episodes = _episode_axis_for_steps(
            steps,
            blob["episode_returns"],
            int(blob.get("global_steps", steps[-1])),
        )
        max_episode = max(max_episode, float(episodes[-1]))
        gamma_runs.append((episodes, gamma))
        lr_runs.append((episodes, lr))

    if not gamma_runs:
        return None

    grid = np.linspace(1.0, max_episode, num=500)
    gamma_stack = []
    lr_stack = []
    for (episodes, gamma), (_, lr) in zip(gamma_runs, lr_runs):
        gamma_stack.append(np.interp(grid, episodes, gamma))
        lr_stack.append(np.interp(grid, episodes, lr))
    gamma_mean = np.mean(np.stack(gamma_stack, axis=0), axis=0)
    gamma_std = np.std(np.stack(gamma_stack, axis=0), axis=0)
    lr_mean = np.mean(np.stack(lr_stack, axis=0), axis=0)
    lr_std = np.std(np.stack(lr_stack, axis=0), axis=0)
    return grid, gamma_mean, gamma_std, lr_mean, lr_std


def plot_reward_mean(
    blobs: list[dict],
    *,
    variant: dict,
    seeds: list[int],
    budget_tag: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    x_plot, mean, std = _reward_mean_stats(blobs, smooth_window)
    color = variant["color"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(x_plot, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.plot(
        x_plot,
        mean,
        color=color,
        lw=2.5,
        label=f"{variant['label']} (mean over {len(seeds)} seeds)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(
        f"CartPole-v1: {variant['label']} "
        f"({budget_tag}, seeds={','.join(map(str, seeds))}, "
        f"{smooth_window}-ep smooth mean ± std)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_knobs_mean(
    blobs: list[dict],
    *,
    variant: dict,
    seeds: list[int],
    budget_tag: str,
    output_path: Path,
) -> None:
    stats = _knob_mean_stats(blobs)
    if stats is None:
        print(f"No controller history for {variant['tag']}; skipping knob plot.", flush=True)
        return
    grid, gamma_mean, gamma_std, lr_mean, lr_std = stats
    color = variant["color"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_gamma, ax_lr = axes
    ax_gamma.fill_between(
        grid, gamma_mean - gamma_std, gamma_mean + gamma_std, color=color, alpha=0.2
    )
    ax_gamma.plot(grid, gamma_mean, color=color, lw=2.0, label="Mean learned γ")
    ax_gamma.axhline(0.99, color="#2ca02c", ls="--", lw=1.2, label="Fixed baseline γ=0.99")
    ax_gamma.set_ylabel("Gamma")
    ax_gamma.grid(True, alpha=0.3)
    ax_gamma.legend(loc="best", fontsize=9)

    ax_lr.fill_between(
        grid, lr_mean - lr_std, lr_mean + lr_std, color=color, alpha=0.2
    )
    ax_lr.plot(grid, lr_mean, color=color, lw=2.0, label="Mean learned lr")
    ax_lr.axhline(
        2.5e-4, color="#2ca02c", ls="--", lw=1.2, label="Fixed baseline lr=2.5e-4"
    )
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Episode")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.grid(True, alpha=0.3)
    ax_lr.legend(loc="best", fontsize=9)
    ax_gamma.set_title(
        f"{variant['label']}: learned hyperparameters by episode "
        f"({budget_tag}, seeds={','.join(map(str, seeds))}, mean ± std)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


    plt.close(fig)


def plot_reward_comparison(
    variant_blobs: dict[str, list[dict]],
    *,
    seeds: list[int],
    budget_tag: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for variant in VARIANTS:
        blobs = variant_blobs[variant["tag"]]
        x_plot, mean, std = _reward_mean_stats(blobs, smooth_window)
        color = variant["color"]
        ax.fill_between(x_plot, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x_plot,
            mean,
            color=color,
            lw=2.5,
            label=f"{variant['label']} (n={len(seeds)} seeds)",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(
        f"CartPole-v1: vanilla vs confidence meta-gradient "
        f"({budget_tag}, seeds={','.join(map(str, seeds))}, "
        f"{smooth_window}-ep smooth mean ± std)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_knobs_comparison(
    variant_blobs: dict[str, list[dict]],
    *,
    seeds: list[int],
    budget_tag: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_gamma, ax_lr = axes

    for variant in VARIANTS:
        stats = _knob_mean_stats(variant_blobs[variant["tag"]])
        if stats is None:
            continue
        grid, gamma_mean, gamma_std, lr_mean, lr_std = stats
        color = variant["color"]
        label = variant["label"]
        ax_gamma.fill_between(
            grid, gamma_mean - gamma_std, gamma_mean + gamma_std, color=color, alpha=0.18
        )
        ax_gamma.plot(grid, gamma_mean, color=color, lw=2.0, label=label)
        ax_lr.fill_between(
            grid, lr_mean - lr_std, lr_mean + lr_std, color=color, alpha=0.18
        )
        ax_lr.plot(grid, lr_mean, color=color, lw=2.0, label=label)

    ax_gamma.axhline(0.99, color="#2ca02c", ls="--", lw=1.2, label="Fixed baseline γ=0.99")
    ax_lr.axhline(2.5e-4, color="#2ca02c", ls="--", lw=1.2, label="Fixed baseline lr=2.5e-4")
    ax_gamma.set_ylabel("Gamma")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Episode")
    ax_gamma.set_title(
        f"Learned hyperparameters: vanilla vs confidence meta-gradient "
        f"({budget_tag}, seeds={','.join(map(str, seeds))}, mean ± std by episode)"
    )
    ax_gamma.grid(True, alpha=0.3)
    ax_lr.grid(True, alpha=0.3)
    ax_gamma.legend(loc="best", fontsize=9)
    ax_lr.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed runs and plots for both meta-gradient variants."
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--total-episodes", type=int, default=5000)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    cli_args = parse_args()
    seeds = [int(s.strip()) for s in cli_args.seeds.split(",") if s.strip()]
    budget_tag = f"{cli_args.total_episodes}ep"
    seed_tag = "-".join(str(s) for s in seeds)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variant_blobs: dict[str, list[dict]] = {}
    for variant in VARIANTS:
        blobs: list[dict] = []
        for seed in seeds:
            blobs.append(
                _load_or_run(variant, seed, budget_tag, plot_only=cli_args.plot_only)
            )
        variant_blobs[variant["tag"]] = blobs

        reward_path = (
            FIG_DIR / f"reward_{variant['tag']}_seeds{seed_tag}_{budget_tag}_mean.png"
        )
        knobs_path = (
            FIG_DIR
            / f"gamma_lr_{variant['tag']}_seeds{seed_tag}_{budget_tag}_mean_by_episode.png"
        )
        plot_reward_mean(
            blobs,
            variant=variant,
            seeds=seeds,
            budget_tag=budget_tag,
            smooth_window=cli_args.smooth_window,
            output_path=reward_path,
        )
        plot_knobs_mean(
            blobs,
            variant=variant,
            seeds=seeds,
            budget_tag=budget_tag,
            output_path=knobs_path,
        )
        print(f"Saved {reward_path}", flush=True)
        print(f"Saved {knobs_path}", flush=True)

        last50 = [np.mean(blob["episode_returns"][-50:]) for blob in blobs]
        print(
            f"{variant['tag']}: last50 mean = {np.mean(last50):.1f} ± {np.std(last50):.1f}",
            flush=True,
        )

    compare_reward_path = (
        FIG_DIR
        / f"reward_meta_gradient_dqn_vs_confidence_seeds{seed_tag}_{budget_tag}_mean.png"
    )
    compare_knobs_path = (
        FIG_DIR
        / f"gamma_lr_meta_gradient_dqn_vs_confidence_seeds{seed_tag}_{budget_tag}_mean_by_episode.png"
    )
    plot_reward_comparison(
        variant_blobs,
        seeds=seeds,
        budget_tag=budget_tag,
        smooth_window=cli_args.smooth_window,
        output_path=compare_reward_path,
    )
    plot_knobs_comparison(
        variant_blobs,
        seeds=seeds,
        budget_tag=budget_tag,
        output_path=compare_knobs_path,
    )
    print(f"Saved {compare_reward_path}", flush=True)
    print(f"Saved {compare_knobs_path}", flush=True)


if __name__ == "__main__":
    main()
