"""
Run opt2b temporal disagreement and plot against baseline (single seed).

Example:
    python code/run_opt2b_vs_baseline.py --seed 1 --total-episodes 5000
    python code/run_opt2b_vs_baseline.py --plot-only --seed 1 --total-episodes 5000
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

COLOR_BASELINE = "#2ca02c"
COLOR_OPT2B = "#17becf"


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def plot_comparison(
    baseline_returns: list[float],
    opt2b_returns: list[float],
    *,
    seed: int,
    budget_tag: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    opt2b = np.asarray(opt2b_returns, dtype=np.float64)
    baseline_episodes = np.arange(1, len(baseline) + 1)
    opt2b_episodes = np.arange(1, len(opt2b) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(baseline_episodes, baseline, color=COLOR_BASELINE, alpha=0.25, lw=1)
    ax.plot(opt2b_episodes, opt2b, color=COLOR_OPT2B, alpha=0.25, lw=1)

    if smooth_window > 1:
        baseline_sm = _smooth(baseline, smooth_window)
        opt2b_sm = _smooth(opt2b, smooth_window)
        baseline_x = np.arange(smooth_window, smooth_window + len(baseline_sm))
        opt2b_x = np.arange(smooth_window, smooth_window + len(opt2b_sm))
        ax.plot(
            baseline_x,
            baseline_sm,
            color=COLOR_BASELINE,
            lw=2.5,
            label=f"CleanRL DQN baseline (seed={seed})",
        )
        ax.plot(
            opt2b_x,
            opt2b_sm,
            color=COLOR_OPT2B,
            lw=2.5,
            label=f"Opt2b temporal disagreement (seed={seed})",
        )
    else:
        ax.plot(
            baseline_episodes,
            baseline,
            color=COLOR_BASELINE,
            lw=2.0,
            label=f"CleanRL DQN baseline (seed={seed})",
        )
        ax.plot(
            opt2b_episodes,
            opt2b,
            color=COLOR_OPT2B,
            lw=2.0,
            label=f"Opt2b temporal disagreement (seed={seed})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(
        f"CartPole-v1: baseline vs opt2b temporal disagreement "
        f"({budget_tag.replace('ep', ' episodes')})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/plot opt2b vs baseline comparison."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=5000)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    cli_args = parse_args()
    budget_tag = f"{cli_args.total_episodes}ep"
    baseline_path = RESULTS_DIR / f"baseline_seed{cli_args.seed}_{budget_tag}.json"
    opt2b_path = RESULTS_DIR / f"opt2b_seed{cli_args.seed}_{budget_tag}.json"
    plot_path = (
        FIG_DIR
        / f"reward_baseline_vs_opt2b_seed{cli_args.seed}_{budget_tag}_by_episode.png"
    )

    if not cli_args.plot_only:
        from opt2b_temporal_disagreement import Args as Opt2bArgs
        from opt2b_temporal_disagreement import run_training

        print("Running opt2b temporal disagreement...", flush=True)
        result = run_training(
            Opt2bArgs(
                seed=cli_args.seed,
                total_episodes=cli_args.total_episodes,
                total_timesteps=500_000,
                meta_gradient=True,
                cuda=cli_args.cuda,
                tensorboard=False,
                log_interval=1_000,
                output_json=str(opt2b_path),
            )
        )
        returns = result["episode_returns"]
        print(
            f"Opt2b finished: {len(returns)} episodes, "
            f"last50={np.mean(returns[-50:]):.1f}, "
            f"final gamma={result['final_gamma']:.5f}, "
            f"final lr={result['final_learning_rate']:.2e}",
            flush=True,
        )

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline results: {baseline_path}")
    if not opt2b_path.exists():
        raise FileNotFoundError(f"Missing opt2b results: {opt2b_path}")

    with baseline_path.open(encoding="utf-8") as handle:
        baseline_blob = json.load(handle)
    with opt2b_path.open(encoding="utf-8") as handle:
        opt2b_blob = json.load(handle)

    plot_comparison(
        baseline_blob["episode_returns"],
        opt2b_blob["episode_returns"],
        seed=cli_args.seed,
        budget_tag=budget_tag,
        smooth_window=cli_args.smooth_window,
        output_path=plot_path,
    )
    print(f"Saved plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
