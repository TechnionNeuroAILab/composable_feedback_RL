"""
Run CleanRL DQN with and without meta-gradient hyperparameter tuning,
then plot episodic returns.

Example:
    python code/run_metagradient_comparison.py --total-episodes 200 --seed 1
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
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cleanrl_dqn_meta_gradient import Args, run_training


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures" / "cleanrl_dqn_meta_gradient"
RESULTS_DIR = FIG_DIR / "results"

COLOR_BASELINE = "#2ca02c"
COLOR_META = "#ff7f0e"


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def plot_comparison(
    baseline_returns: list[float],
    meta_returns: list[float],
    *,
    seed: int,
    env_id: str,
    title_suffix: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    meta = np.asarray(meta_returns, dtype=np.float64)
    baseline_episodes = np.arange(1, len(baseline) + 1)
    meta_episodes = np.arange(1, len(meta) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(baseline_episodes, baseline, color=COLOR_BASELINE, alpha=0.25, lw=1)
    ax.plot(meta_episodes, meta, color=COLOR_META, alpha=0.25, lw=1)

    if smooth_window > 1:
        baseline_sm = _smooth(baseline, smooth_window)
        meta_sm = _smooth(meta, smooth_window)
        baseline_smooth_episodes = np.arange(
            smooth_window, smooth_window + len(baseline_sm)
        )
        meta_smooth_episodes = np.arange(
            smooth_window, smooth_window + len(meta_sm)
        )
        ax.plot(
            baseline_smooth_episodes,
            baseline_sm,
            color=COLOR_BASELINE,
            lw=2.5,
            label=f"CleanRL DQN baseline (seed={seed})",
        )
        ax.plot(
            meta_smooth_episodes,
            meta_sm,
            color=COLOR_META,
            lw=2.5,
            label=f"Meta-gradient DQN (seed={seed})",
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
            meta_episodes,
            meta,
            color=COLOR_META,
            lw=2.0,
            label=f"Meta-gradient DQN (seed={seed})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"{env_id}: baseline vs meta-gradient DQN ({title_suffix})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CleanRL DQN baseline vs meta-gradient tuning."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    if cli_args.total_episodes is None and cli_args.total_timesteps is None:
        cli_args.total_timesteps = 500_000
    if cli_args.total_episodes is not None and cli_args.total_timesteps is not None:
        raise ValueError("Specify either --total-episodes or --total-timesteps, not both.")

    if cli_args.learning_starts is None:
        cli_args.learning_starts = (
            10_000 if cli_args.total_timesteps is not None else 1_000
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if cli_args.total_timesteps is not None:
        budget_tag = f"{cli_args.total_timesteps // 1000}ksteps"
        title_suffix = f"{cli_args.total_timesteps:,} timesteps, by episode"
    else:
        budget_tag = f"{cli_args.total_episodes}ep"
        title_suffix = f"{cli_args.total_episodes} episodes"

    baseline_path = RESULTS_DIR / f"baseline_seed{cli_args.seed}_{budget_tag}.json"
    meta_path = RESULTS_DIR / f"meta_seed{cli_args.seed}_{budget_tag}.json"
    plot_path = FIG_DIR / f"reward_seed{cli_args.seed}_{budget_tag}_by_episode.png"

    shared = dict(
        env_id=cli_args.env_id,
        seed=cli_args.seed,
        total_episodes=cli_args.total_episodes,
        total_timesteps=cli_args.total_timesteps or 500_000,
        learning_starts=cli_args.learning_starts,
        cuda=cli_args.cuda,
        tensorboard=False,
        log_interval=5_000,
    )

    if not cli_args.plot_only:
        print("Running baseline DQN (no meta-gradient)...", flush=True)
        baseline_results = run_training(
            Args(
                **shared,
                meta_gradient=False,
                returns_output=str(baseline_path),
            )
        )
        print(
            f"Baseline finished: {len(baseline_results['episode_returns'])} episodes, "
            f"mean return last 20 = "
            f"{np.mean(baseline_results['episode_returns'][-20:]):.2f}",
            flush=True,
        )

        print("Running meta-gradient DQN...", flush=True)
        meta_results = run_training(
            Args(
                **shared,
                meta_gradient=True,
                returns_output=str(meta_path),
            )
        )
        print(
            f"Meta finished: {len(meta_results['episode_returns'])} episodes, "
            f"mean return last 20 = "
            f"{np.mean(meta_results['episode_returns'][-20:]):.2f}, "
            f"final gamma={meta_results['final_gamma']:.4f}, "
            f"final lr={meta_results['final_learning_rate']:.2e}",
            flush=True,
        )
    else:
        if not baseline_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing saved results in {RESULTS_DIR}. Run without --plot-only first."
            )

    with baseline_path.open(encoding="utf-8") as handle:
        baseline_blob = json.load(handle)
    with meta_path.open(encoding="utf-8") as handle:
        meta_blob = json.load(handle)

    plot_comparison(
        baseline_blob["episode_returns"],
        meta_blob["episode_returns"],
        seed=cli_args.seed,
        env_id=cli_args.env_id,
        title_suffix=title_suffix,
        smooth_window=cli_args.smooth_window,
        output_path=plot_path,
    )
    print(f"Saved plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
