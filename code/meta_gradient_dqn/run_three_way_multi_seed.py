"""
Train baseline, original confidence meta-gradient, and opt2 across multiple seeds.

Reuses existing JSON results when present. Produces mean ± std reward plots.

Example:
    python code/run_three_way_multi_seed.py --seeds 1,2,3,4,5 --total-episodes 5000
    python code/run_three_way_multi_seed.py --plot-only --seeds 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import importlib
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
    ("baseline", "meta_gradient_dqn", {"meta_gradient": False}),
    ("meta", "meta_gradient_confidence_dqn", {"meta_gradient": True}),
    ("opt2", "opt2_enriched_meta_features", {"meta_gradient": True}),
)

COLORS = {
    "baseline": "#2ca02c",
    "meta": "#ff7f0e",
    "opt2": "#9467bd",
}

LABELS = {
    "baseline": "CleanRL DQN baseline",
    "meta": "Confidence meta (original)",
    "opt2": "Opt2: enriched meta features",
}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def _result_path(tag: str, seed: int, budget_tag: str) -> Path:
    if tag == "baseline":
        return RESULTS_DIR / f"baseline_seed{seed}_{budget_tag}.json"
    if tag == "meta":
        return RESULTS_DIR / f"meta_seed{seed}_{budget_tag}.json"
    return RESULTS_DIR / f"{tag}_seed{seed}_{budget_tag}.json"


def _run_variant(
    module_name: str, seed: int, budget_tag: str, extra: dict, output_path: Path
) -> dict:
    module = importlib.import_module(module_name)
    kwargs = {
        "env_id": "CartPole-v1",
        "seed": seed,
        "total_episodes": int(budget_tag.replace("ep", "")),
        "total_timesteps": 500_000,
        "cuda": False,
        "tensorboard": False,
        "log_interval": 1_000,
        **extra,
    }
    if extra.get("meta_gradient") is False:
        kwargs["returns_output"] = str(output_path)
    else:
        kwargs["output_json"] = str(output_path)
    return module.run_training(module.Args(**kwargs))


def _load_or_run(
    tag: str,
    module_name: str,
    seed: int,
    budget_tag: str,
    extra: dict,
    *,
    plot_only: bool,
) -> dict:
    path = _result_path(tag, seed, budget_tag)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if plot_only:
        raise FileNotFoundError(f"Missing result: {path}")
    print(f"Running {LABELS[tag]} seed={seed}...", flush=True)
    result = _run_variant(module_name, seed, budget_tag, extra, path)
    returns = result["episode_returns"]
    print(
        f"  finished: {len(returns)} episodes, "
        f"last50={np.mean(returns[-50:]):.1f}",
        flush=True,
    )
    return result


def plot_multi_seed(
    all_returns: dict[str, list[list[float]]],
    *,
    seeds: list[int],
    env_id: str,
    budget_tag: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    max_len = max(len(r) for runs in all_returns.values() for r in runs)
    episode_axis = np.arange(1, max_len + 1)

    for tag, seed_runs in all_returns.items():
        color = COLORS[tag]
        smoothed_runs: list[np.ndarray] = []
        for returns in seed_runs:
            values = np.asarray(returns, dtype=np.float64)
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
        # Only plot episodes where every seed has a smoothed value.
        valid = np.all(~np.isnan(stack), axis=0)
        if not np.any(valid):
            continue
        x_plot = episode_axis[valid]
        mean = np.mean(stack[:, valid], axis=0)
        std = np.std(stack[:, valid], axis=0)
        ax.fill_between(
            x_plot,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            x_plot,
            mean,
            color=color,
            lw=2.5,
            label=f"{LABELS[tag]} (n={len(seed_runs)} seeds)",
        )

    seed_str = ",".join(str(s) for s in seeds)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(
        f"{env_id}: baseline vs original meta vs opt2 "
        f"({budget_tag}, seeds={seed_str}, {smooth_window}-ep smooth ± std)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed comparison: baseline, original meta, opt2."
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_returns: dict[str, list[list[float]]] = {tag: [] for tag, _, _ in VARIANTS}
    for seed in seeds:
        for tag, module_name, extra in VARIANTS:
            blob = _load_or_run(
                tag,
                module_name,
                seed,
                budget_tag,
                extra,
                plot_only=cli_args.plot_only,
            )
            all_returns[tag].append(blob["episode_returns"])

    output_path = (
        FIG_DIR
        / f"reward_baseline_meta_opt2_seeds{'-'.join(str(s) for s in seeds)}_{budget_tag}.png"
    )
    plot_multi_seed(
        all_returns,
        seeds=seeds,
        env_id="CartPole-v1",
        budget_tag=budget_tag,
        smooth_window=cli_args.smooth_window,
        output_path=output_path,
    )
    print(f"Saved plot to {output_path}", flush=True)

    print("\nSummary (last 50 ep mean, per seed):", flush=True)
    for tag in all_returns:
        for seed, returns in zip(seeds, all_returns[tag]):
            print(
                f"  {tag:8s} seed={seed}: last50={np.mean(returns[-50:]):6.1f}  "
                f"overall={np.mean(returns):6.1f}",
                flush=True,
            )
        means = [np.mean(r[-50:]) for r in all_returns[tag]]
        print(
            f"  {tag:8s} aggregate last50: {np.mean(means):.1f} ± {np.std(means):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
