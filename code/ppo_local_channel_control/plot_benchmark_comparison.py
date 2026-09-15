"""Overlay the composable-PPO learning curve against its benchmarks.

Benchmarks:
- ``ablation_k1_backprop``: fair v4 ablation — same codebase and routing mode
  (backprop, no B) with ``num_channels=1``.
- ``ablation_k1_sparse_b`` (legacy): older K=1 run with default sparse-B routing;
  kept for reference but not a fair routing-mode match for v4.
- ``cleanrl``: independent standard single-critic PPO (CleanRL baseline).

    python code/ppo_local_channel_control/plot_benchmark_comparison.py \
        --composable results/ppo_local_channel_control/HalfCheetah-v4__local_control_v4__seed1_5000ep \
        --ablation results/ppo_local_channel_control/HalfCheetah-v4__ablation_k1_backprop__seed1_5000ep \
        --ablation-legacy results/ppo_local_channel_control/HalfCheetah-v4__ablation_k1__seed1_5000ep \
        --cleanrl results/cleanrl_ppo_baseline/HalfCheetah-v4__cleanrl__seed1_5000ep \
        --composable-label "Composable PPO v4 (K=4, backprop, no B)" \
        --title "HalfCheetah-v4: v4 vs. fair K=1 ablation and benchmarks" \
        --output paper/figures/ppo_local_channel_control_halfcheetah_v4/06_benchmark_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_returns(results_dir: Path) -> np.ndarray:
    return np.asarray(json.loads((results_dir / "episode_returns.json").read_text(encoding="utf-8")), dtype=float)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size < width:
        return values
    return np.convolve(values, np.ones(width) / width, mode="valid")


def plot_comparison(
    runs: list[tuple[str, Path, str]], output_path: Path, smooth_width: int = 50, title: str = ""
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for label, path, color in runs:
        if not path.exists():
            print(f"Skipping missing run: {path}")
            continue
        returns = load_returns(path)
        if returns.size == 0:
            continue
        episodes = np.arange(1, returns.size + 1)
        axis.plot(episodes, returns, alpha=0.12, color=color)
        width = min(smooth_width, returns.size)
        smoothed = smooth(returns, width)
        axis.plot(
            episodes[width - 1 :],
            smoothed,
            color=color,
            linewidth=2.2,
            label=f"{label} ({returns[-100:].mean():.0f} last-100-ep mean)",
        )
    axis.set(xlabel="Episode", ylabel="Episode return", title=title or "HalfCheetah-v4: learning curve vs. benchmarks")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composable", type=Path, required=True)
    parser.add_argument(
        "--ablation",
        type=Path,
        required=True,
        help="Fair ablation (K=1, same routing mode as composable run)",
    )
    parser.add_argument("--cleanrl", type=Path, required=True)
    parser.add_argument(
        "--ablation-legacy",
        type=Path,
        default=None,
        help="Optional legacy K=1 run with sparse-B routing (v3 default)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--composable-label",
        default="Composable PPO (K=4, local control)",
        help="Legend label for the composable run",
    )
    parser.add_argument(
        "--ablation-label",
        default="Ablation: K=1, backprop (no B)",
        help="Legend label for the fair ablation run",
    )
    parser.add_argument(
        "--ablation-legacy-label",
        default="Legacy ablation: K=1, sparse B routing",
        help="Legend label for the optional legacy sparse-B ablation",
    )
    parser.add_argument("--title", default="", help="Plot title override")
    args = parser.parse_args(argv)

    runs: list[tuple[str, Path, str]] = [
        (args.composable_label, args.composable, "tab:blue"),
        (args.ablation_label, args.ablation, "tab:orange"),
    ]
    if args.ablation_legacy is not None:
        runs.append((args.ablation_legacy_label, args.ablation_legacy, "tab:olive"))
    runs.append(("CleanRL PPO baseline (K=1, independent impl.)", args.cleanrl, "tab:green"))
    plot_comparison(runs, args.output, title=args.title)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
