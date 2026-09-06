#!/usr/bin/env python3
"""Plot HOOF HalfCheetah-v2 results (median ± IQR).

Expects author layout under results/hoof/results_A2C and results_NPG
(or vendor/hoof/hoof/results_*), matching plots_a2c.py / plot_tnpg_and_hypers.py.

Paper targets (HalfCheetah):
  - Fig 1a: HOOF-A2C LR vs Baseline A2C vs meta-gradients
  - Table 1: KL ε=0.03 median return ≈ 1524 at 5e6 steps
  - Fig 2a: HOOF-TNPG vs TRPO
  - Fig 3: (δ, γ, λ) schedules
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "hoof"
DEFAULT_FIG = ROOT / "figures"

# Paper Table 1 HalfCheetah row (median returns for KL values)
TABLE1_HC = {
    0.01: 1203,
    0.02: 1451,
    0.03: 1524,
    0.04: 1325,
    0.05: 1388,
    0.06: 1301,
    0.07: 1504,
}


def _discover_run_dirs(root: Path) -> list[Path]:
    candidates = []
    for base in [
        root / "results_A2C",
        root / "results_NPG",
        ROOT / "vendor" / "hoof" / "hoof" / "results_A2C",
        ROOT / "vendor" / "hoof" / "hoof" / "results_NPG",
    ]:
        if base.is_dir():
            candidates.append(base)
    return candidates


def _load_monitor_returns(run_dir: Path) -> list[np.ndarray]:
    """Best-effort load of episode returns from common Baselines dump formats."""
    arrays: list[np.ndarray] = []
    for path in run_dir.rglob("*"):
        if path.suffix not in {".npz", ".csv", ".npz.npz"} and "monitor" not in path.name.lower():
            if path.suffix != ".npz":
                continue
        try:
            if path.suffix == ".npz" or path.name.endswith(".npz.npz"):
                data = np.load(path, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    for key in ("r", "reward", "rewards", "episode_rewards"):
                        if key in data:
                            arrays.append(np.asarray(data[key], dtype=float))
                            break
                elif isinstance(data, np.ndarray):
                    arrays.append(data.astype(float).ravel())
            elif path.suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(path)
                for col in ("r", "reward", "rewards", "episode_reward"):
                    if col in df.columns:
                        arrays.append(df[col].to_numpy(dtype=float))
                        break
        except Exception:
            continue
    return arrays


def plot_table1_reference(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(TABLE1_HC.keys())
    ys = [TABLE1_HC[k] for k in xs]
    ax.plot(xs, ys, "o-", label="Paper Table 1 HalfCheetah medians")
    ax.axvline(0.03, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("KL ε")
    ax.set_ylabel("Median return @ 5e6 steps")
    ax.set_title("HOOF A2C Table 1 reference (reproduce from logs)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_hoof] wrote {out_path}")


def plot_discovered(out_path: Path, result_roots: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    n = 0
    for root in result_roots:
        for seed_dir in sorted(root.rglob("rs_*"))[:30]:
            arrays = _load_monitor_returns(seed_dir)
            for arr in arrays[:1]:
                if arr.size < 2:
                    continue
                # crude x = episode index; author plots use env steps
                ax.plot(arr, alpha=0.4)
                n += 1
    if n == 0:
        ax.text(
            0.5,
            0.5,
            "No HOOF monitor dumps found yet.\n"
            "Run scripts/run_hoof_halfcheetah.sh after placing mjkey.txt.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        ax.set_xlabel("Episode index (approx)")
        ax.set_ylabel("Return")
        ax.set_title(f"HOOF discovered curves ({n})")
        ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_hoof] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot_table1_reference(args.fig_dir / "hoof_halfcheetah_table1_reference.jpg")
    plot_discovered(
        args.fig_dir / "hoof_halfcheetah_discovered.jpg",
        _discover_run_dirs(args.results),
    )


if __name__ == "__main__":
    main()
