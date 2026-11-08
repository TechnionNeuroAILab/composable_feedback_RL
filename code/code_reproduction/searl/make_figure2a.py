#!/usr/bin/env python
"""Reproduce Figure 2 panel (a) (HalfCheetah) of the SEARL paper: reward vs.
total environment steps (log-x), SEARL vs. modified PBT vs. random search,
mean +/- std over seeds.

For each method the plotted curve is the best fitness found so far (the
best individual in the population for SEARL/modified-PBT; the single best
sampled configuration for random search -- exactly what the paper's own
Random Search baseline reports, Section 4.1).

Random search's x-axis is rescaled per the paper's "fair evaluation
protocol" (Section 4.3): since finding the best of N sampled configurations
required running all N of them, the reported curve's env-step counter is
multiplied by N (`num_configs_tried.txt`, written by run_reproduction.py).
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABEL = {"searl": "SEARL (ours)", "modified_pbt": "modified PBT", "random_search": "Random Search"}
METHOD_COLOR = {"searl": "#2a78d6", "modified_pbt": "#1baf7a", "random_search": "#e34948"}
METHOD_ORDER = ["searl", "modified_pbt", "random_search"]


def load_curves(folder, method):
    dirs = sorted(glob.glob(os.path.join(folder, f"HalfCheetah-v4_{method}_seed*")))
    curves = []
    for d in dirs:
        csv_path = os.path.join(d, "history.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        x = df["num_frames"].values.astype(float)
        y = df["best_fitness"].values if "best_fitness" in df.columns else df["reward"].values

        if method == "random_search":
            n_tried_path = os.path.join(d, "num_configs_tried.txt")
            if os.path.exists(n_tried_path):
                n_tried = int(open(n_tried_path).read().strip())
                x = x * n_tried  # fair-evaluation-protocol rescaling, see module docstring

        curves.append((x, y))
    return curves


def aggregate_on_log_grid(curves, n_points=100):
    x_min = min(c[0][c[0] > 0].min() for c in curves if (c[0] > 0).any())
    x_max = max(c[0].max() for c in curves)
    grid = np.logspace(np.log10(x_min), np.log10(x_max), n_points)

    interped = []
    for x, y in curves:
        mask = x > 0
        x, y = x[mask], y[mask]
        yi = np.interp(grid, x, y, left=y[0], right=y[-1])
        interped.append(yi)
    stacked = np.stack(interped)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot(folder, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for method in METHOD_ORDER:
        curves = load_curves(folder, method)
        if not curves:
            print(f"warning: no curves found for method={method} in {folder}")
            continue

        for x, y in curves:
            ax.plot(x, y, color=METHOD_COLOR[method], alpha=0.25, linewidth=1, linestyle="--")

        grid, mean, std = aggregate_on_log_grid(curves)
        ax.plot(grid, mean, color=METHOD_COLOR[method], linewidth=2.5, label=METHOD_LABEL[method])
        ax.fill_between(grid, mean - std, mean + std, color=METHOD_COLOR[method], alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel("Average Return")
    ax.set_title("HalfCheetah")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot(args.folder, os.path.join(args.out_dir, "figure2a_halfcheetah.png"))
