#!/usr/bin/env python
"""CartPole-v1: SEARL-DQN (B) and SEARL + ConfGate aux heads (A) vs CleanRL DQN and Rainbow DQN.

SEARL curves: greedy `test_return` of the best individual per epoch (searl_discrete convention), 10 seeds (1-10).
CleanRL / Rainbow: training-episode returns, 10-episode trailing mean, 10 seeds (0-9). x = environment steps.
Mean +/- 1 std over seeds on a common grid; All four run to 1M steps.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CR = os.path.join(ROOT, "code", "code_reproduction")
SEARL = os.path.join(ROOT, "results", "searl_confgate_cartpole_1M")
OUT = os.path.join(SEARL, "comparison_cartpole_1M.png")


def episodes(pattern, smooth=10):
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        yield d["global_step"].values.astype(float), d["episodic_return"].rolling(smooth, min_periods=1).mean().values


def searl(cfg):
    for s in range(1, 11):
        h = pd.read_csv(f"{SEARL}/{cfg}/seed{s}/history.csv")
        yield h["num_frames"].values.astype(float), h["test_return"].rolling(5, min_periods=1).mean().values


def band(curves, grid):
    ys = np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    with np.errstate(all="ignore"):
        return np.nanmean(ys, 0), np.nanstd(ys, 0)


grid = np.linspace(10_000, 1_000_000, 400)
series = [  # label, curves, color, linestyle
    ("SEARL DQN (B)", list(searl("B_full")), "#1f77b4", "-"),
    ("SEARL + ConfGate heads (A)", list(searl("A_full")), "#d95f02", "-"),
    ("Rainbow DQN (Kaixhin port)", list(episodes(f"{CR}/rainbow_dqn/results/CartPole-v1/seed*.csv")), "#222222", "--"),
    ("CleanRL DQN", list(episodes(f"{CR}/multi_benchmark/results_cartpole/cleanrl_dqn/seed*.csv")), "#7b3fa0", "--"),
]
fig, ax = plt.subplots(figsize=(9, 5.2))
for label, curves, color, ls in series:
    m, s = band(curves, grid)
    last = min(c[0].max() for c in curves)  # only plot where every seed is still present
    m, s = np.where(grid <= last, m, np.nan), np.where(grid <= last, s, np.nan)
    ax.plot(grid, m, color=color, ls=ls, lw=2.2, label=f"{label} (n={len(curves)})")
    ax.fill_between(grid, m - s, m + s, color=color, alpha=0.10, lw=0)
ax.set_xlabel("environment steps")
ax.set_ylabel("episode return (max 500)")
ax.set_title("CartPole-v1: SEARL variants vs CleanRL DQN and Rainbow DQN")
ax.set_ylim(0, 520)
ax.grid(alpha=0.25)
ax.legend(loc="lower right", frameon=False)
fig.text(0.01, 0.005, "SEARL: greedy test return of best individual, 5-epoch mean (1M budget, pop 4). Baselines: training-episode return, "
         "10-ep mean, 1M steps. Mean ± 1 std.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
