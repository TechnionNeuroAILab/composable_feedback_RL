#!/usr/bin/env python
"""Discrete HalfCheetah-v4 (6 joints x {-1,0,+1} = 18 outputs): SEARL DQN (B) and SEARL + ConfGate heads (A) vs
CleanRL DQN and Rainbow DQN. 500k steps, seeds 0-2.

SEARL curves: greedy `test_return` of the best individual per epoch. CleanRL / Rainbow: training-episode returns,
10-episode trailing mean. Mean +/- 1 std over seeds on a common grid, cut where any seed has ended.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CR = os.path.join(ROOT, "code", "code_reproduction")
import sys
LONG = len(sys.argv) > 1 and sys.argv[1] == "1M"  # 1M: SEARL seeds 0-9, baselines (3 seeds) rerun at 1M
SEARL = os.path.join(ROOT, "results", "searl_confgate_halfcheetah_1M" if LONG else "searl_confgate_halfcheetah")
OUT = os.path.join(SEARL, "comparison_halfcheetah_1M.png" if LONG else "comparison_halfcheetah.png")
SEEDS = tuple(range(10)) if LONG else (0, 1, 2)


def episodes(pattern, smooth=10):
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        yield d["global_step"].values.astype(float), d["episodic_return"].rolling(smooth, min_periods=1).mean().values


def searl(cfg):
    for s in SEEDS:
        h = pd.read_csv(f"{SEARL}/{cfg}/seed{s}/history.csv")
        yield h["num_frames"].values.astype(float), h["test_return"].rolling(5, min_periods=1).mean().values


def band(curves, grid):
    ys = np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    with np.errstate(all="ignore"):
        return np.nanmean(ys, 0), np.nanstd(ys, 0)


grid = np.linspace(10_000, 1_000_000 if LONG else 500_000, 600 if LONG else 300)
series = [
    ("SEARL DQN (B)", list(searl("B_full")), "#1f77b4", "-"),
    ("SEARL + ConfGate heads (A)", list(searl("A_full")), "#d95f02", "-"),
    ("Rainbow DQN (Kaixhin port)", list(episodes(f"{CR}/rainbow_dqn/{'results_1M' if LONG else 'results'}/HalfCheetah-v4/seed*.csv")), "#222222", "--"),
    ("CleanRL DQN", list(episodes(f"{CR}/multi_benchmark/{'results_halfcheetah_1M' if LONG else 'results_halfcheetah'}/cleanrl_dqn/seed*.csv")), "#7b3fa0", "--"),
]
fig, ax = plt.subplots(figsize=(9, 5.2))
for label, curves, color, ls in series:
    m, s = band(curves, grid)
    last = min(c[0].max() for c in curves)
    m, s = np.where(grid <= last, m, np.nan), np.where(grid <= last, s, np.nan)
    ax.plot(grid, m, color=color, ls=ls, lw=2.2, label=f"{label} (n={len(curves)})")
    ax.fill_between(grid, m - s, m + s, color=color, alpha=0.10, lw=0)
ax.set_xlabel("environment steps")
ax.set_ylabel("episode return")
ax.set_title("HalfCheetah-v4 (discrete, 18 outputs): SEARL variants vs CleanRL DQN and Rainbow DQN")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", frameon=False)
note = ("SEARL: greedy test return of best individual, 5-epoch mean (1M budget, pop 4, 10 seeds). Baselines: training-episode "
        "return, 10-ep mean (3 seeds). Mean ± 1 std." if LONG else
        "SEARL: greedy test return of best individual, 5-epoch mean (500k budget, pop 4). Baselines: training-episode "
        "return, 10-ep mean. Mean ± 1 std, 3 seeds.")
fig.text(0.01, 0.005, note, fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
