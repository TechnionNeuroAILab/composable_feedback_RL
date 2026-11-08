#!/usr/bin/env python
"""Learning rate over training for the four algorithms, 1M-step runs.

  python plot_confgate_lr.py cartpole | halfcheetah | halfcheetah500k

SEARL DQN (B) / SEARL + ConfGate heads (A): population-mean learning rate used in each training phase, from
confgate_log.csv (4 rows per generation, one per individual; row r belongs to the generation after epoch r // 4).
  B: lr_base (= lr_eff, controller off).   A: lr_base (SEARL's mutated lr) and lr_eff = lr_base * (floor + slope * m).
Rainbow (Kaixhin port) and CleanRL DQN: fixed 2.5e-4 (Adam, no schedule), drawn as horizontal lines.
Mean over seeds taken in log10 space, band = +/- 1 std of log10 lr. y axis is logarithmic.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
POP = 4
CFG = {"cartpole": ("searl_confgate_cartpole_1M", range(1, 11), "CartPole-v1"),
       "halfcheetah": ("searl_confgate_halfcheetah_1M", range(10), "HalfCheetah-v4 (discrete, 18 outputs)"),
       "halfcheetah500k": ("searl_confgate_halfcheetah", range(3), "HalfCheetah-v4 (discrete, 18 outputs), 500k")}
env = sys.argv[1]
folder, seeds, title = CFG[env]
base = os.path.join(ROOT, "results", folder)
FIXED_LR = 2.5e-4


def per_epoch(cfg, col):
    out = []
    for s in seeds:
        h = pd.read_csv(f"{base}/{cfg}/seed{s}/history.csv")
        g = pd.read_csv(f"{base}/{cfg}/seed{s}/confgate_log.csv")
        epoch = np.minimum(np.arange(len(g)) // POP, len(h) - 1)
        lr = pd.Series(np.log10(g[col].values)).groupby(epoch).mean()
        out.append((h["num_frames"].values[lr.index.values].astype(float), lr.values))
    return out


def band(curves, grid):
    ys = np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    with np.errstate(all="ignore"):
        return np.nanmean(ys, 0), np.nanstd(ys, 0)


frames = max(c[0].max() for c in per_epoch("B_full", "lr_base"))
grid = np.linspace(10_000, frames, 300)
fig, ax = plt.subplots(figsize=(9, 5.2))
series = [("SEARL DQN (B): lr", per_epoch("B_full", "lr_base"), "#1f77b4", "-"),
          ("SEARL + ConfGate heads (A): base lr (SEARL-mutated)", per_epoch("A_full", "lr_base"), "#d95f02", ":"),
          ("SEARL + ConfGate heads (A): effective lr = base × (floor + slope·m)", per_epoch("A_full", "lr_eff"), "#d95f02", "-")]
for label, curves, color, ls in series:
    m, s = band(curves, grid)
    last = min(c[0].max() for c in curves)
    m, s = np.where(grid <= last, m, np.nan), np.where(grid <= last, s, np.nan)
    ax.plot(grid, 10 ** m, color=color, ls=ls, lw=2.2, label=f"{label} (n={len(curves)})")
    if ls == "-":
        ax.fill_between(grid, 10 ** (m - s), 10 ** (m + s), color=color, alpha=0.10, lw=0)
ax.axhline(FIXED_LR, color="#222222", ls="--", lw=2.2, label="Rainbow DQN (fixed 2.5e-4)")
ax.axhline(FIXED_LR, color="#7b3fa0", ls=(0, (2, 6)), lw=2.2, label="CleanRL DQN (fixed 2.5e-4)")
ax.set_yscale("log")
ax.set_xlabel("environment steps")
ax.set_ylabel("learning rate")
ax.set_title(f"{title}: learning rate over training")
ax.grid(alpha=0.25, which="both")
ax.legend(loc="lower left", frameon=False, fontsize=8)
fig.text(0.01, 0.005, "SEARL: population mean per generation, geometric mean over seeds ± 1 std (log space). "
         "Rainbow and CleanRL DQN use a constant lr.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = os.path.join(base, f"lr_{env}.png" if env.endswith("500k") else f"lr_{env}_1M.png")
fig.savefig(out, dpi=160)
print("wrote", out)
