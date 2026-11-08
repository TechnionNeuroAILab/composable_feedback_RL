#!/usr/bin/env python
"""Bike grid: lanes {2,3,5,10} x solutions {1,8,100}, 7 variants, one shared greedy-evaluation curve per run
(results/bike_grid/L<l>_S<s>/<variant>/seed<k>/eval.csv; every method scored by common_bike.greedy_return on the same
10 fresh boards, no exploration). Outputs in results/bike_grid_summary/:
  grid_curves.png          12 panels (rows = lanes, columns = solutions), mean +/- std over seeds
  summary_final.csv/.txt   final level   = mean of the last 5 evaluations (last ~50k steps), per seed, then mean +/- sd
  summary_auc.csv          sample efficiency = mean greedy return over all evaluations / 4900 (area under the curve)
  summary_steps95.csv      first evaluation step with greedy return >= 95% of 4900 (median over seeds; 'not reached' if < half the seeds)
Missing runs are skipped, so this can be run while the grid is still going.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GRID = os.path.join(ROOT, "results", "bike_grid")
OUT = os.path.join(ROOT, "results", "bike_grid_summary")
os.makedirs(OUT, exist_ok=True)
LANES, SOLS, MAXR = (2, 3, 5, 10), (1, 8, 100), 4900.0
VARIANTS = [  # key, label, color, linestyle
    ("searl_B", "SEARL DQN", "#1f77b4", "-"), ("searl_A2", "SEARL + 2 heads", "#d95f02", "-"),
    ("searl_Aobs", "SEARL + per-obs heads", "#2ca02c", "-"), ("dqn", "CleanRL DQN", "#7b3fa0", "--"),
    ("dqn_A2", "DQN + 2 heads", "#e6ab02", "--"), ("dqn_Aobs", "DQN + per-obs heads", "#17becf", "--"),
    ("rainbow", "Rainbow", "#222222", ":")]


def load(L, S, v):
    out = []
    for f in sorted(glob.glob(f"{GRID}/L{L}_S{S}/{v}/seed*/eval.csv")):
        d = pd.read_csv(f)
        if len(d):
            out.append(d)
    return out


def common_grid(runs, n=48, hi=480_000):
    g = np.linspace(10_000, hi, n)
    ys = np.stack([np.interp(g, d.global_step.values, d.greedy_return.values, left=np.nan, right=np.nan) for d in runs])
    return g, ys


fig, axes = plt.subplots(len(LANES), len(SOLS), figsize=(16, 12), sharex=True, sharey=True)
rows_final, rows_auc, rows_s95 = [], [], []
for i, L in enumerate(LANES):
    for j, S in enumerate(SOLS):
        ax = axes[i, j]
        for v, label, color, ls in VARIANTS:
            runs = load(L, S, v)
            if not runs:
                continue
            g, ys = common_grid(runs)
            with np.errstate(all="ignore"):
                m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(g, m, color=color, ls=ls, lw=1.8, label=f"{label} (n={len(runs)})" if (i, j) == (0, 0) else None)
            ax.fill_between(g, m - s, m + s, color=color, alpha=0.08, lw=0)
            fin = [float(d.greedy_return.tail(5).mean()) for d in runs]
            auc = [float(np.nanmean(np.interp(g, d.global_step.values, d.greedy_return.values)) / MAXR) for d in runs]
            hit = [d.global_step[d.greedy_return >= 0.95 * MAXR].min() for d in runs]
            reached = [h for h in hit if not np.isnan(h)]
            rows_final.append((L, S, label, np.mean(fin), np.std(fin), len(runs)))
            rows_auc.append((L, S, label, np.mean(auc), np.std(auc), len(runs)))
            rows_s95.append((L, S, label, float(np.median(reached)) if len(reached) * 2 > len(runs) else np.nan, len(reached), len(runs)))
        ax.set_title(f"{L} lanes, {S} solution{'s' if S > 1 else ''}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.axhline(MAXR, color="grey", lw=0.6, ls=":")
        if j == 0:
            ax.set_ylabel("greedy return")
        if i == len(LANES) - 1:
            ax.set_xlabel("environment steps")
axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
fig.suptitle("bike_env v2.1: greedy return on 10 fixed fresh boards (same protocol for every method), mean ± std over seeds", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{OUT}/grid_curves.png", dpi=140)

cols = ["lanes", "solutions", "variant", "mean", "sd", "n"]
fin = pd.DataFrame(rows_final, columns=cols); auc = pd.DataFrame(rows_auc, columns=cols)
s95 = pd.DataFrame(rows_s95, columns=["lanes", "solutions", "variant", "median_step", "n_reached", "n"])
fin.to_csv(f"{OUT}/summary_final.csv", index=False); auc.to_csv(f"{OUT}/summary_auc.csv", index=False); s95.to_csv(f"{OUT}/summary_steps95.csv", index=False)
order = [lab for _, lab, _, _ in VARIANTS]
for name, df, fmt in [("Final greedy return (last 5 evals, mean over seeds)", fin, "{:.0f}"), ("Area under curve (mean greedy return / 4900)", auc, "{:.2f}"),
                      ("Steps to reach 95% of max (median over seeds)", s95.rename(columns={"median_step": "mean"}), "{:.0f}")]:
    print("\n==", name)
    p = df.pivot_table(index=["lanes", "solutions"], columns="variant", values="mean")[[o for o in order if o in df.variant.values]]
    print(p.to_string(float_format=lambda x: fmt.format(x)))
print("\nwrote", OUT)
