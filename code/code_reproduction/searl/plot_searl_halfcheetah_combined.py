#!/usr/bin/env python
"""Discrete HalfCheetah-v4, 1M steps: one figure combining comparison_halfcheetah_1M.png and the SEARL panel of
all_variants_halfcheetah_1M.png, with the 1M baselines.

  SEARL DQN                    results/searl_confgate_halfcheetah_1M/B_full    seeds 0-9
  SEARL + 2 heads (lr gate)    .../A_full     (pose / velocity heads)          seeds 0-9
  SEARL + 17 per-obs heads     .../A_perobs   (one head per observation, lr gate) seeds 0-2
  Rainbow DQN                  rainbow_dqn/results_1M/HalfCheetah-v4           seeds 0-2
  CleanRL DQN                  multi_benchmark/results_halfcheetah_1M/cleanrl_dqn  seeds 0-2

Left: learning curves, mean +/- 1 std over seeds (SEARL: greedy test return of the best individual, 5-epoch mean;
DQN / Rainbow: training-episode return, 10-episode mean). Right: final level = mean over the last 10% of steps
(900k-1M), dot = seed; open dots mark seeds 0-2 of the 10-seed arms (the seeds the 3-seed arms share).
Writes results/searl_confgate_summary/searl_halfcheetah_combined_1M.png
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CR = f"{ROOT}/code/code_reproduction"
S = f"{ROOT}/results/searl_confgate_halfcheetah_1M"
OUT = f"{ROOT}/results/searl_confgate_summary/searl_halfcheetah_combined_1M.png"


def searl(cfg, seeds):
    out = []
    for s in seeds:
        h = pd.read_csv(f"{S}/{cfg}/seed{s}/history.csv").dropna(subset=["test_return"])
        out.append((h.num_frames.values.astype(float), h.test_return.rolling(5, min_periods=1).mean().values,
                    h[h.num_frames >= 900_000].test_return.mean()))
    return out


def episodes(pattern, seeds):
    out = []
    for s in seeds:
        d = pd.read_csv(pattern.format(s=s))
        out.append((d.global_step.values.astype(float), d.episodic_return.rolling(10, min_periods=1).mean().values,
                    d[d.global_step >= 900_000].episodic_return.mean()))
    return out


ARMS = [
    ("SEARL DQN", "#0072B2", "-", searl("B_full", range(10))),
    ("SEARL + 2 heads (pose / velocity), lr gate", "#D55E00", "-", searl("A_full", range(10))),
    ("SEARL + 17 per-obs heads, lr gate", "#009E73", "-", searl("A_perobs", range(3))),
    ("Rainbow DQN", "#222222", ":", episodes(f"{CR}/rainbow_dqn/results_1M/HalfCheetah-v4/seed{{s}}.csv", range(3))),
    ("CleanRL DQN", "#8E44AD", "--", episodes(f"{CR}/multi_benchmark/results_halfcheetah_1M/cleanrl_dqn/seed{{s}}.csv", range(3))),
]

grid = np.linspace(10_000, 1_000_000, 600)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
for name, col, ls, cs in ARMS:
    ys = np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y, _ in cs])
    m, sd = np.nanmean(ys, 0), np.nanstd(ys, 0)
    ax.plot(grid, m, color=col, ls=ls, lw=2.2, label=f"{name} (n={len(cs)})")
    ax.fill_between(grid, m - sd, m + sd, color=col, alpha=0.08, lw=0)
ax.set_xlabel("environment steps")
ax.set_ylabel("episode return")
ax.set_title("Discrete HalfCheetah-v4 (6 joints x 3 torques = 18 outputs), 1M steps", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[1]
for i, (name, col, ls, cs) in enumerate(ARMS):
    v = np.array([f for _, _, f in cs])
    ax.bar(i, v.mean(), color=col, alpha=0.25, width=0.65)
    xs = i + np.linspace(-0.2, 0.2, len(v))
    for j, (xx, vv) in enumerate(zip(xs, v)):
        shared = j < 3 and len(v) > 3
        ax.scatter(xx, vv, s=26, zorder=3, color="white" if shared else col, edgecolor=col, lw=1.3)
    ax.text(i, 3550, f"{v.mean():.0f}\n± {v.std():.0f}", ha="center", va="bottom", fontsize=8.5, color=col)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels(["SEARL", "SEARL\n+2 heads", "SEARL\n+17 heads", "Rainbow", "CleanRL"], fontsize=9)
ax.set_title("final level: mean return, 900k-1M steps", fontsize=11, pad=34)
ax.set_ylim(-650, 3500)
ax.set_clip_on(False)
for a in axes:
    a.axhline(0, color="#888", lw=0.6)
    a.grid(alpha=0.2)
    a.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "SEARL: greedy test return of the best individual (5-epoch mean). Rainbow / CleanRL: training-episode "
         "return (10-episode mean), so not strictly comparable. Mean ± 1 std over seeds.\nRight: dot = seed; open dots: seeds 0-2 of the "
         "10-seed arms (shared with the 3-seed arms). Heads never act: they only scale the learning rate.",
         fontsize=7.8, color="#555")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
for name, _, _, cs in ARMS:
    v = np.array([f for _, _, f in cs])
    print(f"{name:45s} final {v.mean():6.0f} ± {v.std():4.0f}   seeds 0-2: {v[:3].mean():6.0f}   {np.round(v)}")
