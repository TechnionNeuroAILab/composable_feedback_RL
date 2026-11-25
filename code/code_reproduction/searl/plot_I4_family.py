#!/usr/bin/env python
"""4-head family on discrete HalfCheetah-v4, with and without SEARL.  usage: plot_I4_family.py RUN_ROOT [--frames N]
Same colour = same design; solid = SEARL (population 4), dashed = no SEARL (population 1, no selection/mutation,
fixed lr 3e-4; pruning, if any, random on a fixed schedule). Writes RUN_ROOT/figures/I4_family.png.
Curve = greedy test return; finished runs: history.csv smoothed over ~20k steps; unfinished: run.log prints."""
import argparse
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--frames", type=int, default=400_000)
a = p.parse_args()
R, FR = a.root, a.frames
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASE = f"{ROOT}/results/searl_confgate_halfcheetah_1M/B_full"
# (label, colour, linestyle, folder, tick)
ARMS = [("plain SEARL (1 network)", "#0072B2", "-", BASE, "SEARL\n1 net"),
        ("DQN, 1 network", "#0072B2", "--", f"{R}/DQN1", "DQN\n1 net"),
        ("SEARL, 4 heads, no pruning", "#555555", "-", f"{R}/I4_K4_ctrl", "SEARL\n4 heads"),
        ("DQN, 4 heads, no pruning", "#555555", "--", f"{R}/DQN4_ctrl", "DQN\n4 heads"),
        ("SEARL prunes h1-h3 (full head kept)", "#882255", "-", f"{R}/I4_K4_remove", "SEARL\nprune 3"),
        ("DQN, random prune of h1-h3", "#882255", "--", f"{R}/DQN4_prune3", "DQN\nprune 3"),
        ("SEARL prunes all 4 heads", "#009E73", "-", f"{R}/I4_K4_remove_all", "SEARL\nprune all"),
        ("DQN, random prune of all 4", "#009E73", "--", f"{R}/DQN4_pruneall", "DQN\nprune all")]


def curve(d, s):
    h = f"{d}/seed{s}/history.csv"
    if os.path.exists(h):
        h = pd.read_csv(h).dropna(subset=["test_return"])
        h = h[h.num_frames <= FR]
        w = max(1, int(round(20_000 / np.median(np.diff(h.num_frames.values)))))
        return h.num_frames.values, h.test_return.rolling(w, min_periods=1).mean().values, True
    if not os.path.exists(f"{d}/seed{s}/run.log"):
        return np.zeros(0), np.zeros(0), False
    rows = [(int(m[1]), float(m[2])) for m in
            re.finditer(r"frames=(\d+) .*?test=(-?[\d.]+)", open(f"{d}/seed{s}/run.log").read())]
    x = np.array([r for r in rows if r[0] <= FR]) if rows else np.zeros((0, 2))
    if len(x):   # 20k-step bins so SEARL (print every 20k) and DQN (print every 5k) are smoothed alike
        b = (x[:, 0] - 1) // 20_000
        xs = np.array([x[b == k, 0].max() for k in np.unique(b)])
        ys = np.array([x[b == k, 1].mean() for k in np.unique(b)])
        return xs, ys, False
    return np.zeros(0), np.zeros(0), False


data = []
for name, col, ls, d, tick in ARMS:
    cs = [curve(d, s) for s in range(5)]
    if all(len(x) for x, _, _ in cs):
        data.append((name, col, ls, tick, cs))
done = all(c[2] for *_, cs in data for c in cs)
reach = min(x.max() for *_, cs in data for x, _, _ in cs)
win = (0.9 * FR, FR) if done else (max(20_000, reach - 40_000), reach)

fig, ax = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.8, 1]})
for name, col, ls, tick, cs in data:
    for x, y, _ in cs:
        ax[0].plot(x, y, color=col, ls=ls, lw=0.7, alpha=0.25)
    lim = min(x.max() for x, _, _ in cs)
    g = np.arange(20_000, lim + 1, 20_000)
    ax[0].plot(g, np.mean([np.interp(g, x, y) for x, y, _ in cs], 0), color=col, ls=ls, lw=2.4,
               label=f"{name} (n={len(cs)})")
for i, (name, col, ls, tick, cs) in enumerate(data):
    v = np.array([y[(x > win[0]) & (x <= win[1])].mean() for x, y, _ in cs])
    ax[1].bar(i, v.mean(), color=col, alpha=0.35 if ls == "-" else 0.12, width=0.65, edgecolor=col,
              ls=ls, lw=1.2)
    ax[1].scatter(i + np.linspace(-0.15, 0.15, len(v)), v, color=col if ls == "-" else "white", edgecolor=col,
                  s=26, zorder=3, lw=1.2)
    ax[1].text(i, max(v.max(), v.mean()) + 60, f"{v.mean():.0f}\n± {v.std():.0f}", ha="center", va="bottom",
               fontsize=8, color=col)
ax[1].set_xticks(range(len(data)))
ax[1].set_xticklabels([t for *_, t, _ in data], fontsize=8.5)
ax[1].set_title(f"mean test return, {int(win[0]) // 1000}k-{int(win[1]) // 1000}k steps (dot = seed)", fontsize=11)
ax[0].set_xlabel("environment steps")
ax[0].set_ylabel("greedy test return")
ax[0].set_title(f"Discrete HalfCheetah-v4, 4-head ensembles with and without SEARL, seeds 0-4"
                + ("" if done else " (runs in progress)"), fontsize=11)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
for x in ax:
    x.axhline(0, color="#888", lw=0.6)
    x.grid(alpha=0.2)
    x.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Solid = SEARL (population 4, test return of the best individual). Dashed = no SEARL (population 1, no "
         "selection or mutation, lr 3e-4; pruning random, one head loses 25% of its inputs every 24k steps). "
         "Thick = mean over seeds up to the shortest seed.", fontsize=7.8, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
o = f"{R}/figures/I4_family.png"
fig.savefig(o, dpi=135)
print("wrote", o, "| arms drawn:", len(data), "| window", win)
