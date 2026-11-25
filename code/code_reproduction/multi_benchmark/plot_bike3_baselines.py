#!/usr/bin/env python
"""bike v3.0 learnability ladder: SEARL DQN vs CleanRL DQN (vanilla) vs Rainbow DQN on L1 (route), L2 (+ speed with
speed limits) and L4.1 (route + balance, free speed, no speed limits, turn kick 2 at speed 5).

Greedy return on 10 fixed boards (seeds 5000-5009), every 10k steps, 3 seeds, 480k steps.
Reads ../../../results/bike3/L<l>/<method>/seed<s>/eval.csv; writes figures/bike3_baselines.png.
All three methods use ONE Q-head over the 9 joint (steer, throttle) actions (results/.../<searl|dqn|rainbow>_vanilla), with
the same code path and settings as on bike v2.1 (run_bike_grid.sh).
  --vanilla: vanilla DQN with the standard 13-number observation vs with the 4-row hole lookahead; writes
  figures/bike3_vanilla_dqn.png.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../../results/bike3")
VANILLA = "--vanilla" in sys.argv
OUT = os.path.join(HERE, "figures/bike3_vanilla_dqn.png" if VANILLA else "figures/bike3_baselines.png")
METHODS = [("SEARL DQN", "searl_vanilla", "#2a78d6", "-"),          # all three: ONE head over the 9 joint actions,
           ("CleanRL DQN", "dqn_vanilla", "#52514e", (0, (4, 2))),     # otherwise the v2.1 bike-grid code and settings
           ("Rainbow DQN", "rainbow_vanilla", "#e34948", (0, (6, 1.5, 1, 1.5)))]
if VANILLA:
    METHODS = [("vanilla DQN (13-number obs)", "dqn_vanilla", "#52514e", (0, (4, 2))),
               ("vanilla DQN + 4-row lookahead", "dqn_vanilla_look4", "#2a78d6", "-")]
LEVELS = [(1, "L1: route"), (2, "L2: route + speed"), (4.1, "L4.1: route + balance (free speed)")]
NL = len(LEVELS)
POS = {l: i + 1 for i, (l, _) in enumerate(LEVELS)}   # x position in the final panel
GRID = np.arange(10_000, 480_001, 10_000)
INK, MUTED = "#52514e", "#898781"
OPT = pd.read_csv(os.path.join(ROOT, "optimum.csv")).groupby("level").optimum.agg(["mean", "min", "max"])   # optimum_bike3.py
UPPER = 4900          # every row at full progress, no costs (= the L1 maximum)


def curves(level, method):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/L{level}/{method}/seed*/eval.csv")):
        d = pd.read_csv(f)
        out.append(np.interp(GRID, d.global_step, d.greedy_return, right=np.nan))
    return np.array(out)


def final(c):
    return np.nanmean(c[:, -3:], 1)          # mean of the last 3 evaluations per seed


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(1, NL + 1, figsize=(5 * NL, 3.9), gridspec_kw={"width_ratios": [1] * NL + [0.9 + 0.2 * (NL - 3)]})
finals = {}
for ax, (lvl, title) in zip(axes[:NL], LEVELS):
    for name, m, col, ls in METHODS:
        c = curves(lvl, m)
        if not len(c):
            continue
        finals[(lvl, name)] = final(c)
        for row in c:
            ax.plot(GRID / 1e3, row, color=col, lw=0.6, alpha=0.25)
        ax.plot(GRID / 1e3, np.nanmean(c, 0), color=col, lw=2, ls=ls, label=name)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("environment steps (k)")
    ax.set_ylim(-2000, 5300)
    o = OPT.loc[lvl]
    ax.axhline(UPPER, color=INK, lw=1.0, ls=":")
    ax.axhspan(o["min"], o["max"], color="#eda100", alpha=0.10, lw=0)
    ax.axhline(o["mean"], color="#b07800", lw=1.3, ls=(0, (5, 2)))
    ax.text(475, UPPER + 60, f"max possible {UPPER}", ha="right", va="bottom", fontsize=7.5, color=INK)
    key = (f"dashed: best achievable {o['mean']:.0f}\n(range over boards {o['min']:.0f}-{o['max']:.0f})" if lvl > 1
           else "dashed: best achievable = max possible")
    ax.text(475 if lvl > 1 else 20, -1850 if lvl > 1 else 300, key, ha="right" if lvl > 1 else "left", va="bottom",
            fontsize=7.5, color="#b07800")
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
axes[0].set_ylabel("greedy return (10 fixed boards)")
axes[0].legend(loc="lower right", frameon=False, fontsize=8)

ax = axes[NL]
off = ({METHODS[0][0]: -0.15, METHODS[1][0]: 0.15} if VANILLA else dict(zip([m[0] for m in METHODS], (-0.22, 0.0, 0.22))))
for name, m, col, _ in METHODS:
    for lvl, _t in LEVELS:
        v = finals.get((lvl, name))
        if v is None:
            continue
        x = POS[lvl] + off[name]
        ax.scatter(np.full(len(v), x), v, s=26, color=col, edgecolor="white", linewidth=1, zorder=3)
        ax.plot([x - 0.08, x + 0.08], [v.mean()] * 2, color=col, lw=2.2, zorder=4)
        ax.text(x, -1850, f"{v.mean():.0f}", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
ax.text(0.6, -1150, "means:", fontsize=7, color=MUTED, va="bottom")
for lvl, _t in LEVELS:
    ax.plot([POS[lvl] - 0.38, POS[lvl] + 0.38], [OPT.loc[lvl, "mean"]] * 2, color="#b07800", lw=1.3, ls=(0, (5, 2)), zorder=2)
ax.axhline(UPPER, color=INK, lw=1.0, ls=":")
ax.set_xticks([POS[l] for l, _ in LEVELS], [f"L{l}" for l, _ in LEVELS])
ax.set_ylim(-2000, 5300)
ax.set_title("final (last 3 evals), dots = seeds", fontsize=10, color=INK)
ax.grid(axis="y", color="#e6e5e1", lw=0.6)
fig.suptitle("bike v3.0, vanilla DQN (one Q-head over the 9 joint actions)  (5 lanes, 8 solutions, 3 seeds, 480k steps; thin lines = seeds, thick = mean)"
             if VANILLA else
             "bike v3.0: SEARL vs CleanRL DQN vs Rainbow, one Q-head over the 9 joint actions  (5 lanes, 8 solutions, 3 seeds, 480k steps; thin lines = seeds, thick = mean)",
             fontsize=10.5, color=INK)
fig.text(0.5, -0.05, ("Lookahead = the hole grid of the next 4 rows around the ego lane added to the observation (13 -> 41 numbers). "
          if VANILLA else "") +
         "Curves below -2000 are clipped.\n"
         "Dotted: max possible (every row at full progress). Dashed: best achievable = exact fall-free optimum on the same 10 boards (dynamic programming), band = range over boards.", ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("wrote", OUT)
for k, v in sorted(finals.items()):
    print(k, np.round(v), round(v.mean()))
