#!/usr/bin/env python
"""Model A (18 heads, fixed 1/18), model B (heads toggled by SEARL) and model C (head strengths in [0, 1] changed by
SEARL) vs plain SEARL on discrete HalfCheetah-v4, seeds 0-4.

Writes results/searl_obs_heads_halfcheetah_1M/figures/:
  progress_A_B_vs_searl.png  left: greedy test return over steps (thin = seed, thick = mean up to the shortest seed);
                             right: per-seed final level (mean over the last 10% of steps; while some run is
                             unfinished, the last 40k steps that EVERY run has reached instead, marked in the title).
  (model B's head flags: plot_obs_heads_q_and_survival.py -> B_head_survival.png;
   model C's head strengths: plot_obs_heads_weights.py -> C_weight_trends.png, C_weight_gates_heatmap.png)
The I2 / I4 arms (searl_obs_heads_evolve.py) are not drawn here; see plot_I4_family.py and plot_pruning_compare.py.
Finished runs are read from history.csv (test return every epoch, 5-epoch mean); unfinished runs from run.log (every 20k).
"""
import argparse
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_p = argparse.ArgumentParser()
_p.add_argument("--res", default="searl_obs_heads_halfcheetah_1M", help="results subfolder with A_fixed / B_toggle")
_p.add_argument("--frames", type=int, default=1_000_000, help="run length; curves (also plain SEARL) are cut here")
_a = _p.parse_args()
FRAMES = _a.frames
R = f"{ROOT}/results/{_a.res}"
BASE = f"{ROOT}/results/searl_confgate_halfcheetah_1M/B_full"
OUT = f"{R}/figures"
os.makedirs(OUT, exist_ok=True)
SEEDS = range(5)
ARMS = [("plain SEARL", "#0072B2", BASE), ("A: 18 heads, fixed 1/18", "#E69F00", f"{R}/A_fixed"),
        ("B: heads toggled by SEARL", "#009E73", f"{R}/B_toggle")]
# later arms, drawn when their folder exists: (label, colour, folder, short tick label)
EXTRA = [("C: head strengths in [0, 1] changed by SEARL", "#D55E00", f"{R}/C_weight", "C\nweight"),
         ("C from full head: others start at 0, +0.02 per head mutation", "#222222", f"{R}/C_add_from_full", "C add\nfrom full"),
         ("A + TD target from full head", "#CC79A7", f"{R}/A_target_full", "A\ntarget full")]
TICKS = ["plain\nSEARL", "A\nfixed", "B\ntoggle"]
for name, col, d, tick in EXTRA:
    if all(os.path.exists(f"{d}/seed{s}/run.log") for s in SEEDS):
        ARMS.append((name, col, d))
        TICKS.append(tick)
HEADS = ["full"] + [f"obs{i}" for i in range(17)]


def curve(d, s):
    """(steps, return, finished)."""
    h = f"{d}/seed{s}/history.csv"
    if os.path.exists(h):
        h = pd.read_csv(h).dropna(subset=["test_return"])
        h = h[h.num_frames <= FRAMES]
        return h.num_frames.values, h.test_return.rolling(5, min_periods=1).mean().values, True
    rows = [(int(m[1]), float(m[2])) for m in re.finditer(r"frames=(\d+) .*?test=(-?[\d.]+)", open(f"{d}/seed{s}/run.log").read())]
    a = np.array([r for r in rows if r[0] <= FRAMES]) if rows else np.zeros((0, 2))
    return a[:, 0], a[:, 1], False


curves = {name: [curve(d, s) for s in SEEDS] for name, _, d in ARMS}
keep = [i for i, (name, _, _) in enumerate(ARMS) if all(len(x) for x, _, _ in curves[name])]   # every seed has data
ARMS, TICKS = [ARMS[i] for i in keep], [TICKS[i] for i in keep]
all_done = all(c[2] for cs in curves.values() for c in cs)
# while runs are in progress: compare all arms on the last 40k steps that every drawn run has reached
HI = int(min(x.max() for name, _, _ in ARMS for x, _, _ in curves[name]))
LO = max(0, HI - 40_000)

fig, axes = plt.subplots(1, 2, figsize=(13 + 1.2 * (len(ARMS) - 3), 4.8), gridspec_kw={"width_ratios": [2, 1]})
for i, (name, col, _) in enumerate(ARMS):
    cs = curves[name]
    for x, y, _ in cs:
        axes[0].plot(x, y, color=col, lw=0.8, alpha=0.35)
    lim = min(x.max() for x, _, _ in cs)
    g = np.arange(20_000, lim + 1, 20_000)
    axes[0].plot(g, np.mean([np.interp(g, x, y) for x, y, _ in cs], 0), color=col, lw=2.4, label=f"{name} (n={len(cs)})")
    if all_done:
        v = [y[x >= 0.9 * FRAMES].mean() for x, y, _ in cs]
    else:
        v = [y[(x > LO) & (x <= HI)].mean() for x, y, _ in cs]
    axes[1].bar(i, np.mean(v), color=col, alpha=0.3, width=0.6)
    axes[1].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=col, s=30, zorder=3)
    axes[1].text(i, np.mean(v), f"{np.mean(v):.0f} ± {np.std(v):.0f}", ha="center", va="bottom", fontsize=9)
axes[0].set_xlabel("environment steps")
axes[0].set_ylabel("greedy test return (best individual)")
axes[0].set_title(f"Discrete HalfCheetah-v4, SEARL, seeds 0-4, {FRAMES // 1000}k steps" + ("" if all_done else " (runs still in progress)"), fontsize=11)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
axes[1].set_xticks(range(len(ARMS)))
axes[1].set_xticklabels(TICKS, fontsize=9)
axes[1].set_title(f"final level, {int(0.9 * FRAMES) // 1000}k-{FRAMES // 1000}k steps (dot = seed)" if all_done else f"mean test return, {LO // 1000}k-{HI // 1000}k steps (dot = seed)",
                  fontsize=11)
for ax in axes:
    ax.axhline(0, color="#888", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
fig.text(0.01, 0.01, "Thin = seeds, thick = mean over seeds (drawn up to the shortest seed). Finished runs: 5-epoch mean of the "
         "per-epoch test return; unfinished: test return printed every 20k steps.", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/progress_A_B_vs_searl.png", dpi=140)
print("wrote", f"{OUT}/progress_A_B_vs_searl.png")
