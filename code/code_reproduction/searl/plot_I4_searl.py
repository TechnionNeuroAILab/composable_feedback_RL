#!/usr/bin/env python
"""I4 and its SEARL versions vs plain SEARL, plus the same 4 designs without SEARL (dashed), discrete HalfCheetah-v4, 400k steps, seeds 0-4.
usage: plot_I4_searl.py RUN_ROOT  -> RUN_ROOT/figures/I4_searl_versions.png
a) greedy test return of the best individual (thin = seed, thick = mean up to the shortest seed);
b) level over the last 40k steps every seed has reached (360k-400k once all are done), dot = seed;
c) inputs per head over training (population mean, mean over seeds): the 3 pruned heads, and in prune-all the
   full head (h0) separately. One SEARL generation = 4 individuals x 1000 steps = 4000 environment steps; without
   SEARL one generation = 1 individual = 1000 steps.
Same colour = same design; solid = SEARL (population 4), dashed / open = no SEARL (population 1, no selection or
mutation, lr 3e-4; pruning random, one head loses 25% of its inputs every 24k steps)."""
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
FR = 400_000
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# (label, colour, folder, tick, searl, steps per generation)
ARMS = [("plain SEARL (1 network)", "#0072B2", f"{ROOT}/results/searl_confgate_halfcheetah_1M/B_full", "SEARL\n1 net", True, 4000),
        ("DQN, 1 network", "#0072B2", f"{R}/DQN1", "DQN\n1 net", False, 1000),
        ("DQN, 1 network, 512 units (~ 4 heads in weights)", "#CC79A7", f"{R}/DQN1_wide", "DQN 1 net\n512 units", False, 1000),
        ("I4 ctrl: 4 full heads, no pruning", "#555555", f"{R}/I4_K4_ctrl", "I4\nctrl", True, 4000),
        ("DQN, 4 full heads, no pruning", "#555555", f"{R}/DQN4_ctrl", "DQN\n4 heads", False, 1000),
        ("I4 prune-3: h1-h3 pruned, full head kept", "#882255", f"{R}/I4_K4_remove", "I4\nprune-3", True, 4000),
        ("DQN, random prune of h1-h3", "#882255", f"{R}/DQN4_prune3", "DQN\nprune-3", False, 1000),
        ("I4 prune-all: all 4 heads pruned", "#009E73", f"{R}/I4_K4_remove_all", "I4\nprune-all", True, 4000),
        ("DQN, random prune of all 4", "#009E73", f"{R}/DQN4_pruneall", "DQN\nprune-all", False, 1000)]


def curve(d, s):
    h = f"{d}/seed{s}/history.csv"
    if os.path.exists(h):
        h = pd.read_csv(h).dropna(subset=["test_return"])
        h = h[h.num_frames <= FR]
        w = max(1, int(round(20_000 / np.median(np.diff(h.num_frames.values)))))   # ~20k steps for both
        return h.num_frames.values, h.test_return.rolling(w, min_periods=1).mean().values, True
    rows = [(int(m[1]), float(m[2])) for m in
            re.finditer(r"frames=(\d+) .*?test=(-?[\d.]+)", open(f"{d}/seed{s}/run.log").read())]
    a = np.array([r for r in rows if r[0] <= FR])
    return a[:, 0], a[:, 1], False


def n_inputs(st):
    return [17 if x == "all" else len(x.split("+")) for x in st.split("|")]


def style(d, se):
    return "-" if se or d.endswith("_wide") else "--"   # the 512-unit network: solid pink, filled


FOUR = [a for a in ARMS if not a[2].endswith(("B_full", "DQN1", "DQN1_wide"))]   # the 4-head arms


C = {a[0]: [curve(a[2], s) for s in range(5)] for a in ARMS}
done = all(c[2] for cs in C.values() for c in cs)
reach = min(x.max() for cs in C.values() for x, _, _ in cs)
win = (reach - 40_000, reach)

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1.15, 1])
ax = fig.add_subplot(gs[0, 0])
for name, col, d, _, se, _ in ARMS:
    cs = C[name]
    ls = style(d, se)
    for x, y, _ in cs:
        ax.plot(x, y, color=col, ls=ls, lw=0.6, alpha=0.2)
    g = np.arange(20_000, min(x.max() for x, _, _ in cs) + 1, 20_000)
    ax.plot(g, np.mean([np.interp(g, x, y) for x, y, _ in cs], 0), color=col, ls=ls, lw=2.4, label=f"{name} (n=5)")
ax.axvspan(*win, color="#ddd", alpha=0.5, lw=0)
ax.set_xlabel("environment steps")
ax.set_ylabel("greedy test return (best individual)")
ax.set_title("a) learning curves (solid = SEARL, dashed = same design without SEARL)"
             + ("" if done else "; I4 prune-all still running"), fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

ax = fig.add_subplot(gs[0, 1])
for i, (name, col, d, tick, se, _) in enumerate(ARMS):
    v = np.array([y[(x > win[0]) & (x <= win[1])].mean() for x, y, _ in C[name]])
    ax.bar(i, v.mean(), color=col, alpha=0.32 if style(d, se) == "-" else 0.1, width=0.68, edgecolor=col, ls=style(d, se), lw=1.1)
    ax.scatter(i + np.linspace(-0.16, 0.16, 5), v, color=col if style(d, se) == "-" else "white", edgecolor=col, lw=1.2, s=26, zorder=3)
    ax.text(i, max(v.max(), 0) + 80, f"{v.mean():.0f}\n±{v.std():.0f}", ha="center", va="bottom", fontsize=8, color=col)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([a[3] for a in ARMS], fontsize=8.3)
ax.set_title(f"b) mean test return, {int(win[0]) // 1000}k-{int(win[1]) // 1000}k steps (shaded in a; dot = seed)",
             fontsize=11)
ax.set_ylim(top=ax.get_ylim()[1] * 1.12)

ax = fig.add_subplot(gs[1, 0])
for name, col, d, _, se, spg in [a for a in FOUR if "remove" in a[2] or "prune" in a[2]]:
    ls = "-" if se else "--"
    P = pd.concat([pd.read_csv(f"{d}/seed{s}/population_log.csv").assign(s=s) for s in range(5)])
    N = np.array(P.inputs.map(n_inputs).tolist())                       # [rows, 4]
    P["pruned3"], P["h0"] = N[:, 1:].mean(1), N[:, 0]
    m = P.groupby(["s", "generation"])[["pruned3", "h0"]].mean().groupby("generation").agg(["mean", "min", "max"])
    steps = m.index.values * spg
    short = name.split(":")[0]
    ax.plot(steps, m[("pruned3", "mean")], color=col, ls=ls, lw=2.4, label=f"{short}: mean of h1-h3")
    if se:
        ax.fill_between(steps, m[("pruned3", "min")], m[("pruned3", "max")], color=col, alpha=0.12, lw=0)
    if "all" in d:
        ax.plot(steps, m[("h0", "mean")], color=col, ls=ls, lw=1.1, alpha=0.8, label=f"{short}: full head (h0)")
ax.axhline(17, color="#555", lw=1, ls=":", label="I4 ctrl / never pruned (17)")
ax.set_ylim(0, 18)
ax.set_xlim(0, FR)
ax.set_xlabel("environment steps (1 generation = 4000 steps with SEARL, 1000 without)")
ax.set_ylabel("inputs per head (of 17)")
ax.set_title("c) inputs per head over training (mean over seeds; band = range over seeds, SEARL only; thin = h0)",
             fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
rows = []
for name, col, d, tick, se, _ in FOUR:
    for s in range(5):
        P = pd.read_csv(f"{d}/seed{s}/population_log.csv")
        last = P[P.generation == P.generation.max()]
        b = last.loc[last.fitness.idxmax()]
        rows.append((tick.replace("\n", " "), s, int(P.generation.max()), n_inputs(b.inputs)))
txt = "d) inputs per head of the best individual in the latest generation\n\n" \
      f"{'arm':14s}{'seed':>5s}{'gen':>6s}   full/h0  h1  h2  h3\n"
for tick, s, g, n in rows:
    txt += f"{tick:14s}{s:>5d}{g:>6d}   {n[0]:>7d} {n[1]:>3d} {n[2]:>3d} {n[3]:>3d}\n"
ax.text(0.0, 1.0, txt, family="monospace", fontsize=7.6, va="top")
for a in fig.axes[:3]:
    a.axhline(0, color="#888", lw=0.6) if a is not fig.axes[2] else None
    a.grid(alpha=0.2)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Discrete HalfCheetah-v4: I4 (4-head ensembles, equal vote) with SEARL (solid) and without SEARL (dashed), "
             "seeds 0-4, 400k steps", fontsize=12)
fig.text(0.01, 0.005, "No SEARL = population 1, no selection or mutation, lr 3e-4; pruning random: one head loses 25% of its "
         "inputs every 24k steps. SEARL: test return of the best individual; prune = SEARL mutation (25% of one head's inputs).",
         fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.015, 1, 1))
o = f"{R}/figures/I4_searl_versions.png"
fig.savefig(o, dpi=130)
print("wrote", o, "| window", win, "| all done:", done)
