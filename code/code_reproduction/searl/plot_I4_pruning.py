#!/usr/bin/env python
"""I4 (4 heads: full + h1..h3 that start with all 17 observation entries; SEARL prunes blocks of inputs): what gets
pruned, and does selection care.  usage: plot_I4_pruning.py RUN_ROOT  -> RUN_ROOT/figures/I4_pruning.png
a) mean number of inputs per pruned head (population mean) per generation, one line per seed;
b) share of pruned heads (3 per individual x population) that still see each observation entry, mean over seeds;
c) the same at each seed's latest generation (entry x seed);
d) fitness of children relative to their generation's mean, by the mutation that produced them (I4 prune vs ctrl)."""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OBS = ["rootz", "rooty", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "vx", "vz", "vangle",
       "bthigh'", "bshin'", "bfoot'", "fthigh'", "fshin'", "ffoot'"]
R = sys.argv[1]
MUT = {"no": "none", "rl": "learning rate", "classic": "weights", "architecture": "architecture",
       "activation": "activation", "head": "prune inputs"}


def load(arm):
    out = []
    for s in range(5):
        p = f"{R}/{arm}/seed{s}/population_log.csv"
        if os.path.exists(p):
            out.append(pd.read_csv(p, dtype={"mask": str}).assign(s=s))
    P = pd.concat(out)
    P["m"] = P.mutation.str.split("_").str[0].map(MUT)
    P["rel"] = P.fitness - P.groupby(["s", "generation"]).fitness.transform("mean")
    return P


P, C = load("I4_K4_remove"), load("I4_K4_ctrl")
heads = P.inputs.str.split("|").map(lambda h: [set(map(int, x.split("+"))) for x in h[1:]])
P["nin"] = heads.map(lambda h: np.mean([len(x) for x in h]))
B = np.array([[np.mean([i in x for x in h]) for i in range(17)] for h in heads])   # [rows, 17] share of heads seeing i
for i in range(17):
    P[f"e{i}"] = B[:, i]
E_ = [f"e{i}" for i in range(17)]

fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.6, 0.8], height_ratios=[1, 1])
ax = fig.add_subplot(gs[0, 0])
cols = plt.cm.viridis(np.linspace(0, 0.85, 5))
for s, x in P.groupby("s"):
    g = x.groupby("generation").nin.mean()
    ax.plot(g.index, g.values, color=cols[s], lw=1.8, label=f"seed {s} (gen {g.index.max()})")
ax.axhline(17, color="#888", ls=":", lw=1)
ax.set_ylim(0, 18)
ax.set_xlabel("generation")
ax.set_ylabel("inputs per pruned head (of 17)")
ax.set_title("a) how fast the heads are pruned (population mean)", fontsize=10)
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[0, 1:])
G = int(P.groupby("s").generation.max().min())
H = P[P.generation <= G].groupby("generation")[E_].mean().T.values
im = ax.imshow(H, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest",
               extent=(0.5, G + 0.5, 16.5, -0.5))
ax.set_yticks(range(17))
ax.set_yticklabels([f"obs{i} {n}" for i, n in enumerate(OBS)], fontsize=7.5)
ax.set_xlabel("generation")
ax.set_title(f"b) share of pruned heads that still see each entry (mean over 5 seeds, up to gen {G} = shortest seed)",
             fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

ax = fig.add_subplot(gs[1, 0])
last = P[P.generation == P.groupby("s").generation.transform("max")]
F = last.groupby("s")[E_].mean().T.values                                            # [17, seeds]
ax.imshow(F, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
for (i, j), v in np.ndenumerate(F):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else "#333")
ax.set_yticks(range(17))
ax.set_yticklabels([f"obs{i} {n}" for i, n in enumerate(OBS)], fontsize=7.5)
ax.set_xticks(range(F.shape[1]))
ax.set_xticklabels([f"s{s}\ng{g}" for s, g in last.groupby("s").generation.max().items()], fontsize=8)
ax.set_title("c) latest generation, per seed", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
order = ["none", "learning rate", "weights", "architecture", "prune inputs", "activation"]
for k, (D, col, lab) in enumerate([(P, "#882255", "I4 prune"), (C, "#555555", "I4 ctrl")]):
    st = D.groupby("m").rel.agg(["mean", "sem", "count"]).reindex(order)
    xs = np.arange(len(order)) + (k - 0.5) * 0.38
    ax.bar(xs, st["mean"], 0.36, yerr=st["sem"], color=col, alpha=0.8, label=lab, capsize=2)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=8.5)
ax.axhline(0, color="#888", lw=0.6)
ax.set_ylabel("child fitness - generation mean")
ax.set_title("d) fitness of children by the mutation that made them (mean ± sem)", fontsize=10)
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[1, 2])
x = P.assign(nr=P.nin - P.groupby(["s", "generation"]).nin.transform("mean"))
ax.scatter(x.nr, x.rel, s=5, alpha=0.25, color="#882255")
r = np.corrcoef(x.nr, x.rel)[0, 1]
ax.axhline(0, color="#888", lw=0.6)
ax.axvline(0, color="#888", lw=0.6)
ax.set_xlabel("inputs per head - generation mean")
ax.set_ylabel("fitness - generation mean")
ax.set_title(f"e) more inputs -> fitter?  r = {r:.2f}", fontsize=10)
for a in fig.axes[:-1]:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("I4: 4 heads (full + 3 that start with all 17 inputs), SEARL prunes 25% of one head's inputs per prune "
             "mutation -- runs in progress", fontsize=11.5)
fig.tight_layout()
o = f"{R}/figures/I4_pruning.png"
fig.savefig(o, dpi=125)
print("wrote", o)
