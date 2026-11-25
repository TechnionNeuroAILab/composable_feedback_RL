#!/usr/bin/env python
"""Compare the 4 pruning versions (prune h1-h3 / prune all 4 heads, with / without SEARL), discrete HalfCheetah-v4.
usage: plot_pruning_compare.py RUN_ROOT  -> RUN_ROOT/figures/pruning_compare.png
a) inputs per prunable head over training (mean over seeds);
b) return against inputs: each arm's path through (inputs per prunable head, test return) every 20k steps, with the
   unpruned 4-head versions (17 inputs) as reference;
c) which observation entries survive: share of prunable heads that still see each entry at the end, per seed
   (SEARL: mean over the final population; no SEARL: the one agent), with the mean pairwise correlation of these
   keep-profiles across seeds (r ~ 0: different seeds keep different entries, i.e. chance decides);
d) the same, mean over seeds, per arm;
e) SEARL: fitness of children made by each mutation, relative to their generation's mean;
f) no SEARL: change in test return across a prune (mean over the 10k steps after minus the 10k before), against the
   same change at steps with no prune (baseline = ordinary learning drift)."""
import itertools
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
OBS = ["rootz", "rooty", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "vx", "vz", "vangle",
       "bthigh'", "bshin'", "bfoot'", "fthigh'", "fshin'", "ffoot'"]
# (label, colour, linestyle, folder, searl, env steps per generation, prunable head indices)
ARMS = [("SEARL prune-3 (full head kept)", "#882255", "-", "I4_K4_remove", True, 4000, [1, 2, 3]),
        ("no SEARL, random prune-3", "#882255", "--", "DQN4_prune3", False, 1000, [1, 2, 3]),
        ("SEARL prune-all", "#009E73", "-", "I4_K4_remove_all", True, 4000, [0, 1, 2, 3]),
        ("no SEARL, random prune-all", "#009E73", "--", "DQN4_pruneall", False, 1000, [0, 1, 2, 3])]
REF = [("SEARL, 4 heads, no pruning", "#555555", "-", "I4_K4_ctrl"),
       ("no SEARL, 4 heads, no pruning", "#555555", "--", "DQN4_ctrl")]
SEEDS = range(5)
MUT = {"no": "none", "rl": "lr", "classic": "weights", "architecture": "architecture", "activation": "activation",
       "head": "prune"}


def heads_sets(st):
    return [set(range(17)) if x == "all" else set(map(int, x.split("+"))) for x in st.split("|")]


def history(folder, s):
    h = pd.read_csv(f"{R}/{folder}/seed{s}/history.csv").dropna(subset=["test_return"])
    return h.num_frames.values, h.test_return.values


def smooth_at(x, y, grid, half=10_000):
    return np.array([y[(x > g - 2 * half) & (x <= g)].mean() for g in grid])


pops = {}
for name, col, ls, f, searl, spg, ph in ARMS:
    P = pd.concat([pd.read_csv(f"{R}/{f}/seed{s}/population_log.csv").assign(s=s) for s in SEEDS])
    H = P.inputs.map(heads_sets)
    P["nin"] = H.map(lambda h: np.mean([len(h[k]) for k in ph]))
    K = np.array([[np.mean([i in h[k] for k in ph]) for i in range(17)] for h in H])
    for i in range(17):
        P[f"e{i}"] = K[:, i]
    P["steps"] = P.generation * spg
    pops[name] = P
E = [f"e{i}" for i in range(17)]

fig = plt.figure(figsize=(20, 15.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.25, 0.95], width_ratios=[1, 1, 0.75])

# a) inputs over training
ax = fig.add_subplot(gs[0, 0])
for name, col, ls, f, searl, spg, ph in ARMS:
    m = pops[name].groupby(["s", "steps"]).nin.mean().groupby("steps").mean()
    ax.plot(m.index, m.values, color=col, ls=ls, lw=2.3, label=name)
ax.axhline(17, color="#555", ls=":", lw=1)
ax.set_ylim(0, 18)
ax.set_xlim(0, 400_000)
ax.set_xlabel("environment steps")
ax.set_ylabel("inputs per prunable head (of 17)")
ax.set_title("a) how much the prunable heads see (population mean, mean over 5 seeds)", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

# b) return vs inputs
ax = fig.add_subplot(gs[0, 1])
grid = np.arange(40_000, 400_001, 20_000)
for name, col, ls, f, searl, spg, ph in ARMS:
    nin = pops[name].groupby(["s", "steps"]).nin.mean().groupby("steps").mean()
    xin = np.interp(grid, nin.index.values, nin.values)
    ret = np.mean([smooth_at(*history(f, s), grid) for s in SEEDS], 0)
    ax.plot(xin, ret, color=col, ls=ls, lw=2.2, marker="o", ms=3, label=name)
    for g in (100_000, 200_000, 300_000, 400_000):
        j = int(np.argmin(np.abs(grid - g)))
        ax.annotate(f"{g // 1000}k", (xin[j], ret[j]), fontsize=7, color=col, xytext=(4, 4), textcoords="offset points")
for name, col, ls, f in REF:
    ret = np.mean([smooth_at(*history(f, s), grid) for s in SEEDS], 0)
    ax.scatter(np.full(len(grid), 17.0) + (0.15 if ls == "--" else -0.15), ret, color=col if ls == "-" else "white",
               edgecolor=col, s=14, zorder=3, label=f"{name} (17 inputs, 40k-400k)")
ax.invert_xaxis()
ax.set_xlabel("inputs per prunable head (training runs right to left)")
ax.set_ylabel("test return (20k-step mean, mean over seeds)")
ax.set_title("b) return as the heads lose inputs", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

# e) SEARL: child fitness by mutation
ax = fig.add_subplot(gs[0, 2])
order = ["none", "lr", "weights", "architecture", "prune", "activation"]
searl_arms = [a for a in ARMS if a[4]]
for k, (name, col, ls, *_rest) in enumerate(searl_arms):
    P = pops[name].copy()
    P["m"] = P.mutation.fillna("").str.split("_").str[0].map(MUT)
    P["rel"] = P.fitness - P.groupby(["s", "generation"]).fitness.transform("mean")
    st = P.groupby("m").rel.agg(["mean", "sem"]).reindex(order)
    ax.barh(np.arange(len(order)) + (k - 0.5) * 0.38, st["mean"], 0.36, xerr=st["sem"], color=col, alpha=0.85,
            label=name, capsize=2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.axvline(0, color="#888", lw=0.6)
ax.set_xlabel("child fitness - generation mean")
ax.set_title("e) SEARL: what each mutation does to fitness\n(mean ± sem over all children)", fontsize=10.5)
ax.legend(frameon=False, fontsize=8)

# c) survival per seed
ax = fig.add_subplot(gs[1, :2])
cols, labels, cons = [], [], []
for name, col, ls, f, searl, spg, ph in ARMS:
    P = pops[name]
    last = P[P.generation == P.groupby("s").generation.transform("max")]
    prof = last.groupby("s")[E].mean()                                   # [seeds, 17]
    cons.append(np.mean([np.corrcoef(prof.iloc[i], prof.iloc[j])[0, 1]
                         for i, j in itertools.combinations(range(len(prof)), 2)]))
    for s in prof.index:
        cols.append(prof.loc[s].values)
        labels.append(f"s{s}")
    cols.append(np.full(17, np.nan))
    labels.append("")
M = np.array(cols[:-1]).T
ax.imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
for (i, j), v in np.ndenumerate(M):
    if not np.isnan(v):
        ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=6.3,
                color="white" if v > 0.6 else "#333")
ax.set_yticks(range(17))
ax.set_yticklabels([f"obs{i} {n}" for i, n in enumerate(OBS)], fontsize=8)
ax.set_xticks(range(len(labels) - 1))
ax.set_xticklabels(labels[:-1], fontsize=7.5)
for k, (name, col, *_r) in enumerate(ARMS):
    ax.text(k * 6 + 2, -1.0, f"{name}\nseed agreement r = {cons[k]:+.2f}", ha="center", va="bottom", fontsize=8.5,
            color=col, fontweight="bold")
ax.set_title("c) share of prunable heads that still see each entry at the end (SEARL: final population; no SEARL: "
             "the agent)", fontsize=10.5, pad=34)

# d) mean over seeds
ax = fig.add_subplot(gs[1, 2])
y = np.arange(17)
for k, (name, col, ls, f, searl, spg, ph) in enumerate(ARMS):
    P = pops[name]
    last = P[P.generation == P.groupby("s").generation.transform("max")]
    prof = last.groupby("s")[E].mean()
    ax.plot(prof.mean().values, y + (k - 1.5) * 0.15, ls=ls, color=col, marker="o" if searl else "s",
            mfc=col if searl else "white", ms=5, lw=1, label=name)
ax.set_yticks(y)
ax.set_yticklabels([f"obs{i} {n}" for i, n in enumerate(OBS)], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of prunable heads seeing the entry, mean over seeds")
ax.set_title("d) which entries survive, per arm", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

# f) no SEARL: change in test return across prune events
ax = fig.add_subplot(gs[2, :2])
W = 10_000
xs = 0
for name, col, ls, f, searl, spg, ph in ARMS:
    if searl:
        continue
    ev, base = [], []
    for s in SEEDS:
        x, yv = history(f, s)
        P = pops[name][pops[name].s == s]
        pr = set(P[P.mutation.fillna("").str.startswith("head_input")].steps.values)
        for t in np.arange(W + 1000, 400_000 - W, 1000):
            d = yv[(x > t) & (x <= t + W)].mean() - yv[(x > t - W) & (x <= t)].mean()
            (ev if t in pr else base).append(d)
    for j, (v, lab) in enumerate([(base, "no prune"), (ev, "prune")]):
        v = np.array(v)
        ax.bar(xs + j, v.mean(), 0.75, yerr=v.std() / np.sqrt(len(v)), color=col, alpha=0.25 if j == 0 else 0.85,
               capsize=3)
        ax.text(xs + j, 0, f"{lab}\n(n={len(v)})", ha="center", va="top", fontsize=8, color="#333")
        ax.text(xs + j, v.mean() + (12 if v.mean() >= 0 else -12), f"{v.mean():+.0f}", ha="center",
                va="bottom" if v.mean() >= 0 else "top", fontsize=9, color=col)
    ax.text(xs + 0.5, ax.get_ylim()[1] if False else 0, "", fontsize=1)
    ax.annotate(name, (xs + 0.5, 1), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=9, color=col,
                fontweight="bold")
    xs += 3
ax.axhline(0, color="#888", lw=0.6)
ax.set_xticks([])
ax.set_xlim(-0.8, xs - 1.2)
ax.set_ylabel("test return: mean over next 10k steps\nminus mean over previous 10k (± sem)")
ax.set_title("f) no SEARL: what one random prune does to the return (vs the same window at steps with no prune)",
             fontsize=10.5)

ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
rows = []
for name, col, ls, f, searl, spg, ph in ARMS:
    v = [(lambda xy: xy[1][xy[0] >= 360_000].mean())(history(f, s)) for s in SEEDS]
    nin = pops[name][pops[name].generation == pops[name].groupby("s").generation.transform("max")].groupby("s").nin.mean()
    rows.append((name, np.mean(v), np.std(v), nin.mean()))
for name, col, ls, f in REF:
    v = [(lambda xy: xy[1][xy[0] >= 360_000].mean())(history(f, s)) for s in SEEDS]
    rows.append((name, np.mean(v), np.std(v), 17.0))
t = "final level, 360k-400k steps (5 seeds)\n\n" + f"{'arm':33s}{'return':>14s}{'inputs':>8s}\n"
for name, m, sd, n in rows:
    t += f"{name:33s}{m:7.0f} ± {sd:4.0f}{n:8.1f}\n"
ax.text(0, 0.95, t, family="monospace", fontsize=8.6, va="top")

for a in fig.axes:
    if a.axison:
        a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Discrete HalfCheetah-v4: pruning the 4-head ensemble with SEARL (solid) and without SEARL (dashed), "
             "400k steps, seeds 0-4", fontsize=13)
fig.text(0.01, 0.004, "Prune = one head loses a random 25% of its inputs. SEARL: a mutation (1 in 6 per individual per "
         "generation), kept or lost by selection. No SEARL: one prune every 24k steps on the single agent, no selection. "
         "prune-3: the full head is never pruned; prune-all: all 4 heads can be.", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.012, 1, 0.98))
o = f"{R}/figures/pruning_compare.png"
fig.savefig(o, dpi=120)
print("wrote", o)
for (name, *_r), c in zip(ARMS, cons):
    print(f"{name:32s} seed agreement r = {c:+.2f}")
