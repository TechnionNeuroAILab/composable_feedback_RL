#!/usr/bin/env python
"""Two diagnostic figures for the 18-head SEARL model on discrete HalfCheetah-v4.

usage: plot_obs_heads_q_and_survival.py RUN_DIR [RUN_DIR ...] --out FIG_DIR
  RUN_DIR = one seed's output folder (heads_log.csv, population_log.csv, q_dump.npz). Runs are grouped by their
  parent folder name (e.g. A_fixed, B_toggle).

1. q_full_vs_obs.png   (needs the Q columns of heads_log.csv / q_dump.npz, logged from the Q-logging version on)
   a) mean Q level per head over training (full head thick, single-observation heads thin);
   b) how much each head's Q changes across states (std over states) -- a head that sees one number can only
      vary along that number;
   c) correlation of each head's per-state torque preference (Q minus its per-state mean over the 3 torques, the
      part that decides argmax) with the full head's;
   d) scatter of one late phase: full-head Q vs each single-observation head's Q on the same 32 states x 18 outputs.
2. B_head_survival.png (population_log.csv of toggle runs)
   a) heatmap: rows = heads, columns = generations, colour = share of the population with that head on (weight > 0);
   b) per head: share of individuals x generations on, and the generation it was last on in the elite (individual 0
      after selection = the parent kept by elitism), with the no-selection drift level 0.5.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEADS = ["full"] + [f"obs{i}" for i in range(17)]
# HalfCheetah-v4 observation entries (gymnasium docs)
OBS_NAMES = ["rootz", "rooty(angle)", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "vx", "vz", "vangle",
             "bthigh'", "bshin'", "bfoot'", "fthigh'", "fshin'", "ffoot'"]
LABELS = ["full"] + [f"obs{i} {n}" for i, n in enumerate(OBS_NAMES)]
FULL_C, OBS_C = "#0072B2", "#E69F00"


def group(d):
    return os.path.basename(os.path.dirname(os.path.normpath(d)))


def q_figure(runs, out):
    hs = []
    for d in runs:
        p = f"{d}/heads_log.csv"
        if os.path.exists(p):
            h = pd.read_csv(p)
            if "q_mean" in h:
                hs.append(h.assign(run=d, arm=group(d)))
    if not hs:
        print("no Q logs yet")
        return
    h = pd.concat(hs)
    arms = list(dict.fromkeys(h.arm))
    fig, axes = plt.subplots(len(arms), 4, figsize=(19, 4.2 * len(arms)), squeeze=False)
    for r, arm in enumerate(arms):
        ha = h[h.arm == arm]
        n_runs = ha.run.nunique()
        # mean over individuals of a phase-bin and over seeds; phases are binned to 20 bins of training progress
        ha = ha.assign(prog=ha.groupby("run").phase.transform(lambda p: (p / p.max() * 19.999).astype(int)))
        on = ha[ha.on == 1]   # an off head is not trained toward the ensemble's vote either way; show only heads that vote
        for c, (col, title) in enumerate([("q_mean", "mean Q (all states, outputs)"),
                                          ("q_state_std", "Q spread across states (std)"),
                                          ("adv_corr_full", "corr. of torque preference with full head")]):
            ax = axes[r, c]
            m = on.groupby(["head", "prog"])[col].mean().unstack(0)
            for hd in HEADS:
                if hd in m:
                    full = hd == "full"
                    ax.plot(m.index / 20, m[hd], color=FULL_C if full else OBS_C, lw=2.6 if full else 0.8,
                            alpha=1 if full else 0.6, zorder=3 if full else 2)
            if col != "adv_corr_full":
                ob = m[[x for x in m if x != "full"]].mean(1)
                ax.plot(m.index / 20, ob, color=OBS_C, lw=2.6, ls="--")
            ax.set_title(f"{arm} (n={n_runs} seeds): {title}", fontsize=10)
            ax.set_xlabel("training progress (fraction of logged phases)")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 2].axhline(0, color="#888", lw=0.6)
        axes[r, 0].plot([], [], color=FULL_C, lw=2.6, label="full head")
        axes[r, 0].plot([], [], color=OBS_C, lw=0.8, label="single-obs heads")
        axes[r, 0].plot([], [], color=OBS_C, lw=2.6, ls="--", label="mean of single-obs heads")
        axes[r, 0].legend(frameon=False, fontsize=8)
        # d) scatter of the last phase of the first run of this arm
        ax = axes[r, 3]
        d0 = [d for d in runs if group(d) == arm and os.path.exists(f"{d}/q_dump.npz")]
        if d0:
            z = np.load(f"{d0[0]}/q_dump.npz")
            i = len(z["phase"]) - 1
            q, mask = z["q"][i], z["mask"][i]                                   # [K, 32, 18]
            for k in range(1, q.shape[0]):
                ax.scatter(q[0].ravel(), q[k].ravel(), s=3, alpha=0.25, color=OBS_C if mask[k] else "#bbb")
            lo, hi = q.min(), q.max()
            ax.plot([lo, hi], [lo, hi], color="#555", lw=0.8, ls=":")
            ax.set_xlabel("full head Q")
            ax.set_ylabel("single-obs head Q (grey = head off)")
            ax.set_title(f"{os.path.basename(d0[0])}, phase {z['phase'][i]}: 32 states x 18 outputs", fontsize=10)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out}/q_full_vs_obs.png", dpi=130)
    print("wrote", f"{out}/q_full_vs_obs.png")


def survival_figure(runs, out):
    pops = []
    for d in runs:
        p = f"{d}/population_log.csv"
        if os.path.exists(p):
            x = pd.read_csv(p, dtype={"mask": str})
            if (x.n_on < 18).any() or group(d).startswith("B"):
                pops.append(x.assign(run=d))
    if not pops:
        print("no toggle population logs")
        return
    n = len(pops)
    fig = plt.figure(figsize=(7.5 + 2.4 * n, 7.2))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.6])
    share_all, last_elite = [], []
    for c, x in enumerate(pops):
        bits = np.array([[int(ch) for ch in m.lstrip("m")] for m in x["mask"]])
        x = pd.concat([x.reset_index(drop=True), pd.DataFrame(bits, columns=HEADS)], axis=1)
        g = x.groupby("generation")[HEADS].mean().T                        # [18, G] share of population on
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(g.values, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest",
                  extent=(g.columns.min() - 0.5, g.columns.max() + 0.5, 17.5, -0.5))
        ax.set_yticks(range(18))
        ax.set_yticklabels(LABELS if c == 0 else [], fontsize=7)
        ax.set_xlabel("generation")
        ax.set_title(f"{os.path.basename(os.path.normpath(x.run[0]))}\nfitness end {x[x.generation == x.generation.max()].fitness.max():.0f}",
                     fontsize=9)
        share_all.append(bits.mean(0))
        # best individual of each generation (the one elitism keeps): its mask
        best = x.loc[x.groupby("generation").fitness.idxmax()]
        last_elite.append(best[HEADS].values[-1])
    ax = fig.add_subplot(gs[0, n])
    share_all, last_elite = np.array(share_all), np.array(last_elite)
    y = np.arange(18)
    ax.barh(y, share_all.mean(0), color="#009E73", alpha=0.35, label="mean over seeds")
    for s in share_all:
        ax.scatter(s, y, color="#009E73", s=12, zorder=3)
    ax.scatter(last_elite.mean(0), y, marker="D", color="#CC79A7", s=22, zorder=4,
               label="share of seeds where the final best individual has it on")
    # no-selection reference: every head starts on; per generation each individual is mutated once, head_toggle is 1 of 6
    # operators and flips 1 of 18 heads -> flip prob p = 1/108 per head per generation along a lineage.
    G = int(max(x.generation.max() for x in pops))
    p = 1 / 108
    on_g = 0.5 + 0.5 * (1 - 2 * p) ** np.arange(1, G + 1)
    ax.axvline(on_g.mean(), color="#888", ls="--", lw=1, label=f"no selection, mean over {G} gens ({on_g.mean():.2f})")
    ax.axvline(on_g[-1], color="#888", ls=":", lw=1, label=f"no selection, at the last gen ({on_g[-1]:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of individuals x generations with the head on")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model B: which heads survive mutation + selection (green = on and voting, white = weight 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/B_head_survival.png", dpi=130)
    print("wrote", f"{out}/B_head_survival.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q_figure(a.runs, a.out)
    survival_figure(a.runs, a.out)
