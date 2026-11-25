#!/usr/bin/env python
"""Limb heads on discrete HalfCheetah-v4 (400k steps, seeds 0-4), with and without SEARL, against the 1-network
and 4-head baselines.  usage: plot_limbs.py RUN_ROOT  -> RUN_ROOT/figures/limb_heads.png
Limb heads: Q-values summed (not averaged), one loss on the sum.
  L3 split   back leg obs -> back joints; front leg obs -> front joints; torso obs -> a term for all 6 joints
  L2 overlap back leg + torso obs -> back joints; front leg + torso obs -> front joints
  L3 random  as split, observations dealt at random into groups of 6 / 6 / 5 (fixed per seed)
a) SEARL curves, b) no-SEARL curves (thin = seed, thick = mean), c) final level 360k-400k (dot = seed),
d) late-training torque preferences: how strongly each head's term separates the 3 torques of the joints it covers
   (max - min of its centred Q), and how often the third head changes the chosen torque."""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PLAIN = f"{ROOT}/results/searl_confgate_halfcheetah_1M/B_full"
# (label, colour, folder, tick, weights)
SEARL = [("1 network (plain SEARL)", "#0072B2", PLAIN, "1 net", "4.6k"),
         ("4 full heads, averaged", "#555555", f"{R}/I4_K4_ctrl", "4 heads", "18.5k"),
         ("L2 overlap: leg+torso per leg", "#E69F00", f"{R}/L2_overlap", "L2\noverlap", "5.4k"),
         ("L3 split: back / front / torso", "#882255", f"{R}/L3_split", "L3\nsplit", "7.2k"),
         ("L3 random groups 6/6/5", "#999933", f"{R}/L3_random", "L3\nrandom", "7.2k")]
DQN = [("1 network, 128 units", "#0072B2", f"{R}/DQN1", "1 net", "4.6k"),
       ("1 network, 512 units", "#CC79A7", f"{R}/DQN1_wide", "1 net\n512", "18.5k"),
       ("4 full heads, averaged", "#555555", f"{R}/DQN4_ctrl", "4 heads", "18.5k"),
       ("L2 overlap: leg+torso per leg", "#E69F00", f"{R}/DQN_L2_overlap", "L2\noverlap", "5.4k"),
       ("L3 split: back / front / torso", "#882255", f"{R}/DQN_L3_split", "L3\nsplit", "7.2k"),
       ("L3 random groups 6/6/5", "#999933", f"{R}/DQN_L3_random", "L3\nrandom", "7.2k")]


def curve(d, s):
    h = pd.read_csv(f"{d}/seed{s}/history.csv").dropna(subset=["test_return"])
    h = h[h.num_frames <= 400_000]
    w = max(1, int(round(20_000 / np.median(np.diff(h.num_frames.values)))))
    return h.num_frames.values, h.test_return.rolling(w, min_periods=1).mean().values, h


def final(d):
    out = []
    for s in range(5):
        h = pd.read_csv(f"{d}/seed{s}/history.csv").dropna(subset=["test_return"])
        out.append(h[(h.num_frames >= 360_000) & (h.num_frames <= 400_000)].test_return.mean())
    return np.array(out)


fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], height_ratios=[1, 1])
for c, (arms, title) in enumerate([(SEARL, "a) with SEARL (population 4)"),
                                   (DQN, "b) without SEARL (population 1, lr 3e-4)")]):
    ax = fig.add_subplot(gs[0, c])
    for name, col, d, _, wts in arms:
        cs = [curve(d, s) for s in range(5)]
        for x, y, _ in cs:
            ax.plot(x, y, color=col, lw=0.6, alpha=0.22)
        g = np.arange(20_000, 400_001, 20_000)
        ax.plot(g, np.mean([np.interp(g, x, y) for x, y, _ in cs], 0), color=col, lw=2.5,
                label=f"{name} ({wts} weights)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylim(-800, 4600)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("greedy test return")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")

ax = fig.add_subplot(gs[1, :2])
pos, ticks = 0, []
for arms, se in [(SEARL, True), (DQN, False)]:
    for name, col, d, tick, _ in arms:
        v = final(d)
        ax.bar(pos, v.mean(), color=col, alpha=0.35 if se else 0.12, width=0.7, edgecolor=col,
               ls="-" if se else "--", lw=1.2)
        ax.scatter(pos + np.linspace(-0.17, 0.17, 5), v, s=26, zorder=3, color=col if se else "white",
                   edgecolor=col, lw=1.2)
        ax.text(pos, max(v.max(), 0) + 90, f"{v.mean():.0f}\n±{v.std():.0f}", ha="center", va="bottom",
                fontsize=8.3, color=col)
        ticks.append((pos, tick))
        pos += 1
    if se:
        ax.axvline(pos - 0.0, color="#bbb", lw=1)
        pos += 1
ax.set_xticks([p for p, _ in ticks])
ax.set_xticklabels([t for _, t in ticks], fontsize=9)
ax.text(2, 4700, "with SEARL", ha="center", fontsize=10.5, fontweight="bold")
ax.text(len(SEARL) + 1 + 2.5, 4700, "without SEARL", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylim(-600, 5100)
ax.axhline(0, color="#888", lw=0.6)
ax.set_ylabel("mean test return, 360k-400k steps")
ax.set_title("c) final level (dot = seed; filled = SEARL, open = no SEARL)", fontsize=11)

# d) how much each head moves the decision, late training
ax = fig.add_subplot(gs[0, 2])
rows = []
for lab, d in [("L3 split, SEARL", "L3_split"), ("L3 split, no SEARL", "DQN_L3_split"),
               ("L3 random, SEARL", "L3_random"), ("L3 random, no SEARL", "DQN_L3_random")]:
    sp, fl = [], []
    for s in range(5):
        q = np.load(f"{R}/{d}/seed{s}/q_dump.npz")["q"]
        q = q[int(q.shape[0] * 0.8):].reshape(-1, q.shape[1], 32, 6, 3)
        q = q - q.mean(-1, keepdims=True)
        spr = q.max(-1) - q.min(-1)
        legs = spr[:, 0, :, :3].mean() + spr[:, 1, :, 3:].mean()
        sp.append(spr[:, 2].mean() / (spr[:, 2].mean() + legs / 2))
        fl.append((q[:, :2].sum(1).argmax(-1) != q.sum(1).argmax(-1)).mean())
    rows.append((lab, np.mean(sp), np.mean(fl), sp, fl))
y = np.arange(len(rows))
ax.barh(y - 0.18, [r[1] for r in rows], 0.34, color="#882255", alpha=0.75,
        label="third head's share of the torque preference")
ax.barh(y + 0.18, [r[2] for r in rows], 0.34, color="#E69F00", alpha=0.75,
        label="decisions the third head changes")
for i, r in enumerate(rows):
    ax.scatter(r[3], np.full(5, i - 0.18), color="#882255", s=10, zorder=3)
    ax.scatter(r[4], np.full(5, i + 0.18), color="#E69F00", s=10, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.axvline(0.5, color="#888", lw=0.6, ls=":")
ax.set_xlabel("share (dot = seed), last 20% of training")
ax.set_title("d) is the torso / third head used?", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="lower right")

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
txt = ("Heads (inputs -> joints whose Q they output)\n\n"
       "L3 split    back:  obs 2-4, 11-13      -> bthigh bshin bfoot\n"
       "            front: obs 5-7, 14-16      -> fthigh fshin ffoot\n"
       "            torso: obs 0, 1, 8-10      -> a term for all 6\n"
       "L2 overlap  back:  back leg + torso    -> back joints\n"
       "            front: front leg + torso   -> front joints\n"
       "L3 random   17 obs dealt into 6/6/5 at random (per seed),\n"
       "            same outputs as L3 split\n\n"
       "Q(joint, torque) = sum of the heads' terms for that joint;\n"
       "one Huber loss on the sum (heads trained jointly).\n"
       "torso: rootz, rooty, vx, vz, vangle\n"
       "leg:   3 joint angles + 3 joint velocities\n\n"
       "Averaged-head arms (4 heads): each head sees all 17\n"
       "and fits the target on its own; Q = mean of the heads.")
ax.text(0, 1, txt, family="monospace", fontsize=8.8, va="top")
for a in fig.axes[:4]:
    a.grid(alpha=0.2)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Discrete HalfCheetah-v4: limb heads (each head sees a body part and controls its joints), 400k steps, seeds 0-4",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
o = f"{R}/figures/limb_heads.png"
fig.savefig(o, dpi=125)
print("wrote", o)
for lab, a, b, *_ in rows:
    print(f"{lab:22s} third-head share {a:.2f}  changes {b:.2f}")
