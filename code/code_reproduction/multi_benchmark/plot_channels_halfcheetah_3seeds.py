"""Channel variants on discrete HalfCheetah-v4, seeds 0-2, 1M steps: learning curves (mean over seeds of the 20-episode
running mean, on a common grid), loop vs. loop-off (alpha_i = 1) controls, final level and speed per seed.
Writes figures/channels_halfcheetah_3seeds.png and prints the per-arm numbers."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "results_halfcheetah_1M/"
SEEDS = [0, 1, 2]
GRID = np.linspace(0, 1_000_000, 201)
ARMS = [  # label, dir, colour, linestyle
    ("Rainbow DQN", "../rainbow_dqn/results_1M/HalfCheetah-v4", "#e34948", (0, (6, 1.5, 1, 1.5))),
    ("CleanRL DQN", "cleanrl_dqn", "#52514e", (0, (4, 2))),
    ("current 2 heads (lr gate)", "cleanrl_dqn_confgate", "#898781", (0, (1, 1.5))),
    ("current per-obs heads (lr gate)", "cleanrl_dqn_confgate_perobs", "#e87ba4", (0, (1, 1.5))),
    ("additive, no loop", "channels_additive", "#2a78d6", "-"),
    ("A+C", "channels_ac", "#eb6834", "-"),
    ("F-lite", "channels_flite", "#1baf7a", "-"),
    ("F-lite, loop off", "channels_flite_noloop", "#1baf7a", (0, (3, 1.5))),
    ("F", "channels_f", "#6250d6", "-"),
    ("F, loop off", "channels_f_noloop", "#6250d6", (0, (3, 1.5))),
]
THR = 800


def load(d, s):
    df = pd.read_csv(f"{d if d.startswith('..') else R + d}/seed{s}.csv")
    sm = df.episodic_return.rolling(20, min_periods=1).mean()
    curve = np.interp(GRID, df.global_step, sm)
    fin = float(df.episodic_return[df.global_step >= 900_000].mean())
    hit = df.global_step[sm >= THR]
    return curve, fin, float(sm.max()), (float(hit.iloc[0]) / 1e3 if len(hit) else np.nan)


data = {lab: [load(d, s) for s in SEEDS] for lab, d, _, _ in ARMS}
print(f"{'arm':33s} {'final (last 10%)':>28s}  mean   {'peak':>5s}   steps to {THR} (k)")
for lab, *_ in ARMS:
    f = [x[1] for x in data[lab]]; p = [x[2] for x in data[lab]]; h = [x[3] for x in data[lab]]
    print(f"{lab:33s} {str(np.round(f).astype(int)):>28s} {np.mean(f):6.0f} {np.mean(p):6.0f}   {np.round(h).tolist()}")

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
panels = [("Baselines vs. additive channels", ["Rainbow DQN", "CleanRL DQN", "current 2 heads (lr gate)", "current per-obs heads (lr gate)", "additive, no loop", "A+C"]),
          ("F-lite: loop vs. loop off", ["additive, no loop", "F-lite", "F-lite, loop off"]),
          ("F: loop vs. loop off", ["additive, no loop", "F", "F, loop off"])]
sty = {lab: (c, ls) for lab, _, c, ls in ARMS}
for i, (title, labs) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    for lab in labs:
        c, ls = sty[lab]
        ys = np.array([x[0] for x in data[lab]])
        ax.plot(GRID, ys.mean(0), color=c, ls=ls, lw=2.2 if ls == "-" else 2, label=lab)
        ax.fill_between(GRID, ys.min(0), ys.max(0), color=c, alpha=0.10, lw=0)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlabel("environment steps"); ax.set_ylim(-600, 2600); ax.grid(alpha=0.25)
    if i == 0:
        ax.set_ylabel("training return (20-ep mean; mean of 3 seeds, band = range)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
for j, (title, k) in enumerate([("Final level per seed (last 10% of steps); bar = mean", 1), (f"Steps until the 20-episode mean first reaches {THR} (thousands)", 3)]):
    ax = fig.add_subplot(gs[1, 0:2] if j == 0 else gs[1, 2])
    for x, (lab, _, c, ls) in enumerate(ARMS):
        v = np.array([d[k] for d in data[lab]])
        ax.scatter(np.full(3, x) + np.array([-0.12, 0, 0.12]), v, color=c, s=30, zorder=3,
                   facecolors=c if ls == "-" or isinstance(ls, tuple) and lab.startswith(("Rainbow", "Clean", "current")) is False else c)
        ax.hlines(np.nanmean(v), x - 0.3, x + 0.3, color=c, lw=2.4)
    if j == 0:
        ax.set_xticks(range(len(ARMS)), [a[0].replace(" (lr gate)", "\n(lr gate)").replace(", ", ",\n").replace("current ", "current\n") for a in ARMS], fontsize=7.6)
    else:
        ax.set_xticks(range(len(ARMS)), [a[0].replace(" (lr gate)", "") for a in ARMS], fontsize=7.6, rotation=40, ha="right")
        ax.text(0.99, 0.98, "F-lite seed 2 never reaches 800", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#52514e")
    ax.set_title(title, loc="left", fontsize=10.5); ax.grid(alpha=0.25, axis="y")
fig.suptitle("Discrete HalfCheetah-v4, 1M steps, seeds 0-2: feature-specific channel variants (training-episode return; Rainbow = Kaixhin port, noisy nets, no epsilon)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig("figures/channels_halfcheetah_3seeds.png", dpi=150)
print("wrote figures/channels_halfcheetah_3seeds.png")

# loop statistics: cancellation and alpha spread, per arm, last half of training
print("\nchannel stats, second half of training (mean over seeds): alpha / snr / |err| per channel, cancellation")
for lab, d, *_ in ARMS[4:]:
    rows = []
    for s in SEEDS:
        p = f"{R}{d}/seed{s}_channels.csv"
        if os.path.exists(p):
            c = pd.read_csv(p); rows.append(c[c.global_step >= 500_000])
    c = pd.concat(rows)
    g = c.groupby("channel")[["alpha", "snr", "err_abs"]].mean().round(3)
    print(f"-- {lab}: cancellation {c.cancellation.mean():.2f}\n{g.to_string()}")
