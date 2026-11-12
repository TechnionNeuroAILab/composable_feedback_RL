"""Plain SEARL DQN vs. SEARL + ConfGate heads (lr gate; 2 left/right heads or one head per observation entry) on bike_env
v2.1 with 3/5/7/10 lanes (8 solutions, pop 4, 480k steps, seeds 0-2, same default evaluation: 3 greedy episodes of the best
individual per epoch). Writes figures/searl_lanes_confgate.png."""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANES = [3, 5, 7, 10]
METHODS = [("SEARL DQN", "results_lanes/lanes{L}", "#2a78d6"),
           ("SEARL + 2 heads (lr gate)", "results_lanes_confgate/2h/lanes{L}", "#eb6834"),
           ("SEARL + per-obs heads (lr gate)", "results_lanes_confgate/perobs/lanes{L}", "#1baf7a")]
GRID = np.linspace(5_000, 480_000, 200)
MAX = 4900


def runs(pattern):
    return [pd.read_csv(p) for p in sorted(glob.glob(pattern + "/seed*/history.csv"))]


fig = plt.figure(figsize=(17, 8.2))
top = [fig.add_subplot(2, 4, i + 1) for i in range(4)]
fin_ax, reach_ax = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
offsets = np.linspace(-0.25, 0.25, len(METHODS))
for k, (name, pat, col) in enumerate(METHODS):
    for j, L in enumerate(LANES):
        hs = runs(pat.format(L=L))
        if not hs:
            continue
        ys = np.stack([np.interp(GRID, h.num_frames, h.test_return.rolling(10, min_periods=1).mean()) for h in hs])
        top[j].plot(GRID, ys.mean(0), color=col, lw=2, label=f"{name} (n={len(hs)})")
        top[j].fill_between(GRID, ys.min(0), ys.max(0), color=col, alpha=0.12)
        finals = [h.test_return[h.num_frames >= 0.9 * h.num_frames.max()].mean() for h in hs]
        reach = []
        for h in hs:
            hit = h.num_frames[h.test_return.rolling(10, min_periods=1).mean() >= 0.95 * MAX]
            reach.append(hit.iloc[0] / 1000 if len(hit) else np.nan)
        for ax, vals in ((fin_ax, finals), (reach_ax, reach)):
            ax.scatter([j + offsets[k]] * len(vals), vals, color=col, s=36, zorder=3, edgecolor="white",
                       label=name if j == 0 else None)
            ax.plot([j + offsets[k] - 0.09, j + offsets[k] + 0.09], [np.nanmean(vals)] * 2, color=col, lw=3)
        print(f"{name:32s} {L:2d} lanes  final {np.round(finals).astype(int)} mean {np.mean(finals):5.0f}   "
              f"95% at {np.round(reach, 1)}k steps")
for j, L in enumerate(LANES):
    a = top[j]
    a.axhline(MAX, color="#c3c2b7", ls="--", lw=1)
    a.set_ylim(-2200, 5300)
    a.set_title(f"{L} lanes", loc="left", fontsize=10)
    a.set_xlabel("environment steps")
    if j == 0:
        a.set_ylabel("return (max 4900)")
        a.legend(fontsize=8, frameon=False, loc="lower right")
fin_ax.axhline(MAX, color="#c3c2b7", ls="--", lw=1)
fin_ax.set_title("Final level per seed (last 10% of steps); bar = mean", loc="left", fontsize=9.5)
reach_ax.set_title("Steps until the 10-epoch mean first reaches 95% of the maximum (thousands)", loc="left", fontsize=9.5)
for ax in (fin_ax, reach_ax):
    ax.set_xticks(range(len(LANES)))
    ax.set_xticklabels([f"{L} lanes" for L in LANES])
fin_ax.legend(fontsize=8.5, frameon=False, loc="lower left")
for a in top + [fin_ax, reach_ax]:
    a.grid(alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("bike_env v2.1, 8 solutions: plain SEARL DQN vs. SEARL + ConfGate heads (population 4, 480k steps, seeds 0-2). Top: greedy test return, 10-epoch mean, band = seed range",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig("figures/searl_lanes_confgate.png", dpi=160)
print("wrote figures/searl_lanes_confgate.png")
