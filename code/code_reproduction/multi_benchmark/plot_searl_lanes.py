"""SEARL DQN (searl_discrete.py --task bike) on bike_env v2.1 with 3/5/7/10 lanes, 8 solutions, pop 4, 480k steps, 3 seeds.
Writes figures/searl_lanes.png: greedy test return over training, final level per seed, and the best individual's lr."""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANES = [3, 5, 7, 10]
COL = {3: "#2a78d6", 5: "#eb6834", 7: "#1baf7a", 10: "#6250d6"}   # categorical: blue, orange, aqua, violet
GRID = np.linspace(5_000, 480_000, 200)
MAX = 4900


def runs(lanes):
    return [pd.read_csv(p) for p in sorted(glob.glob(f"results_lanes/lanes{lanes}/seed*/history.csv"))]


fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [2.2, 1, 1.4]})
rows = []
for L in LANES:
    hs = runs(L)
    ys = np.stack([np.interp(GRID, h.num_frames, h.test_return.rolling(10, min_periods=1).mean()) for h in hs])
    m, s = ys.mean(0), ys.std(0)
    a1.plot(GRID, m, color=COL[L], lw=2.2, label=f"{L} lanes")
    a1.fill_between(GRID, m - s, m + s, color=COL[L], alpha=0.15)
    finals = [h.test_return[h.num_frames >= 0.9 * h.num_frames.max()].mean() for h in hs]
    a2.scatter([L] * len(finals), finals, color=COL[L], s=40, zorder=3, edgecolor="white")
    a2.plot([L - 0.6, L + 0.6], [np.mean(finals)] * 2, color=COL[L], lw=3)
    reach = []
    for h in hs:
        sm = h.test_return.rolling(10, min_periods=1).mean()
        hit = h.num_frames[sm >= 0.95 * MAX]
        reach.append(int(hit.iloc[0]) if len(hit) else np.nan)
    lrs = np.stack([np.interp(GRID, h.num_frames, h.best_lr) for h in hs])
    a3.plot(GRID, np.exp(np.log(lrs).mean(0)), color=COL[L], lw=2.2, label=f"{L} lanes")
    rows.append((L, finals, reach, [h.best_lr.iloc[-1] for h in hs]))

a1.axhline(MAX, color="#c3c2b7", ls="--", lw=1)
a1.set_ylim(-2200, 5300)
a3.legend(frameon=False, fontsize=8.5)
a1.set_title("Greedy test return of the best individual (10-epoch mean; mean ± sd of 3 seeds)", loc="left", fontsize=9.5)
a1.set_xlabel("environment steps (agent decisions)")
a1.set_ylabel("return (max 4900)")
a1.legend(frameon=False, fontsize=9, loc="lower right")
a2.set_title("Final level per seed (last 10% of steps)", loc="left", fontsize=9.5)
a2.set_xticks(LANES)
a2.set_xlabel("lanes")
a2.axhline(MAX, color="#c3c2b7", ls="--", lw=1)
a3.set_yscale("log")
a3.set_title("lr of the best individual (geometric mean of seeds)", loc="left", fontsize=9.5)
a3.set_xlabel("environment steps")
for a in (a1, a2, a3):
    a.grid(alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("SEARL DQN on bike_env v2.1: number of lanes (8 solutions, population 4, 480k steps)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig("figures/searl_lanes.png", dpi=170)
print("wrote figures/searl_lanes.png")
for L, f, r, lr in rows:
    print(f"{L:2d} lanes  final {np.round(f).astype(int)} mean {np.mean(f):.0f}   steps to 95% {r}   final best lr {np.round(lr, 6)}")
