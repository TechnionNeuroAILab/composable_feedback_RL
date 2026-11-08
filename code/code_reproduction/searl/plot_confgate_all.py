#!/usr/bin/env python
"""All variants on CartPole-v1 and discrete HalfCheetah-v4, 1M steps. One figure per environment, two panels:
  left  = SEARL family (SEARL DQN, SEARL + 2 heads, SEARL + per-observation heads) with Rainbow for reference
  right = single-agent family (CleanRL DQN, + 2 heads, + per-observation heads) with Rainbow for reference
Curves: SEARL = greedy test return of the best individual (5-epoch mean); others = training-episode return (10-episode mean).
Mean +/- 1 std over seeds on a common grid; a series is cut where any of its seeds has ended. Missing series are skipped.
Also prints a table of the final level (mean over the last 10% of steps, per seed) and m / lr_eff for the gated runs.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CR = os.path.join(ROOT, "code", "code_reproduction")
MB = os.path.join(CR, "multi_benchmark")
RES = os.path.join(ROOT, "results")
OUTDIR = os.path.join(RES, "searl_confgate_summary")
os.makedirs(OUTDIR, exist_ok=True)
FRAMES = 1_000_000

COL = {"B": "#1f77b4", "A": "#d95f02", "Aobs": "#2ca02c", "base": "#7b3fa0", "S2": "#d95f02", "Sobs": "#2ca02c", "rb": "#222222"}


def episodes(pattern, smooth=10):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        if len(d):
            out.append((d["global_step"].values.astype(float), d["episodic_return"].rolling(smooth, min_periods=1).mean().values))
    return out


def searl(root, cfg, seeds):
    out = []
    for s in seeds:
        f = f"{root}/{cfg}/seed{s}/history.csv"
        if os.path.exists(f):
            h = pd.read_csv(f)
            out.append((h["num_frames"].values.astype(float), h["test_return"].rolling(5, min_periods=1).mean().values))
    return out


def band(curves, grid):
    ys = np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    last = min(c[0].max() for c in curves)
    with np.errstate(all="ignore"):
        m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
    return np.where(grid <= last, m, np.nan), np.where(grid <= last, s, np.nan)


def final_level(curves, frac=0.1):
    """Per seed: mean of the (smoothed) curve over the last `frac` of the budget."""
    return [float(np.mean(y[x >= (1 - frac) * FRAMES])) for x, y in curves if (x >= (1 - frac) * FRAMES).any()]


def build(env):
    if env == "cartpole":
        sroot = f"{RES}/searl_confgate_cartpole_1M"; ss = range(1, 11); sp = range(1, 4)
        cl = f"{MB}/results_cartpole"
        series = {
            "SEARL DQN": ("B", "-", searl(sroot, "B_full", ss)),
            "SEARL + 2 heads": ("A", "-", searl(sroot, "A_full", ss)),
            "SEARL + per-obs heads (4)": ("Aobs", "-", searl(sroot, "A_perobs", sp)),
            "CleanRL DQN": ("base", "--", episodes(f"{cl}/cleanrl_dqn/seed?.csv")),
            "DQN + 2 heads": ("S2", "--", episodes(f"{cl}/cleanrl_dqn_confgate/seed?.csv")),
            "DQN + per-obs heads (4)": ("Sobs", "--", episodes(f"{cl}/cleanrl_dqn_confgate_perobs/seed?.csv")),
            "Rainbow DQN": ("rb", ":", episodes(f"{CR}/rainbow_dqn/results/CartPole-v1/seed?.csv")),
        }
        title, ylim = "CartPole-v1", (0, 520)
    else:
        sroot = f"{RES}/searl_confgate_halfcheetah_1M"; ss = range(10); sp = range(3)
        cl = f"{MB}/results_halfcheetah_1M"
        series = {
            "SEARL DQN": ("B", "-", searl(sroot, "B_full", ss)),
            "SEARL + 2 heads": ("A", "-", searl(sroot, "A_full", ss)),
            "SEARL + per-obs heads (17)": ("Aobs", "-", searl(sroot, "A_perobs", sp)),
            "CleanRL DQN": ("base", "--", episodes(f"{cl}/cleanrl_dqn/seed?.csv")),
            "DQN + 2 heads": ("S2", "--", episodes(f"{cl}/cleanrl_dqn_confgate/seed?.csv")),
            "DQN + per-obs heads (17)": ("Sobs", "--", episodes(f"{cl}/cleanrl_dqn_confgate_perobs/seed?.csv")),
            "Rainbow DQN": ("rb", ":", episodes(f"{CR}/rainbow_dqn/results_1M/HalfCheetah-v4/seed?.csv")),
        }
        title, ylim = "HalfCheetah-v4 (discrete, 18 outputs)", None
    return title, ylim, {k: v for k, v in series.items() if v[2]}


grid = np.linspace(10_000, FRAMES, 500)
for env in sys.argv[1:] or ["cartpole", "halfcheetah"]:
    title, ylim, series = build(env)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    panels = [("SEARL family", ["SEARL DQN", "SEARL + 2 heads", "SEARL + per-obs heads", "Rainbow DQN"]),
              ("Single-agent DQN family", ["CleanRL DQN", "DQN + 2 heads", "DQN + per-obs heads", "Rainbow DQN"])]
    for ax, (ptitle, keys) in zip(axes, panels):
        for name, (c, ls, curves) in series.items():
            if not any(name.startswith(k) for k in keys):
                continue
            m, s = band(curves, grid)
            ax.plot(grid, m, color=COL[c], ls=ls, lw=2.1, label=f"{name} (n={len(curves)})")
            if name != "Rainbow DQN":
                ax.fill_between(grid, m - s, m + s, color=COL[c], alpha=0.09, lw=0)
        ax.set_title(f"{title}: {ptitle}", fontsize=11)
        ax.set_xlabel("environment steps"); ax.grid(alpha=0.25); ax.legend(loc="lower right", frameon=False, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("episode return")
    fig.text(0.01, 0.005, "SEARL: greedy test return of best individual, 5-epoch mean. DQN / Rainbow: training-episode return, 10-ep mean. "
             "Mean ± 1 std over seeds; a curve stops where any seed ends.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = f"{OUTDIR}/all_variants_{env}_1M.png"; fig.savefig(out, dpi=150); print("wrote", out)

    print(f"\n== {title}: final level (mean over last 10% of steps), per-seed mean ± sd, n seeds")
    for name, (c, ls, curves) in series.items():
        v = final_level(curves)
        print(f"  {name:30s} {np.mean(v):8.0f} ± {np.std(v):5.0f}   n={len(v)}")
