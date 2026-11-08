#!/usr/bin/env python
"""ConfGate-mix (the aux heads also take part in acting) on CartPole-v1 and discrete HalfCheetah-v4, 1M steps, 3 seeds.

Writes to results/searl_confgate_summary/:
  mix_curves_{cartpole,halfcheetah}.png  learning curves; left = 2 heads, right = one head per observation.
                                         SEARL DQN, lr gate only, mix only, mix + lr gate (+ Rainbow); thin lines = seeds.
  mix_lambda.png                         lam (weight of the heads in the acting policy) over training, SEARL runs.
  mix_final_levels.png                   final level per seed for every variant, SEARL and single-agent.
Colour = arm (same in every panel). SEARL curves: greedy test return of the best individual, 5-epoch mean.
Single-agent / Rainbow: training-episode return, 10-episode mean. Final level = mean over the last 10% of steps.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CR = os.path.join(ROOT, "code", "code_reproduction")
MB = os.path.join(CR, "multi_benchmark")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "searl_confgate_summary")
os.makedirs(OUT, exist_ok=True)
FRAMES = 1_000_000
SMOOTH = 25   # epochs (about 35k steps) for the SEARL curves in the learning-curve figure; the final levels use the raw evaluations
# Okabe-Ito: colour = what the heads do (arm), identical in every panel
ARM = {"plain": ("#0072B2", "no heads"), "gate": ("#E69F00", "lr gate only"),
       "mix": ("#D55E00", "heads act (mix only)"), "mixgate": ("#009E73", "heads act + lr gate"), "rb": ("#222222", "Rainbow")}
ENV = {
    "cartpole": dict(title="CartPole-v1", sroot=f"{RES}/searl_confgate_cartpole_1M", seeds=[1, 2, 3], sa=f"{MB}/results_cartpole",
                     rb=f"{CR}/rainbow_dqn/results/CartPole-v1/seed?.csv", ylim=(0, 520)),
    "halfcheetah": dict(title="HalfCheetah-v4 (discrete)", sroot=f"{RES}/searl_confgate_halfcheetah_1M", seeds=[0, 1, 2],
                        sa=f"{MB}/results_halfcheetah_1M", rb=f"{CR}/rainbow_dqn/results_1M/HalfCheetah-v4/seed?.csv", ylim=None),
}
SEARL_CFG = {  # (heads, arm) -> run folder
    ("2h", "plain"): "B_full", ("2h", "gate"): "A_full", ("2h", "mix"): "M_2h", ("2h", "mixgate"): "MG_2h",
    ("obs", "plain"): "B_full", ("obs", "gate"): "A_perobs", ("obs", "mix"): "M_obs", ("obs", "mixgate"): "MG_obs"}
SA_DIR = {("2h", "plain"): "cleanrl_dqn", ("2h", "gate"): "cleanrl_dqn_confgate", ("2h", "mix"): "cleanrl_dqn_confgate_mix_2h",
          ("2h", "mixgate"): "cleanrl_dqn_confgate_mixgate_2h", ("obs", "plain"): "cleanrl_dqn",
          ("obs", "gate"): "cleanrl_dqn_confgate_perobs", ("obs", "mix"): "cleanrl_dqn_confgate_mix_obs",
          ("obs", "mixgate"): "cleanrl_dqn_confgate_mixgate_obs"}


def searl_curves(env, cfg):
    e = ENV[env]; out = []
    for s in e["seeds"]:
        f = f"{e['sroot']}/{cfg}/seed{s}/history.csv"
        if os.path.exists(f):
            h = pd.read_csv(f).dropna(subset=["test_return"])
            out.append((h.num_frames.values.astype(float), h.test_return.rolling(SMOOTH, min_periods=1).mean().values))
    return out


def episode_curves(pattern, seeds=None, smooth=10):
    out = []
    for f in sorted(glob.glob(pattern)):
        if seeds is not None and int(os.path.basename(f)[4]) not in seeds:
            continue
        d = pd.read_csv(f)
        if len(d):
            out.append((d["global_step"].values.astype(float), d["episodic_return"].rolling(smooth, min_periods=1).mean().values))
    return out


def on_grid(curves, grid):
    return np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])


def final_level(curves, frac=0.1):
    return [float(np.mean(y[x >= (1 - frac) * FRAMES])) for x, y in curves if (x >= (1 - frac) * FRAMES).any()]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, lw=0.6)
    ax.tick_params(labelsize=8)


# ----------------------------------------------------------------------------------------------- 1. learning curves
grid = np.linspace(10_000, FRAMES, 400)
for env, e in ENV.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    rb = episode_curves(e["rb"])
    for ax, (hd, ttl) in zip(axes, [("2h", "2 aux heads (two observation groups)"), ("obs", "one aux head per observation entry")]):
        for arm in ("plain", "gate", "mix", "mixgate"):
            c = searl_curves(env, SEARL_CFG[(hd, arm)])
            if not c:
                continue
            Y = on_grid(c, grid)
            for y in Y:
                ax.plot(grid, y, color=ARM[arm][0], lw=0.5, alpha=0.18)
            ax.plot(grid, np.nanmean(Y, 0), color=ARM[arm][0], lw=2.2, label=f"{ARM[arm][1]} (n={len(c)})")
        if rb:
            ax.plot(grid, np.nanmean(on_grid(rb, grid), 0), color=ARM["rb"][0], lw=1.6, ls=":", label=f"Rainbow (n={len(rb)})")
        ax.set_title(f"{e['title']} - SEARL, {ttl}", fontsize=10.5)
        ax.set_xlabel("environment steps"); style(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8.5)
        if e["ylim"]:
            ax.set_ylim(*e["ylim"])
    axes[0].set_ylabel("greedy test return (SEARL)")
    fig.text(0.01, 0.005, "Thick = mean of 3 paired seeds, thin = single seeds. SEARL: greedy test return of the best individual (25-epoch mean, about 35k steps). "
             "Rainbow: training-episode return (10-ep mean).", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = f"{OUT}/mix_curves_{env}.png"; fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# ----------------------------------------------------------------------------------------------- 2. lambda over time
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), sharey=True)
bins = np.linspace(0, FRAMES, 41)
for r, (env, e) in enumerate(ENV.items()):
    for c, (hd, ttl) in enumerate([("2h", "2 heads"), ("obs", "per-observation heads")]):
        ax = axes[r, c]
        for arm in ("mix", "mixgate"):
            runs = []
            for s in e["seeds"]:
                d = f"{e['sroot']}/{SEARL_CFG[(hd, arm)]}/seed{s}"
                if not os.path.exists(f"{d}/confgate_log.csv"):
                    continue
                lg, hi = pd.read_csv(f"{d}/confgate_log.csv"), pd.read_csv(f"{d}/history.csv")
                fr = hi.num_frames.values[np.minimum(np.arange(len(lg)) // 4, len(hi) - 1)]   # 4 individuals trained per epoch
                idx = np.digitize(fr, bins) - 1
                y = np.array([lg.lam.values[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)])
                runs.append(y)
                ax.plot((bins[:-1] + bins[1:]) / 2, y, color=ARM[arm][0], lw=0.6, alpha=0.3)
            if runs:
                ax.plot((bins[:-1] + bins[1:]) / 2, np.nanmean(runs, 0), color=ARM[arm][0], lw=2.2, label=f"{ARM[arm][1]} (n={len(runs)})")
        ax.axhline(0.2, color="#888", lw=0.8, ls="--")
        ax.set_title(f"{e['title']} - {ttl}", fontsize=10.5); style(ax); ax.set_ylim(0, 1)
        if r == 1:
            ax.set_xlabel("environment steps")
        if c == 0:
            ax.set_ylabel("lam = weight of the heads when acting")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.text(0.01, 0.005, "lam per training phase averaged over the 4 individuals (not only the best). Thick = mean of 3 seeds, thin = seeds. "
         "Dashed line = initial lam 0.2. lam = 0: heads ignored; 1: heads alone choose the action.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
p = f"{OUT}/mix_lambda.png"; fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# ----------------------------------------------------------------------------------------------- 3. final levels per seed
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
ROWS = [("SEARL, 2 heads", "searl", "2h"), ("SEARL, per-obs heads", "searl", "obs"),
        ("single-agent DQN, 2 heads", "sa", "2h"), ("single-agent DQN, per-obs heads", "sa", "obs")]
table = []
for ax, (env, e) in zip(axes, ENV.items()):
    y = 0; yt, yl = [], []
    for gname, kind, hd in ROWS:
        for arm in ("plain", "gate", "mix", "mixgate"):
            if kind == "searl":
                cv = searl_curves(env, SEARL_CFG[(hd, arm)])
                vals = [float(np.mean(h.test_return[h.num_frames >= 0.9 * FRAMES])) for h in
                        [pd.read_csv(f"{e['sroot']}/{SEARL_CFG[(hd, arm)]}/seed{s}/history.csv").dropna(subset=["test_return"]) for s in e["seeds"]]]
            else:
                vals = final_level(episode_curves(f"{e['sa']}/{SA_DIR[(hd, arm)]}/seed?.csv", seeds=[0, 1, 2]))
            if not vals:
                y += 1; continue
            ax.scatter(vals, [y] * len(vals), s=34, color=ARM[arm][0], zorder=3, edgecolor="white", linewidth=0.8)
            ax.plot([np.mean(vals)] * 2, [y - 0.32, y + 0.32], color=ARM[arm][0], lw=2.4, zorder=2)
            yt.append(y); yl.append(f"{gname}: {ARM[arm][1]}")
            table.append((env, gname, ARM[arm][1], np.mean(vals), np.std(vals), len(vals)))
            y += 1
        y += 0.9
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5); ax.invert_yaxis()
    ax.set_title(f"{e['title']}: dot = seed, bar = mean", fontsize=10.5)
    ax.set_xlabel("return over the last 10% of steps"); style(ax); ax.grid(axis="y", alpha=0)
fig.tight_layout()
p = f"{OUT}/mix_final_levels.png"; fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)
pd.DataFrame(table, columns=["env", "family", "arm", "mean", "std", "n"]).round(1).to_csv(f"{OUT}/mix_final_levels.csv", index=False)
