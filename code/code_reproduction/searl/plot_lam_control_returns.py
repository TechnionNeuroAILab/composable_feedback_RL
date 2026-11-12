"""Greedy return: real ConfGate-mix SEARL runs (act with Q_mix) vs. the neutral-lambda control (acts with Q_full only).
Writes results/searl_confgate_summary/lambda_control_returns.png and prints final levels (mean of last 10% of evals)."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results")
REAL = {"CartPole-v1": "searl_confgate_cartpole_1M", "HalfCheetah-v4": "searl_confgate_halfcheetah_1M"}
ARMS = [("M_2h", "2 heads, mix only"), ("MG_2h", "2 heads, mix + lr gate"),
        ("M_obs", "per-obs heads, mix only"), ("MG_obs", "per-obs heads, mix + lr gate")]
REAL_C, CTRL_C, RB_C = "#2a78d6", "#898781", "#e34948"
RB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../rainbow_dqn")
RB_DIR = {"CartPole-v1": f"{RB}/results/CartPole-v1", "HalfCheetah-v4": f"{RB}/results_1M/HalfCheetah-v4"}
GRID = np.linspace(0, 1_000_000, 201)


def rainbow(env):
    """Kaixhin Rainbow port: training-episode returns, each seed interpolated on a common grid (binned means)."""
    ys, finals = [], []
    for p in sorted(glob.glob(f"{RB_DIR[env]}/seed*.csv")):
        d = pd.read_csv(p)
        b = np.digitize(d.global_step.values, GRID)
        m = d.groupby(b).episodic_return.mean()
        y = np.interp(GRID, GRID[np.clip(m.index.values - 1, 0, len(GRID) - 1)], m.values)
        ys.append(y)
        finals.append(float(d.episodic_return.values[d.global_step.values >= 900_000].mean()))
    return np.mean(ys, 0), finals


def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if os.path.exists(os.path.join(os.path.dirname(p), "history.csv")):
            out.append(pd.read_csv(p))
    return out


def final(d):
    n = max(1, len(d) // 10)
    return float(d.greedy_return.values[-n:].mean())


fig, axes = plt.subplots(2, 5, figsize=(19, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]})
for r, (env, real_dir) in enumerate(REAL.items()):
    finals = []
    rb_curve, rb_fin = rainbow(env)
    print(f"{env:15s} Rainbow final (last 10% of training episodes) {np.round(rb_fin, 0)} mean {np.mean(rb_fin):7.1f}")
    for c, (arm, title) in enumerate(ARMS):
        ax = axes[r, c]
        ax.plot(GRID, rb_curve, color=RB_C, lw=2, ls="--", label=f"Rainbow DQN mean (n={len(rb_fin)})")
        real = load(f"{RES}/{real_dir}/{arm}/seed*/eval.csv")
        ctrl = load(f"{RES}/searl_confgate_lam_neutral/{env}/{arm}/seed*/eval.csv")
        for runs, col, lab in [(ctrl, CTRL_C, "control: acts with Q_full"), (real, REAL_C, "real: acts with Q_mix")]:
            if not runs:
                continue
            for d in runs:
                ax.plot(d.global_step, d.greedy_return, color=col, lw=0.6, alpha=0.35)
            n = min(len(d) for d in runs)
            m = np.mean([d.greedy_return.values[:n] for d in runs], 0)
            ax.plot(runs[0].global_step.values[:n], pd.Series(m).rolling(5, min_periods=1).mean(), color=col, lw=2.4,
                    label=f"{lab} (n={len(runs)})")
        ax.set_title(f"{env}: {title}", fontsize=9.5, loc="left")
        ax.grid(alpha=0.25)
        if r == 1:
            ax.set_xlabel("environment steps")
        if c == 0:
            ax.set_ylabel("greedy return")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        fr, fc = [final(d) for d in real], [final(d) for d in ctrl]
        finals.append((arm, fr, fc))
        print(f"{env:15s} {arm:7s} real {np.round(fr, 0)} mean {np.mean(fr):7.1f} | control {np.round(fc, 0)} mean {np.mean(fc):7.1f}")
    ax = axes[r, 4]
    for i, (arm, fr, fc) in enumerate(finals):
        ax.scatter(np.full(len(fc), i - 0.15) + np.random.uniform(-0.05, 0.05, len(fc)), fc, color=CTRL_C, s=22)
        ax.scatter(np.full(len(fr), i + 0.15) + np.random.uniform(-0.05, 0.05, len(fr)), fr, color=REAL_C, s=22)
        ax.hlines(np.mean(fc), i - 0.3, i - 0.0, color=CTRL_C, lw=2)
        ax.hlines(np.mean(fr), i + 0.0, i + 0.3, color=REAL_C, lw=2)
    ax.axhline(np.mean(rb_fin), color=RB_C, lw=1.6, ls="--", label="Rainbow DQN mean")
    if r == 0:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xticks(range(4), [a for a, _ in ARMS], fontsize=8.5)
    ax.set_title("final level (last 10%)", fontsize=9.5, loc="left")
    ax.grid(alpha=0.25, axis="y")
np.random.seed(0)
fig.text(0.01, 0.005, "Thin lines = seeds, thick = mean (5-eval smoothing). Control = --neutral-lam: lambda mutates but the "
         "agent acts with Q_full, so it equals the same arm without the mix. Without the lr gate, M_2h and M_obs controls are the same runs. "
         "Red dashed = Rainbow DQN (Kaixhin port) mean of training-episode returns (noisy nets, no epsilon).",
         fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = f"{RES}/searl_confgate_summary/lambda_control_returns.png"
fig.savefig(out, dpi=150)
print("wrote", out)
