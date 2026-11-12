"""lambda over training: real ConfGate-mix SEARL runs vs. the neutral-lambda control (lambda mutated and inherited but never
used to act, so selection cannot see it). Writes results/searl_confgate_summary/lambda_control.png and prints, per arm, the
final lambda (last 10% of training phases) of the real seeds against the spread of the control seeds."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results")
REAL = {"CartPole-v1": "searl_confgate_cartpole_1M", "HalfCheetah-v4": "searl_confgate_halfcheetah_1M"}
ARMS = [("M_2h", "2 heads, mix only"), ("MG_2h", "2 heads, mix + lr gate"),
        ("M_obs", "per-obs heads, mix only"), ("MG_obs", "per-obs heads, mix + lr gate")]
BINS = np.linspace(0, 1, 41)
REAL_C, CTRL_C = "#2a78d6", "#898781"


def curves(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if not os.path.exists(os.path.join(os.path.dirname(p), "history.csv")):
            continue   # run not finished
        lam = pd.read_csv(p).lam.values
        t = np.arange(len(lam)) / len(lam)
        idx = np.digitize(t, BINS) - 1
        out.append(np.array([lam[idx == b].mean() for b in range(len(BINS) - 1)]))
    return out


fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
x = (BINS[:-1] + BINS[1:]) / 2
rows = []
for r, (env, real_dir) in enumerate(REAL.items()):
    for c, (arm, title) in enumerate(ARMS):
        ax = axes[r, c]
        real = curves(f"{RES}/{real_dir}/{arm}/seed*/confgate_log.csv")
        ctrl = curves(f"{RES}/searl_confgate_lam_neutral/{env}/{arm}/seed*/confgate_log.csv")
        for y in ctrl:
            ax.plot(x, y, color=CTRL_C, lw=0.8, alpha=0.6)
        for y in real:
            ax.plot(x, y, color=REAL_C, lw=0.9, alpha=0.6)
        if ctrl:
            ax.plot(x, np.mean(ctrl, 0), color=CTRL_C, lw=2.4, label=f"control: selection blind to lambda (n={len(ctrl)})")
        if real:
            ax.plot(x, np.mean(real, 0), color=REAL_C, lw=2.4, label=f"real runs (n={len(real)})")
        ax.axhline(0.2, color="#c3c2b7", ls="--", lw=1)
        ax.set_title(f"{env}: {title}", fontsize=9.5, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        if r == 1:
            ax.set_xlabel("fraction of training")
        if c == 0:
            ax.set_ylabel("lambda (mean of 4 individuals)")
        if r == 0 and c == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        fin = lambda ys: [float(np.nanmean(y[-4:])) for y in ys]   # last 10% of training
        rows.append((env, arm, fin(real), fin(ctrl)))
fig.text(0.01, 0.005, "Thin lines = seeds, thick = mean. Dashed = initial lambda 0.2. Control: same code with --neutral-lam "
         "(lambda mutates and is inherited but is not used to act). Without the lr gate, the control is identical for 2-head and "
         "per-obs heads.", fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = f"{RES}/searl_confgate_summary/lambda_control.png"
fig.savefig(out, dpi=160)
print("wrote", out)
for env, arm, re_, ct in rows:
    ct = np.array(ct)
    print(f"{env:15s} {arm:7s} real final lam {np.round(re_, 2)}  mean {np.mean(re_):.2f} | control mean {ct.mean():.2f} "
          f"range [{ct.min():.2f}, {ct.max():.2f}] (n={len(ct)})")
