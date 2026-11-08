#!/usr/bin/env python
"""Builds the LaTeX table fragments and copies the figures for paper/searl_confgate_results.tex.

Final level, one definition everywhere: per seed, the mean over the last 10% of the budget of
  SEARL            the raw greedy test return of the best individual (one evaluation per epoch),
  DQN / Rainbow    the training-episode return (episodes that end in the last 10% of steps).
Tables report mean +- sd over seeds (sd = population sd, ddof 0) and n.
Writes paper/searl_confgate_tables/*.tex and paper/figures/searl_confgate/*.png.
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CR, RES = os.path.join(ROOT, "code", "code_reproduction"), os.path.join(ROOT, "results")
MB = os.path.join(CR, "multi_benchmark")
PAPER = os.path.join(ROOT, "paper")
TAB, FIG = os.path.join(PAPER, "searl_confgate_tables"), os.path.join(PAPER, "figures", "searl_confgate")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 1_000_000


def searl_final(root, cfg, seeds):
    out = []
    for s in seeds:
        f = f"{root}/{cfg}/seed{s}/history.csv"
        if os.path.exists(f):
            h = pd.read_csv(f).dropna(subset=["test_return"])
            out.append(float(h[h.num_frames >= 0.9 * N].test_return.mean()))
    return out


def ep_final(pattern, seeds=None):
    out = []
    for f in sorted(glob.glob(pattern)):
        if seeds is not None and int(os.path.basename(f)[4]) not in seeds:
            continue
        d = pd.read_csv(f)
        x = d[d.global_step >= 0.9 * N].episodic_return
        if len(x):
            out.append(float(x.mean()))
    return out


def cell(v, bold=False):
    if not v:
        return "--"
    s = f"{np.mean(v):.0f} $\\pm$ {np.std(v):.0f}"
    return (f"\\textbf{{{s}}}" if bold else s) + f" ({len(v)})"


CP = dict(sroot=f"{RES}/searl_confgate_cartpole_1M", s10=range(1, 11), s3=[1, 2, 3], sa=f"{MB}/results_cartpole",
          rb=f"{CR}/rainbow_dqn/results/CartPole-v1/seed?.csv")
HC = dict(sroot=f"{RES}/searl_confgate_halfcheetah_1M", s10=range(10), s3=[0, 1, 2], sa=f"{MB}/results_halfcheetah_1M",
          rb=f"{CR}/rainbow_dqn/results_1M/HalfCheetah-v4/seed?.csv")

# ---- Table 1: all variants, all available seeds
rows = [("SEARL DQN", lambda e: searl_final(e["sroot"], "B_full", e["s10"])),
        ("SEARL + 2 heads (lr gate)", lambda e: searl_final(e["sroot"], "A_full", e["s10"])),
        ("SEARL + per-obs heads (lr gate)", lambda e: searl_final(e["sroot"], "A_perobs", e["s3"])),
        ("CleanRL DQN", lambda e: ep_final(f"{e['sa']}/cleanrl_dqn/seed?.csv")),
        ("DQN + 2 heads (lr gate)", lambda e: ep_final(f"{e['sa']}/cleanrl_dqn_confgate/seed?.csv")),
        ("DQN + per-obs heads (lr gate)", lambda e: ep_final(f"{e['sa']}/cleanrl_dqn_confgate_perobs/seed?.csv")),
        ("Rainbow", lambda e: ep_final(e["rb"]))]
L = ["\\begin{tabular}{lrr}", "\\toprule", "Variant & CartPole-v1 & HalfCheetah-v4 \\\\", "\\midrule"]
for i, (name, fn) in enumerate(rows):
    if i in (3, 6):
        L.append("\\midrule")
    L.append(f"{name} & {cell(fn(CP))} & {cell(fn(HC))} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{TAB}/main_final.tex", "w").write("\n".join(L) + "\n")

# ---- paired SEARL + 2 heads vs SEARL DQN (10 seeds)
paired = {}
for env, e in (("CartPole-v1", CP), ("HalfCheetah-v4", HC)):
    a, b = np.array(searl_final(e["sroot"], "A_full", e["s10"])), np.array(searl_final(e["sroot"], "B_full", e["s10"]))
    d = a - b; rng = np.random.default_rng(0)
    bs = np.percentile([rng.choice(d, len(d)).mean() for _ in range(20000)], [2.5, 97.5])
    paired[env] = (a.mean(), b.mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d), *bs)
L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
     "Task & SEARL + 2 heads & SEARL DQN & mean diff $\\pm$ s.e. & seeds ahead & bootstrap 95\\% CI \\\\", "\\midrule"]
for env, (ma, mb, md, se, k, n, lo, hi) in paired.items():
    L.append(f"{env} & {ma:.0f} & {mb:.0f} & {md:+.0f} $\\pm$ {se:.0f} & {k}/{n} & [{lo:+.0f}, {hi:+.0f}] \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{TAB}/paired.tex", "w").write("\n".join(L) + "\n")

# ---- Table: heads-act (mix), 3 paired seeds
CFG = {("2h", "gate"): "A_full", ("2h", "mix"): "M_2h", ("2h", "mixgate"): "MG_2h",
       ("obs", "gate"): "A_perobs", ("obs", "mix"): "M_obs", ("obs", "mixgate"): "MG_obs"}
SA = {("2h", "gate"): "cleanrl_dqn_confgate", ("2h", "mix"): "cleanrl_dqn_confgate_mix_2h", ("2h", "mixgate"): "cleanrl_dqn_confgate_mixgate_2h",
      ("obs", "gate"): "cleanrl_dqn_confgate_perobs", ("obs", "mix"): "cleanrl_dqn_confgate_mix_obs", ("obs", "mixgate"): "cleanrl_dqn_confgate_mixgate_obs"}
ARM = {"gate": "lr gate only", "mix": "heads act (mix only)", "mixgate": "heads act + lr gate"}
L = ["\\begin{tabular}{llrr}", "\\toprule", "Family & Variant & CartPole-v1 & HalfCheetah-v4 \\\\", "\\midrule",
     f"SEARL & no heads (SEARL DQN) & {cell(searl_final(CP['sroot'], 'B_full', CP['s3']))} & {cell(searl_final(HC['sroot'], 'B_full', HC['s3']))} \\\\"]
for hd, hn in (("2h", "2 heads"), ("obs", "per-obs heads")):
    for arm in ("gate", "mix", "mixgate"):
        L.append(f"SEARL & {hn}, {ARM[arm]} & {cell(searl_final(CP['sroot'], CFG[(hd, arm)], CP['s3']))} & "
                 f"{cell(searl_final(HC['sroot'], CFG[(hd, arm)], HC['s3']))} \\\\")
L.append("\\midrule")
L.append(f"single-agent DQN & no heads (CleanRL DQN) & {cell(ep_final(CP['sa'] + '/cleanrl_dqn/seed?.csv', [0, 1, 2]))} & "
         f"{cell(ep_final(HC['sa'] + '/cleanrl_dqn/seed?.csv', [0, 1, 2]))} \\\\")
for hd, hn in (("2h", "2 heads"), ("obs", "per-obs heads")):
    for arm in ("gate", "mix", "mixgate"):
        L.append(f"single-agent DQN & {hn}, {ARM[arm]} & {cell(ep_final(CP['sa'] + '/' + SA[(hd, arm)] + '/seed?.csv', [0, 1, 2]))} & "
                 f"{cell(ep_final(HC['sa'] + '/' + SA[(hd, arm)] + '/seed?.csv', [0, 1, 2]))} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{TAB}/mix_final.tex", "w").write("\n".join(L) + "\n")

# ---- lambda: early (first 10% of training phases) -> late (last 10%), mean over seeds and the 4 individuals
L = ["\\begin{tabular}{llrrl}", "\\toprule", "Task & Variant & $\\lambda$ first 10\\% & $\\lambda$ last 10\\% & final $\\lambda$ per seed \\\\", "\\midrule"]
for env, e, tn in (("cp", CP, "CartPole-v1"), ("hc", HC, "HalfCheetah-v4")):
    for hd, hn in (("2h", "2 heads"), ("obs", "per-obs")):
        for arm in ("mix", "mixgate"):
            early, late = [], []
            for s in e["s3"]:
                lg = pd.read_csv(f"{e['sroot']}/{CFG[(hd, arm)]}/seed{s}/confgate_log.csv")
                k = len(lg); early.append(lg.lam.iloc[:k // 10].mean()); late.append(lg.lam.iloc[-k // 10:].mean())
            L.append(f"{tn} & {hn}, {ARM[arm]} & {np.mean(early):.2f} & {np.mean(late):.2f} & {', '.join(f'{x:.2f}' for x in late)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{TAB}/lambda.tex", "w").write("\n".join(L) + "\n")

# ---- bike grid (final level, last 5 evaluations, mean over 3 seeds)
g = pd.read_csv(f"{RES}/bike_grid_summary/summary_final.csv")
order = ["SEARL DQN", "SEARL + 2 heads", "SEARL + per-obs heads", "CleanRL DQN", "DQN + 2 heads", "DQN + per-obs heads", "Rainbow"]
short = ["S", "S+2h", "S+obs", "DQN", "D+2h", "D+obs", "Rb"]
L = ["\\begin{tabular}{rr" + "r" * 7 + "}", "\\toprule", "Lanes & Sol. & " + " & ".join(short) + " \\\\", "\\midrule"]
for (lanes, sol), d in g.groupby(["lanes", "solutions"], sort=False):
    m = {r.variant: r["mean"] for _, r in d.iterrows()}
    best = max(m.values())
    L.append(f"{lanes} & {sol} & " + " & ".join((f"\\textbf{{{m[v]:.0f}}}" if m[v] == best else f"{m[v]:.0f}") for v in order) + " \\\\")
    if sol == 100 and lanes != 10:
        L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{TAB}/bike_final.tex", "w").write("\n".join(L) + "\n")

# ---- figures
for src, dst in [(f"{RES}/searl_confgate_summary/all_variants_cartpole_1M.png", "all_variants_cartpole_1M.png"),
                 (f"{RES}/searl_confgate_summary/all_variants_halfcheetah_1M.png", "all_variants_halfcheetah_1M.png"),
                 (f"{RES}/searl_confgate_cartpole_1M/lr_cartpole_1M.png", "lr_cartpole_1M.png"),
                 (f"{RES}/searl_confgate_halfcheetah_1M/lr_halfcheetah_1M.png", "lr_halfcheetah_1M.png"),
                 (f"{RES}/searl_confgate_summary/mix_curves_cartpole.png", "mix_curves_cartpole.png"),
                 (f"{RES}/searl_confgate_summary/mix_curves_halfcheetah.png", "mix_curves_halfcheetah.png"),
                 (f"{RES}/searl_confgate_summary/mix_lambda.png", "mix_lambda.png"),
                 (f"{RES}/searl_confgate_summary/mix_final_levels.png", "mix_final_levels.png"),
                 (f"{RES}/bike_grid_summary/grid_curves.png", "bike_grid_curves.png")]:
    shutil.copy(src, f"{FIG}/{dst}")
print("tables:", sorted(os.listdir(TAB)), "\nfigures:", sorted(os.listdir(FIG)))
for f in sorted(os.listdir(TAB)):
    print("\n==", f); print(open(f"{TAB}/{f}").read())
