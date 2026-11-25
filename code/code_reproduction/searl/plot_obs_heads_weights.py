#!/usr/bin/env python
"""Head strengths g_k of the 18-head SEARL model (model C, --weight-heads): what each agent ended with, and how the
strengths moved during the runs. Also reads model B runs (g_k is then 0 or 1).

usage: plot_obs_heads_weights.py ARM_DIR [--out FIG_DIR]
  ARM_DIR = results/<...>/C_weight, holding seed*/{gates_log.csv, gates_final.csv, config.json}
  FIG_DIR defaults to ARM_DIR/../figures; every output name starts with the arm name.

Reminder: the vote of head k is w_k = g_k / sum(g), so only the ratios of g matter. "relative vote" below is K * w_k:
1 = the equal share every head has in model A, 0 = the head does not vote.

Figures
  <arm>_gates_heatmap.png  rows = heads, columns = generations; top row of panels: population-mean g_k per seed (and
                           the mean over seeds); bottom row: population-mean relative vote.
  <arm>_trends.png         a) effective number of voting heads n_eff = 1 / sum w_k^2;  b) the full head's share of the
                           vote;  c) mean g of the full head vs of the single-observation heads;  d) g_k of every head
                           at the end of the run (one dot per seed = mean over its agents).
Tables
  <arm>_final_gates.csv       every agent of the final population of every seed x every head: g, w, fitness, is_best.
  <arm>_final_gates_best.csv  heads x seeds: g_k of the best final agent of each seed (+ mean and sd over seeds).
  <arm>_head_trends.csv       per head: mean g in each quarter of the run, final g and relative vote (mean, sd over
                              seeds), slope of g per 100 generations, share of final agents whose g is still the
                              head's start value, mean g among those that changed, and the head's mutations.
  <arm>_mutations.csv         every head_weight mutation: seed, generation, agent, head, g before -> after, the
                              mutant's fitness, and whether it left a copy in the next generation (kept).

Reading the trends: with `resample` and NO selection, a g_k that has been mutated is U[0, 1], so mean g drifts from 1
to 0.5 for every head alike and n_eff drifts to the dashed "no selection" level. Selection shows up as heads that
differ from each other in the same direction across seeds, not as the common decline. The 4 agents of a run are close
relatives, so the independent unit is the seed.
Runs with --init-gates full --weight-mut add (full head at 1, the others start at 0 and only gain): the full head is
left out of the g panels (it stays 1) and the colour / axis ranges follow the single-observation heads. With NO
selection a lineage gets a head mutation in 1 of 6 generations, so at generation t the summed g of the
single-observation heads is expected to be add * (t - 1) / 6; the dashed lines show that. Below the line = lineages
that gained strength were removed more often than chance; on the line = the strength had no effect on survival.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

OBS_NAMES = ["rootz", "rooty(angle)", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "vx", "vz", "vangle",
             "bthigh'", "bshin'", "bfoot'", "fthigh'", "fshin'", "ffoot'"]
FULL_C, OBS_C, SEED_C, REF_C = "#0072B2", "#E69F00", "#9a9a9a", "#444444"


def label(head):
    m = re.fullmatch(r"obs(\d+)", head)
    return f"{head} {OBS_NAMES[int(m.group(1))]}" if m and int(m.group(1)) < len(OBS_NAMES) else head


def load(arm_dir):
    runs = {}
    for d in sorted(glob.glob(os.path.join(arm_dir, "seed*"))):
        p = os.path.join(d, "gates_log.csv")
        if not os.path.exists(p):
            continue
        g = pd.read_csv(p)
        if g.empty:
            continue
        f = os.path.join(d, "gates_final.csv")
        c = os.path.join(d, "config.json")
        runs[os.path.basename(d)] = {
            "log": g,
            "final": pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 2 else None,
            "cfg": json.load(open(c)) if os.path.exists(c) else {}}
    return runs


def drift_n_eff(k, protect_full, n=20000):
    """Mean n_eff when every g_k is an independent U[0, 1] draw (the long-run state of `resample` with no selection)."""
    g = np.random.RandomState(0).uniform(0, 1, (n, k))
    if protect_full:
        g[:, 0] = 1.0
    return float(((g.sum(1) ** 2) / (g ** 2).sum(1)).mean())


def per_generation(log, heads, prefix):
    """[K, G] population mean of g_ or w_ per generation."""
    m = log.groupby("generation")[[f"{prefix}_{h}" for h in heads]].mean()
    return m.index.to_numpy(), m.to_numpy().T


def heatmap_figure(runs, heads, arm, out, from_full=False):
    k, seeds = len(heads), list(runs)
    gtop = 1.0
    if from_full:   # the full head stays 1; scale the colour to the other heads
        gtop = max(0.02, max(float(per_generation(r["log"], heads[1:], "g")[1].max()) for r in runs.values()))
    cols = len(seeds) + (1 if len(seeds) > 1 else 0)
    fig, axes = plt.subplots(2, cols, figsize=(3.6 * cols + 2.2, 9.5), squeeze=False, sharey=True)
    gmax = min(r["log"].generation.max() for r in runs.values())
    g_all, v_all = [], []
    for c, s in enumerate(seeds):
        gen, g = per_generation(runs[s]["log"], heads, "g")
        _, w = per_generation(runs[s]["log"], heads, "w")
        g_all.append(g[:, :gmax])
        v_all.append(k * w[:, :gmax])
        ims = draw_heat(axes[0, c], g, gen[-1], 0, f"{s}: g", gtop)
        imv = draw_heat(axes[1, c], k * w, gen[-1], 1, f"{s}: relative vote")
    if len(seeds) > 1:
        ims = draw_heat(axes[0, -1], np.mean(g_all, 0), gmax, 0, f"mean of {len(seeds)} seeds: g", gtop)
        imv = draw_heat(axes[1, -1], np.mean(v_all, 0), gmax, 1, f"mean of {len(seeds)} seeds: relative vote")
    for r in range(2):
        axes[r, 0].set_yticks(range(k))
        axes[r, 0].set_yticklabels([label(h) for h in heads], fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("generation")
    fig.colorbar(ims, ax=axes[0].tolist(), fraction=0.02, pad=0.01, extend="max" if from_full else "neither",
                 label="population-mean g  (full head = 1, above the scale; others start at 0)" if from_full
                 else "population-mean g  (1 = start)")
    fig.colorbar(imv, ax=axes[1].tolist(), fraction=0.02, pad=0.01, extend="max",
                 label="relative vote K*w  (1 = equal share, 0 = no vote)")
    fig.suptitle(f"{arm}: head strengths over the run (mean over the agents of each generation)", fontsize=12)
    p = os.path.join(out, f"{arm}_gates_heatmap.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def draw_heat(ax, z, gen_max, row, title, gtop=1.0):
    if row == 0:   # magnitude 0..gtop: one hue, light -> dark
        im = ax.imshow(z, aspect="auto", cmap="Blues", vmin=0, vmax=gtop, interpolation="nearest",
                       extent=(0.5, gen_max + 0.5, z.shape[0] - 0.5, -0.5))
    else:          # below / above the equal share: two hues around a neutral midpoint at 1
        im = ax.imshow(z, aspect="auto", cmap="PuOr_r", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2),
                       interpolation="nearest", extent=(0.5, gen_max + 0.5, z.shape[0] - 0.5, -0.5))
    ax.set_title(title, fontsize=10)
    return im


def trends_figure(runs, heads, arm, out, cfg):
    k, seeds = len(heads), list(runs)
    has_full = heads[0] == "full"
    obs = [h for h in heads if h != "full"]
    resample = bool(cfg.get("weight_heads")) and cfg.get("weight_mut", "resample") == "resample"
    from_full = cfg.get("init_gates") == "full" and has_full
    add = bool(cfg.get("weight_heads")) and cfg.get("weight_mut") == "add"
    step = float(cfg.get("weight_add", 0.02))
    gens = np.arange(1, max(r["log"].generation.max() for r in runs.values()) + 1)
    drift_sum = step * (gens - 1) / 6.0   # expected summed g of the candidate heads with no selection (see docstring)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    (a, b), (c, d) = axes

    def lines(ax, series, color, name):
        """thin line per seed, thick line = mean over seeds (on the generations all seeds have)."""
        for s in series:
            ax.plot(s.index, s.values, color=color, lw=0.8, alpha=0.45)
        if len(series) > 1:
            m = pd.concat(series, axis=1).dropna().mean(1)
            ax.plot(m.index, m.values, color=color, lw=2.4, label=f"{name} (mean of {len(series)} seeds)")
        else:
            ax.plot(series[0].index, series[0].values, color=color, lw=2.4, label=name)

    # a) effective number of voting heads
    lines(a, [r["log"].groupby("generation").n_eff.mean() for r in runs.values()], FULL_C, "n_eff")
    if from_full:   # starts at 1 (only the full head votes); zoom on the data instead of 0..18
        a.set_ylim(0.95, max(1.2, 1.05 * max(r["log"].groupby("generation").n_eff.mean().max() for r in runs.values())))
        a.axhline(1, color=REF_C, lw=1, ls=":", label="only the full head votes (1)")
    else:
        a.axhline(k, color=REF_C, lw=1, ls=":", label=f"all heads equal ({k})")
        if resample:
            a.axhline(drift_n_eff(k, cfg.get("protect_full", False)), color=REF_C, lw=1, ls="--",
                      label="no selection, long run")
        a.set_ylim(0, k + 1)
    a.set_title("a) effective number of voting heads, 1 / sum w^2")
    a.set_ylabel("n_eff (population mean)")

    # b) the full head's share of the vote
    if has_full:
        lines(b, [r["log"].groupby("generation").w_full.mean() for r in runs.values()], FULL_C, "full head")
        if from_full and add:
            b.plot(gens, 1 / (1 + drift_sum), color=REF_C, lw=1, ls="--", label="no selection (expected)")
            b.set_ylim(min(0.5, b.get_ylim()[0]), 1.02)
        else:
            b.axhline(1 / k, color=REF_C, lw=1, ls=":", label=f"equal share (1/{k})")
            b.set_ylim(0, None)
    b.set_title("b) the full head's share of the vote, w_full")
    b.set_ylabel("share of the vote (population mean)")

    # c) mean strength: full head vs single-observation heads
    if from_full:   # the full head stays 1: show the summed strength the other heads have gained
        lines(c, [r["log"].groupby("generation")[[f"g_{h}" for h in obs]].mean().sum(1) for r in runs.values()], OBS_C,
              f"sum over the {len(obs)} single-observation heads")
        if add:
            c.plot(gens, drift_sum, color=REF_C, lw=1, ls="--", label=f"no selection (expected, {step:g} per head mutation)")
        c.set_ylim(0, None)
        c.set_title("c) summed strength g of the single-observation heads (full head = 1)")
        c.set_ylabel("sum of g (population mean)")
    else:
        if has_full:
            lines(c, [r["log"].groupby("generation").g_full.mean() for r in runs.values()], FULL_C, "full head")
        lines(c, [r["log"].groupby("generation")[[f"g_{h}" for h in obs]].mean().mean(1) for r in runs.values()], OBS_C,
              f"mean of the {len(obs)} single-observation heads")
        if resample:
            c.axhline(0.5, color=REF_C, lw=1, ls="--", label="no selection, long run (0.5)")
        c.set_ylim(-0.03, 1.03)
        c.set_title("c) strength g")
        c.set_ylabel("g (population mean)")
    for ax in (a, b, c):
        ax.set_xlabel("generation")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=9, frameon=False)

    # d) g_k of every head at the end of the run
    fin = np.array([[r["log"][r["log"].generation == r["log"].generation.max()][f"g_{h}"].mean() for h in heads]
                    for r in runs.values()])                                         # [seeds, K]
    fin_gen = int(min(r["log"].generation.max() for r in runs.values()))
    y = np.arange(k)
    for i in range(len(seeds)):
        d.scatter(fin[i], y, s=34, color=[FULL_C if h == "full" else OBS_C for h in heads], alpha=0.55,
                  edgecolor="white", linewidth=0.8, zorder=3)
    d.scatter(fin.mean(0), y, s=150, marker="|", color="black", linewidth=2.2, zorder=4, label="mean over seeds")
    d.scatter([], [], s=34, color=OBS_C, label="one seed (mean over its final agents)")
    if resample:
        d.axvline(0.5, color=REF_C, lw=1, ls="--", label="no selection, long run (0.5)")
    d.set_yticks(y)
    d.set_yticklabels([label(h) for h in heads], fontsize=8)
    d.invert_yaxis()
    if from_full:   # the full head's dots (g = 1) are off the axis on purpose
        top = max(step, float(fin[:, 1:].max()))
        d.set_xlim(-0.04 * top, 1.08 * top)
        if add:
            d.axvline(drift_sum[fin_gen - 1] / len(obs), color=REF_C, lw=1, ls="--", label="no selection (expected)")
        d.set_title("d) final strength of every single-observation head (full head = 1, off the axis)")
    else:
        d.set_xlim(-0.03, 1.03)
        d.set_title("d) final strength of every head")
    d.set_xlabel("g at the end of the run")
    d.grid(axis="x", alpha=0.25, lw=0.5)
    d.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    fig.suptitle(f"{arm}: trends of the head strengths within the runs (thin lines = single seeds)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out, f"{arm}_trends.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def mutation_events(runs, heads):
    rows = []
    pat = re.compile(r"head_weight_(.+)_([0-9.]+)_to_([0-9.]+)$")
    for s, r in runs.items():
        log = r["log"]
        for gen, pop in log.groupby("generation"):
            nxt = log[log.generation == gen + 1]
            for _, x in pop.iterrows():
                m = pat.match(str(x.mutation))
                if not m:
                    continue
                kept = float("nan") if nxt.empty else int((nxt.individual == x.individual).any()
                                                          or (nxt.parent == x.individual).any())
                rows.append({"seed": s, "generation": gen, "individual": x.individual, "head": m.group(1),
                             "g_before": float(m.group(2)), "g_after": float(m.group(3)),
                             "fitness": x.fitness, "population_mean_fitness": pop.fitness.mean(), "kept": kept})
    return pd.DataFrame(rows, columns=["seed", "generation", "individual", "head", "g_before", "g_after", "fitness",
                                       "population_mean_fitness", "kept"])


def head_trends(runs, heads, events):
    k = len(heads)
    per_seed = {key: [] for key in ("q", "final_g", "final_v", "slope", "still1", "changed_mean")}
    for r in runs.values():
        log = r["log"]
        gen, g = per_generation(log, heads, "g")
        _, w = per_generation(log, heads, "w")
        edges = np.linspace(gen.min() - 1, gen.max(), 5)
        per_seed["q"].append(np.stack([g[:, (gen > lo) & (gen <= hi)].mean(1) for lo, hi in zip(edges[:-1], edges[1:])], 1))
        per_seed["final_g"].append(g[:, -1])
        per_seed["final_v"].append(k * w[:, -1])
        per_seed["slope"].append(np.array([np.polyfit(gen, row, 1)[0] * 100 if len(gen) > 1 else np.nan for row in g]))
        last = log[log.generation == gen.max()][[f"g_{h}" for h in heads]].to_numpy()     # [agents, K]
        start = log[log.generation == gen.min()][[f"g_{h}" for h in heads]].to_numpy()[:1]   # same for all agents
        per_seed["still1"].append((last == start).mean(0))
        per_seed["changed_mean"].append(pd.DataFrame(last).where(last != start).mean(0).to_numpy())   # NaN: none changed
    q = np.mean(per_seed["q"], 0)
    fg, fv, sl = (np.array(per_seed[x]) for x in ("final_g", "final_v", "slope"))
    changed = pd.DataFrame(per_seed["changed_mean"]).mean(0).to_numpy()   # mean over the seeds that have a value
    out = pd.DataFrame({
        "head": heads, "name": [label(h) for h in heads],
        "g_q1": q[:, 0], "g_q2": q[:, 1], "g_q3": q[:, 2], "g_q4": q[:, 3],
        "g_final_mean": fg.mean(0), "g_final_sd_seeds": fg.std(0, ddof=1) if len(fg) > 1 else np.nan,
        "g_final_min_seed": fg.min(0), "g_final_max_seed": fg.max(0),
        "rel_vote_final_mean": fv.mean(0), "rel_vote_final_sd_seeds": fv.std(0, ddof=1) if len(fv) > 1 else np.nan,
        "seeds_rel_vote_above_1": (fv > 1).sum(0), "n_seeds": len(fg),
        "g_slope_per_100_gen": sl.mean(0), "seeds_slope_negative": (sl < 0).sum(0),
        "share_final_agents_g_unchanged": np.mean(per_seed["still1"], 0), "g_mean_of_changed_final": changed})
    if len(events):
        ev = events.assign(up=events.g_after > events.g_before)
        agg = ev.groupby("head").agg(n_mutations=("up", "size"), n_up=("up", "sum"), share_kept=("kept", "mean"))
        agg["share_kept_up"] = ev[ev.up].groupby("head").kept.mean()
        agg["share_kept_down"] = ev[~ev.up].groupby("head").kept.mean()
        out = out.merge(agg.reset_index(), on="head", how="left")
        out["n_mutations"] = out["n_mutations"].fillna(0).astype(int)
    return out.round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("arm_dir")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    arm_dir = os.path.normpath(a.arm_dir)
    arm = os.path.basename(arm_dir)
    out = a.out or os.path.join(os.path.dirname(arm_dir), "figures")
    os.makedirs(out, exist_ok=True)
    runs = load(arm_dir)
    if not runs:
        raise SystemExit(f"no seed*/gates_log.csv with rows under {arm_dir}")
    first = next(iter(runs.values()))
    heads = [c[2:] for c in first["log"].columns if c.startswith("g_")]
    cfg = first["cfg"]

    written = [heatmap_figure(runs, heads, arm, out, from_full=cfg.get("init_gates") == "full" and heads[0] == "full"),
               trends_figure(runs, heads, arm, out, cfg)]

    finals = [r["final"].assign(seed=s) for s, r in runs.items() if r["final"] is not None]
    if finals:
        fin = pd.concat(finals)[["seed", "individual", "parent", "is_best", "fitness", "n_eff", "head", "gate", "weight"]]
        fin.to_csv(os.path.join(out, f"{arm}_final_gates.csv"), index=False)
        best = fin[fin.is_best == 1].pivot(index="head", columns="seed", values="gate").reindex(heads)
        best = best.assign(mean=best.mean(1), sd=best.std(1, ddof=1) if best.shape[1] > 1 else np.nan)
        best.round(4).to_csv(os.path.join(out, f"{arm}_final_gates_best.csv"))
        written += [os.path.join(out, f"{arm}_final_gates.csv"), os.path.join(out, f"{arm}_final_gates_best.csv")]
    else:
        print("no gates_final.csv yet (runs not finished): final tables skipped")

    events = mutation_events(runs, heads)
    events.to_csv(os.path.join(out, f"{arm}_mutations.csv"), index=False)
    trends = head_trends(runs, heads, events)
    trends.to_csv(os.path.join(out, f"{arm}_head_trends.csv"), index=False)
    written += [os.path.join(out, f"{arm}_mutations.csv"), os.path.join(out, f"{arm}_head_trends.csv")]

    print(f"{arm}: {len(runs)} seed(s), {len(heads)} heads, {len(events)} head_weight mutations")
    print(trends[["head", "g_q1", "g_q4", "g_final_mean", "rel_vote_final_mean", "g_slope_per_100_gen"]].to_string(index=False))
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
