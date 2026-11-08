#!/usr/bin/env python
"""Two figures. The continuous one is four benchmarks (HalfCheetah-v4, bike v2.1, CartPole this
session, CartPole previous runs). The discrete one is a 3x3: those four, plus Acrobot, LunarLander,
factored Hopper, Freeway RAM, and FrozenLake.

Pong and Enduro are omitted: at the reduced budgets (SEARL 60k agent steps, CleanRL/PPO 250k) Pong stays
near -21 for every method, and Enduro's SEARL/PPO lines stay near zero (CleanRL DQN did learn Enduro and is
still left out). x = total environment steps (log).

Solid lines = hyperparameter-optimisation / adaptive methods (SEARL, random search, HyperController, PB2,
confidence gates); dashed lines = fixed-hyperparameter baselines (CleanRL DQN / TD3 / PPO).
Every curve is the mean over seeds (NaN-aware) with +/- 1 std shaded.
This script writes two files:
  figures/multi_benchmark.png          continuous HalfCheetah (SEARL-TD3, CleanRL TD3, continuous PPO, ...)
  figures/multi_benchmark_discrete.png same panels, plus Acrobot, LunarLander, Hopper (9 outputs),
                                       Freeway RAM, and FrozenLake. HalfCheetah: each joint in {-1, 0, +1}
                                       (18 outputs), SEARL DQN / CleanRL DQN / discrete PPO only. The five
                                       extra tasks use the full bike-panel set (also random search,
                                       HyperController, PB2), 3 seeds, 480k steps (FrozenLake 100k).

y-axis per panel
  HalfCheetah (continuous) : episode return (HyperController/PB2 log mean step reward per 1000-step rollout: x 1000)
  HalfCheetah (discrete)   : episode return; each joint is its own 3-way choice (SEARL: greedy test return)
  bike v2.1   : raw episode return (perfect ~4900); per-episode CSVs smoothed with a trailing mean over 10 episodes
  CartPole    : episode return (max 500), same 10-episode smoothing
x-axis: agent steps (bike: one decision per row). Random search's x is multiplied by the number of
configurations it trained (the SEARL paper's fair-evaluation protocol).
"""
import argparse
import glob
import importlib.util
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.join(HERE, "..")
SMOOTH = 10

STYLE = {  # label -> (color, linestyle, linewidth)
    "SEARL": ("#1f77b4", "-", 2.6), "SEARL TD3": ("#1f77b4", "-", 2.6), "SEARL DQN": ("#1f77b4", "-", 2.6),
    "Random search": ("#d62728", "-", 2.2),
    "HyperController": ("#2ca02c", "-", 2.2), "PB2": ("#ff7f0e", "-", 2.2),
    "Analytical gate": ("#9467bd", "-", 2.2), "Learnable gate": ("#e377c2", "-", 2.2),
    "RNN gate": ("#bcbd22", "-", 2.2), "DQN baseline": ("#17becf", "--", 2.4),
    "Rainbow DQN (fixed HPs)": ("#8c564b", "--", 2.2), "Rainbow DQN (Kaixhin port)": ("#222222", "-.", 2.4), "CleanRL DQN": ("#4a3aa7", "--", 2.4),
    "CleanRL TD3": ("#4a3aa7", "--", 2.4), "CleanRL PPO": ("#7f7f7f", "--", 2.4),
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def aggregate(curves, n_points=120):
    curves = [(np.asarray(x, float), np.asarray(y, float)) for x, y in curves if len(x) > 0]
    curves = [(x[x > 0], y[x > 0]) for x, y in curves]
    curves = [c for c in curves if len(c[0]) > 0]
    if not curves:
        return None
    lo, hi = min(c[0].min() for c in curves), max(c[0].max() for c in curves)
    grid = np.logspace(np.log10(lo), np.log10(hi), n_points)
    stacked = np.stack([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    in_range = np.stack([(grid >= x.min()) & (grid <= x.max()) for x, _ in curves]).sum(0)  # seeds still running
    with np.errstate(all="ignore"):
        return grid, np.nanmean(stacked, 0), np.nanstd(stacked, 0), in_range


def smooth_csv(path, column):
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df["global_step"].values.astype(float), df[column].rolling(SMOOTH, min_periods=1).mean().values


def csv_curves(pattern, column="clipped_return"):
    out = [smooth_csv(p, column) for p in sorted(glob.glob(pattern))]
    return [c for c in out if c is not None]


BEST_RE = re.compile(r"best_fitness: (\S+)\s+step:(\d+)")
TEST_RE = re.compile(r"test_mean_reward: (\S+)\s+step:(\d+)")


def parse_log(path, regex):
    xs, ys = [], []
    for line in open(path):
        m = regex.search(line)
        if m:
            ys.append(float(m.group(1)))
            xs.append(float(m.group(2)))
    return np.array(xs), np.array(ys)


def first_log(d):
    logs = sorted(glob.glob(os.path.join(d, "**", "log_file.txt"), recursive=True))
    return logs[-1] if logs else None


def discrete_searl(name, column):
    """Unified discrete SEARL (../searl/searl_discrete.py): history.csv per seed."""
    base = os.path.join(CR, "searl/results_discrete")
    curves = []
    for p in sorted(glob.glob(os.path.join(base, f"{name}_seed*", "history.csv"))):
        df = pd.read_csv(p)
        if len(df):
            curves.append((df["num_frames"].values.astype(float), df[column].values))
    return curves


def atari_searl(env):
    return discrete_searl(env, "best_fitness")


def atari_rainbow(env):
    base = os.path.join(CR, "searl_dqn/reproduction/results/rainbow")
    curves = []
    for d in sorted(glob.glob(os.path.join(base, f"{env}_seed*/"))):
        lg = first_log(d)
        if lg:
            x, y = parse_log(lg, TEST_RE)
            if len(x):
                curves.append((x, y))
    return curves


def atari_rs(env, n_configs=10):
    base = os.path.join(CR, "searl_dqn/reproduction/results/rs")
    runs = {}
    for d in sorted(glob.glob(os.path.join(base, f"{env}_run*_cfg*/"))):
        m = re.search(r"_run(\d+)_cfg(\d+)/?$", d.rstrip("/") + "/")
        lg = first_log(d)
        if m and lg:
            x, y = parse_log(lg, TEST_RE)
            if len(x):
                runs.setdefault(int(m.group(1)), []).append((x, y))
    curves = []
    for _, cfgs in sorted(runs.items()):
        if len(cfgs) >= n_configs:
            best = max(cfgs, key=lambda c: c[1][-1])
            curves.append((best[0] * len(cfgs), best[1]))
    return curves


def bike_history(pattern, scale_file=True):
    curves = []
    for d in sorted(glob.glob(pattern)):
        p = os.path.join(d, "history.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        x = df["num_frames"].values.astype(float)
        y = (df["test_return"] if "test_return" in df else df["best_fitness"] if "best_fitness" in df else df["reward"]).values
        n = os.path.join(d, "num_configs_tried.txt")
        if scale_file and os.path.exists(n):
            x = x * int(open(n).read())
        curves.append((x, y))
    return curves


def halfcheetah_lines():
    """Continuous HalfCheetah-v4: SEARL-TD3, CleanRL TD3, continuous PPO, HyperController, PB2, random search."""
    cl = _load_module("cl_fig", os.path.join(CR, "cleanrl/make_figures.py"))
    lines = {}
    searl_long = os.path.join(CR, "searl/results_500k")
    lines["SEARL TD3"] = cl.load_searl_curves(searl_long, "searl")
    lines["Random search"] = cl.load_searl_curves(os.path.join(CR, "searl/results"), "random_search")
    lines["CleanRL TD3"] = cl.load_cleanrl_curves(os.path.join(CR, "cleanrl/results"), "td3", "HalfCheetah-v4")
    lines["CleanRL PPO"] = cl.load_cleanrl_curves(os.path.join(CR, "cleanrl/results"), "ppo", "HalfCheetah-v4")
    hc = cl.load_hc_runs(os.path.join(CR, "hypercontroller/results_full"), "HalfCheetah-v4")
    for m in ("HyperController", "PB2"):
        if m in hc:
            n = min(len(df) for df in hc[m])
            lines[m] = [(np.arange(1, n + 1) * 1000.0, df["reward"].values[:n] * 1000.0) for df in hc[m]]
    return lines


def halfcheetah_discrete_lines():
    """Discrete HalfCheetah-v4 (6 joints x 3 torques = 18 outputs). SEARL is a greedy test return; DQN and PPO are training returns."""
    hc = os.path.join(HERE, "results_halfcheetah")
    return {
        "SEARL DQN": bike_history(os.path.join(hc, "searl/seed*"), scale_file=False),
        "CleanRL DQN": csv_curves(os.path.join(hc, "cleanrl_dqn/seed*.csv"), "episodic_return"),
        "CleanRL PPO": csv_curves(os.path.join(hc, "ppo/none_seed*.csv"), "episodic_return"),
    }


def atari_lines(env):
    ppo = os.path.join(HERE, "results/ppo")
    return {
        "SEARL": atari_searl(env), "Random search": atari_rs(env), "Rainbow DQN (fixed HPs)": atari_rainbow(env),
        "CleanRL DQN": csv_curves(os.path.join(CR, "..", "cleanrl_dqn_baseline/results", f"{env}_seed*.csv")),
        "CleanRL PPO": csv_curves(os.path.join(ppo, f"{env}_none_seed*.csv")),
        "HyperController": csv_curves(os.path.join(ppo, f"{env}_HyperController_seed*.csv")),
        "PB2": csv_curves(os.path.join(ppo, f"{env}_PB2_seed*.csv")),
    }


def bike_lines():
    ppo = os.path.join(HERE, "results/ppo")
    return {
        "SEARL DQN": discrete_searl("bike", "test_return"),
        "Random search": bike_history(os.path.join(HERE, "results/bike_rs/seed*")),
        "CleanRL DQN": csv_curves(os.path.join(HERE, "results/cleanrl_dqn/bike_seed*.csv"), "episodic_return"),
        "CleanRL PPO": csv_curves(os.path.join(ppo, "bike_none_seed*.csv"), "episodic_return"),
        "HyperController": csv_curves(os.path.join(ppo, "bike_HyperController_seed*.csv"), "episodic_return"),
        "PB2": csv_curves(os.path.join(ppo, "bike_PB2_seed*.csv"), "episodic_return"),
    }


FOURWAY_DIR = os.path.join(CR, "..", "..", "paper", "_tmp_b_feedb_cg", "ckpt_fourway_decay3500")
FOURWAY_FILES = {"Analytical gate": "b_feedb_cg_gate_seed{}.pt", "Learnable gate": "b_feedb_cg_learnable_seed{}.pt",
                 "RNN gate": "b_feedb_cg_rnn_ctrl_seed{}.pt"}


def fourway_curves(label):
    """code/conf_gate/confgate_fourway.py results (5000 episodes each): x = cumulative env steps (CartPole reward is 1 per
    step, so steps = cumsum of episode returns), y = episode return smoothed over 10 episodes."""
    import torch
    curves = []
    for seed in range(1, 11):
        p = os.path.join(FOURWAY_DIR, FOURWAY_FILES[label].format(seed))
        if not os.path.exists(p):
            continue
        r = np.asarray(torch.load(p, map_location="cpu", weights_only=False)["episode_returns"], float)
        curves.append((np.cumsum(r), pd.Series(r).rolling(SMOOTH, min_periods=1).mean().values))
    return curves


PAPER_CK = os.path.join(CR, "..", "..", "paper", "_tmp_b_feedb_cg")
# results stored BEFORE the seeds added in this session: (folder, file pattern, seeds)
PREVIOUS_SPECS = {
    "DQN baseline": ("ckpt_fourway_decay3500", "b_feedb_cg_seed{}.pt", range(1, 11)),
    "Analytical gate": ("ckpt_fourway_decay3500", "b_feedb_cg_gate_seed{}.pt", [1]),
    "Learnable gate": ("ckpt_learnable_decay3500_ep5000", "b_feedb_cg_learnable_seed{}.pt", range(1, 11)),
    "RNN gate": ("ckpt_fourway_decay3500", "b_feedb_cg_rnn_ctrl_seed{}.pt", [1]),
}


def previous_cartpole_lines():
    import torch
    lines = {}
    for label, (folder, pat, seeds) in PREVIOUS_SPECS.items():
        curves = []
        for seed in seeds:
            p = os.path.join(PAPER_CK, folder, pat.format(seed))
            if os.path.exists(p):
                r = np.asarray(torch.load(p, map_location="cpu", weights_only=False)["episode_returns"], float)
                curves.append((np.cumsum(r), pd.Series(r).rolling(SMOOTH, min_periods=1).mean().values))
        lines[label] = curves
    return lines


def extra_lines(env_id):
    """Bike-panel methods on one extra discrete task (results_extra/<env>)."""
    base = os.path.join(HERE, "results_extra", env_id.replace("/", "_"))
    return {
        "SEARL DQN": bike_history(os.path.join(base, "searl", "seed*"), scale_file=False),
        "Random search": bike_history(os.path.join(base, "rs", "seed*")),
        "CleanRL DQN": csv_curves(os.path.join(base, "cleanrl_dqn", "seed*.csv"), "episodic_return"),
        "CleanRL PPO": csv_curves(os.path.join(base, "ppo", "none_seed*.csv"), "episodic_return"),
        "HyperController": csv_curves(os.path.join(base, "ppo", "HyperController_seed*.csv"), "episodic_return"),
        "PB2": csv_curves(os.path.join(base, "ppo", "PB2_seed*.csv"), "episodic_return"),
    }


def cartpole_lines():
    cp = os.path.join(HERE, "results_cartpole")
    lines = {lab: fourway_curves(lab) for lab in FOURWAY_FILES}
    lines.update({
        "SEARL DQN": bike_history(os.path.join(cp, "searl/seed*")),
        "Random search": bike_history(os.path.join(cp, "rs/seed*")),
        "CleanRL DQN": csv_curves(os.path.join(cp, "cleanrl_dqn/seed*.csv"), "episodic_return"),
        "CleanRL PPO": csv_curves(os.path.join(cp, "ppo/none_seed*.csv"), "episodic_return"),
        "HyperController": csv_curves(os.path.join(cp, "ppo/HyperController_seed*.csv"), "episodic_return"),
        "PB2": csv_curves(os.path.join(cp, "ppo/PB2_seed*.csv"), "episodic_return"),
    })
    return lines


def draw(ax, lines, title, ylabel, legend=True):
    order = ["SEARL TD3", "SEARL DQN", "SEARL", "Random search", "HyperController", "PB2", "Analytical gate", "Learnable gate", "RNN gate",
             "DQN baseline", "Rainbow DQN (fixed HPs)", "Rainbow DQN (Kaixhin port)", "CleanRL DQN", "CleanRL TD3", "CleanRL PPO"]
    for name in order:
        curves = lines.get(name)
        agg = aggregate(curves) if curves else None
        if agg is None:
            continue
        g, m, s, cnt = agg
        color, ls, lw = STYLE[name]
        full = cnt >= len(curves)
        # solid where every seed has data; faded dotted tail where only some seeds ran that far
        # (e.g. one seed ends later than the others) -- the tail is NOT a full-seed mean
        mm = np.where(full, m, np.nan)
        ax.plot(g, mm, color=color, linestyle=ls, linewidth=lw, label=f"{name} (n={len(curves)})")
        if (~full & np.isfinite(m)).any():
            i0 = max(int(np.argmax(~full)) - 1, 0)
            ax.plot(g[i0:], np.where(np.isfinite(m[i0:]), m[i0:], np.nan), color=color, linestyle=":", linewidth=1.4,
                    alpha=0.45)
        ax.fill_between(g, mm - s, mm + s, color=color, alpha=0.10, linewidth=0)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Total environment steps", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


def add_rainbow(lines, slug):
    """Adds the ported Kaixhin Rainbow (code_reproduction/rainbow_dqn): training-episode returns, one CSV per seed."""
    out = dict(lines)
    out["Rainbow DQN (Kaixhin port)"] = csv_curves(
        os.path.join(CR, "rainbow_dqn", "results", slug, "seed*.csv"), "episodic_return")
    return out


def render_discrete(out):
    """3x3 discrete figure: the original four panels plus the five extra tasks."""
    panels = [
        (add_rainbow(halfcheetah_discrete_lines(), "HalfCheetah-v4"), "HalfCheetah-v4 (discrete, 18 outputs)", "Episode return", None),
        (add_rainbow(bike_lines(), "bike"), "bike_env v2.1 (discrete)", "Episode return (perfect ~ 4900)", None),
        (add_rainbow(cartpole_lines(), "CartPole-v1"), "CartPole-v1 (this session)", "Episode return (max 500)", (-20, 560)),
        (add_rainbow(extra_lines("Acrobot-v1"), "Acrobot-v1"), "Acrobot-v1", "Episode return", None),
        (add_rainbow(extra_lines("LunarLander-v2"), "LunarLander-v2"), "LunarLander-v2", "Episode return", None),
        (add_rainbow(extra_lines("Hopper-v4"), "Hopper-v4"), "Hopper-v4 (discrete, 9 outputs)", "Episode return", None),
        (add_rainbow(extra_lines("ALE/Freeway-ram-v5"), "ALE_Freeway-ram-v5"), "Freeway RAM (ALE/Freeway-ram-v5)", "Episode return", None),
        (add_rainbow(extra_lines("FrozenLake-v1"), "FrozenLake-v1"), "FrozenLake-v1 (sanity check, 100k steps)", "Episode return", None),
        (previous_cartpole_lines(), "CartPole-v1 (previous stored runs)", "Episode return (max 500)", (-20, 560)),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16.8, 16.2))
    for ax, (lines, title, ylabel, ylim) in zip(axes.ravel(), panels):
        draw(ax, lines, title, ylabel, legend=False)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_xlim(5, 2.5e6)
    handles, labels, seen = [], [], set()
    order = ["SEARL TD3", "SEARL DQN", "SEARL", "Random search", "HyperController", "PB2",
             "Analytical gate", "Learnable gate", "RNN gate", "DQN baseline",
             "Rainbow DQN (fixed HPs)", "Rainbow DQN (Kaixhin port)", "CleanRL DQN", "CleanRL TD3", "CleanRL PPO"]
    found = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            found.setdefault(lab.split(" (n=")[0], h)
    for name in order:
        if name in found:
            handles.append(found[name])
            labels.append(name)
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.045))
    foot = (FOOT_COMMON + FOOT_DISCRETE + FOOT_REST).replace(" (previous panel: n shown in legend)", "")
    fig.text(0.5, 0.0, foot, ha="center", va="top", fontsize=7.5, color="#444")
    fig.suptitle("Discrete benchmarks: SEARL vs. HPO methods vs. fixed-hyperparameter baselines (mean +/- std)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def render(halfcheetah, title, ylabel, footnote, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    draw(axes[0, 0], halfcheetah, title, ylabel)
    draw(axes[0, 1], bike_lines(), "bike_env v2.1 (discrete: SEARL-DQN / CleanRL DQN / PPO)",
         "Episode return (perfect ~ 4900)")
    draw(axes[1, 0], cartpole_lines(), "CartPole-v1 (task of conf_gate/confgate_fourway.py)", "Episode return (max 500)")
    draw(axes[1, 1], previous_cartpole_lines(), "CartPole-v1: previous stored runs (DQN, analytical, learnable, RNN gate)",
         "Episode return (max 500)")
    for ax in (axes[1, 0], axes[1, 1]):          # same axes so the two CartPole panels compare directly
        ax.set_ylim(-20, 560)
        ax.set_xlim(5, 2.5e6)
    axes[1, 0].set_title("CartPole-v1 (discrete: SEARL-DQN / CleanRL DQN / PPO + gates): this session", fontsize=11)
    fig.text(0.5, 0.0, footnote, ha="center", va="top", fontsize=7.5, color="#444")
    fig.suptitle("SEARL vs. HPO methods vs. fixed-hyperparameter baselines vs. conf_gate gating methods (mean +/- std)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


FOOT_COMMON = (
    "solid = HPO / adaptive method, dashed = fixed hyperparameters; faded dotted tail = only some seeds ran that long (not a full mean). "
    "HalfCheetah and bike: 3 seeds; CartPole: 10 seeds (previous panel: n shown in legend). "
)
FOOT_DISCRETE = (
    "HalfCheetah: each of 6 joints chosen separately in {-1, 0, +1} (18 outputs), 500k steps, SEARL DQN / CleanRL DQN / PPO only. "
    "Hopper: each of 3 joints in {-1, 0, +1} (9 outputs). "
    "Acrobot, LunarLander, Hopper, and Freeway RAM: 480k steps, 3 seeds, same six methods as bike. "
    "FrozenLake: 100k steps, slippery default, return 0 or 1. Freeway uses the 128-byte RAM observation.\n"
)


FOOT_REST = (
    "Pong and Enduro omitted: at 60k-250k steps they do not show a learning curve.\n"
    "bike, CartPole: SEARL and random search = greedy test returns; CleanRL, PPO-family and the confgate / previous DQN curves = training returns (include exploration). "
    "CartPole budgets: confgate methods 5000 episodes (0.6-1.2M steps); other algorithms 1M steps.\n"
    "PB2 on CartPole: its GP fit failed in all 10 runs (iterations 1141-1746, steps 584k-894k) and it keeps its last hyperparameters afterwards; its signal has 0.1% jitter. "
    "Previous panel unchanged: DQN baseline = 10 seeds, learnable = earlier version of confgate_learnable.py (10 seeds), analytical and RNN gate = seed 1 only (before the new seeds 2-10)."
)


def main():
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    render(
        halfcheetah_lines(),
        "HalfCheetah-v4 (continuous: SEARL-TD3 / CleanRL TD3 / PPO)",
        "Episode return (HC/PB2: step reward x 1000)",
        FOOT_COMMON + "TD3 only on HalfCheetah (continuous actions). " + FOOT_REST,
        os.path.join(HERE, "figures/multi_benchmark.png"),
    )
    render_discrete(os.path.join(HERE, "figures/multi_benchmark_discrete.png"))


if __name__ == "__main__":
    main()
