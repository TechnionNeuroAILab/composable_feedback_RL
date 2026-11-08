"""v2.1 plots, multi-seed: one subfolder per lane count + cross-lane comparison -> ../../plotting/v2.1_plots/{lanes2,lanes3,lanes5,lanes10,lane_comparison}.
Lines = mean over seeds, bands = min-max over seeds."""
import glob, json, re, importlib.util, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent / "plotting" / "v2.1_plots"
RES = HERE / "training_results_v2_1"
LANES = [2, 3, 5, 10]
SOL = [(1, "#2a78d6"), (3, "#eb6834"), (8, "#1baf7a"), (20, "#eda100"), (100, "#e87ba4")]   # colour = solutions
LANE_COL = {2: "#2a78d6", 3: "#eb6834", 5: "#1baf7a", 10: "#eda100"}                          # colour = lanes
MAX_REWARD, EP = 4900.0, 2000
bg, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.facecolor": bg, "figure.facecolor": bg,
                     "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
roll = lambda x, k=50: np.convolve(np.asarray(x, float), np.ones(min(k, len(x))) / min(k, len(x)), mode="valid")
def style(a):
    a.grid(axis="y", color=grid); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)

runs = {}
for d in sorted(RES.glob("lanes_*__sol_*__rows_240__seed_*")):
    if not (d / "final_greedy_eval_50ep.json").exists(): continue
    L, k, seed = map(int, re.search(r"lanes_(\d+)__sol_(\d+)__rows_240__seed_(\d+)", d.name).groups())
    runs.setdefault((L, k), []).append({"seed": seed, "greedy": pd.read_csv(d / "greedy_eval.csv"),
        "train": pd.read_csv(d / "training_episodes.csv"), "final": json.load(open(d / "final_greedy_eval_50ep.json")),
        "board": json.load(open(d / "config.json"))["initial_board"]})
n_seeds = {key: len(v) for key, v in runs.items()}
print("runs found:", sum(n_seeds.values()), {k: v for k, v in sorted(n_seeds.items()) if v != 5})

def agg_greedy(L, k, col):
    g = [r["greedy"].set_index("episode")[col] for r in runs[(L, k)]]
    df = pd.concat(g, axis=1); return df.index.values, df.mean(1).values, df.min(1).values, df.max(1).values
def agg_train(L, k, col):
    cur = np.array([roll(r["train"][col]) for r in runs[(L, k)]])
    x = np.arange(EP - cur.shape[1] + 1, EP + 1) if cur.ndim == 2 else None
    return x, cur.mean(0), cur.min(0), cur.max(0)
def line(a, x, m, lo, hi, c, label=None, lw=2, marker=None):
    a.fill_between(x, lo, hi, color=c, alpha=.35, lw=0); a.plot(x, m, c=c, lw=lw, marker=marker, ms=3, label=label)
def first_below(L, k, thr=10.0):
    x, m, *_ = agg_train(L, k, "falls"); idx = np.where(m < thr)[0]; return int(x[idx[0]]) if len(idx) else np.nan

spec = importlib.util.spec_from_file_location("bike_v2_0", HERE.parent / "bike_dqn_multi_lanes_v2.0.py")
v20 = importlib.util.module_from_spec(spec); sys.modules["bike_v2_0"] = v20; spec.loader.exec_module(v20)

# ---------------- per lane count ----------------
for L in LANES:
    sols = [(k, c) for k, c in SOL if (L, k) in runs]
    if not sols: continue
    out = ROOT / f"lanes{L}"; (out / "boards").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, c in sols:
        lab = f"{k} solution{'s' if k > 1 else ''} (board: {runs[(L, k)][0]['board']['solutions']}; n={n_seeds[(L, k)]})"
        for a, col in zip(ax, ["greedy_falls", "greedy_reward", "greedy_zero_fall_episodes"]):
            line(a, *agg_greedy(L, k, col), c, label=lab if col == "greedy_falls" else None, marker="o")
    for a, t in zip(ax, ["Greedy policy: falls per episode", "Greedy policy: reward per episode", "Greedy policy: share of episodes with 0 falls"]):
        a.set_title(t, loc="left", fontsize=10.5); a.set_xlabel("training episode"); style(a)
    ax[1].axhline(MAX_REWARD, color=ink2, ls="--", lw=1); ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Bike board v2.1: {L} lanes, 240 rows; greedy evaluation every 100 episodes on 10 fresh boards; line = mean of seeds, band = min-max", x=.01, ha="left", fontsize=10.5)
    plt.tight_layout(); plt.savefig(out / "greedy_evaluation.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, c in sols:
        for a, col in zip(ax, ["reward", "falls", "holes_passed"]):
            line(a, *agg_train(L, k, col), c, label=f"{k} solution{'s' if k > 1 else ''}", lw=1.8)
    for a, t in zip(ax, ["Reward per training episode", "Falls per training episode", "Holes passed without collision"]):
        a.set_title(t + " (50-ep rolling)", loc="left", fontsize=10.5); a.set_xlabel("episode"); style(a)
    ax[0].axhline(MAX_REWARD, color=ink2, ls="--", lw=1); ax[0].legend(frameon=False)
    fig.suptitle(f"v2.1, {L} lanes: training episodes (epsilon-greedy exploration included); line = mean of seeds, band = min-max", x=.01, ha="left", fontsize=10.5)
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=140); plt.close()

    rows = []
    for k, _ in sols:
        for r in runs[(L, k)]:
            g = r["greedy"]; ok = g[g.greedy_zero_fall_episodes >= 1.0]
            rows.append({"lanes": L, "solutions_requested": k, "solutions_in_board": r["board"]["solutions"], "seed": r["seed"],
                         "final_greedy_falls_50ep": r["final"]["greedy_falls"], "final_greedy_reward_50ep": r["final"]["greedy_reward"],
                         "first_episode_all_10_greedy_eps_fall_free": int(ok.episode.iloc[0]) if len(ok) else np.nan,
                         "train_reward_last100": r["train"].tail(100)["reward"].mean(), "train_falls_last100": r["train"].tail(100)["falls"].mean()})
    pd.DataFrame(rows).to_csv(out / "per_seed_summary.csv", index=False)
    for k, _ in sols:
        b = v20.Board(v20.BoardConfig(L, 240, k, 1, 2.0, 10.0, 1))
        v20.plot_board(b, out / "boards" / f"board_lanes{L}_rows240_sol{k}.png", strip_rows=60, max_routes=40)

# ---------------- comparison across lane counts (training episodes: reward + falls) ----------------
cmp_dir = ROOT / "lane_comparison"; cmp_dir.mkdir(parents=True, exist_ok=True)
Ks = [k for k, _ in SOL if any((L, k) in runs for L in LANES)]
TCOLS = [("reward", "Reward per training episode (50-ep rolling)"), ("falls", "Falls per training episode (50-ep rolling)")]
SUB = "240 rows, training episodes; line = mean of seeds, band = min-max"

fig, ax = plt.subplots(len(Ks), 2, figsize=(12, 3.3 * len(Ks)), sharex=True, squeeze=False)
for i, k in enumerate(Ks):
    for L in LANES:
        if (L, k) not in runs: continue
        for j, (col, _) in enumerate(TCOLS):
            line(ax[i, j], *agg_train(L, k, col), LANE_COL[L], label=f"{L} lanes (n={n_seeds[(L, k)]})" if (i == 0 and j == 0) else None, lw=1.8)
    ax[i, 0].set_ylabel(f"{k} solution{'s' if k > 1 else ''}", color=ink, fontsize=11); ax[i, 0].axhline(MAX_REWARD, color=ink2, ls="--", lw=1)
    for a in ax[i]: style(a)
for a, (_, t) in zip(ax[0], TCOLS): a.set_title(t, loc="left", fontsize=10.5)
for a in ax[-1]: a.set_xlabel("training episode")
ax[0, 0].legend(frameon=False)
fig.suptitle(f"Bike board v2.1: effect of the number of lanes (rows = number of solutions); {SUB}", x=.01, ha="left", fontsize=10.5)
plt.tight_layout(); plt.savefig(cmp_dir / "compare_lanes_all_solutions.png", dpi=130); plt.close()

for k in Ks:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for L in LANES:
        if (L, k) not in runs: continue
        for a, (col, _) in zip(ax, TCOLS):
            line(a, *agg_train(L, k, col), LANE_COL[L], lw=1.8,
                 label=f"{L} lanes (board: {runs[(L, k)][0]['board']['solutions']} solutions, n={n_seeds[(L, k)]})" if col == "reward" else None)
    for a, (_, t) in zip(ax, TCOLS): a.set_title(t, loc="left", fontsize=10.5); a.set_xlabel("training episode"); style(a)
    ax[0].axhline(MAX_REWARD, color=ink2, ls="--", lw=1); ax[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"v2.1, {k} solution{'s' if k > 1 else ''}: 2 vs 3 vs 5 vs 10 lanes; {SUB}", x=.01, ha="left", fontsize=10)
    plt.tight_layout(); plt.savefig(cmp_dir / f"compare_lanes_{k}_solutions.png", dpi=140); plt.close()

def per_seed_first_below(L, k, thr=10.0):
    out = []
    for r in runs[(L, k)]:
        y = roll(r["train"]["falls"]); idx = np.where(y < thr)[0]; out.append(int(EP - len(y) + 1 + idx[0]) if len(idx) else np.nan)
    return out
fig, ax = plt.subplots(1, 3, figsize=(17, 3.9))
heat = [(lambda L, k: per_seed_first_below(L, k), "Episode when training falls first < 10 (mean ± std over seeds)", "{:.0f} ± {:.0f}"),
        (lambda L, k: [r["train"].tail(100)["falls"].mean() for r in runs[(L, k)]], "Training falls per episode, last 100 episodes (mean ± std)", "{:.2f} ± {:.2f}"),
        (lambda L, k: [r["final"]["greedy_falls"] for r in runs[(L, k)]], "Final greedy falls per episode (mean ± std)", "{:.2f} ± {:.2f}")]
for a, (fn, ttl, fmt) in zip(ax, heat):
    M = np.full((len(LANES), len(Ks)), np.nan); S = np.full_like(M, np.nan)
    for i, L in enumerate(LANES):
        for j, k in enumerate(Ks):
            if (L, k) in runs: v = np.array(fn(L, k), float); M[i, j], S[i, j] = np.nanmean(v), np.nanstd(v)
    a.imshow(M, cmap="Blues", aspect="auto"); a.set_xticks(range(len(Ks))); a.set_xticklabels(Ks); a.set_yticks(range(len(LANES))); a.set_yticklabels(LANES)
    a.set_xlabel("requested solutions"); a.set_ylabel("lanes"); a.set_title(ttl, loc="left", fontsize=9)
    for i in range(len(LANES)):
        for j in range(len(Ks)):
            if not np.isnan(M[i, j]):
                dark = (M[i, j] - np.nanmin(M)) / max(np.nanmax(M) - np.nanmin(M), 1e-9) > .5
                a.text(j, i, fmt.format(M[i, j], S[i, j]), ha="center", va="center", fontsize=8, color="white" if dark else ink)
plt.tight_layout(); plt.savefig(cmp_dir / "summary_heatmaps.png", dpi=140); plt.close()

allrows = []
for (L, k) in sorted(runs):
    rs = runs[(L, k)]
    allrows.append({"lanes": L, "solutions_requested": k, "solutions_in_board": rs[0]["board"]["solutions"], "obs_size": L + L // 2 + 1, "seeds": len(rs),
                    "train_reward_last100_mean": np.mean([r["train"].tail(100)["reward"].mean() for r in rs]),
                    "train_reward_last100_std": np.std([r["train"].tail(100)["reward"].mean() for r in rs]),
                    "train_falls_last100_mean": np.mean([r["train"].tail(100)["falls"].mean() for r in rs]),
                    "train_falls_last100_std": np.std([r["train"].tail(100)["falls"].mean() for r in rs]),
                    "train_episode_falls_below_10": first_below(L, k),
                    "final_greedy_falls_mean": np.mean([r["final"]["greedy_falls"] for r in rs]),
                    "seeds_with_zero_greedy_falls": int(sum(r["final"]["greedy_falls"] == 0 for r in rs)),
                    "final_greedy_reward_mean": np.mean([r["final"]["greedy_reward"] for r in rs])})
pd.DataFrame(allrows).to_csv(cmp_dir / "all_runs_summary.csv", index=False)
print(pd.DataFrame(allrows).round(2).to_string(index=False))
