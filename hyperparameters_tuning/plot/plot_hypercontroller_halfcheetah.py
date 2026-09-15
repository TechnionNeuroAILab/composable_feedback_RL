#!/usr/bin/env python3
"""Plot HyperController HalfCheetah-v4 paper-style figures.

Reads logs.csv / eval_logs.csv from results/hypercontroller/... folders
named like HalfCheetah-v4_{method}_seed{i}/.

Produces:
  - median train reward vs HPO wall-clock (Fig 1 left)
  - median eval reward sum vs wall-clock (Fig 1 right)
  - boxplots at final iteration (Fig 2)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "hypercontroller"
DEFAULT_FIG = ROOT / "figures"

METHODS = [
    "HyperController",
    "Random",
    "Random_Start",
    "HyperBand",
    "GP-UCB",
    "PB2",
]
DIR_RE = re.compile(
    r"HalfCheetah-v4_(?P<method>.+)_seed(?P<seed>\d+)$"
)


def _discover_runs(root: Path) -> dict[str, list[Path]]:
    by_method: dict[str, list[Path]] = {m: [] for m in METHODS}
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        m = DIR_RE.search(path.name)
        if not m:
            continue
        method = m.group("method")
        if method in by_method:
            by_method[method].append(path)
    return by_method


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / name
    if not path.exists():
        # author may nest differently
        matches = list(run_dir.rglob(name))
        if not matches:
            return None
        path = matches[0]
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _time_and_metric(df: pd.DataFrame, metric_candidates: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    cols = {c.lower(): c for c in df.columns}
    t_col = None
    for key in ("hpo_time", "time", "wall_time", "wallclock", "minutes", "t"):
        if key in cols:
            t_col = cols[key]
            break
    y_col = None
    for key in metric_candidates:
        if key.lower() in cols:
            y_col = cols[key.lower()]
            break
    if y_col is None:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            return None
        # last numeric as metric, first as time if possible
        y = numeric.iloc[:, -1].to_numpy(dtype=float)
        t = (
            numeric.iloc[:, 0].to_numpy(dtype=float)
            if t_col is None
            else df[t_col].to_numpy(dtype=float)
        )
        return t, y
    t = (
        np.arange(len(df), dtype=float)
        if t_col is None
        else df[t_col].to_numpy(dtype=float)
    )
    y = df[y_col].to_numpy(dtype=float)
    return t, y


def _median_curve(runs: list[Path], csv_name: str, metrics: list[str]):
    series = []
    for run in runs:
        df = _read_csv(run, csv_name)
        if df is None or df.empty:
            continue
        pair = _time_and_metric(df, metrics)
        if pair is None:
            continue
        series.append(pair)
    return series


def plot_fig1(by_method: dict[str, list[Path]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, runs in by_method.items():
        if not runs:
            continue
        train = _median_curve(
            runs, "logs.csv", ["reward", "train_reward", "r", "return"]
        )
        ev = _median_curve(
            runs,
            "eval_logs.csv",
            ["eval_reward_sum", "reward_sum", "reward", "return"],
        )
        for ax, series, title in (
            (axes[0], train, "Median train reward vs time"),
            (axes[1], ev, "Median eval reward sum vs time"),
        ):
            if not series:
                continue
            # interpolate onto common length via index if times mismatch
            ys = [y for _, y in series]
            min_len = min(len(y) for y in ys)
            if min_len < 1:
                continue
            stack = np.vstack([y[:min_len] for y in ys])
            med = np.median(stack, axis=0)
            ts = series[0][0][:min_len]
            # convert seconds->minutes if looks like seconds
            if np.nanmax(ts) > 60:
                ts = ts / 60.0
            ax.plot(ts, med, label=method)
    for ax, title in zip(axes, ("Train", "Eval")):
        ax.set_xscale("log")
        ax.set_xlabel("HPO wall-clock (minutes, log)")
        ax.set_ylabel("Reward")
        ax.set_title(f"HyperController HalfCheetah-v4 — {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_hc] wrote {out_path}")


def plot_fig2(by_method: dict[str, list[Path]], out_path: Path) -> None:
    train_final = []
    eval_final = []
    labels = []
    for method, runs in by_method.items():
        t_vals, e_vals = [], []
        for run in runs:
            df_t = _read_csv(run, "logs.csv")
            df_e = _read_csv(run, "eval_logs.csv")
            if df_t is not None and not df_t.empty:
                num = df_t.select_dtypes(include=[np.number])
                if num.shape[1]:
                    t_vals.append(float(num.iloc[-1, -1]))
            if df_e is not None and not df_e.empty:
                num = df_e.select_dtypes(include=[np.number])
                if num.shape[1]:
                    e_vals.append(float(num.iloc[-1, -1]))
        if t_vals or e_vals:
            labels.append(method)
            train_final.append(t_vals)
            eval_final.append(e_vals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if labels:
        axes[0].boxplot(train_final, tick_labels=labels, showfliers=False)
        axes[1].boxplot(eval_final, tick_labels=labels, showfliers=False)
    axes[0].set_title("Final train reward (t≈1000)")
    axes[1].set_title("Final eval reward sum (t≈1000)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_hc] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    by_method = _discover_runs(args.results)
    n = sum(len(v) for v in by_method.values())
    print(f"[plot_hc] discovered {n} run dirs under {args.results}")
    plot_fig1(by_method, args.fig_dir / "hypercontroller_halfcheetah_fig1.jpg")
    plot_fig2(by_method, args.fig_dir / "hypercontroller_halfcheetah_fig2.jpg")


if __name__ == "__main__":
    main()
