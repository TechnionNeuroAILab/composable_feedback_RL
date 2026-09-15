#!/usr/bin/env python3
"""Plot SEARL HalfCheetah logs with paper-style fair env-step x-axis.

Looks under results/searl/ for CSV/JSON produced by the vendor logger and
writes figures/searl_*.jpg. Exact filenames depend on SEARL's FolderHandler;
this script discovers common patterns.

Fair protocol reminder: x-axis must be TOTAL population environment steps
(not per-worker steps). Random-search baselines should multiply by #configs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "searl"
DEFAULT_FIG = ROOT / "figures"


def _find_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def _load_numeric_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    # Prefer explicit fair-step columns if present
    x_keys = [
        "total_frames",
        "total_env_steps",
        "frames",
        "env_steps",
        "steps",
        "timestep",
    ]
    y_keys = [
        "fitness",
        "eval_return",
        "return",
        "reward",
        "mean_reward",
        "test_reward",
        "episode_reward",
    ]
    x_col = next((cols[k] for k in x_keys if k in cols), None)
    y_col = next((cols[k] for k in y_keys if k in cols), None)
    if x_col is None or y_col is None:
        # fall back to first two numeric columns
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return None
        x = numeric.iloc[:, 0].to_numpy(dtype=float)
        y = numeric.iloc[:, 1].to_numpy(dtype=float)
        return x, y
    return df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float)


def plot_learning_curves(result_dir: Path, out_path: Path) -> None:
    csvs = _find_csvs(result_dir)
    if not csvs:
        # try json summaries
        jsons = sorted(result_dir.rglob("*.json"))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            f"No CSV logs under\n{result_dir}\n({len(jsons)} json files found)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_searl] placeholder -> {out_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for csv_path in csvs[:40]:
        curve = _load_numeric_curve(csv_path)
        if curve is None:
            continue
        x, y = curve
        label = csv_path.parent.name[:40]
        ax.plot(x, y, alpha=0.7, label=label)
        plotted += 1
    ax.set_xscale("log")
    ax.set_xlabel("Total environment steps (fair protocol)")
    ax.set_ylabel("Return / fitness")
    ax.set_title("SEARL HalfCheetah (discovered CSV curves)")
    ax.grid(True, alpha=0.3)
    if plotted and plotted <= 12:
        ax.legend(fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_searl] wrote {out_path} ({plotted} curves)")


def plot_architecture_summary(result_dir: Path, out_path: Path) -> None:
    """Scan logs for architecture-related fields; paper Table 3 ~2.8 layers / ~1019 nodes."""
    rows = []
    for path in result_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("SEARL architecture scan (paper Table 3: ~2.8 layers, ~1019 nodes)")
    ax.text(
        0.5,
        0.5,
        f"Found {len(rows)} JSON blobs under {result_dir.name}\n"
        "Inspect actor hidden_size traces in SEARL checkpoints for Fig 4.",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_searl] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot_learning_curves(args.results, args.fig_dir / "searl_halfcheetah_learning.jpg")
    plot_architecture_summary(
        args.results, args.fig_dir / "searl_halfcheetah_architecture_note.jpg"
    )


if __name__ == "__main__":
    main()
