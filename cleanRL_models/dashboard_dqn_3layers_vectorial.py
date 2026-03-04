"""
Dashboard for dqn_3layers_vectorial.py runs.
Reads TensorBoard events from runs/. Output figure: dqn_3layers_vectorial.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np

FEATURE_NAMES = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
FEATURE_LABELS = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Ang. Vel"]


def find_latest_run(runs_dir: Path, run_prefix: str = "dqn_3layers_vectorial") -> Path | None:
    """Find the latest run directory under runs_dir whose name contains run_prefix."""
    if not runs_dir.is_dir():
        return None
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and run_prefix in d.name]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_tb_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Load TensorBoard scalar tags from a run directory. Returns {tag: [(step, value), ...]}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()

    result = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]
    return result


def smooth(y: np.ndarray, w: int = 30) -> np.ndarray:
    """Rolling mean, same length as y (min_periods=1)."""
    if len(y) == 0:
        return y
    w = min(w, len(y))
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        start = max(0, i - w + 1)
        out[i] = np.nanmean(y[start : i + 1])
    return np.nan_to_num(out, nan=y[0] if len(y) else 0)


def _placeholder_no_data(ax, msg: str = "No data"):
    ax.set_axis_on()
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, fontsize=9, color="#718096",
            ha="center", va="center", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard for dqn_3layers_vectorial from TensorBoard runs")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Path to runs directory")
    parser.add_argument("--run-path", type=Path, default=None, help="Path to a specific run directory")
    parser.add_argument("--figure-name", type=str, default="dqn_3layers_vectorial", help="Output figure base name")
    parser.add_argument("--smooth-window", type=int, default=30, help="Smoothing window")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    runs_dir = args.runs_dir or script_dir / "runs"
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.run_path is not None:
        run_dir = Path(args.run_path)
        if not run_dir.is_dir():
            raise SystemExit(f"Run path is not a directory: {run_dir}")
    else:
        run_dir = find_latest_run(runs_dir)
        if run_dir is None:
            raise SystemExit(f"No run found under {runs_dir} matching 'dqn_3layers_vectorial'. Run dqn_3layers_vectorial.py first.")

    data = load_tb_scalars(run_dir)
    if not data:
        raise SystemExit(f"No scalar events found in {run_dir}")

    episodic_return = np.array(data.get("charts/episodic_return", []))
    td_loss = np.array(data.get("losses/td_loss", []))
    q_values = np.array(data.get("losses/q_values", []))
    vectorial_loss = np.array(data.get("losses/vectorial_loss", []))
    reward_per_feature = np.array(data.get("vectorial/reward_per_feature_mean", []))
    contribution_norm = np.array(data.get("vectorial/contribution_norm_mean", []))
    L_i = [np.array(data.get(f"losses/L_i_feature_{i}", [])) for i in range(4)]

    plt.rcParams.update({
        "figure.facecolor": "#0a0e1a",
        "axes.facecolor": "#0f1525",
        "axes.edgecolor": "#1c2540",
        "axes.labelcolor": "#ffffff",
        "axes.titlecolor": "#ffffff",
        "xtick.color": "#ffffff",
        "ytick.color": "#ffffff",
        "grid.color": "#1c2540",
        "grid.linewidth": 0.6,
        "text.color": "#ffffff",
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    COLORS = {
        "reward": "#43e97b",
        "loss": "#ff5f6d",
        "td": "#ff5f6d",
        "vec": "#a78bfa",
        "q": "#f5c518",
        "features": ["#00e5ff", "#f5c518", "#ff5f6d", "#43e97b"],
        "fill": 0.15,
    }

    def styled_ax(ax, title: str, xlabel: str = "Step", ylabel: str = ""):
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="#e2e8f0")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))

    def fill_between(ax, x, y, color, alpha=0.12):
        ax.fill_between(x, y, alpha=alpha, color=color)

    def plot_series(ax, arr, color, label=None):
        if len(arr) == 0:
            return
        steps = arr[:, 0].astype(int)
        vals = arr[:, 1]
        smo = smooth(vals, args.smooth_window)
        ax.plot(steps, vals, color=color, alpha=0.2, linewidth=0.6)
        ax.plot(steps, smo, color=color, linewidth=1.8, label=label)
        fill_between(ax, steps, smo, color)

    fig = plt.figure(figsize=(18, 14), facecolor="#0a0e1a")
    fig.suptitle("DQN 3-Layers Vectorial — CartPole-v1", fontsize=14, fontweight="bold", color="#00e5ff", y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38, left=0.06, right=0.97, top=0.93, bottom=0.06)

    # Row 0: Episode reward | TD loss | Vectorial loss | Mean Q
    ax_rew = fig.add_subplot(gs[0, 0])
    if len(episodic_return) > 0:
        arr = np.array(episodic_return)
        if arr.ndim == 1:
            arr = np.column_stack([np.arange(len(arr)), arr])
        plot_series(ax_rew, arr, COLORS["reward"])
    else:
        _placeholder_no_data(ax_rew, "No episodic return")
    styled_ax(ax_rew, "Episode Reward", ylabel="Return")

    ax_td = fig.add_subplot(gs[0, 1])
    if len(td_loss) > 0:
        plot_series(ax_td, np.array(td_loss), COLORS["td"])
    else:
        _placeholder_no_data(ax_td, "No TD loss")
    styled_ax(ax_td, "TD Loss", ylabel="Loss")

    ax_vec = fig.add_subplot(gs[0, 2])
    if len(vectorial_loss) > 0:
        plot_series(ax_vec, np.array(vectorial_loss), COLORS["vec"])
    else:
        _placeholder_no_data(ax_vec, "No vectorial loss")
    styled_ax(ax_vec, "Vectorial Loss $L_{\\mathrm{vec}}$", ylabel="Loss")

    ax_q = fig.add_subplot(gs[0, 3])
    if len(q_values) > 0:
        plot_series(ax_q, np.array(q_values), COLORS["q"])
    else:
        _placeholder_no_data(ax_q, "No Q values")
    styled_ax(ax_q, "Mean Q (chosen action)", ylabel="Q")

    # Row 1: Reward per feature | Contribution norm | L_i for features 0,1
    ax_rpf = fig.add_subplot(gs[1, 0])
    if len(reward_per_feature) > 0:
        plot_series(ax_rpf, np.array(reward_per_feature), COLORS["features"][0])
    else:
        _placeholder_no_data(ax_rpf, "No reward/feature")
    styled_ax(ax_rpf, "Reward per feature $r/n$", ylabel="$r/n$")

    ax_cn = fig.add_subplot(gs[1, 1])
    if len(contribution_norm) > 0:
        plot_series(ax_cn, np.array(contribution_norm), COLORS["features"][1])
    else:
        _placeholder_no_data(ax_cn, "No contribution norm")
    styled_ax(ax_cn, "Mean $\\|$contribution$\\|$", ylabel="Norm")

    ax_l0 = fig.add_subplot(gs[1, 2])
    if len(L_i[0]) > 0:
        plot_series(ax_l0, np.array(L_i[0]), COLORS["features"][0], FEATURE_LABELS[0])
    else:
        _placeholder_no_data(ax_l0, f"No $L_i$ feature 0")
    styled_ax(ax_l0, f"$L_0$ ({FEATURE_LABELS[0]})", ylabel="$L_i$")

    ax_l1 = fig.add_subplot(gs[1, 3])
    if len(L_i[1]) > 0:
        plot_series(ax_l1, np.array(L_i[1]), COLORS["features"][1], FEATURE_LABELS[1])
    else:
        _placeholder_no_data(ax_l1, f"No $L_i$ feature 1")
    styled_ax(ax_l1, f"$L_1$ ({FEATURE_LABELS[1]})", ylabel="$L_i$")

    # Row 2: L_2, L_3, and all L_i
    ax_l2 = fig.add_subplot(gs[2, 0])
    if len(L_i[2]) > 0:
        plot_series(ax_l2, np.array(L_i[2]), COLORS["features"][2], FEATURE_LABELS[2])
    else:
        _placeholder_no_data(ax_l2, f"No $L_i$ feature 2")
    styled_ax(ax_l2, f"$L_2$ ({FEATURE_LABELS[2]})", ylabel="$L_i$")

    ax_l3 = fig.add_subplot(gs[2, 1])
    if len(L_i[3]) > 0:
        plot_series(ax_l3, np.array(L_i[3]), COLORS["features"][3], FEATURE_LABELS[3])
    else:
        _placeholder_no_data(ax_l3, f"No $L_i$ feature 3")
    styled_ax(ax_l3, f"$L_3$ ({FEATURE_LABELS[3]})", ylabel="$L_i$")

    # All L_i on one plot
    ax_all = fig.add_subplot(gs[2, 2:])
    has_any = False
    for i in range(4):
        if len(L_i[i]) > 0:
            arr = np.array(L_i[i])
            steps = arr[:, 0].astype(int)
            vals = arr[:, 1]
            smo = smooth(vals, args.smooth_window)
            ax_all.plot(steps, smo, color=COLORS["features"][i], linewidth=1.4, label=FEATURE_LABELS[i])
            has_any = True
    if has_any:
        ax_all.legend(fontsize=8, framealpha=0.15)
    else:
        _placeholder_no_data(ax_all, "No $L_i$ data")
    styled_ax(ax_all, "Per-feature loss $L_i$ (smoothed)", ylabel="$L_i$")

    out_path = figures_dir / f"{args.figure_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
