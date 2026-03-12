"""
Dashboard for CleanRL DQN runs (e.g. dqn_original.py).
Same graph layout as 2layers_dqn_linear.ipynb. Reads TensorBoard events from runs/.
Figure name matches the source script (e.g. dqn_original.png).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np

# Same as notebook
FEATURE_NAMES = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
FEATURE_LABELS = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Ang. Vel"]


def find_latest_run(runs_dir: Path, run_prefix: str = "CartPole") -> Path | None:
    """Find the latest run directory under runs_dir matching prefix."""
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


def _placeholder_no_data(ax, msg: str = "Not logged by CleanRL DQN"):
    ax.set_axis_on()
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, fontsize=9, color="#718096",
            ha="center", va="center", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN Training Dashboard from TensorBoard runs")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Path to runs directory")
    parser.add_argument("--run-path", type=Path, default=None, help="Path to a specific run directory")
    parser.add_argument("--figure-name", type=str, default="dqn_original", help="Output figure base name")
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
            raise SystemExit(f"No run found under {runs_dir}. Run dqn_original.py first.")

    data = load_tb_scalars(run_dir)
    if not data:
        raise SystemExit(f"No scalar events found in {run_dir}")

    episodic_return = np.array(data.get("charts/episodic_return", []))
    episodic_length = np.array(data.get("charts/episodic_length", []))
    td_loss = np.array(data.get("losses/td_loss", []))
    q_values = np.array(data.get("losses/q_values", []))

    # Debug: always print loaded run and data counts
    print("[DEBUG] Run directory:", run_dir)
    print("[DEBUG] Runs dir searched:", runs_dir)
    if runs_dir.is_dir():
        candidates = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print("[DEBUG] All run dirs (newest first):", [d.name for d in candidates if d.is_dir()])
    print("[DEBUG] TensorBoard tags found:", list(data.keys()))
    print("[DEBUG] charts/episodic_return: len=%s shape=%s" % (len(episodic_return), getattr(episodic_return, "shape", "N/A")))
    if len(episodic_return) > 0:
        print("         first 3 (step, value):", episodic_return[:3].tolist())
        print("         last 3 (step, value):", episodic_return[-3:].tolist())
    else:
        print("         (empty — reward plot will be blank)")
    print("[DEBUG] charts/episodic_length: len=%s" % len(episodic_length))
    print("[DEBUG] losses/td_loss: len=%s" % len(td_loss))
    print("[DEBUG] losses/q_values: len=%s" % len(q_values))
    print("[DEBUG] Will plot reward:", len(episodic_return) > 0)

    # Style: match 2layers_dqn_linear.ipynb exactly
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
        "q": "#f5c518",
        "features": ["#00e5ff", "#f5c518", "#ff5f6d", "#43e97b"],
        "left": "#a78bfa",
        "right": "#f97316",
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

    # Same layout as notebook: 3 rows x 4 columns, figsize=(18, 14)
    fig = plt.figure(figsize=(18, 14), facecolor="#0a0e1a")
    fig.suptitle("DQN Training Dashboard — CartPole-v1", fontsize=14, fontweight="bold", color="#00e5ff", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38, left=0.06, right=0.97, top=0.93, bottom=0.06)
    w = args.smooth_window

    # ── Row 0: Reward | Loss | Mean Q | Dominant Feature ─────────────────────────
    # 1. Episode Reward (notebook: "Episode Reward", ylabel "Steps survived")
    ax_rew = fig.add_subplot(gs[0, 0])
    if len(episodic_return) > 0:
        steps_r = episodic_return[:, 0].astype(int)
        vals_r = episodic_return[:, 1]
        smo_r = smooth(vals_r, w)
        ax_rew.plot(steps_r, vals_r, color=COLORS["reward"], alpha=0.25, linewidth=0.6)
        ax_rew.plot(steps_r, smo_r, color=COLORS["reward"], linewidth=1.8, label="Smoothed")
        fill_between(ax_rew, steps_r, smo_r, COLORS["reward"])
    styled_ax(ax_rew, "Episode Reward", ylabel="Steps survived")

    # 2. Training Loss (MSE)
    ax_loss = fig.add_subplot(gs[0, 1])
    if len(td_loss) > 0:
        steps_l = td_loss[:, 0].astype(int)
        vals_l = td_loss[:, 1]
        smo_l = smooth(vals_l, w)
        ax_loss.plot(steps_l, vals_l, color=COLORS["loss"], alpha=0.2, linewidth=0.6)
        ax_loss.plot(steps_l, smo_l, color=COLORS["loss"], linewidth=1.8)
        fill_between(ax_loss, steps_l, smo_l, COLORS["loss"])
    styled_ax(ax_loss, "Training Loss (MSE)", ylabel="Loss")

    # 3. Mean Q Value (chosen action)
    ax_q = fig.add_subplot(gs[0, 2])
    if len(q_values) > 0:
        steps_q = q_values[:, 0].astype(int)
        vals_q = q_values[:, 1]
        smo_q = smooth(vals_q, w)
        ax_q.plot(steps_q, vals_q, color=COLORS["q"], alpha=0.2, linewidth=0.6)
        ax_q.plot(steps_q, smo_q, color=COLORS["q"], linewidth=1.8)
        fill_between(ax_q, steps_q, smo_q, COLORS["q"])
    styled_ax(ax_q, "Mean Q Value (chosen action)", ylabel="Q")

    # 4. Dominant Feature (steps) — notebook: barh with FEATURE_LABELS; we have no data
    ax_dom = fig.add_subplot(gs[0, 3])
    dom_counts = np.zeros(len(FEATURE_NAMES))
    bars = ax_dom.barh(FEATURE_LABELS, dom_counts, color=COLORS["features"], edgecolor="none", height=0.55)
    ax_dom.invert_yaxis()
    styled_ax(ax_dom, "Dominant Feature (steps)", xlabel="Step count", ylabel="")
    ax_dom.text(0.5, 0.5, "Not logged\nby CleanRL DQN", transform=ax_dom.transAxes,
                fontsize=8, color="#718096", ha="center", va="center")

    # ── Row 1: Q left vs Q right (span 2) | Mean |Feature Contribution| (span 2) ──
    # 5. Q Value per Action over Episodes (Q left vs Q right)
    ax_ql = fig.add_subplot(gs[1, :2])
    if len(q_values) > 0:
        steps_q = q_values[:, 0].astype(int)
        vals_q = q_values[:, 1]
        smo_q = smooth(vals_q, w)
        ax_ql.plot(steps_q, smo_q, color=COLORS["q"], linewidth=1.6, label="Mean Q (both actions)")
        fill_between(ax_ql, steps_q, smo_q, COLORS["q"])
        ax_ql.text(0.02, 0.98, "Q left/right not logged; showing mean Q", transform=ax_ql.transAxes,
                   fontsize=7, color="#718096", va="top")
    else:
        _placeholder_no_data(ax_ql, "Q left/right not logged by CleanRL DQN")
    styled_ax(ax_ql, "Q Value per Action over Episodes", ylabel="Mean Q")
    ax_ql.legend(fontsize=8, framealpha=0.15)

    # 6. Mean |Feature Contribution| to Chosen Action — no data
    ax_fc = fig.add_subplot(gs[1, 2:])
    _placeholder_no_data(ax_fc, "Mean |Feature Contribution| not logged by CleanRL DQN")
    styled_ax(ax_fc, "Mean |Feature Contribution| to Chosen Action", ylabel="|Contribution|")

    # ── Row 2: Individual feature contribution plots [4 subplots] ─────────────────
    for fi, flabel in enumerate(FEATURE_LABELS):
        ax = fig.add_subplot(gs[2, fi])
        _placeholder_no_data(ax, f"{flabel}\n(not logged)")
        styled_ax(ax, flabel, ylabel="Contribution")

    out_path = figures_dir / f"{args.figure_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
