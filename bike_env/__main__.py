"""
CLI entry: train DQN or generate figures without importing from a notebook.

Examples::

    python -m bike_env train --scenario lanes3 --seed 1 --timesteps 500000
    python -m bike_env sweep-lanes --timesteps 500000
    python -m bike_env sweep-lanes --plots-only --base-dir bike_outputs
    python -m bike_env plots --log-dir bike_outputs/lanes3/seed_1
    python -m bike_env plots-advanced --model bike_outputs/lanes3/seed_1/bike_model_final.zip
    python -m bike_env compare --log-dirs run_a run_b --labels A B -o comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_train(ns: argparse.Namespace) -> None:
    from .config import TrainConfig
    from .training import train_dqn

    base = Path(ns.base_dir) if ns.base_dir else Path("bike_outputs")
    cfg = TrainConfig(
        scenario_name=ns.scenario,
        n_lanes=ns.lanes,
        seed=ns.seed,
        base_output_dir=base,
        total_timesteps=ns.timesteps,
        checkpoint_save_freq=ns.checkpoint_freq,
        verbose=ns.verbose,
    )
    if ns.log_dir:
        cfg.log_dir = Path(ns.log_dir)
    out = train_dqn(cfg)
    print("Training finished.")
    print(f"  log_dir: {out['log_dir']}")
    print(f"  model:   {out['final_model_path']}")


def _cmd_plots(ns: argparse.Namespace) -> None:
    from .plotting.monitor import generate_all_monitor_figures

    log_dir = Path(ns.log_dir)
    out_dir = Path(ns.output_dir) if ns.output_dir else None
    generate_all_monitor_figures(log_dir, output_dir=out_dir, show=ns.show)
    print("Monitor figures written to:", out_dir or (log_dir / "visualizations"))


def _cmd_plots_advanced(ns: argparse.Namespace) -> None:
    from .plotting.advanced import run_all_advanced_visualizations

    paths = run_all_advanced_visualizations(
        ns.model,
        n_lanes=ns.lanes,
        output_dir=Path(ns.output_dir) if ns.output_dir else None,
        n_eval_episodes=ns.eval_episodes,
        n_heatmap_episodes=ns.heatmap_episodes,
        gif_max_steps=ns.gif_steps,
        show=ns.show,
    )
    print("Advanced figures:", paths)


def _cmd_compare(ns: argparse.Namespace) -> None:
    from .plotting.compare import plot_episode_rewards_comparison

    plot_episode_rewards_comparison(
        ns.log_dirs,
        ns.labels,
        output_path=ns.output,
        rolling_window=ns.window,
        title=ns.title,
        show=ns.show,
    )
    if ns.output:
        print("Saved:", ns.output)


def _cmd_sweep_lanes(ns: argparse.Namespace) -> None:
    from .sweep import DEFAULT_LANE_COUNTS, make_comparison_plots, run_lane_sweep

    lane_counts = tuple(ns.lanes) if ns.lanes else DEFAULT_LANE_COUNTS
    base = Path(ns.base_dir) if ns.base_dir else Path("bike_outputs")

    if not ns.plots_only:
        print("Training lane counts:", lane_counts)
        out = run_lane_sweep(
            lane_counts,
            base_output_dir=base,
            seed=ns.seed,
            total_timesteps=ns.timesteps,
            checkpoint_save_freq=ns.checkpoint_freq,
            verbose=ns.verbose,
        )
        log_dirs = [str(p) for p in out["log_dirs"]]
        print("Log directories:", log_dirs)
    elif ns.log_dirs:
        log_dirs = list(ns.log_dirs)
    else:
        log_dirs = [str(base / f"lanes{n}" / f"seed_{ns.seed}") for n in lane_counts]

    if ns.labels:
        labels = list(ns.labels)
    elif not ns.plots_only:
        labels = [f"{n} lanes" for n in lane_counts]
    else:
        labels = []
        for p in log_dirs:
            folder = Path(p).parent.parent.name
            if folder.startswith("lanes") and folder[5:].isdigit():
                labels.append(f"{int(folder[5:])} lanes")
            else:
                labels.append(folder)

    if len(labels) != len(log_dirs):
        raise SystemExit("--labels must have one entry per run (same length as --log-dirs or lane list).")

    plot_out = Path(ns.plot_output) if ns.plot_output else base / "lane_sweep_comparison"
    paths = make_comparison_plots(
        log_dirs,
        labels,
        output_dir=plot_out,
        show=ns.show,
        reward_window=ns.reward_window,
        falls_window=ns.falls_window,
        loss_smooth_window=ns.loss_smooth,
    )
    print("Comparison figures:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="python -m bike_env")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Run DQN training")
    t.add_argument("--scenario", default="default", help="Scenario name (folder under base_dir)")
    t.add_argument("--lanes", type=int, default=3)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--timesteps", type=int, default=10_000_000)
    t.add_argument("--checkpoint-freq", type=int, default=100_000)
    t.add_argument("--base-dir", default=None, help="Default: bike_outputs")
    t.add_argument("--log-dir", default=None, help="Override output directory entirely")
    t.add_argument("--verbose", type=int, default=0)
    t.set_defaults(func=_cmd_train)

    pl = sub.add_parser("plots", help="Generate monitor-based figures from a log dir")
    pl.add_argument("--log-dir", required=True)
    pl.add_argument("--output-dir", default=None, help="Default: <log-dir>/visualizations")
    pl.add_argument("--show", action="store_true", help="Also open interactive windows")
    pl.set_defaults(func=_cmd_plots)

    pa = sub.add_parser("plots-advanced", help="Fall/survival/behavior/heatmap/GIF for a saved model")
    pa.add_argument("--model", required=True, help="Path to .zip checkpoint")
    pa.add_argument("--lanes", type=int, default=3)
    pa.add_argument("--output-dir", default=None)
    pa.add_argument("--eval-episodes", type=int, default=100)
    pa.add_argument("--heatmap-episodes", type=int, default=50)
    pa.add_argument("--gif-steps", type=int, default=300)
    pa.add_argument("--show", action="store_true")
    pa.set_defaults(func=_cmd_plots_advanced)

    c = sub.add_parser("compare", help="Overlay rolling rewards from multiple log dirs")
    c.add_argument("--log-dirs", nargs="+", required=True)
    c.add_argument("--labels", nargs="+", required=True)
    c.add_argument("-o", "--output", default=None)
    c.add_argument("--window", type=int, default=50)
    c.add_argument("--title", default="Episode reward (rolling mean) — multiple runs")
    c.add_argument("--show", action="store_true")
    c.set_defaults(func=_cmd_compare)

    sl = sub.add_parser(
        "sweep-lanes",
        help="Train 2,3,5,10,20 lane scenarios (or chosen --lanes) and plot reward, loss, falls",
    )
    sl.add_argument(
        "--lanes",
        type=int,
        nargs="+",
        default=None,
        help="Lane counts (default: 2 3 5 10 20)",
    )
    sl.add_argument("--seed", type=int, default=1)
    sl.add_argument("--timesteps", type=int, default=10_000_000)
    sl.add_argument("--checkpoint-freq", type=int, default=100_000)
    sl.add_argument("--base-dir", default=None, help="Default: bike_outputs")
    sl.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip training; only build plots from existing logs",
    )
    sl.add_argument(
        "--log-dirs",
        nargs="+",
        default=None,
        help="With --plots-only: explicit log dirs (same order as --labels / lane list)",
    )
    sl.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (default: 'N lanes' for each lane count)",
    )
    sl.add_argument(
        "--plot-output",
        default=None,
        help="Where to save compare_*.png (default: <base-dir>/lane_sweep_comparison)",
    )
    sl.add_argument("--reward-window", type=int, default=50)
    sl.add_argument("--falls-window", type=int, default=50)
    sl.add_argument("--loss-smooth", type=int, default=50, help="Moving average window for loss curve")
    sl.add_argument("--verbose", type=int, default=0)
    sl.add_argument("--show", action="store_true")
    sl.set_defaults(func=_cmd_sweep_lanes)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
