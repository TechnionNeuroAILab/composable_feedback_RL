"""Train multiple lane counts and build comparison figures (reward, loss, falls)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import TrainConfig
from .plotting.compare import plot_lane_sweep_comparison
from .training import train_dqn

DEFAULT_LANE_COUNTS: tuple[int, ...] = (2, 3, 5, 10, 20)


def scenario_name_for_lanes(n_lanes: int) -> str:
    return f"lanes{n_lanes}"


def default_log_dirs(base_dir: Path, lane_counts: Sequence[int]) -> List[Path]:
    return [base_dir / scenario_name_for_lanes(n) / "seed_1" for n in lane_counts]


def run_lane_sweep(
    lane_counts: Sequence[int] = DEFAULT_LANE_COUNTS,
    *,
    base_output_dir: Path | str = "bike_outputs",
    seed: int = 1,
    total_timesteps: int = 10_000_000,
    checkpoint_save_freq: int = 100_000,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Train one DQN run per lane count; each run uses ``bike_outputs/lanes{N}/seed_{seed}`` (unless overridden).

    Returns the last training return dict and a list of all log directories.
    """
    base = Path(base_output_dir)
    log_dirs: List[Path] = []
    last: Dict[str, Any] = {}
    for n in lane_counts:
        cfg = TrainConfig(
            scenario_name=scenario_name_for_lanes(n),
            n_lanes=n,
            seed=seed,
            base_output_dir=base,
            total_timesteps=total_timesteps,
            checkpoint_save_freq=checkpoint_save_freq,
            verbose=verbose,
        )
        last = train_dqn(cfg)
        log_dirs.append(Path(last["log_dir"]))
    return {"log_dirs": log_dirs, "last_run": last}


def make_comparison_plots(
    log_dirs: Sequence[Path | str],
    labels: Sequence[str],
    output_dir: Path | str | None = None,
    *,
    base_output_dir: Path | str | None = None,
    show: bool = False,
    reward_window: int = 50,
    falls_window: int = 50,
    loss_smooth_window: int = 50,
) -> dict:
    """
    Write ``compare_reward.png``, ``compare_loss.png``, ``compare_falls.png``.

    If ``output_dir`` is None, uses ``<base_output_dir>/lane_sweep_comparison`` (default ``bike_outputs/lane_sweep_comparison``).
    """
    dirs = [Path(p) for p in log_dirs]
    lab = list(labels)
    out = Path(output_dir) if output_dir is not None else Path(base_output_dir or "bike_outputs") / "lane_sweep_comparison"
    return plot_lane_sweep_comparison(
        dirs,
        lab,
        out,
        reward_window=reward_window,
        falls_window=falls_window,
        loss_smooth_window=loss_smooth_window,
        title_prefix="Bike env",
        show=show,
    )
