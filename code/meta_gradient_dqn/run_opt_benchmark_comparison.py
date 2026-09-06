"""
Run opt1/opt2/opt3 confidence meta-gradient benchmarks and plot comparisons.

Example:
    python code/run_opt_benchmark_comparison.py --seed 1 --total-episodes 5000
    python code/run_opt_benchmark_comparison.py --plot-only --seed 1 --total-episodes 5000
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
FIG_DIR = ROOT / "paper" / "figures" / "meta_gradient_confidence"
RESULTS_DIR = FIG_DIR / "results"

OPT_VARIANTS = (
    ("opt1", "opt1_dedicated_subencoders", "Opt1: dedicated sub-encoders"),
    ("opt2", "opt2_enriched_meta_features", "Opt2: enriched meta features"),
    ("opt3", "opt3_dynamic_aux_weights", "Opt3: dynamic aux weights"),
)

COLORS = {
    "baseline": "#2ca02c",
    "meta": "#ff7f0e",
    "opt1": "#1f77b4",
    "opt2": "#9467bd",
    "opt3": "#d62728",
}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def _episode_axis_for_steps(
    steps: np.ndarray, episode_returns: list[float], global_steps: int
) -> np.ndarray:
    """Map training steps to episode index using CartPole cumulative returns."""
    if len(episode_returns) == 0:
        return steps.astype(np.float64)
    episode_end_steps = np.cumsum(np.asarray(episode_returns, dtype=np.float64))
    if global_steps > episode_end_steps[-1]:
        episode_end_steps = np.concatenate(
            [
                episode_end_steps,
                np.array([float(global_steps)], dtype=np.float64),
            ]
        )
        episode_axis = np.arange(1, len(episode_end_steps) + 1, dtype=np.float64)
    else:
        episode_axis = np.arange(1, len(episode_end_steps) + 1, dtype=np.float64)
    return np.interp(steps, episode_end_steps, episode_axis)


def _plot_reward_curves(
    series: dict[str, list[float]],
    *,
    color_keys: dict[str, str] | None,
    seed: int,
    env_id: str,
    title_suffix: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, returns in series.items():
        values = np.asarray(returns, dtype=np.float64)
        episodes = np.arange(1, len(values) + 1)
        color_key = (color_keys or {}).get(label, label)
        color = COLORS.get(color_key, "#333333")
        ax.plot(episodes, values, color=color, alpha=0.2, lw=1)
        if smooth_window > 1 and len(values) >= smooth_window:
            smoothed = _smooth(values, smooth_window)
            smooth_episodes = np.arange(
                smooth_window, smooth_window + len(smoothed)
            )
            ax.plot(
                smooth_episodes,
                smoothed,
                color=color,
                lw=2.2,
                label=f"{label} (seed={seed})",
            )
        else:
            ax.plot(
                episodes, values, color=color, lw=2.0, label=f"{label} (seed={seed})"
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"{env_id}: {title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_hyperparameters(
    histories: dict[str, dict],
    *,
    color_keys: dict[str, str] | None,
    seed: int,
    title_suffix: str,
    output_path: Path,
    by_episode: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_gamma, ax_lr = axes

    for key, blob in histories.items():
        history = blob.get("controller_history", [])
        if not history:
            continue
        steps = np.asarray([entry["step"] for entry in history], dtype=np.float64)
        gamma = np.asarray([entry["gamma"] for entry in history], dtype=np.float64)
        lr = np.asarray([entry["learning_rate"] for entry in history], dtype=np.float64)
        if by_episode:
            x = _episode_axis_for_steps(
                steps,
                blob["episode_returns"],
                int(blob.get("global_steps", steps[-1])),
            )
            x_label = "Episode"
        else:
            x = steps
            x_label = "Global step"
        color_key = (color_keys or {}).get(key, key)
        color = COLORS.get(color_key, "#333333")
        label = key
        ax_gamma.plot(x, gamma, color=color, lw=1.5, alpha=0.9, label=label)
        ax_lr.plot(x, lr, color=color, lw=1.5, alpha=0.9, label=label)

    ax_gamma.set_ylabel("Gamma")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_xlabel(x_label)
    ax_gamma.set_title(f"Meta-controller hyperparameters ({title_suffix}, seed={seed})")
    ax_gamma.grid(True, alpha=0.3)
    ax_lr.grid(True, alpha=0.3)
    ax_gamma.legend(loc="best", fontsize=9)
    ax_lr.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    has_aux = any(
        "auxiliary_weight" in entry
        for blob in histories.values()
        for entry in blob.get("controller_history", [])
    )
    if not has_aux:
        return

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for key, blob in histories.items():
        history = blob.get("controller_history", [])
        if not history or "auxiliary_weight" not in history[0]:
            continue
        steps = np.asarray([entry["step"] for entry in history], dtype=np.float64)
        aux = np.asarray(
            [entry["auxiliary_weight"] for entry in history], dtype=np.float64
        )
        if by_episode:
            x = _episode_axis_for_steps(
                steps,
                blob["episode_returns"],
                int(blob.get("global_steps", steps[-1])),
            )
            x_label = "Episode"
        else:
            x = steps
            x_label = "Global step"
        color_key = (color_keys or {}).get(key, key)
        ax.plot(x, aux, color=COLORS.get(color_key, "#333333"), lw=1.5, alpha=0.9, label=key)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Auxiliary weight")
    ax.set_title(f"Dynamic auxiliary weight ({title_suffix}, seed={seed})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    aux_path = output_path.with_name(
        output_path.stem.replace("gamma_lr", "aux_weight") + output_path.suffix
    )
    fig.savefig(aux_path, dpi=160)
    plt.close(fig)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_variant(module_name: str, args_kwargs: dict, output_path: Path) -> dict:
    module = importlib.import_module(module_name)
    args_cls = module.Args
    run_training = module.run_training
    args_kwargs = dict(args_kwargs)
    args_kwargs["output_json"] = str(output_path)
    return run_training(args_cls(**args_kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and plot opt1/opt2/opt3 benchmark comparisons."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=5000)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    cli_args = parse_args()
    budget_tag = f"{cli_args.total_episodes}ep"
    title_suffix = f"{cli_args.total_episodes} episodes"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    opt_paths: dict[str, Path] = {}
    opt_results: dict[str, dict] = {}
    common_kwargs = {
        "env_id": cli_args.env_id,
        "seed": cli_args.seed,
        "total_episodes": cli_args.total_episodes,
        "total_timesteps": 500_000,
        "meta_gradient": True,
        "cuda": cli_args.cuda,
        "tensorboard": False,
        "log_interval": 1_000,
    }

    for tag, module_name, label in OPT_VARIANTS:
        output_path = RESULTS_DIR / f"{tag}_seed{cli_args.seed}_{budget_tag}.json"
        opt_paths[tag] = output_path
        if cli_args.plot_only:
            if not output_path.exists():
                raise FileNotFoundError(f"Missing results: {output_path}")
            opt_results[tag] = _load_json(output_path)
            continue
        print(f"Running {label}...", flush=True)
        opt_results[tag] = _run_variant(module_name, common_kwargs, output_path)
        returns = opt_results[tag]["episode_returns"]
        print(
            f"{label} finished: {len(returns)} episodes, "
            f"last 50 mean = {np.mean(returns[-50:]):.2f}, "
            f"final gamma = {opt_results[tag]['final_gamma']:.5f}, "
            f"final lr = {opt_results[tag]['final_learning_rate']:.2e}",
            flush=True,
        )

    baseline_path = RESULTS_DIR / f"baseline_seed{cli_args.seed}_{budget_tag}.json"
    meta_path = RESULTS_DIR / f"meta_seed{cli_args.seed}_{budget_tag}.json"
    if not baseline_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing baseline/meta results in {RESULTS_DIR}. "
            "Run run_meta_gradient_confidence_comparison.py first."
        )
    baseline_blob = _load_json(baseline_path)
    meta_blob = _load_json(meta_path)

    opt_labels = {tag: label for tag, _, label in OPT_VARIANTS}
    opt_color_keys = {label: tag for tag, label in opt_labels.items()}
    _plot_reward_curves(
        {
            opt_labels[tag]: opt_results[tag]["episode_returns"]
            for tag, _, _ in OPT_VARIANTS
        },
        color_keys=opt_color_keys,
        seed=cli_args.seed,
        env_id=cli_args.env_id,
        title_suffix=f"benchmark variants ({title_suffix})",
        smooth_window=cli_args.smooth_window,
        output_path=FIG_DIR / f"reward_opt_benchmarks_seed{cli_args.seed}_{budget_tag}.png",
    )

    opt_histories = {
        opt_labels[tag]: opt_results[tag] for tag, _, _ in OPT_VARIANTS
    }
    meta_histories = {
        "Confidence meta (original)": meta_blob,
        **opt_histories,
    }
    all_color_keys = {
        "Confidence meta (original)": "meta",
        **opt_color_keys,
    }
    _plot_hyperparameters(
        opt_histories,
        color_keys=opt_color_keys,
        seed=cli_args.seed,
        title_suffix=f"opt benchmarks ({title_suffix})",
        output_path=FIG_DIR / f"gamma_lr_opt_benchmarks_seed{cli_args.seed}_{budget_tag}.png",
        by_episode=False,
    )
    _plot_hyperparameters(
        opt_histories,
        color_keys=opt_color_keys,
        seed=cli_args.seed,
        title_suffix=f"opt benchmarks ({title_suffix})",
        output_path=FIG_DIR
        / f"gamma_lr_opt_benchmarks_seed{cli_args.seed}_{budget_tag}_by_episode.png",
        by_episode=True,
    )
    _plot_hyperparameters(
        meta_histories,
        color_keys=all_color_keys,
        seed=cli_args.seed,
        title_suffix=f"original + opt benchmarks ({title_suffix})",
        output_path=FIG_DIR
        / f"gamma_lr_all_meta_seed{cli_args.seed}_{budget_tag}_by_episode.png",
        by_episode=True,
    )

    all_series = {
        "CleanRL DQN baseline": baseline_blob["episode_returns"],
        "Confidence meta (original)": meta_blob["episode_returns"],
        **{opt_labels[tag]: opt_results[tag]["episode_returns"] for tag, _, _ in OPT_VARIANTS},
    }
    full_color_keys = {
        "CleanRL DQN baseline": "baseline",
        "Confidence meta (original)": "meta",
        **opt_color_keys,
    }
    _plot_reward_curves(
        all_series,
        color_keys=full_color_keys,
        seed=cli_args.seed,
        env_id=cli_args.env_id,
        title_suffix=f"baseline vs original meta vs opt benchmarks ({title_suffix})",
        smooth_window=cli_args.smooth_window,
        output_path=FIG_DIR / f"reward_all_variants_seed{cli_args.seed}_{budget_tag}_by_episode.png",
    )

    print("Saved plots:", flush=True)
    print(f"  {FIG_DIR / f'reward_opt_benchmarks_seed{cli_args.seed}_{budget_tag}.png'}")
    print(
        f"  {FIG_DIR / f'gamma_lr_opt_benchmarks_seed{cli_args.seed}_{budget_tag}.png'}"
    )
    print(
        f"  {FIG_DIR / f'gamma_lr_opt_benchmarks_seed{cli_args.seed}_{budget_tag}_by_episode.png'}"
    )
    print(
        f"  {FIG_DIR / f'gamma_lr_all_meta_seed{cli_args.seed}_{budget_tag}_by_episode.png'}"
    )
    print(
        f"  {FIG_DIR / f'reward_all_variants_seed{cli_args.seed}_{budget_tag}_by_episode.png'}"
    )


if __name__ == "__main__":
    main()
