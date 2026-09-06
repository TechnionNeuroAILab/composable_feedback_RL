"""
Compare vanilla CleanRL DQN against Optuna-tuned vanilla DQN.

All four conditions (vanilla, Optuna A/B/C) share the same non-optimal
initial hyperparameters. Each Optuna space only searches its own knobs;
untuned knobs stay at that shared initialization.

Example:
    python code/run_optuna_vanilla_comparison.py \\
        --seed 1 --total-episodes 5000 --search-episodes 1000 --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from meta_gradient_dqn import run_training as run_vanilla_training
from optuna_cleanrl_dqn import (
    SPACE_IDS,
    SPACE_LABELS,
    SPACE_NAMES,
    SHARED_BASE_HPARAMS,
    dump_study_json,
    make_vanilla_args,
    run_optuna_search,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures" / "optuna_cleanrl_dqn"
RESULTS_DIR = FIG_DIR / "results" / "nonoptimal"

LABELS = {
    "vanilla": "Vanilla (shared non-optimal init)",
    **SPACE_LABELS,
}

COLORS = {
    "vanilla": "#2ca02c",
    "A": "#ff7f0e",
    "B": "#9467bd",
    "C": "#d62728",
}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def _last_mean(returns: list[float], n: int = 50) -> float:
    if not returns:
        return float("nan")
    return float(np.mean(returns[-min(n, len(returns)) :]))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, blob: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)


def vanilla_path(seed: int, budget_tag: str) -> Path:
    return RESULTS_DIR / f"vanilla_seed{seed}_{budget_tag}.json"


def study_db_path(space_id: str) -> Path:
    return RESULTS_DIR / f"optuna_{space_id}.db"


def study_json_path(space_id: str) -> Path:
    return RESULTS_DIR / f"optuna_{space_id}_study.json"


def best_retrain_path(space_id: str, seed: int, budget_tag: str) -> Path:
    return RESULTS_DIR / f"optuna_{space_id}_best_seed{seed}_{budget_tag}.json"


def summary_path(seed: int, budget_tag: str) -> Path:
    return RESULTS_DIR / f"summary_seed{seed}_{budget_tag}.json"


def plot_reward_comparison(
    curves: dict[str, list[float]],
    *,
    seed: int,
    env_id: str,
    title_suffix: str,
    smooth_window: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tag, returns in curves.items():
        values = np.asarray(returns, dtype=np.float64)
        episodes = np.arange(1, len(values) + 1)
        color = COLORS[tag]
        ax.plot(episodes, values, color=color, alpha=0.22, lw=1)
        if smooth_window > 1 and len(values) >= smooth_window:
            smoothed = _smooth(values, smooth_window)
            smooth_episodes = np.arange(
                smooth_window, smooth_window + len(smoothed)
            )
            ax.plot(
                smooth_episodes,
                smoothed,
                color=color,
                lw=2.5,
                label=f"{LABELS[tag]} (seed={seed})",
            )
        else:
            ax.plot(
                episodes,
                values,
                color=color,
                lw=2.0,
                label=f"{LABELS[tag]} (seed={seed})",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic return")
    ax.set_title(
        f"{env_id}: vanilla vs Optuna, shared non-optimal init ({title_suffix})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_optimization_history(
    histories: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    n_trials: int,
    search_episodes: int,
    output_path: Path,
) -> None:
    space_ids = list(histories)
    n_panels = len(space_ids)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.2 * n_panels, 4.4),
        squeeze=False,
    )
    for ax, space_id in zip(axes[0], space_ids):
        trials = histories[space_id]
        numbers = [t["number"] for t in trials]
        values = [t["value"] for t in trials]
        color = COLORS[space_id]
        ax.scatter(numbers, values, color=color, s=28, zorder=3)
        if values:
            best_so_far = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
            ax.plot(
                numbers,
                best_so_far,
                color=color,
                lw=2.0,
                label="best so far",
            )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Last-100 mean return")
        ax.set_title(LABELS[space_id])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle(
        f"Optuna search history (seed={seed}, {n_trials} trials x "
        f"{search_episodes} ep)",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_search_spaces(raw: str) -> list[str]:
    spaces = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = [item for item in spaces if item not in SPACE_IDS]
    if unknown:
        raise ValueError(f"Unknown search spaces {unknown}. Use A, B, and/or C.")
    if not spaces:
        raise ValueError("Provide at least one search space.")
    return spaces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vanilla CleanRL DQN vs Optuna-tuned vanilla DQN."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=5000)
    parser.add_argument("--search-episodes", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--search-spaces", default="A,B,C")
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_or_load_vanilla(
    *,
    env_id: str,
    seed: int,
    total_episodes: int,
    budget_tag: str,
    cuda: bool,
    plot_only: bool,
) -> dict[str, Any]:
    path = vanilla_path(seed, budget_tag)
    if path.exists():
        print(f"Loading vanilla results from {path}", flush=True)
        return _load_json(path)
    if plot_only:
        raise FileNotFoundError(f"Missing vanilla results: {path}")
    print("Running vanilla DQN with shared non-optimal init...", flush=True)
    results = run_vanilla_training(
        make_vanilla_args(
            env_id=env_id,
            seed=seed,
            total_episodes=total_episodes,
            cuda=cuda,
            hparams=SHARED_BASE_HPARAMS,
            returns_output=str(path),
        )
    )
    print(
        f"Vanilla finished: {len(results['episode_returns'])} episodes, "
        f"last50={_last_mean(results['episode_returns']):.2f}",
        flush=True,
    )
    results["hparams"] = dict(SHARED_BASE_HPARAMS)
    _write_json(path, results)
    return results


def run_or_load_optuna_space(
    space_id: str,
    *,
    env_id: str,
    seed: int,
    total_episodes: int,
    search_episodes: int,
    n_trials: int,
    budget_tag: str,
    cuda: bool,
    plot_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    retrain_path = best_retrain_path(space_id, seed, budget_tag)
    study_path = study_json_path(space_id)
    db_path = study_db_path(space_id)

    if plot_only:
        if not retrain_path.exists() or not study_path.exists():
            raise FileNotFoundError(
                f"Missing Optuna {space_id} results in {RESULTS_DIR}"
            )
        return _load_json(retrain_path), _load_json(study_path)

    print(
        f"Running Optuna {space_id} ({SPACE_NAMES[space_id]}) search...",
        flush=True,
    )
    study = run_optuna_search(
        space_id,
        storage_path=db_path,
        n_trials=n_trials,
        search_episodes=search_episodes,
        seed=seed,
        env_id=env_id,
        cuda=cuda,
    )
    study_blob = dump_study_json(study, space_id, study_path)
    best_params = study_blob["best_params"]
    print(
        f"Optuna {space_id} best trial={study_blob['best_trial_number']} "
        f"search last100={study_blob['best_value']:.2f} "
        f"params={best_params}",
        flush=True,
    )

    if retrain_path.exists():
        print(f"Loading Optuna {space_id} retrain from {retrain_path}", flush=True)
        return _load_json(retrain_path), study_blob

    print(
        f"Retraining Optuna {space_id} winner for {total_episodes} episodes "
        f"on seed={seed}...",
        flush=True,
    )
    results = run_vanilla_training(
        make_vanilla_args(
            env_id=env_id,
            seed=seed,
            total_episodes=total_episodes,
            cuda=cuda,
            hparams=best_params,
            returns_output=str(retrain_path),
        )
    )
    results["best_params"] = best_params
    results["best_trial_value"] = study_blob["best_value"]
    results["best_trial_number"] = study_blob["best_trial_number"]
    results["search_space"] = space_id
    results["search_space_name"] = SPACE_NAMES[space_id]
    _write_json(retrain_path, results)
    print(
        f"Optuna {space_id} retrain finished: "
        f"{len(results['episode_returns'])} episodes, "
        f"last50={_last_mean(results['episode_returns']):.2f}",
        flush=True,
    )
    return results, study_blob


def build_summary(
    vanilla: dict[str, Any],
    optuna_runs: dict[str, dict[str, Any]],
    studies: dict[str, dict[str, Any]],
    *,
    seed: int,
    env_id: str,
    total_episodes: int,
    search_episodes: int,
    n_trials: int,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "vanilla": {
            "label": LABELS["vanilla"],
            "hparams": dict(SHARED_BASE_HPARAMS),
            "last50_mean": _last_mean(vanilla["episode_returns"]),
            "n_episodes": len(vanilla["episode_returns"]),
        }
    }
    for space_id, blob in optuna_runs.items():
        conditions[space_id] = {
            "label": LABELS[space_id],
            "hparams": blob.get("best_params", studies[space_id]["best_params"]),
            "search_best_last100": studies[space_id]["best_value"],
            "last50_mean": _last_mean(blob["episode_returns"]),
            "n_episodes": len(blob["episode_returns"]),
        }
    return {
        "seed": seed,
        "env_id": env_id,
        "total_episodes": total_episodes,
        "search_episodes": search_episodes,
        "n_trials": n_trials,
        "shared_base_hparams": dict(SHARED_BASE_HPARAMS),
        "vanilla_defaults": dict(SHARED_BASE_HPARAMS),
        "conditions": conditions,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nSummary (last 50 episode mean):", flush=True)
    for tag, info in summary["conditions"].items():
        extra = ""
        if tag != "vanilla":
            extra = f"  search_last100={info['search_best_last100']:.2f}"
        print(
            f"  {info['label']}: last50={info['last50_mean']:.2f}"
            f"{extra}\n    hparams={info['hparams']}",
            flush=True,
        )


def main() -> None:
    cli_args = parse_args()
    space_ids = parse_search_spaces(cli_args.search_spaces)
    budget_tag = f"{cli_args.total_episodes}ep"
    title_suffix = f"{cli_args.total_episodes} episodes, seed={cli_args.seed}"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vanilla = run_or_load_vanilla(
        env_id=cli_args.env_id,
        seed=cli_args.seed,
        total_episodes=cli_args.total_episodes,
        budget_tag=budget_tag,
        cuda=cli_args.cuda,
        plot_only=cli_args.plot_only,
    )

    optuna_runs: dict[str, dict[str, Any]] = {}
    studies: dict[str, dict[str, Any]] = {}
    for space_id in space_ids:
        retrain, study_blob = run_or_load_optuna_space(
            space_id,
            env_id=cli_args.env_id,
            seed=cli_args.seed,
            total_episodes=cli_args.total_episodes,
            search_episodes=cli_args.search_episodes,
            n_trials=cli_args.n_trials,
            budget_tag=budget_tag,
            cuda=cli_args.cuda,
            plot_only=cli_args.plot_only,
        )
        optuna_runs[space_id] = retrain
        studies[space_id] = study_blob

    curves = {"vanilla": vanilla["episode_returns"]}
    curves.update(
        {space_id: blob["episode_returns"] for space_id, blob in optuna_runs.items()}
    )
    reward_path = (
        FIG_DIR
        / f"reward_seed{cli_args.seed}_{budget_tag}_vanilla_vs_optuna_nonoptimal.png"
    )
    plot_reward_comparison(
        curves,
        seed=cli_args.seed,
        env_id=cli_args.env_id,
        title_suffix=title_suffix,
        smooth_window=cli_args.smooth_window,
        output_path=reward_path,
    )
    print(f"Saved plot to {reward_path}", flush=True)

    history_path = FIG_DIR / "optuna_optimization_history_nonoptimal.png"
    plot_optimization_history(
        {space_id: studies[space_id]["trials"] for space_id in space_ids},
        seed=cli_args.seed,
        n_trials=cli_args.n_trials,
        search_episodes=cli_args.search_episodes,
        output_path=history_path,
    )
    print(f"Saved plot to {history_path}", flush=True)

    summary = build_summary(
        vanilla,
        optuna_runs,
        studies,
        seed=cli_args.seed,
        env_id=cli_args.env_id,
        total_episodes=cli_args.total_episodes,
        search_episodes=cli_args.search_episodes,
        n_trials=cli_args.n_trials,
    )
    summary_file = summary_path(cli_args.seed, budget_tag)
    _write_json(summary_file, summary)
    print_summary(summary)
    print(f"Saved summary to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
