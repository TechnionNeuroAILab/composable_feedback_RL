#!/usr/bin/env python3
"""Evaluate CartPole HPO outputs on held-out seeds and plot learning speed.

Compares the repository baseline, the published RL-Zoo-like CartPole settings,
an Optuna best config, a PB2 final config used statically, and optionally the
existing recurrent meta-controller. All fresh evaluations use identical seeds
and episode budgets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from cartpole_hpo_common import CartPoleDQNRun, DQNConfig, speed_metrics
from confgate_meta_hpo import (
    CKPT_DIR,
    END_E,
    EPSILON_DECAY_EPISODES,
    GAMMA,
    LEARNING_RATE,
    START_E,
    MetaHPOController,
    run_inner_task,
)

ROOT = Path(__file__).resolve().parent.parent
HPO_ROOT = ROOT / "paper" / "_tmp_b_feedb_cg" / "cartpole_hpo"
FIG_ROOT = ROOT / "paper" / "figures" / "conf_meta_hpo"


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_config(path: Path) -> DQNConfig:
    return DQNConfig.from_dict(json.loads(path.read_text()))


def _evaluate_static(
    config: DQNConfig,
    seeds: list[int],
    episodes: int,
    max_steps: int,
    device: torch.device,
) -> list[Dict[str, Any]]:
    results = []
    for seed in seeds:
        run = CartPoleDQNRun(
            config,
            seed=seed,
            device=device,
            max_steps=max_steps,
        )
        try:
            metrics = run.train_to_episode(episodes)
            results.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "episode_returns": run.episode_returns,
                }
            )
        finally:
            run.close()
    return results


def _evaluate_meta_controller(
    checkpoint: Path,
    seeds: list[int],
    episodes: int,
    max_steps: int,
    device: torch.device,
) -> list[Dict[str, Any]]:
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    tasks = list(blob.get("tasks", ["CartPole-v1"]))
    if "CartPole-v1" not in tasks:
        raise ValueError(f"Controller checkpoint has no CartPole-v1: {tasks}")
    task_idx = tasks.index("CartPole-v1")
    controller = MetaHPOController(n_tasks=len(tasks)).to(device)
    controller.load_state_dict(blob["controller"])
    controller.eval()
    frozen_opt = optim.Adam(controller.parameters(), lr=0.0)
    results = []
    for seed in seeds:
        result = run_inner_task(
            env_id="CartPole-v1",
            task_idx=task_idx,
            controller=controller,
            opt_ctrl=frozen_opt,
            device=device,
            inner_steps=max_steps,
            seed=seed,
            use_controller=True,
            update_controller=False,
            deterministic_ctrl=True,
            total_episodes=episodes,
        )
        returns = [float(v) for v in result["episode_returns"]]
        results.append(
            {
                "seed": seed,
                "metrics": speed_metrics(returns, episodes),
                "episode_returns": returns,
            }
        )
    return results


def _aggregate(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = (
        "speed_score",
        "auc_normalized",
        "mean_return",
        "last100_mean",
    )
    summary: Dict[str, Any] = {"n_seeds": len(results)}
    for key in metric_keys:
        values = np.asarray(
            [result["metrics"][key] for result in results],
            dtype=np.float64,
        )
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }
    solve_values = [
        result["metrics"]["first_solve_episode"]
        for result in results
        if result["metrics"]["first_solve_episode"] is not None
    ]
    summary["solve_rate"] = len(solve_values) / max(1, len(results))
    summary["median_first_solve_episode"] = (
        float(np.median(solve_values)) if solve_values else None
    )
    return summary


def _plot(
    all_results: Dict[str, list[Dict[str, Any]]],
    out_path: Path,
    episodes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("tab10")
    for index, (name, results) in enumerate(all_results.items()):
        curves = []
        for result in results:
            values = np.asarray(result["episode_returns"], dtype=np.float64)
            padded = np.full(episodes, np.nan)
            padded[: min(episodes, values.size)] = values[:episodes]
            curves.append(padded)
        matrix = np.vstack(curves)
        mean = np.nanmean(matrix, axis=0)
        sem = np.nanstd(matrix, axis=0) / np.sqrt(max(1, matrix.shape[0]))
        window = max(1, episodes // 50)
        kernel = np.ones(window) / window
        mean_sm = np.convolve(mean, kernel, mode="valid")
        sem_sm = np.convolve(sem, kernel, mode="valid")
        x = np.arange(window, episodes + 1)
        color = colors(index % 10)
        ax.plot(x, mean_sm, label=name, color=color, lw=2)
        ax.fill_between(
            x,
            mean_sm - sem_sm,
            mean_sm + sem_sm,
            color=color,
            alpha=0.15,
        )
    ax.axhline(475, color="black", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (smoothed mean ± SEM)")
    ax.set_title("CartPole learning speed on held-out seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="cartpole_hpo_comparison")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106,107,108,109,110",
    )
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optuna-tag", default="optuna_cartpole_fast")
    parser.add_argument("--pb2-tag", default="pb2_cartpole_fast")
    parser.add_argument(
        "--meta-checkpoint",
        type=Path,
        default=None,
        help="Optional meta_controller_*.pt checkpoint",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    configs: Dict[str, DQNConfig] = {
        "repo baseline": DQNConfig(
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=START_E,
            epsilon_final=END_E,
            epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        ),
        "RL-Zoo-like": DQNConfig(
            learning_rate=2.3e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_final=0.04,
            epsilon_decay_episodes=160,
        ),
    }
    optuna_path = HPO_ROOT / args.optuna_tag / "best_config.json"
    if optuna_path.exists():
        configs["Optuna best"] = _load_config(optuna_path)
    pb2_path = HPO_ROOT / args.pb2_tag / "best_final_config.json"
    if pb2_path.exists():
        configs["PB2 final (static)"] = _load_config(pb2_path)

    all_results: Dict[str, list[Dict[str, Any]]] = {}
    for name, config in configs.items():
        print(f"[eval] {name}: {config}", flush=True)
        all_results[name] = _evaluate_static(
            config,
            seeds,
            args.episodes,
            args.max_steps,
            device,
        )

    if args.meta_checkpoint is not None:
        checkpoint = args.meta_checkpoint
        if not checkpoint.is_absolute():
            checkpoint = CKPT_DIR / checkpoint
        print(f"[eval] meta-controller: {checkpoint}", flush=True)
        all_results["meta-controller"] = _evaluate_meta_controller(
            checkpoint,
            seeds,
            args.episodes,
            args.max_steps,
            device,
        )

    report = {
        "tag": args.tag,
        "episodes": args.episodes,
        "seeds": seeds,
        "methods": {
            name: {
                "summary": _aggregate(results),
                "per_seed": [
                    {"seed": result["seed"], **result["metrics"]}
                    for result in results
                ],
            }
            for name, results in all_results.items()
        },
        "caveat": (
            "PB2 final (static) evaluates only PB2's final hyperparameters. "
            "PB2's online population score also benefits from adaptive schedules "
            "and checkpoint inheritance and is reported in its own summary.json."
        ),
    }
    out_dir = HPO_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    figure = FIG_ROOT / f"{args.tag}.jpg"
    _plot(all_results, figure, args.episodes)
    print(json.dumps(report["methods"], indent=2, sort_keys=True), flush=True)
    print(f"Wrote {out_dir / 'report.json'}", flush=True)
    print(f"Wrote {figure}", flush=True)


if __name__ == "__main__":
    main()
