#!/usr/bin/env python3
"""Resumable, process-parallel Optuna tuning for fast CartPole DQN learning.

The objective is normalized learning-curve AUC with a small time-to-solve
tie-breaker. Weak trials are pruned at intermediate episode checkpoints.

Example:
    python code/cartpole_optuna_hpo.py --trials 40 --workers 4 --episodes 1000

Re-running the same tag resumes its SQLite study. Results are written below:
    paper/_tmp_b_feedb_cg/cartpole_hpo/<tag>/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import optuna
import torch

from cartpole_hpo_common import CartPoleDQNRun, DQNConfig

ROOT = Path(__file__).resolve().parent.parent
HPO_ROOT = ROOT / "paper" / "_tmp_b_feedb_cg" / "cartpole_hpo"
DEFAULT_TAG = "optuna_cartpole_fast"


def _study_paths(tag: str) -> tuple[Path, str]:
    out_dir = HPO_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "study.sqlite3"
    return out_dir, f"sqlite:///{db_path}"


def _sample_config(trial: optuna.Trial) -> DQNConfig:
    # Search one-minus-gamma on a log scale, as recommended by RL Zoo.
    one_minus_gamma = trial.suggest_float(
        "one_minus_gamma", 1e-4, 5e-2, log=True
    )
    epsilon_start = trial.suggest_float("epsilon_start", 0.5, 1.0)
    epsilon_final = trial.suggest_float("epsilon_final", 0.01, 0.2)
    if epsilon_final >= epsilon_start:
        epsilon_final = max(0.01, epsilon_start - 0.01)
    return DQNConfig(
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, 5e-3, log=True
        ),
        gamma=1.0 - one_minus_gamma,
        epsilon_start=epsilon_start,
        epsilon_final=epsilon_final,
        epsilon_decay_episodes=trial.suggest_int(
            "epsilon_decay_episodes", 50, 1500, log=True
        ),
    )


def _objective(
    trial: optuna.Trial,
    *,
    episodes: int,
    report_every: int,
    base_seed: int,
    device: torch.device,
    max_steps: int,
) -> float:
    config = _sample_config(trial)
    seed = base_seed + 1009 * trial.number
    run = CartPoleDQNRun(
        config,
        seed=seed,
        device=device,
        max_steps=max_steps,
    )
    try:
        for target in range(report_every, episodes + report_every, report_every):
            target = min(target, episodes)
            metrics = run.train_to_episode(target)
            trial.report(float(metrics["speed_score"]), step=target)
            trial.set_user_attr("last_intermediate_metrics", metrics)
            if target < episodes and trial.should_prune():
                raise optuna.TrialPruned()
            if target >= episodes:
                break
        final = run.train_to_episode(episodes)
        trial.set_user_attr("config", config.__dict__)
        trial.set_user_attr("metrics", final)
        trial.set_user_attr("seed", seed)
        return float(final["speed_score"])
    finally:
        run.close()


def _create_study(tag: str, storage: str, worker_seed: int) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=worker_seed,
        multivariate=True,
        n_startup_trials=8,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,
        n_warmup_steps=100,
        interval_steps=50,
    )
    return optuna.create_study(
        study_name=tag,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def _write_summary(study: optuna.Study, out_dir: Path, episodes: int) -> None:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = sum(
        trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
    )
    payload: Dict[str, Any] = {
        "study": study.study_name,
        "episodes": episodes,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": pruned,
    }
    if completed:
        best = study.best_trial
        config = dict(best.user_attrs.get("config", {}))
        if not config:
            config = {
                "learning_rate": best.params["learning_rate"],
                "gamma": 1.0 - best.params["one_minus_gamma"],
                "epsilon_start": best.params["epsilon_start"],
                "epsilon_final": best.params["epsilon_final"],
                "epsilon_decay_episodes": best.params[
                    "epsilon_decay_episodes"
                ],
            }
        payload.update(
            {
                "best_trial": best.number,
                "best_speed_score": best.value,
                "best_config": config,
                "best_metrics": best.user_attrs.get("metrics"),
            }
        )
        (out_dir / "best_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True)
        )
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _worker(args: argparse.Namespace) -> None:
    out_dir, storage = _study_paths(args.tag)
    study = _create_study(
        args.tag,
        storage,
        worker_seed=args.base_seed + args.worker_id,
    )
    device = torch.device(args.device)
    study.optimize(
        lambda trial: _objective(
            trial,
            episodes=args.episodes,
            report_every=args.report_every,
            base_seed=args.base_seed,
            device=device,
            max_steps=args.max_steps,
        ),
        n_trials=args.worker_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    _write_summary(study, out_dir, args.episodes)


def _launch_workers(args: argparse.Namespace) -> None:
    out_dir, storage = _study_paths(args.tag)
    study = _create_study(args.tag, storage, worker_seed=args.base_seed)
    if not study.trials:
        # A known strong CartPole baseline anchors TPE and the comparison.
        study.enqueue_trial(
            {
                "learning_rate": 2.3e-3,
                "one_minus_gamma": 0.01,
                "epsilon_start": 1.0,
                "epsilon_final": 0.04,
                "epsilon_decay_episodes": 160,
            }
        )

    workers = max(1, args.workers)
    per_worker = int(math.ceil(args.trials / workers))
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices:
        devices = ["cpu"]
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, Any, Path]] = []

    for worker_id in range(workers):
        device = devices[worker_id % len(devices)]
        log_path = logs_dir / f"worker{worker_id}.log"
        log_file = open(log_path, "a", buffering=1)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-id",
            str(worker_id),
            "--worker-trials",
            str(per_worker),
            "--tag",
            args.tag,
            "--episodes",
            str(args.episodes),
            "--report-every",
            str(args.report_every),
            "--base-seed",
            str(args.base_seed),
            "--device",
            device,
            "--max-steps",
            str(args.max_steps),
        ]
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        print(
            f"[optuna] worker={worker_id} device={device} log={log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_file, log_path))

    failed = []
    for process, log_file, log_path in processes:
        rc = process.wait()
        log_file.close()
        if rc:
            failed.append((rc, log_path))
    if failed:
        raise SystemExit(f"Optuna workers failed: {failed}")

    study = optuna.load_study(study_name=args.tag, storage=storage)
    _write_summary(study, out_dir, args.episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-trials", type=int, default=1, help=argparse.SUPPRESS
    )
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _launch_workers(args)


if __name__ == "__main__":
    main()
