#!/usr/bin/env python3
"""Adaptive online CartPole DQN hyperparameter optimization with Ray PB2.

PB2 keeps a population of DQN lifetimes, periodically copies strong
checkpoints into weak trials, and uses a time-varying Gaussian-process bandit
to choose new learning-rate, gamma, and epsilon-schedule values.

Example:
    python code/cartpole_pb2_hpo.py --population 4 --episodes 1000

Re-run with the same tag and --resume to continue an interrupted Ray run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.schedulers.pb2 import PB2

from cartpole_hpo_common import CartPoleDQNRun, DQNConfig

ROOT = Path(__file__).resolve().parent.parent
HPO_ROOT = ROOT / "paper" / "_tmp_b_feedb_cg" / "cartpole_hpo"
DEFAULT_TAG = "pb2_cartpole_fast"


def _sanitize_config(config: Dict[str, Any]) -> DQNConfig:
    eps_start = float(np.clip(config["epsilon_start"], 0.5, 1.0))
    eps_final = float(np.clip(config["epsilon_final"], 0.01, 0.2))
    eps_final = min(eps_final, eps_start - 0.01)
    return DQNConfig(
        learning_rate=float(np.clip(config["learning_rate"], 1e-5, 5e-3)),
        gamma=float(np.clip(config["gamma"], 0.95, 0.9999)),
        epsilon_start=eps_start,
        epsilon_final=max(0.01, eps_final),
        epsilon_decay_episodes=int(
            np.clip(round(config["epsilon_decay_episodes"]), 50, 1500)
        ),
    )


class PB2CartPoleTrainable(tune.Trainable):
    """Ray adapter around a checkpointable CartPole DQN lifetime."""

    def setup(self, config: Dict[str, Any]) -> None:
        self.budget_episodes = int(config["budget_episodes"])
        self.episodes_per_iteration = int(config["episodes_per_iteration"])
        self.max_steps = int(config["max_steps"])
        self.seed = int(config.get("seed", 0))
        self.dqn_config = _sanitize_config(config)
        self.runner = CartPoleDQNRun(
            self.dqn_config,
            seed=self.seed,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_steps=self.max_steps,
        )

    def step(self) -> Dict[str, Any]:
        target = min(
            self.budget_episodes,
            len(self.runner.episode_returns) + self.episodes_per_iteration,
        )
        metrics = self.runner.train_to_episode(target)
        return {
            **metrics,
            "episodes_total": len(self.runner.episode_returns),
            "learning_rate": self.dqn_config.learning_rate,
            "gamma": self.dqn_config.gamma,
            "epsilon_start": self.dqn_config.epsilon_start,
            "epsilon_final": self.dqn_config.epsilon_final,
            "epsilon_decay_episodes": self.dqn_config.epsilon_decay_episodes,
        }

    def reset_config(self, new_config: Dict[str, Any]) -> bool:
        self.config = new_config
        self.dqn_config = _sanitize_config(new_config)
        self.runner.set_config(self.dqn_config)
        return True

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        path = Path(checkpoint_dir) / "dqn_state.pt"
        self.runner.save(path)
        # Ray 2.57 requires returning the directory it supplied, not a child.
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "dqn_state.pt"
        self.runner.load(path)
        # The receiving PB2 trial's mutated config must override donor config.
        self.dqn_config = _sanitize_config(self.config)
        self.runner.set_config(self.dqn_config)

    def cleanup(self) -> None:
        self.runner.close()


def _explore(config: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce valid epsilon and integer decay values after PB2 mutation."""
    config["epsilon_start"] = float(
        np.clip(config["epsilon_start"], 0.5, 1.0)
    )
    config["epsilon_final"] = float(
        np.clip(config["epsilon_final"], 0.01, 0.2)
    )
    config["epsilon_final"] = min(
        config["epsilon_final"], config["epsilon_start"] - 0.01
    )
    config["epsilon_decay_episodes"] = int(
        np.clip(round(config["epsilon_decay_episodes"]), 50, 1500)
    )
    return config


def _build_tuner(args: argparse.Namespace, experiment_path: Path) -> tune.Tuner:
    if args.resume and tune.Tuner.can_restore(str(experiment_path)):
        print(f"[pb2] restoring {experiment_path}", flush=True)
        return tune.Tuner.restore(
            str(experiment_path),
            trainable=PB2CartPoleTrainable,
            resume_unfinished=True,
            resume_errored=True,
        )

    scheduler = PB2(
        time_attr="training_iteration",
        metric="speed_score",
        mode="max",
        perturbation_interval=args.perturbation_iterations,
        quantile_fraction=0.25,
        hyperparam_bounds={
            "learning_rate": [1e-5, 5e-3],
            "gamma": [0.95, 0.9999],
            "epsilon_start": [0.5, 1.0],
            "epsilon_final": [0.01, 0.2],
            "epsilon_decay_episodes": [50, 1500],
        },
        custom_explore_fn=_explore,
        synch=True,
    )
    iterations = int(
        math.ceil(args.episodes / max(1, args.episodes_per_iteration))
    )
    param_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-3),
        "gamma": tune.uniform(0.95, 0.9999),
        "epsilon_start": tune.uniform(0.5, 1.0),
        "epsilon_final": tune.uniform(0.01, 0.2),
        "epsilon_decay_episodes": tune.uniform(50, 1500),
        "budget_episodes": args.episodes,
        "episodes_per_iteration": args.episodes_per_iteration,
        "max_steps": args.max_steps,
        "seed": tune.randint(args.base_seed, args.base_seed + 1_000_000),
    }
    trainable = tune.with_resources(
        PB2CartPoleTrainable,
        resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )
    return tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.population,
            reuse_actors=True,
        ),
        # Use Tune's RunConfig (not Ray Train v2's stricter config) because
        # this is a class-based Tune Trainable with scheduler-driven stopping.
        run_config=tune.RunConfig(
            name="ray",
            storage_path=str(experiment_path.parent),
            stop={"training_iteration": iterations},
        ),
    )


def _write_summary(
    results: tune.ResultGrid,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    best = results.get_best_result(
        metric="speed_score",
        mode="max",
        scope="all",
    )
    config = _sanitize_config(best.config)
    payload = {
        "method": "PB2",
        "tag": args.tag,
        "population": args.population,
        "episodes": args.episodes,
        "episodes_per_iteration": args.episodes_per_iteration,
        "best_speed_score": best.metrics.get("speed_score"),
        "best_metrics": {
            key: best.metrics.get(key)
            for key in (
                "auc_normalized",
                "last100_mean",
                "first_solve_episode",
                "episodes_total",
                "env_steps",
            )
        },
        "best_final_config": config.__dict__,
        "best_result_path": str(best.path),
        "note": (
            "PB2 is an online population optimizer. Its score includes weight "
            "inheritance and adaptive schedules, not just this final config."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    (out_dir / "best_final_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True)
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-iteration", type=int, default=50)
    parser.add_argument("--perturbation-iterations", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--gpus-per-trial", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = HPO_ROOT / args.tag
    experiment_path = out_dir / "ray"
    ray.init(ignore_reinit_error=True)
    try:
        tuner = _build_tuner(args, experiment_path)
        results = tuner.fit()
        if results.errors:
            raise RuntimeError(f"PB2 trials failed: {results.errors}")
        _write_summary(results, out_dir, args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
