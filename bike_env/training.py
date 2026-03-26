"""Train DQN on BikeEnvAdvanced (stable-baselines3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .config import TrainConfig
from .env import BikeEnvAdvanced, FallCounterWrapper


def train_dqn(
    config: TrainConfig,
    *,
    callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one DQN training job. Creates ``log_dir``, saves checkpoints, final model, and ``run_metadata.json``.

    Returns a dict with paths: ``log_dir``, ``final_model_path``, ``monitor_csv`` (if present).
    """
    log_dir = config.resolved_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        FallCounterWrapper(BikeEnvAdvanced(n_lanes=config.n_lanes)),
        str(log_dir),
        info_keywords=("episode_falls",),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_save_freq,
        save_path=str(log_dir),
        name_prefix=config.checkpoint_name_prefix,
    )
    callbacks = [checkpoint_callback]
    if callback is not None:
        callbacks.append(callback)

    tb_dir = log_dir / "tensorboard"
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=list(config.policy_net_arch)),
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        learning_rate=config.learning_rate,
        verbose=config.verbose,
        seed=config.seed,
        tensorboard_log=str(tb_dir),
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
    )

    stem = log_dir / config.final_model_name
    model.save(str(stem))
    final_path = stem.with_suffix(".zip")

    meta = config.to_metadata_dict()
    meta["final_model"] = str(final_path)
    with open(log_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    monitor_csv = log_dir / "monitor.csv"
    out: Dict[str, Any] = {
        "log_dir": log_dir,
        "final_model_path": final_path,
        "metadata": meta,
    }
    if monitor_csv.exists():
        out["monitor_csv"] = monitor_csv
    return out
