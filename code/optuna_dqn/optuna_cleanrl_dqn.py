"""
Optuna hyperparameter search for vanilla CleanRL DQN.

Wraps ``meta_gradient_dqn.run_training`` with ``meta_gradient=False``.
Search spaces:

  * A ``lr_gamma``: learning_rate and gamma (same knobs as meta-gradient DQN)
  * B ``core_dqn``: core DQN training knobs
  * C ``cleanrl_typical``: usual CleanRL Optuna set

Untuned knobs always stay at ``SHARED_BASE_HPARAMS`` so vanilla and
Optuna A/B/C share the same non-optimal initialization.

Example:
    python code/run_optuna_vanilla_comparison.py --seed 1 --total-episodes 5000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from meta_gradient_dqn import Args
from meta_gradient_dqn import run_training as run_vanilla_training

SPACE_IDS = ("A", "B", "C")

SPACE_NAMES = {
    "A": "lr_gamma",
    "B": "core_dqn",
    "C": "cleanrl_typical",
}

SPACE_LABELS = {
    "A": "Optuna A: lr + gamma",
    "B": "Optuna B: core DQN knobs",
    "C": "Optuna C: CleanRL typical",
}

VANILLA_DEFAULTS = {
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 10_000,
    "tau": 1.0,
    "target_network_frequency": 500,
    "train_frequency": 10,
    "start_e": 1.0,
    "end_e": 0.05,
    "exploration_fraction": 0.5,
}

# Shared non-optimal initialization used by vanilla and every Optuna space.
# Knobs a space does not search stay at these values, so all four models
# start from the same untuned hyperparameters.
SHARED_BASE_HPARAMS: dict[str, float | int] = {
    "learning_rate": 1e-4,
    "gamma": 0.90,
    "batch_size": 32,
    "buffer_size": 10_000,
    "tau": 0.5,
    "target_network_frequency": 1000,
    "train_frequency": 10,
    "start_e": 1.0,
    "end_e": 0.05,
    "exploration_fraction": 0.3,
}

SPACE_PARAM_KEYS = {
    "A": ("learning_rate", "gamma"),
    "B": (
        "learning_rate",
        "gamma",
        "batch_size",
        "buffer_size",
        "tau",
        "target_network_frequency",
        "exploration_fraction",
        "start_e",
        "end_e",
        "train_frequency",
    ),
    "C": (
        "learning_rate",
        "batch_size",
        "buffer_size",
        "gamma",
        "tau",
        "target_network_frequency",
        "exploration_fraction",
    ),
}


def suggest_hparams(trial: optuna.Trial, space_id: str) -> dict[str, float | int]:
    space_id = space_id.upper()
    if space_id == "A":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.80, 0.9999),
        }
    if space_id == "B":
        start_e = trial.suggest_float("start_e", 0.5, 1.0)
        end_e_hi = min(0.1, start_e - 1e-6)
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.80, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "buffer_size": trial.suggest_categorical(
                "buffer_size", [5_000, 10_000, 50_000, 100_000]
            ),
            "tau": trial.suggest_float("tau", 0.1, 1.0),
            "target_network_frequency": trial.suggest_categorical(
                "target_network_frequency", [1, 100, 250, 500, 1000]
            ),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.8),
            "start_e": start_e,
            "end_e": trial.suggest_float("end_e", 0.01, end_e_hi),
            "train_frequency": trial.suggest_categorical("train_frequency", [1, 4, 10]),
        }
    if space_id == "C":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256, 512]
            ),
            "buffer_size": trial.suggest_categorical(
                "buffer_size", [10_000, 50_000, 100_000]
            ),
            "gamma": trial.suggest_float("gamma", 0.9, 0.999),
            "tau": trial.suggest_float("tau", 0.01, 1.0),
            "target_network_frequency": trial.suggest_categorical(
                "target_network_frequency", [1, 100, 500, 1000]
            ),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        }
    raise ValueError(f"Unknown search space {space_id!r}. Expected A, B, or C.")


def merge_with_shared_base(hparams: dict[str, Any] | None = None) -> dict[str, float | int]:
    merged = dict(SHARED_BASE_HPARAMS)
    if hparams:
        merged.update(as_plain_hparams(hparams))
    return merged


def shared_base_for_space(space_id: str) -> dict[str, float | int]:
    space_id = space_id.upper()
    return {key: SHARED_BASE_HPARAMS[key] for key in SPACE_PARAM_KEYS[space_id]}


def as_plain_hparams(params: dict[str, Any]) -> dict[str, float | int]:
    plain: dict[str, float | int] = {}
    for key, value in params.items():
        if isinstance(value, (np.integer,)):
            plain[key] = int(value)
        elif isinstance(value, (np.floating,)):
            plain[key] = float(value)
        else:
            plain[key] = value
    return plain


def objective_value(episode_returns: list[float], window: int = 100) -> float:
    if not episode_returns:
        return 0.0
    n = min(window, len(episode_returns))
    return float(np.mean(episode_returns[-n:]))


def make_vanilla_args(
    *,
    env_id: str,
    seed: int,
    total_episodes: int,
    cuda: bool,
    hparams: dict[str, float | int] | None = None,
    returns_output: str | None = None,
    log_interval: int = 10_000,
) -> Args:
    kwargs: dict[str, Any] = {
        "env_id": env_id,
        "seed": seed,
        "total_episodes": total_episodes,
        "total_timesteps": 500_000,
        "meta_gradient": False,
        "cuda": cuda,
        "tensorboard": False,
        "track": False,
        "log_interval": log_interval,
        "returns_output": returns_output,
        "exp_name": "optuna_cleanrl_dqn",
    }
    kwargs.update(merge_with_shared_base(hparams))
    return Args(**kwargs)


def make_objective(
    space_id: str,
    *,
    env_id: str,
    eval_seed: int,
    search_episodes: int,
    cuda: bool,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_hparams(trial, space_id)
        hparams = merge_with_shared_base(suggested)
        trial_seed = eval_seed * 1000 + trial.number
        print(
            f"[{SPACE_NAMES[space_id]}] trial={trial.number} seed={trial_seed} "
            f"hparams={hparams}",
            flush=True,
        )
        results = run_vanilla_training(
            make_vanilla_args(
                env_id=env_id,
                seed=trial_seed,
                total_episodes=search_episodes,
                cuda=cuda,
                hparams=hparams,
            )
        )
        value = objective_value(results["episode_returns"])
        print(
            f"[{SPACE_NAMES[space_id]}] trial={trial.number} "
            f"last100={value:.2f} episodes={len(results['episode_returns'])}",
            flush=True,
        )
        return value

    return objective


def sqlite_storage(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def n_complete_trials(study: optuna.Study) -> int:
    return sum(trial.state == TrialState.COMPLETE for trial in study.trials)


def load_or_create_study(
    space_id: str,
    storage_path: Path,
    seed: int,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = TPESampler(seed=seed)
    return optuna.create_study(
        study_name=f"cleanrl_dqn_{SPACE_NAMES[space_id]}",
        direction="maximize",
        sampler=sampler,
        storage=sqlite_storage(storage_path),
        load_if_exists=True,
    )


def run_optuna_search(
    space_id: str,
    *,
    storage_path: Path,
    n_trials: int,
    search_episodes: int,
    seed: int,
    env_id: str,
    cuda: bool,
) -> optuna.Study:
    space_id = space_id.upper()
    if space_id not in SPACE_NAMES:
        raise ValueError(f"Unknown search space {space_id!r}.")
    study = load_or_create_study(space_id, storage_path, seed)
    if n_complete_trials(study) == 0 and len(study.trials) == 0:
        study.enqueue_trial(shared_base_for_space(space_id))
        print(
            f"Optuna {space_id}: enqueue shared non-optimal init "
            f"{shared_base_for_space(space_id)}",
            flush=True,
        )
    remaining = n_trials - n_complete_trials(study)
    if remaining > 0:
        print(
            f"Optuna {space_id} ({SPACE_NAMES[space_id]}): "
            f"running {remaining} trials "
            f"({n_complete_trials(study)} already complete of {n_trials})",
            flush=True,
        )
        study.optimize(
            make_objective(
                space_id,
                env_id=env_id,
                eval_seed=seed,
                search_episodes=search_episodes,
                cuda=cuda,
            ),
            n_trials=remaining,
        )
    else:
        print(
            f"Optuna {space_id} ({SPACE_NAMES[space_id]}): "
            f"study already has {n_complete_trials(study)} complete trials",
            flush=True,
        )
    return study


def trial_history(study: optuna.Study) -> list[dict[str, Any]]:
    history = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        history.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": as_plain_hparams(trial.params),
            }
        )
    return history


def study_summary(study: optuna.Study, space_id: str) -> dict[str, Any]:
    suggested = as_plain_hparams(study.best_params) if study.best_trial else {}
    best_params = merge_with_shared_base(suggested) if suggested else dict(SHARED_BASE_HPARAMS)
    return {
        "space_id": space_id.upper(),
        "space_name": SPACE_NAMES[space_id.upper()],
        "shared_base_hparams": dict(SHARED_BASE_HPARAMS),
        "best_params": best_params,
        "best_suggested_params": suggested,
        "best_value": None if study.best_trial is None else float(study.best_value),
        "best_trial_number": None if study.best_trial is None else study.best_trial.number,
        "n_complete": n_complete_trials(study),
        "trials": trial_history(study),
    }


def dump_study_json(study: optuna.Study, space_id: str, path: Path) -> dict[str, Any]:
    summary = study_summary(study, space_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
