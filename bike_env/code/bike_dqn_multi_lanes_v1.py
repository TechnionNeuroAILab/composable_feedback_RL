#!/usr/bin/env python3
"""Configurable multi-lane Bike DQN training and diagnostics (v1).

Examples:
    # Three lateral actions, fixed speed, and falls caused only by holes
    python bike_dqn_multi_lanes_v1.py --lanes 5 --actions 3 \
        --fixed-speed 5 --fall-rules hole_collision --hole-lambda 5

    # Original five actions with no falls
    python bike_dqn_multi_lanes_v1.py --lanes 5 --actions 5 --fall-rules none
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor, load_results


_HERE = Path(__file__).resolve().parent
_DEFAULT_OUTPUT = _HERE / "training_results_v1"
_DEFAULT_PLOT_DIR = _HERE.parent / "plotting" / "bike_dqn_multi_lanes_v1"

ACTION_NAMES = {0: "Left", 1: "Stay", 2: "Right", 3: "Accelerate", 4: "Brake"}
FALL_RULES = ("hole_collision", "turn_slip", "speed_wobble", "slow_unstable")
STAGE_ORDER = ("beginning", "middle", "end")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    """All behavior that defines one environment experiment."""

    n_lanes: int = 5
    action_count: int = 3
    fixed_speed: float = 5.0
    max_speed: float = 12.0
    visibility_range: float = 25.0
    hole_lambda: float = 5.0
    fall_rules: tuple[str, ...] = ("hole_collision",)
    grace_steps: int = 5
    reset_holes_on_fall: bool = False
    episode_steps: int = 1000
    fall_time_penalty: int = 50
    collision_distance: float = 0.8
    collision_penalty: float = -10.0

    def __post_init__(self) -> None:
        if self.n_lanes < 2:
            raise ValueError("n_lanes must be at least 2")
        if self.action_count not in (3, 5):
            raise ValueError("action_count must be 3 or 5")
        if not 0 < self.fixed_speed <= self.max_speed:
            raise ValueError("fixed_speed must be in (0, max_speed]")
        if self.hole_lambda <= 0:
            raise ValueError("hole_lambda must be positive")
        if self.grace_steps < 0:
            raise ValueError("grace_steps cannot be negative")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        unknown = set(self.fall_rules) - set(FALL_RULES)
        if unknown:
            raise ValueError(f"Unknown fall rules: {sorted(unknown)}")


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int | None = 1_000_000
    total_episodes: int | None = None
    seed: int = 1
    exploration_fraction: float = 0.25
    exploration_final_eps: float = 0.025
    learning_starts: int = 5000
    batch_size: int = 128
    checkpoint_freq: int = 100_000
    loss_log_freq: int = 100
    eval_episodes: int = 20


def _rules_label(rules: tuple[str, ...]) -> str:
    return "none" if not rules else "-".join(rule.replace("_", "-") for rule in rules)


def experiment_name(config: EnvConfig, seed: int) -> str:
    lambda_text = f"{config.hole_lambda:g}".replace(".", "p")
    return (
        f"lanes_{config.n_lanes}__actions_{config.action_count}"
        f"__falls_{_rules_label(config.fall_rules)}__lambda_{lambda_text}__seed_{seed}"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BikeEnvAdvancedV1(gym.Env):
    """Bike environment with selectable controls, falls, and Poisson holes.

    The hole process is represented by the nearest upcoming hole in each lane.
    Exponential inter-hole distances make each lane a homogeneous Poisson
    process with mean gap ``hole_lambda`` distance units.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        self.action_space = spaces.Discrete(config.action_count)
        obs_high = np.full(2 + config.n_lanes, 100.0, dtype=np.float32)
        obs_high[0] = float(config.n_lanes - 1)
        obs_high[1] = float(config.max_speed)
        self.observation_space = spaces.Box(
            low=np.zeros(2 + config.n_lanes, dtype=np.float32),
            high=obs_high,
            dtype=np.float32,
        )
        self._next_hole_id = 0

    @staticmethod
    def _sigmoid(value: float, threshold: float, steepness: float = 3.0) -> float:
        exponent = float(np.clip(-steepness * (value - threshold), -700, 700))
        return 1.0 / (1.0 + math.exp(exponent))

    def _new_hole_id(self) -> int:
        hole_id = self._next_hole_id
        self._next_hole_id += 1
        return hole_id

    def _sample_gap(self) -> float:
        return float(self.np_random.exponential(self.config.hole_lambda))

    def _initialize_holes(self) -> None:
        self.holes = np.array(
            [self._sample_gap() for _ in range(self.config.n_lanes)], dtype=np.float64
        )
        self.hole_ids = [self._new_hole_id() for _ in range(self.config.n_lanes)]
        self._threatened_holes: set[int] = set()
        self._collided_holes: set[int] = set()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.x_pos = float(self.config.n_lanes // 2)
        self.velocity = (
            self.config.fixed_speed if self.config.action_count == 3 else 0.0
        )
        self.angle = 0.0
        self.time_left = self.config.episode_steps
        self.distance_traveled = 0.0
        self.grace_period_steps = self.config.grace_steps
        self.holes_passed = 0
        self.holes_avoided = 0
        self.hole_collisions = 0
        self.falls = 0
        self._next_hole_id = 0
        self._initialize_holes()
        return self._get_obs(), self._base_info()

    def _base_info(self) -> dict[str, Any]:
        return {
            "fell": False,
            "reason": "None",
            "grace_remaining": self.grace_period_steps,
            "distance": self.distance_traveled,
            "lane": int(round(self.x_pos)),
            "lane_changed": False,
            "near_hole_lane_change": False,
            "hole_events": [],
        }

    def _fall_reason(
        self,
        current_lane: int,
        projected_holes: np.ndarray,
    ) -> tuple[str | None, int | None]:
        rules = self.config.fall_rules
        if not rules or self.grace_period_steps > 0:
            return None, None

        if "turn_slip" in rules and self.angle != 0:
            probability = self._sigmoid(self.velocity, 8.5)
            if self.np_random.random() < probability:
                return "Turn Slip", None

        if "speed_wobble" in rules:
            probability = self._sigmoid(self.velocity, 9.0, steepness=5.0)
            if self.np_random.random() < probability:
                return "Speed Wobble", None

        if "hole_collision" in rules:
            old_distance = self.holes[current_lane]
            new_distance = projected_holes[current_lane]
            intersects = (
                old_distance >= -self.config.collision_distance
                and new_distance <= self.config.collision_distance
            )
            if intersects:
                return "Hole Collision", current_lane

        if "slow_unstable" in rules and self.velocity < 1.5:
            probability = self._sigmoid(1.5 - self.velocity, 0.5)
            if self.np_random.random() < probability:
                return "Slow Unstable", None

        return None, None

    def step(self, action: int):
        action_int = int(np.asarray(action).item())
        if not self.action_space.contains(action_int):
            raise ValueError(f"Invalid action {action_int} for {self.config.action_count} actions")

        self.time_left -= 1
        old_lane = int(round(self.x_pos))
        old_lane_hole_distance = float(self.holes[old_lane])
        old_lane_hole_id = self.hole_ids[old_lane]
        if 0 <= old_lane_hole_distance <= self.config.visibility_range:
            self._threatened_holes.add(old_lane_hole_id)

        self.angle = -0.45 if action_int == 0 else 0.45 if action_int == 2 else 0.0
        if self.config.action_count == 3:
            self.velocity = self.config.fixed_speed
        elif action_int == 3:
            self.velocity = min(self.config.max_speed, self.velocity + 1.0)
        elif action_int == 4:
            self.velocity = max(0.0, self.velocity * 0.9)

        self.x_pos += math.sin(self.angle) * self.velocity * 0.1
        self.x_pos = float(np.clip(self.x_pos, 0, self.config.n_lanes - 1))
        current_lane = int(round(self.x_pos))
        lane_changed = current_lane != old_lane
        near_hole_lane_change = (
            lane_changed
            and 0 <= old_lane_hole_distance <= self.config.visibility_range
        )

        advance = self.velocity * 0.1
        projected_holes = self.holes - advance
        reason, collision_lane = self._fall_reason(current_lane, projected_holes)
        fell = reason is not None
        hole_events: list[dict[str, Any]] = []

        if collision_lane is not None:
            collision_id = self.hole_ids[collision_lane]
            self._collided_holes.add(collision_id)
            self.hole_collisions += 1
            hole_events.append(
                {
                    "event": "collision",
                    "hole_id": collision_id,
                    "lane": collision_lane,
                    "distance": self.distance_traveled
                    + max(float(self.holes[collision_lane]), 0.0),
                }
            )

        self.distance_traveled += advance
        self.holes = projected_holes

        for lane in range(self.config.n_lanes):
            while self.holes[lane] < 0:
                hole_id = self.hole_ids[lane]
                was_threatened = hole_id in self._threatened_holes
                collided = hole_id in self._collided_holes
                if not collided:
                    if was_threatened and current_lane != lane:
                        event = "avoided"
                        self.holes_avoided += 1
                    elif was_threatened:
                        event = "unprotected_pass"
                    else:
                        event = "passed_other_lane"
                    self.holes_passed += 1
                    hole_events.append(
                        {
                            "event": event,
                            "hole_id": hole_id,
                            "lane": lane,
                            "distance": self.distance_traveled + float(self.holes[lane]),
                        }
                    )

                self._threatened_holes.discard(hole_id)
                self._collided_holes.discard(hole_id)
                self.holes[lane] += self._sample_gap()
                self.hole_ids[lane] = self._new_hole_id()

        reward = self.config.collision_penalty if fell else self.velocity
        if fell:
            self.falls += 1
            self.velocity = (
                self.config.fixed_speed if self.config.action_count == 3 else 0.0
            )
            self.angle = 0.0
            self.time_left -= self.config.fall_time_penalty
            self.grace_period_steps = self.config.grace_steps
            if self.config.reset_holes_on_fall:
                self._initialize_holes()
        elif self.grace_period_steps > 0:
            self.grace_period_steps -= 1

        terminated = self.time_left <= 0
        info = {
            "fell": fell,
            "reason": reason or "None",
            "grace_remaining": self.grace_period_steps,
            "distance": self.distance_traveled,
            "lane": current_lane,
            "lane_changed": lane_changed,
            "near_hole_lane_change": near_hole_lane_change,
            "hole_events": hole_events,
            "holes_passed_total": self.holes_passed,
            "holes_avoided_total": self.holes_avoided,
            "hole_collisions_total": self.hole_collisions,
        }
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        visible_holes = np.where(
            self.holes <= self.config.visibility_range,
            self.holes,
            100.0,
        )
        return np.array(
            [self.x_pos, self.velocity, *visible_holes], dtype=np.float32
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingMetricsCallback(BaseCallback):
    """Capture DQN loss per episode and save staged model snapshots."""

    def __init__(
        self,
        run_dir: Path,
        *,
        total_timesteps: int | None = None,
        total_episodes: int | None = None,
        loss_log_freq: int = 100,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.total_timesteps = total_timesteps
        self.total_episodes = total_episodes
        self.middle_timestep = (
            max(1, total_timesteps // 2) if total_timesteps is not None else None
        )
        self.middle_episode = (
            max(1, total_episodes // 2) if total_episodes is not None else None
        )
        self.loss_log_freq = max(1, loss_log_freq)
        self.loss_rows: list[dict[str, float]] = []
        self.episode_count = 0
        self.middle_saved = False

    def _maybe_save_middle(self) -> None:
        if self.middle_saved:
            return
        by_timestep = (
            self.middle_timestep is not None
            and self.num_timesteps >= self.middle_timestep
        )
        by_episode = (
            self.middle_episode is not None
            and self.episode_count >= self.middle_episode
        )
        if by_timestep or by_episode:
            self.model.save(str(self.run_dir / "model_middle"))
            self.middle_saved = True

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            finished = int(np.sum(dones))
            if finished:
                self.episode_count += finished
                self._maybe_save_middle()

        if self.num_timesteps % self.loss_log_freq == 0:
            loss = self.model.logger.name_to_value.get("train/loss")
            if loss is not None:
                self.loss_rows.append(
                    {
                        "episode": float(max(self.episode_count, 1)),
                        "timesteps": float(self.num_timesteps),
                        "loss": float(loss),
                    }
                )

        if (
            self.total_episodes is not None
            and self.episode_count >= self.total_episodes
        ):
            return False
        return True

    def _on_training_end(self) -> None:
        pd.DataFrame(
            self.loss_rows, columns=["episode", "timesteps", "loss"]
        ).to_csv(self.run_dir / "training_loss.csv", index=False)


def train(
    env_config: EnvConfig,
    train_config: TrainConfig,
    base_output: Path,
) -> tuple[DQN, Path]:
    run_dir = base_output / experiment_name(env_config, train_config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"environment": asdict(env_config), "training": asdict(train_config)},
            file,
            indent=2,
        )

    env = Monitor(BikeEnvAdvancedV1(env_config), str(run_dir))
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [256, 256]},
        exploration_fraction=train_config.exploration_fraction,
        exploration_final_eps=train_config.exploration_final_eps,
        learning_rate=1e-4,
        learning_starts=train_config.learning_starts,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        verbose=0,
    )
    model.save(str(run_dir / "model_beginning"))

    if train_config.total_episodes is not None:
        budget_text = f"{train_config.total_episodes:,} episodes"
        total_timesteps = train_config.total_episodes * env_config.episode_steps
    else:
        budget_text = f"{train_config.total_timesteps:,} steps"
        total_timesteps = train_config.total_timesteps

    metrics_callback = TrainingMetricsCallback(
        run_dir,
        total_timesteps=total_timesteps,
        total_episodes=train_config.total_episodes,
        loss_log_freq=train_config.loss_log_freq,
    )
    callbacks: list[BaseCallback] = [metrics_callback]
    if train_config.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=train_config.checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="bike_model",
            )
        )

    print(
        f"  [{experiment_name(env_config, train_config.seed)}] "
        f"Training {budget_text} ..."
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
    )
    if not metrics_callback.middle_saved:
        model.save(str(run_dir / "model_middle"))
    model.save(str(run_dir / "model_end"))
    env.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Staged evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvaluationData:
    steps: pd.DataFrame
    episodes: pd.DataFrame
    holes: pd.DataFrame
    visible_holes: pd.DataFrame


def evaluate_snapshots(
    run_dir: Path,
    env_config: EnvConfig,
    n_episodes: int,
    seed: int,
) -> EvaluationData:
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    hole_rows: list[dict[str, Any]] = []
    visible_rows: list[dict[str, Any]] = []

    for stage in STAGE_ORDER:
        model = DQN.load(str(run_dir / f"model_{stage}"))
        for episode in range(n_episodes):
            env = BikeEnvAdvancedV1(env_config)
            obs, _ = env.reset(seed=seed + episode)
            done = False
            step = 0
            total_reward = 0.0
            seen_holes: set[int] = set()

            while not done and step < env_config.episode_steps:
                for lane, (hole_id, relative_distance) in enumerate(
                    zip(env.hole_ids, env.holes)
                ):
                    if (
                        hole_id not in seen_holes
                        and relative_distance <= env_config.visibility_range
                    ):
                        seen_holes.add(hole_id)
                        visible_rows.append(
                            {
                                "stage": stage,
                                "episode": episode,
                                "hole_id": hole_id,
                                "lane": lane,
                                "distance": env.distance_traveled
                                + float(relative_distance),
                            }
                        )

                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                obs, reward, terminated, truncated, info = env.step(action_int)
                total_reward += float(reward)
                step_rows.append(
                    {
                        "stage": stage,
                        "episode": episode,
                        "step": step,
                        "distance": info["distance"],
                        "lane_position": env.x_pos,
                        "lane": info["lane"],
                        "velocity": env.velocity,
                        "action": action_int,
                        "action_name": ACTION_NAMES[action_int],
                        "reward": float(reward),
                        "fell": info["fell"],
                        "fall_reason": info["reason"],
                        "lane_changed": info["lane_changed"],
                        "near_hole_lane_change": info["near_hole_lane_change"],
                    }
                )
                for event in info["hole_events"]:
                    hole_rows.append(
                        {
                            "stage": stage,
                            "episode": episode,
                            "step": step,
                            **event,
                        }
                    )
                step += 1
                done = terminated or truncated

            episode_rows.append(
                {
                    "stage": stage,
                    "episode": episode,
                    "reward": total_reward,
                    "steps": step,
                    "distance": env.distance_traveled,
                    "falls": env.falls,
                    "holes_passed": env.holes_passed,
                    "holes_avoided": env.holes_avoided,
                    "hole_collisions": env.hole_collisions,
                }
            )
            env.close()

    columns = [
        "stage",
        "episode",
        "step",
        "event",
        "hole_id",
        "lane",
        "distance",
    ]
    visible_columns = ["stage", "episode", "hole_id", "lane", "distance"]
    data = EvaluationData(
        steps=pd.DataFrame(step_rows),
        episodes=pd.DataFrame(episode_rows),
        holes=pd.DataFrame(hole_rows, columns=columns),
        visible_holes=pd.DataFrame(visible_rows, columns=visible_columns),
    )
    data.steps.to_csv(run_dir / "evaluation_steps.csv", index=False)
    data.episodes.to_csv(run_dir / "evaluation_episodes.csv", index=False)
    data.holes.to_csv(run_dir / "evaluation_holes.csv", index=False)
    data.visible_holes.to_csv(run_dir / "evaluation_visible_holes.csv", index=False)
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


def _episode_indexed_monitor(run_dir: Path) -> pd.DataFrame:
    monitor = load_results(str(run_dir))
    if monitor.empty:
        return monitor
    monitor = monitor.reset_index(drop=True)
    monitor["episode"] = np.arange(1, len(monitor) + 1)
    return monitor


def _episode_losses(run_dir: Path) -> pd.DataFrame:
    loss_path = run_dir / "training_loss.csv"
    if not loss_path.exists():
        return pd.DataFrame()
    losses = pd.read_csv(loss_path)
    if losses.empty or "episode" not in losses.columns:
        return pd.DataFrame()
    return (
        losses.groupby("episode", observed=True)["loss"]
        .mean()
        .reset_index()
        .sort_values("episode")
    )


def write_statistics_methodology(
    out_dir: Path,
    env_config: EnvConfig,
    train_config: TrainConfig,
    n_training_episodes: int,
) -> None:
    eval_eps = train_config.eval_episodes
    middle_ep = (
        train_config.total_episodes // 2
        if train_config.total_episodes is not None
        else "half of total timesteps"
    )
    text = f"""# Statistics Methodology

## Training budget
- Completed training episodes: {n_training_episodes}
- Configured stop criterion: {
        f"{train_config.total_episodes} episodes"
        if train_config.total_episodes is not None
        else f"{train_config.total_timesteps} timesteps"
    }
- Evaluation rollouts per stage: {eval_eps} episodes

## Beginning / Middle / End snapshots
These are **three saved model checkpoints**, not time windows of one long rollout.

| Stage | When saved | Checkpoint file |
|-------|------------|-----------------|
| Beginning | Before any gradient update | `model_beginning.zip` |
| Middle | After ~50% of training ({middle_ep} episodes or timesteps) | `model_middle.zip` |
| End | After training finishes | `model_end.zip` |

Each stage is evaluated independently on **{eval_eps} fresh episodes**
(seeds `{train_config.seed + 10_000}` .. `{train_config.seed + 10_000 + eval_eps - 1}`),
using the deterministic policy (`model.predict(..., deterministic=True)`).

## Per-episode plot conventions
All learning-curve x-axes use **episode index** (1, 2, 3, ...).
Loss curves aggregate logged SGD losses by **mean loss per training episode**.

Evaluation plots aggregate metrics **per evaluation episode first**, then summarize by stage:
- **Action / lane bars**: mean fraction of steps in each action/lane, averaged across the {eval_eps} episodes of that stage (error bars = std across episodes).
- **Hole outcomes**: mean count of each event type **per episode**, with std across episodes.
- **Avoidance rate**: for each eval episode, `avoided / (avoided + collisions)`, then mean ± std across episodes.
- **Near-hole lane changes**: mean count per episode, with std across episodes.
- **Falls**: mean falls per episode by stage; fall-reason counts are per-episode totals summed for display.
- **Trajectories**: episode 0 only (representative single rollout per stage).

## Hole event definitions
- **collision**: bike intersected a hole in its lane.
- **avoided**: hole was visible/threatening and passed in another lane without collision.
- **unprotected_pass**: hole was threatening but passed in the same lane without collision.
- **passed_other_lane**: hole passed while never in the bike's lane and not threatening.

## Environment settings for this run
- Lanes: {env_config.n_lanes}
- Actions: {env_config.action_count}
- Fixed speed: {env_config.fixed_speed}
- Hole mean gap (lambda): {env_config.hole_lambda}
- Fall rules: {_rules_label(env_config.fall_rules)}
- Collision penalty: {env_config.collision_penalty}
- Grace steps: {env_config.grace_steps}
"""
    (out_dir / "00_statistics_methodology.md").write_text(text, encoding="utf-8")
    print("    Saved: 00_statistics_methodology.md")


def plot_reward_and_loss(run_dir: Path, out_dir: Path, title: str) -> None:
    monitor = _episode_indexed_monitor(run_dir)
    losses = _episode_losses(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Training Curves (per episode) - {title}", fontsize=14, fontweight="bold")

    if monitor.empty:
        axes[0].text(0.5, 0.5, "No completed episodes", ha="center", va="center")
    else:
        axes[0].plot(monitor["episode"], monitor["r"], alpha=0.2, linewidth=0.6)
        moving = monitor["r"].rolling(100, min_periods=1).mean()
        axes[0].plot(
            monitor["episode"],
            moving,
            color="crimson",
            linewidth=2,
            label="100-episode mean",
        )
        axes[0].legend()
    axes[0].set(
        xlabel="Training Episode",
        ylabel="Episode Reward",
        title="Reward (raw and 100-episode mean)",
    )
    axes[0].grid(True, alpha=0.3)

    if losses.empty:
        axes[1].text(0.5, 0.5, "Loss unavailable before learning starts", ha="center", va="center")
    else:
        axes[1].plot(losses["episode"], losses["loss"], alpha=0.35, linewidth=0.8)
        smooth = losses["loss"].rolling(20, min_periods=1).mean()
        axes[1].plot(
            losses["episode"],
            smooth,
            color="darkorange",
            linewidth=2,
            label="20-episode mean",
        )
        axes[1].set_yscale("log")
        axes[1].legend()
    axes[1].set(xlabel="Training Episode", ylabel="Mean DQN Loss", title="Training Loss")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "01_reward_and_loss.png")


def _per_episode_action_fractions(data: EvaluationData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stage, episode), group in data.steps.groupby(["stage", "episode"], observed=True):
        total = len(group)
        if total == 0:
            continue
        for action_name, count in group["action_name"].value_counts().items():
            rows.append(
                {
                    "stage": stage,
                    "episode": episode,
                    "action_name": action_name,
                    "fraction": count / total,
                }
            )
    return pd.DataFrame(rows)


def _per_episode_lane_fractions(data: EvaluationData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stage, episode), group in data.steps.groupby(["stage", "episode"], observed=True):
        total = len(group)
        if total == 0:
            continue
        for lane, count in group["lane"].value_counts().items():
            rows.append(
                {
                    "stage": stage,
                    "episode": episode,
                    "lane": lane,
                    "fraction": count / total,
                }
            )
    return pd.DataFrame(rows)


def plot_action_and_lane_distributions(
    data: EvaluationData,
    env_config: EnvConfig,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Behavior Across Learning Stages (mean ± std over eval episodes)",
        fontsize=14,
        fontweight="bold",
    )

    action_fractions = _per_episode_action_fractions(data)
    action_summary = (
        action_fractions.groupby(["stage", "action_name"], observed=True)["fraction"]
        .agg(["mean", "std"])
        .reset_index()
    )
    sns.barplot(
        data=action_summary,
        x="action_name",
        y="mean",
        hue="stage",
        hue_order=STAGE_ORDER,
        ax=axes[0],
    )
    for bar, (_, row) in zip(axes[0].patches, action_summary.iterrows()):
        if pd.notna(row["std"]) and row["std"] > 0:
            bar.set_alpha(0.85)
    axes[0].set(
        xlabel="Action",
        ylabel="Mean Fraction of Steps (per episode)",
        title="Action Distribution",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)

    lane_fractions = _per_episode_lane_fractions(data)
    lane_summary = (
        lane_fractions.groupby(["stage", "lane"], observed=True)["fraction"]
        .agg(["mean", "std"])
        .reset_index()
    )
    sns.barplot(
        data=lane_summary,
        x="lane",
        y="mean",
        hue="stage",
        hue_order=STAGE_ORDER,
        ax=axes[1],
    )
    axes[1].set(
        xlabel="Lane",
        ylabel="Mean Fraction of Steps (per episode)",
        title=f"Lane Distribution ({env_config.n_lanes} lanes)",
    )
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "02_action_and_lane_distributions.png")


def plot_hole_learning(data: EvaluationData, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Hole-Passing Behavior (mean ± std per eval episode)",
        fontsize=14,
        fontweight="bold",
    )

    relevant_events = data.holes[
        data.holes["event"].isin(["avoided", "collision", "unprotected_pass"])
    ]
    if relevant_events.empty:
        axes[0].text(0.5, 0.5, "No threatened-hole outcomes", ha="center", va="center")
        axes[0].axis("off")
    else:
        per_episode_events = (
            relevant_events.groupby(["stage", "episode", "event"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        event_summary = (
            per_episode_events.groupby(["stage", "event"], observed=True)["count"]
            .agg(["mean", "std"])
            .reset_index()
            .fillna(0.0)
        )
        sns.barplot(
            data=event_summary,
            x="stage",
            y="mean",
            hue="event",
            order=STAGE_ORDER,
            ax=axes[0],
        )
        axes[0].set(
            xlabel="Learning Stage",
            ylabel="Mean Events per Episode",
            title="Threatened-Hole Outcomes",
        )
        axes[0].grid(axis="y", alpha=0.3)

    episode_metrics = data.episodes.copy()
    episode_metrics["avoidance_rate"] = np.where(
        (episode_metrics["holes_avoided"] + episode_metrics["hole_collisions"]) > 0,
        episode_metrics["holes_avoided"]
        / (episode_metrics["holes_avoided"] + episode_metrics["hole_collisions"]),
        np.nan,
    )
    near_hole = (
        data.steps.groupby(["stage", "episode"], observed=True)["near_hole_lane_change"]
        .sum()
        .reset_index(name="near_hole_lane_changes")
    )
    episode_metrics = episode_metrics.merge(
        near_hole, on=["stage", "episode"], how="left"
    ).fillna({"near_hole_lane_changes": 0})

    avoidance_summary = (
        episode_metrics.groupby("stage", observed=True)["avoidance_rate"]
        .agg(["mean", "std"])
        .reindex(STAGE_ORDER)
    )
    lane_change_summary = (
        episode_metrics.groupby("stage", observed=True)["near_hole_lane_changes"]
        .agg(["mean", "std"])
        .reindex(STAGE_ORDER)
    )

    axes[1].errorbar(
        avoidance_summary.index,
        avoidance_summary["mean"],
        yerr=avoidance_summary["std"].fillna(0.0),
        marker="o",
        linewidth=2,
        capsize=4,
        label="Avoidance rate",
    )
    axes[1].set(
        xlabel="Learning Stage",
        ylabel="Avoided / (Avoided + Collisions)",
        ylim=(0, 1.05),
        title="Avoidance Success and Near-Hole Maneuvers",
    )
    axes[1].grid(True, alpha=0.3)
    secondary = axes[1].twinx()
    secondary.bar(
        lane_change_summary.index,
        lane_change_summary["mean"],
        yerr=lane_change_summary["std"].fillna(0.0),
        alpha=0.25,
        color="purple",
        capsize=4,
        label="Near-hole lane changes / episode",
    )
    secondary.set_ylabel("Mean Near-Hole Lane Changes per Episode")
    lines, labels = axes[1].get_legend_handles_labels()
    bars, bar_labels = secondary.get_legend_handles_labels()
    axes[1].legend(lines + bars, labels + bar_labels, loc="best")
    fig.tight_layout()
    _save(fig, out_dir / "03_hole_learning.png")


def plot_stage_trajectories(
    data: EvaluationData,
    env_config: EnvConfig,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    fig.suptitle(
        "Representative Trajectories (eval episode 0 per stage)",
        fontsize=14,
        fontweight="bold",
    )
    event_colors = {
        "avoided": "green",
        "collision": "red",
        "unprotected_pass": "orange",
    }

    for ax, stage in zip(axes, STAGE_ORDER):
        trajectory = data.steps[
            (data.steps["stage"] == stage) & (data.steps["episode"] == 0)
        ]
        visible = data.visible_holes[
            (data.visible_holes["stage"] == stage)
            & (data.visible_holes["episode"] == 0)
        ]
        events = data.holes[
            (data.holes["stage"] == stage) & (data.holes["episode"] == 0)
        ]
        ax.scatter(
            visible["distance"],
            visible["lane"],
            marker="x",
            color="black",
            alpha=0.45,
            label="Visible hole",
        )
        ax.plot(
            trajectory["distance"],
            trajectory["lane_position"],
            color="steelblue",
            linewidth=1.5,
            label="Bike",
        )
        for event, color in event_colors.items():
            selected = events[events["event"] == event]
            if not selected.empty:
                ax.scatter(
                    selected["distance"],
                    selected["lane"],
                    color=color,
                    s=35,
                    label=event.replace("_", " ").title(),
                )
        ax.set(
            xlabel="Longitudinal Distance",
            title=stage.title(),
            yticks=range(env_config.n_lanes),
            ylim=(-0.5, env_config.n_lanes - 0.5),
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Lane Position")
    fig.tight_layout()
    _save(fig, out_dir / "04_beginning_middle_end_trajectories.png")


def plot_fall_analysis(data: EvaluationData, out_dir: Path) -> None:
    falls = data.steps[data.steps["fell"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fall Analysis (per eval episode)", fontsize=14, fontweight="bold")

    falls_per_episode = (
        data.episodes.groupby("stage", observed=True)["falls"]
        .agg(["mean", "std"])
        .reindex(STAGE_ORDER)
        .fillna(0.0)
    )
    axes[0].bar(
        falls_per_episode.index,
        falls_per_episode["mean"],
        yerr=falls_per_episode["std"],
        capsize=4,
        color="indianred",
        alpha=0.85,
    )
    axes[0].set(
        xlabel="Learning Stage",
        ylabel="Mean Falls per Episode",
        title="Falls per Episode by Stage",
    )
    axes[0].grid(axis="y", alpha=0.3)

    if falls.empty:
        axes[1].text(0.5, 0.5, "No falls recorded", ha="center", va="center")
        axes[1].axis("off")
    else:
        falls_per_episode_reason = (
            falls.groupby(["stage", "episode", "fall_reason"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        reason_summary = (
            falls_per_episode_reason.groupby(["stage", "fall_reason"], observed=True)[
                "count"
            ]
            .mean()
            .reset_index(name="mean_per_episode")
        )
        sns.barplot(
            data=reason_summary,
            x="fall_reason",
            y="mean_per_episode",
            hue="stage",
            hue_order=STAGE_ORDER,
            ax=axes[1],
        )
        axes[1].set(
            xlabel="Reason",
            ylabel="Mean Falls per Episode",
            title="Fall Reasons by Stage",
        )
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "05_fall_analysis.png")


def generate_all_plots(
    run_dir: Path,
    out_dir: Path,
    env_config: EnvConfig,
    train_config: TrainConfig,
    data: EvaluationData,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = _episode_indexed_monitor(run_dir)
    write_statistics_methodology(
        out_dir,
        env_config,
        train_config,
        n_training_episodes=len(monitor),
    )
    title = (
        f"{env_config.n_lanes} lanes, {env_config.action_count} actions, "
        f"falls={_rules_label(env_config.fall_rules)}, lambda={env_config.hole_lambda:g}"
    )
    plot_reward_and_loss(run_dir, out_dir, title)
    plot_action_and_lane_distributions(data, env_config, out_dir)
    plot_hole_learning(data, out_dir)
    plot_stage_trajectories(data, env_config, out_dir)
    plot_fall_analysis(data, out_dir)
    return monitor


def plot_all_lanes_comparison(all_data: dict[int, pd.DataFrame], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for n_lanes, frame in all_data.items():
        if frame.empty:
            continue
        episodes = frame["episode"] if "episode" in frame.columns else frame.index + 1
        moving = frame["r"].rolling(100, min_periods=1).mean()
        ax.plot(episodes, moving, linewidth=2, label=f"{n_lanes} lanes")
    ax.set(
        xlabel="Training Episode",
        ylabel="Mean Reward (100-episode MA)",
        title="Learning Curves Across Lane Counts",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "all_lanes_comparison.png")


def plot_lambda_comparison(
    reward_data: dict[float, pd.DataFrame],
    eval_data: dict[float, EvaluationData],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Training Reward and Hole-Passing Across Lambda Values",
        fontsize=14,
        fontweight="bold",
    )

    for hole_lambda, frame in sorted(reward_data.items()):
        if frame.empty:
            continue
        episodes = frame["episode"] if "episode" in frame.columns else frame.index + 1
        moving = frame["r"].rolling(100, min_periods=1).mean()
        axes[0].plot(episodes, moving, linewidth=2, label=f"λ={hole_lambda:g}")
    axes[0].set(
        xlabel="Training Episode",
        ylabel="Mean Reward (100-episode MA)",
        title="Training Reward",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    summary_rows: list[dict[str, Any]] = []
    for hole_lambda, data in sorted(eval_data.items()):
        end_episodes = data.episodes[data.episodes["stage"] == "end"]
        if end_episodes.empty:
            continue
        summary_rows.append(
            {
                "hole_lambda": hole_lambda,
                "lambda_label": f"λ={hole_lambda:g}",
                "mean_holes_passed": end_episodes["holes_passed"].mean(),
                "std_holes_passed": end_episodes["holes_passed"].std(ddof=0),
            }
        )
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        sns.barplot(
            data=summary,
            x="lambda_label",
            y="mean_holes_passed",
            order=summary["lambda_label"].tolist(),
            ax=axes[1],
            color="steelblue",
        )
        for index, row in summary.iterrows():
            axes[1].errorbar(
                index,
                row["mean_holes_passed"],
                yerr=row["std_holes_passed"],
                fmt="none",
                ecolor="black",
                capsize=4,
            )
    else:
        axes[1].text(0.5, 0.5, "No end-stage evaluation data", ha="center", va="center")
    axes[1].set(
        xlabel="Hole Mean Gap (λ)",
        ylabel="Mean Holes Passed per Episode",
        title="End-Stage Holes Passed (eval, mean ± std)",
    )
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "lambda_comparison_rewards_and_holes.png")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configurable Bike Rider DQN v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lanes", nargs="+", type=int, default=[5], metavar="N")
    parser.add_argument("--actions", type=int, choices=[3, 5], default=3)
    parser.add_argument("--fixed-speed", type=float, default=5.0)
    parser.add_argument(
        "--fall-rules",
        nargs="+",
        choices=["none", *FALL_RULES],
        default=["hole_collision"],
        help="Enabled fall rules; use 'none' alone to disable all falls",
    )
    parser.add_argument(
        "--hole-lambda",
        nargs="+",
        type=float,
        default=[5.0],
        metavar="LAMBDA",
        help="Mean distance between holes (Poisson process scale); pass multiple values to compare",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=-10.0,
        help="Reward on collision/fall step",
    )
    parser.add_argument("--grace-steps", type=int, default=5)
    parser.add_argument("--reset-holes-on-fall", action="store_true")
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Training budget in environment steps (ignored if --episodes is set)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Stop training after this many completed episodes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-final-eps", type=float, default=0.025)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--loss-log-freq", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=_DEFAULT_PLOT_DIR)
    args = parser.parse_args()
    if "none" in args.fall_rules and len(args.fall_rules) > 1:
        parser.error("'none' cannot be combined with other fall rules")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes is None and args.timesteps is None:
        args.timesteps = 1_000_000
    fall_rules = () if args.fall_rules == ["none"] else tuple(args.fall_rules)
    train_config = TrainConfig(
        total_timesteps=args.timesteps,
        total_episodes=args.episodes,
        seed=args.seed,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        loss_log_freq=args.loss_log_freq,
        eval_episodes=args.eval_episodes,
    )
    all_lane_data: dict[int, pd.DataFrame] = {}
    all_lambda_reward_data: dict[float, pd.DataFrame] = {}
    all_lambda_eval_data: dict[float, EvaluationData] = {}

    for hole_lambda in args.hole_lambda:
        for n_lanes in args.lanes:
            env_config = EnvConfig(
                n_lanes=n_lanes,
                action_count=args.actions,
                fixed_speed=args.fixed_speed,
                hole_lambda=hole_lambda,
                collision_penalty=args.collision_penalty,
                fall_rules=fall_rules,
                grace_steps=args.grace_steps,
                reset_holes_on_fall=args.reset_holes_on_fall,
                episode_steps=args.episode_steps,
            )
            run_name = experiment_name(env_config, args.seed)
            print(f"\n{'=' * 72}\n  {run_name}\n{'=' * 72}")
            _, run_dir = train(env_config, train_config, args.output_dir)
            evaluation = evaluate_snapshots(
                run_dir,
                env_config,
                train_config.eval_episodes,
                train_config.seed + 10_000,
            )
            plot_dir = args.plot_dir / run_name
            monitor = generate_all_plots(
                run_dir,
                plot_dir,
                env_config,
                train_config,
                evaluation,
            )
            if len(args.lanes) == 1:
                all_lambda_reward_data[hole_lambda] = monitor
                all_lambda_eval_data[hole_lambda] = evaluation
            if len(args.hole_lambda) == 1:
                all_lane_data[n_lanes] = monitor

    if len(all_lane_data) > 1:
        plot_all_lanes_comparison(all_lane_data, args.plot_dir)
    if len(all_lambda_reward_data) > 1:
        plot_lambda_comparison(all_lambda_reward_data, all_lambda_eval_data, args.plot_dir)
    print(f"\nAll runs complete. Plots saved under: {args.plot_dir}")


if __name__ == "__main__":
    main()
