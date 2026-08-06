#!/usr/bin/env python3
"""Configurable multi-lane Bike DQN training and diagnostics (v1.1).

v1.1 generates complete hole layouts lane-by-lane. Consecutive holes in one
lane have a shifted-exponential gap, and holes in lane ``n`` are rejected when
they are too close to any hole in lane ``n - 1``. Lateral actions move the
bike discretely to an adjacent lane while forward speed stays unchanged.

Examples:
    python bike_dqn_multi_lanes_v1.1.py --lanes 5 --actions 3 \
        --hole-lambda 5 --min-same-lane-distance 8 \
        --min-adjacent-hole-distance 20

    python bike_dqn_multi_lanes_v1.1.py --lanes 2 3 5 10 \
        --plot-hole-layout-only --preview-distance 500
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
_DEFAULT_OUTPUT = _HERE / "training_results_v1_1"
_DEFAULT_PLOT_DIR = _HERE.parent / "plotting" / "bike_dqn_multi_lanes_v1_1"

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
    min_same_lane_distance: float = 8.0
    min_adjacent_hole_distance: float = 20.0
    adjacent_deletion_multiplier: float = 2.0
    fall_rules: tuple[str, ...] = ("hole_collision",)
    grace_steps: int = 5
    reset_holes_on_fall: bool = False
    episode_steps: int = 1000
    fall_time_penalty: int = 500
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
        if self.min_same_lane_distance < 0:
            raise ValueError("min_same_lane_distance cannot be negative")
        if self.min_adjacent_hole_distance < 0:
            raise ValueError("min_adjacent_hole_distance cannot be negative")
        if self.adjacent_deletion_multiplier < 0:
            raise ValueError("adjacent_deletion_multiplier cannot be negative")
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
    same_sep = f"{config.min_same_lane_distance:g}".replace(".", "p")
    adjacent_sep = f"{config.min_adjacent_hole_distance:g}".replace(".", "p")
    return (
        f"lanes_{config.n_lanes}__actions_{config.action_count}"
        f"__falls_{_rules_label(config.fall_rules)}__lambda_{lambda_text}"
        f"__same_sep_{same_sep}__adj_sep_{adjacent_sep}__seed_{seed}"
    )


def experiment_name_base(config: EnvConfig) -> str:
    lambda_text = f"{config.hole_lambda:g}".replace(".", "p")
    same_sep = f"{config.min_same_lane_distance:g}".replace(".", "p")
    adjacent_sep = f"{config.min_adjacent_hole_distance:g}".replace(".", "p")
    return (
        f"lanes_{config.n_lanes}__actions_{config.action_count}"
        f"__falls_{_rules_label(config.fall_rules)}__lambda_{lambda_text}"
        f"__same_sep_{same_sep}__adj_sep_{adjacent_sep}"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BikeEnvAdvancedV1_1(gym.Env):
    """Bike environment with a pre-generated, lane-by-lane hole layout.

    Same-lane gaps are ``min_same_lane_distance + Exponential(hole_lambda)``.
    Lane 0 is generated first. For lane ``n > 0``, each candidate must be at
    least the adjacent deletion threshold away from every hole already placed
    in lane ``n - 1``.
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

    @property
    def adjacent_deletion_distance(self) -> float:
        return (
            self.config.min_adjacent_hole_distance
            * self.config.adjacent_deletion_multiplier
        )

    def _sample_same_lane_gap(self) -> float:
        """Draw one shifted-exponential same-lane gap."""
        floor = self.config.min_same_lane_distance
        if self.config.n_lanes > 1 and self.adjacent_deletion_distance > 0:
            # Leave enough room for the next lane to place holes between exclusions.
            floor = max(floor, 2.0 * self.adjacent_deletion_distance)
        return floor + float(self.np_random.exponential(self.config.hole_lambda))

    def _candidate_clear_of_previous_lane(
        self,
        candidate: float,
        previous_lane: np.ndarray,
        deletion_distance: float,
    ) -> bool:
        """Return True when ``candidate`` is far enough from every previous-lane hole."""
        if deletion_distance <= 0 or len(previous_lane) == 0:
            return True
        return bool(np.all(np.abs(previous_lane - candidate) >= deletion_distance))

    def _advance_to_next_valid_candidate(
        self,
        candidate: float,
        last_accepted: float,
        previous_lane: np.ndarray,
        deletion_distance: float,
    ) -> float:
        """Move ``candidate`` forward until it clears all previous-lane exclusion zones."""
        floor = self.config.min_same_lane_distance
        if self.config.n_lanes > 1 and deletion_distance > 0:
            floor = max(floor, 2.0 * deletion_distance)
        if not self._candidate_clear_of_previous_lane(
            candidate, previous_lane, deletion_distance
        ):
            blockers = previous_lane[
                np.abs(previous_lane - candidate) < deletion_distance
            ]
            candidate = float(np.max(blockers) + deletion_distance)
        return max(candidate, last_accepted + floor)


    def _generate_hole_layout(
        self,
        horizon: float,
        *,
        origin: float = 0.0,
    ) -> list[np.ndarray]:
        """Generate all lanes from ``origin`` to ``horizon`` in lane order.

        Every lane restarts at ``origin``. For lane ``i > 0``, each candidate
        must stay at least the adjacent deletion threshold away from every
        hole already placed in lane ``i - 1``. Rejected candidates are
        advanced forward to the next valid position; if none exists before
        ``horizon``, hole generation for that lane stops.
        """
        if horizon <= origin:
            raise ValueError("hole-layout horizon must be greater than origin")

        layouts: list[np.ndarray] = []
        deletion_distance = self.adjacent_deletion_distance
        for lane in range(self.config.n_lanes):
            positions: list[float] = []
            last_accepted = origin
            previous_lane = layouts[lane - 1] if lane > 0 else np.array([])
            while last_accepted <= horizon:
                candidate = last_accepted + self._sample_same_lane_gap()
                candidate = self._advance_to_next_valid_candidate(
                    candidate,
                    last_accepted,
                    previous_lane,
                    deletion_distance,
                )
                if candidate > horizon:
                    break
                if not self._candidate_clear_of_previous_lane(
                    candidate, previous_lane, deletion_distance
                ):
                    break
                positions.append(candidate)
                last_accepted = candidate
            layouts.append(np.asarray(positions, dtype=np.float64))
        self._validate_hole_layout(layouts, origin=origin, horizon=horizon)
        return layouts

    def _validate_hole_layout(
        self,
        layouts: list[np.ndarray],
        *,
        origin: float,
        horizon: float,
    ) -> None:
        """Validate ordering and both configured spacing invariants."""
        tolerance = 1e-9
        if len(layouts) != self.config.n_lanes:
            raise RuntimeError("hole layout has the wrong number of lanes")

        for lane, positions in enumerate(layouts):
            if np.any(positions <= origin) or np.any(positions > horizon):
                raise RuntimeError(f"lane {lane} contains holes outside the layout range")
            gaps = np.diff(np.concatenate(([origin], positions)))
            min_gap = self.config.min_same_lane_distance
            if self.config.n_lanes > 1 and self.adjacent_deletion_distance > 0:
                min_gap = max(min_gap, 2.0 * self.adjacent_deletion_distance)
            if np.any(gaps < min_gap - tolerance):
                raise RuntimeError(
                    f"lane {lane} violates the minimum same-lane hole distance"
                )
            if lane == 0:
                continue
            previous = layouts[lane - 1]
            if len(previous) == 0 or len(positions) == 0:
                continue
            separations = np.abs(positions[:, None] - previous[None, :])
            if np.any(separations < self.adjacent_deletion_distance - tolerance):
                raise RuntimeError(
                    f"lane {lane} violates the adjacent-lane deletion distance"
                )

    def _layout_horizon(self) -> float:
        max_distance = self.config.max_speed * 0.1 * self.config.episode_steps
        return (
            self.distance_traveled
            + max_distance
            + self.config.visibility_range
            + self.config.min_same_lane_distance
        )

    def _set_current_hole(self, lane: int) -> None:
        index = self._hole_indices[lane]
        layout = self.hole_layouts[lane]
        if index >= len(layout):
            self.holes[lane] = np.inf
            self.hole_ids[lane] = self._new_hole_id()
            return
        self.holes[lane] = float(layout[index] - self.distance_traveled)
        self.hole_ids[lane] = self._new_hole_id()

    def _initialize_holes(self, *, layout_horizon: float | None = None) -> None:
        horizon = layout_horizon if layout_horizon is not None else self._layout_horizon()
        self.hole_layouts = self._generate_hole_layout(
            horizon,
            origin=self.distance_traveled,
        )
        self._hole_indices = [0] * self.config.n_lanes
        self.holes = np.empty(self.config.n_lanes, dtype=np.float64)
        self.hole_ids = [0] * self.config.n_lanes
        for lane in range(self.config.n_lanes):
            self._set_current_hole(lane)
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
        layout_horizon = None
        if options and "layout_horizon" in options:
            layout_horizon = float(options["layout_horizon"])
        self._initialize_holes(layout_horizon=layout_horizon)
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
            "falls": self.falls,
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

        if self.config.action_count == 3:
            self.velocity = self.config.fixed_speed
        elif action_int == 3:
            self.velocity = min(self.config.max_speed, self.velocity + 1.0)
        elif action_int == 4:
            self.velocity = max(0.0, self.velocity * 0.9)

        if action_int == 0:
            current_lane = max(0, old_lane - 1)
        elif action_int == 2:
            current_lane = min(self.config.n_lanes - 1, old_lane + 1)
        else:
            current_lane = old_lane
        self.x_pos = float(current_lane)
        self.angle = 0.0
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
                self._hole_indices[lane] += 1
                self._set_current_hole(lane)

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
            "falls": self.falls,
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
        self.episode_rows: list[dict[str, float | int]] = []
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
        infos = self.locals.get("infos")
        if dones is not None:
            for done, info in zip(np.atleast_1d(dones), infos or []):
                if not done:
                    continue
                self.episode_count += 1
                episode_stats = info.get("episode", {})
                self.episode_rows.append(
                    {
                        "episode": self.episode_count,
                        "timesteps": float(self.num_timesteps),
                        "reward": float(episode_stats.get("r", np.nan)),
                        "steps": float(episode_stats.get("l", np.nan)),
                        "falls": int(info.get("falls", 0)),
                    }
                )
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
        pd.DataFrame(
            self.episode_rows,
            columns=["episode", "timesteps", "reward", "steps", "falls"],
        ).to_csv(self.run_dir / "training_episodes.csv", index=False)


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

    env = Monitor(BikeEnvAdvancedV1_1(env_config), str(run_dir))
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
            env = BikeEnvAdvancedV1_1(env_config)
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


def _same_lane_gap_rows(
    layouts: list[np.ndarray],
    preview_distance: float,
    *,
    run_index: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for lane, positions in enumerate(layouts):
        visible = positions[positions <= preview_distance]
        if len(visible) < 2:
            continue
        for gap_index, gap in enumerate(np.diff(visible), start=1):
            row = {
                "lane": lane,
                "gap_index": gap_index,
                "distance": float(gap),
            }
            if run_index is not None:
                row["run_index"] = run_index
            if seed is not None:
                row["seed"] = seed
            rows.append(row)
    return pd.DataFrame(rows)


def _adjacent_lane_min_distance_rows(
    layouts: list[np.ndarray],
    preview_distance: float,
    *,
    run_index: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """For each hole in lane i+1, record min distance to any hole in lane i."""
    rows: list[dict[str, float | int | str]] = []
    for lane in range(len(layouts) - 1):
        left = layouts[lane][layouts[lane] <= preview_distance]
        right = layouts[lane + 1][layouts[lane + 1] <= preview_distance]
        if len(left) == 0 or len(right) == 0:
            continue
        pair_label = f"{lane}-{lane + 1}"
        for hole_index, position in enumerate(right):
            min_distance = float(np.min(np.abs(left - position)))
            row = {
                "lane_pair": pair_label,
                "left_lane": lane,
                "right_lane": lane + 1,
                "hole_index": hole_index,
                "distance": min_distance,
            }
            if run_index is not None:
                row["run_index"] = run_index
            if seed is not None:
                row["seed"] = seed
            rows.append(row)
    return pd.DataFrame(rows)


def _spacing_thresholds(env_config: EnvConfig) -> tuple[float, float]:
    deletion_distance = (
        env_config.min_adjacent_hole_distance
        * env_config.adjacent_deletion_multiplier
    )
    effective_same_lane_floor = env_config.min_same_lane_distance
    if env_config.n_lanes > 1 and deletion_distance > 0:
        effective_same_lane_floor = max(
            effective_same_lane_floor,
            2.0 * deletion_distance,
        )
    return effective_same_lane_floor, deletion_distance


def plot_aggregated_hole_spacing_distributions(
    same_lane: pd.DataFrame,
    adjacent_lane: pd.DataFrame,
    out_dir: Path,
    env_config: EnvConfig,
    *,
    n_runs: int,
    preview_distance: float,
) -> None:
    """Plot pooled spacing distributions across many preview runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_same_lane_floor, deletion_distance = _spacing_thresholds(env_config)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Aggregated Hole Spacing — {env_config.n_lanes} lanes, "
        f"{n_runs} runs, 0–{preview_distance:g} distance",
        fontsize=14,
        fontweight="bold",
    )

    if same_lane.empty:
        axes[0].text(0.5, 0.5, "No same-lane gaps collected", ha="center", va="center")
    else:
        sns.histplot(
            data=same_lane,
            x="distance",
            bins=25,
            stat="density",
            kde=True,
            color="steelblue",
            ax=axes[0],
        )
        axes[0].axvline(
            effective_same_lane_floor,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Effective min ({effective_same_lane_floor:g})",
        )
        axes[0].legend()
    axes[0].set(
        xlabel="Distance Between Consecutive Holes in Same Lane",
        ylabel="Density",
        title=f"Same-Lane Gap Distribution ({n_runs} runs pooled)",
    )
    axes[0].grid(True, alpha=0.25)

    if adjacent_lane.empty:
        axes[1].text(
            0.5,
            0.5,
            "No adjacent-lane distances collected",
            ha="center",
            va="center",
        )
    else:
        sns.histplot(
            data=adjacent_lane,
            x="distance",
            bins=25,
            stat="density",
            kde=True,
            color="darkorange",
            ax=axes[1],
        )
        axes[1].axvline(
            deletion_distance,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Deletion threshold ({deletion_distance:g})",
        )
        axes[1].legend()
    axes[1].set(
        xlabel="Min Distance to Nearest Hole in Previous Lane",
        ylabel="Density",
        title=(
            f"Adjacent-Lane Min Distance per Hole ({n_runs} runs pooled)"
        ),
    )
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir / f"02_hole_spacing_distributions_{n_runs}_runs.png")

    if not same_lane.empty:
        same_lane.to_csv(out_dir / f"same_lane_gaps_{n_runs}_runs.csv", index=False)
    if not adjacent_lane.empty:
        adjacent_lane.to_csv(
            out_dir / f"adjacent_lane_min_distances_{n_runs}_runs.csv",
            index=False,
        )


def plot_hole_spacing_distributions(
    layouts: list[np.ndarray],
    out_dir: Path,
    env_config: EnvConfig,
    *,
    seed: int,
    preview_distance: float,
) -> None:
    """Plot same-lane and adjacent-lane hole distance distributions."""
    same_lane = _same_lane_gap_rows(layouts, preview_distance)
    adjacent_lane = _adjacent_lane_min_distance_rows(layouts, preview_distance)
    effective_same_lane_floor, deletion_distance = _spacing_thresholds(env_config)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Hole Spacing Distributions — {env_config.n_lanes} lanes, seed={seed}",
        fontsize=14,
        fontweight="bold",
    )

    if same_lane.empty:
        axes[0].text(0.5, 0.5, "Not enough holes for same-lane gaps", ha="center", va="center")
    else:
        sns.histplot(
            data=same_lane,
            x="distance",
            hue="lane",
            multiple="stack",
            bins=min(20, max(5, len(same_lane) // 3)),
            ax=axes[0],
        )
        axes[0].axvline(
            effective_same_lane_floor,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Effective min ({effective_same_lane_floor:g})",
        )
    axes[0].set(
        xlabel="Distance Between Consecutive Holes in Same Lane",
        ylabel="Count",
        title="Same-Lane Gap Distribution",
    )
    axes[0].grid(True, alpha=0.25)
    if not same_lane.empty:
        axes[0].legend(title="Lane", fontsize=8)

    if adjacent_lane.empty:
        axes[1].text(
            0.5,
            0.5,
            "Not enough holes for adjacent-lane distances",
            ha="center",
            va="center",
        )
    else:
        sns.histplot(
            data=adjacent_lane,
            x="distance",
            hue="lane_pair",
            multiple="stack",
            bins=min(20, max(5, len(adjacent_lane) // 5)),
            ax=axes[1],
        )
        axes[1].axvline(
            deletion_distance,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Deletion threshold ({deletion_distance:g})",
        )
    axes[1].set(
        xlabel="Min Distance to Nearest Hole in Previous Lane",
        ylabel="Count",
        title="Adjacent-Lane Min Distance per Hole",
    )
    axes[1].grid(True, alpha=0.25)
    if not adjacent_lane.empty:
        axes[1].legend(title="Lane pair", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "01_hole_spacing_distributions.png")

    if not same_lane.empty:
        same_lane.to_csv(out_dir / "same_lane_gaps.csv", index=False)
    if not adjacent_lane.empty:
        adjacent_lane.to_csv(
            out_dir / "adjacent_lane_min_distances.csv",
            index=False,
        )


def plot_hole_layout_preview(
    env_config: EnvConfig,
    out_dir: Path,
    *,
    seed: int,
    preview_distance: float,
) -> tuple[Path, list[np.ndarray]]:
    """Generate and plot a seeded hole layout without creating an agent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = BikeEnvAdvancedV1_1(env_config)
    env.reset(seed=seed, options={"layout_horizon": preview_distance})
    layouts = [lane.copy() for lane in env.hole_layouts]

    fig, ax = plt.subplots(figsize=(16, max(5, env_config.n_lanes * 0.6)))
    for lane, positions in enumerate(layouts):
        visible = positions[positions <= preview_distance]
        ax.scatter(
            visible,
            np.full(len(visible), lane),
            marker="x",
            s=35,
            linewidths=1.5,
            label=f"Lane {lane}",
        )

    deletion_distance = (
        env_config.min_adjacent_hole_distance
        * env_config.adjacent_deletion_multiplier
    )
    ax.set(
        xlim=(0, preview_distance),
        ylim=(-0.5, env_config.n_lanes - 0.5),
        yticks=range(env_config.n_lanes),
        xlabel="Longitudinal Distance",
        ylabel="Lane",
        title=(
            f"Seeded Hole Layout Preview — {env_config.n_lanes} lanes, "
            f"λ={env_config.hole_lambda:g}, same-lane minimum="
            f"{env_config.min_same_lane_distance:g}, adjacent base="
            f"{env_config.min_adjacent_hole_distance:g}, "
            f"deletion threshold={deletion_distance:g}, seed={seed}"
        ),
    )
    ax.grid(True, alpha=0.25)
    if env_config.n_lanes <= 10:
        ax.legend(loc="upper right", ncol=min(env_config.n_lanes, 5), fontsize=8)
    fig.tight_layout()
    path = out_dir / "00_hole_layout_preview.png"
    _save(fig, path)
    plot_hole_spacing_distributions(
        layouts,
        out_dir,
        env_config,
        seed=seed,
        preview_distance=preview_distance,
    )
    env.close()
    return path, layouts


def plot_hole_layout_multi_preview(
    env_config: EnvConfig,
    base_out_dir: Path,
    *,
    seeds: list[int],
    preview_distance: float,
) -> None:
    """Generate per-run previews and an aggregated spacing distribution."""
    base_out_dir.mkdir(parents=True, exist_ok=True)
    all_same_lane_frames: list[pd.DataFrame] = []
    all_adjacent_lane_frames: list[pd.DataFrame] = []

    for run_index, seed in enumerate(seeds, start=1):
        run_dir = base_out_dir / f"run_{run_index:02d}_seed_{seed}"
        print(f"\n{'=' * 72}\n  Preview run {run_index}/{len(seeds)}: seed={seed}\n{'=' * 72}")
        _, layouts = plot_hole_layout_preview(
            env_config,
            run_dir,
            seed=seed,
            preview_distance=preview_distance,
        )
        all_same_lane_frames.append(
            _same_lane_gap_rows(
                layouts,
                preview_distance,
                run_index=run_index,
                seed=seed,
            )
        )
        all_adjacent_lane_frames.append(
            _adjacent_lane_min_distance_rows(
                layouts,
                preview_distance,
                run_index=run_index,
                seed=seed,
            )
        )

    aggregated_dir = base_out_dir / f"aggregated_{len(seeds)}_runs"
    same_lane = (
        pd.concat(all_same_lane_frames, ignore_index=True)
        if all_same_lane_frames
        else pd.DataFrame()
    )
    adjacent_lane = (
        pd.concat(all_adjacent_lane_frames, ignore_index=True)
        if all_adjacent_lane_frames
        else pd.DataFrame()
    )
    plot_aggregated_hole_spacing_distributions(
        same_lane,
        adjacent_lane,
        aggregated_dir,
        env_config,
        n_runs=len(seeds),
        preview_distance=preview_distance,
    )


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
- Exponential gap scale (lambda): {env_config.hole_lambda}
- Minimum same-lane gap: {env_config.min_same_lane_distance}
- Adjacent-lane base distance: {env_config.min_adjacent_hole_distance}
- Adjacent deletion multiplier: {env_config.adjacent_deletion_multiplier}
- Effective adjacent deletion distance: {
        env_config.min_adjacent_hole_distance
        * env_config.adjacent_deletion_multiplier
    }
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
        f"falls={_rules_label(env_config.fall_rules)}, lambda={env_config.hole_lambda:g}, "
        f"same_sep={env_config.min_same_lane_distance:g}, "
        f"adjacent_threshold={env_config.min_adjacent_hole_distance * env_config.adjacent_deletion_multiplier:g}"
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
        description="Configurable Bike Rider DQN v1.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lanes", nargs="+", type=int, default=[5], metavar="N")
    parser.add_argument("--actions", type=int, choices=[3, 5], default=3)
    parser.add_argument("--fixed-speed", type=float, default=5.0)
    parser.add_argument(
        "--visibility-range",
        type=float,
        default=25.0,
        metavar="DISTANCE",
        help="Longitudinal range within which upcoming holes are visible in observations",
    )
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
        "--min-same-lane-distance",
        type=float,
        default=8.0,
        metavar="X",
        help="Minimum same-lane gap added to every exponential gap sample",
    )
    parser.add_argument(
        "--min-adjacent-hole-distance",
        type=float,
        default=20.0,
        metavar="X",
        help=(
            "Base adjacent-lane distance; multiplied by "
            "--adjacent-deletion-multiplier for candidate rejection"
        ),
    )
    parser.add_argument(
        "--adjacent-deletion-multiplier",
        type=float,
        default=2.0,
        metavar="M",
        help="Multiplier applied to the adjacent-lane base distance",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=-10.0,
        help="Reward on collision/fall step",
    )
    parser.add_argument(
        "--fall-time-penalty",
        type=int,
        default=500,
        metavar="STEPS",
        help="Environment steps removed from the episode after each fall",
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
    parser.add_argument(
        "--plot-hole-layout-only",
        action="store_true",
        help="Plot seeded hole layouts and exit without creating or training an agent",
    )
    parser.add_argument(
        "--preview-distance",
        type=float,
        default=500.0,
        metavar="DISTANCE",
        help="Longitudinal range shown by --plot-hole-layout-only",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of preview runs to generate with --plot-hole-layout-only; "
            "uses seeds seed, seed+1, ..., seed+N-1"
        ),
    )
    args = parser.parse_args()
    if "none" in args.fall_rules and len(args.fall_rules) > 1:
        parser.error("'none' cannot be combined with other fall rules")
    if args.preview_distance <= 0:
        parser.error("--preview-distance must be positive")
    if args.preview_count <= 0:
        parser.error("--preview-count must be positive")
    return args


def _env_config_from_args(
    args: argparse.Namespace,
    *,
    n_lanes: int,
    hole_lambda: float,
    fall_rules: tuple[str, ...],
) -> EnvConfig:
    return EnvConfig(
        n_lanes=n_lanes,
        action_count=args.actions,
        fixed_speed=args.fixed_speed,
        visibility_range=args.visibility_range,
        hole_lambda=hole_lambda,
        min_same_lane_distance=args.min_same_lane_distance,
        min_adjacent_hole_distance=args.min_adjacent_hole_distance,
        adjacent_deletion_multiplier=args.adjacent_deletion_multiplier,
        collision_penalty=args.collision_penalty,
        fall_time_penalty=args.fall_time_penalty,
        fall_rules=fall_rules,
        grace_steps=args.grace_steps,
        reset_holes_on_fall=args.reset_holes_on_fall,
        episode_steps=args.episode_steps,
    )


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    fall_rules = () if args.fall_rules == ["none"] else tuple(args.fall_rules)

    if args.plot_hole_layout_only:
        preview_seeds = [args.seed + offset for offset in range(args.preview_count)]
        for hole_lambda in args.hole_lambda:
            for n_lanes in args.lanes:
                env_config = _env_config_from_args(
                    args,
                    n_lanes=n_lanes,
                    hole_lambda=hole_lambda,
                    fall_rules=fall_rules,
                )
                base_name = experiment_name_base(env_config)
                base_out_dir = args.plot_dir / base_name
                print(f"\n{'#' * 72}\n  {base_name} — {args.preview_count} preview runs\n{'#' * 72}")
                if args.preview_count == 1:
                    plot_hole_layout_preview(
                        env_config,
                        base_out_dir / f"run_01_seed_{preview_seeds[0]}",
                        seed=preview_seeds[0],
                        preview_distance=args.preview_distance,
                    )
                else:
                    plot_hole_layout_multi_preview(
                        env_config,
                        base_out_dir,
                        seeds=preview_seeds,
                        preview_distance=args.preview_distance,
                    )
        print(f"\nHole-layout previews saved under: {args.plot_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes is None and args.timesteps is None:
        args.timesteps = 1_000_000
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
            env_config = _env_config_from_args(
                args,
                n_lanes=n_lanes,
                hole_lambda=hole_lambda,
                fall_rules=fall_rules,
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
