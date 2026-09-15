#!/usr/bin/env python3
"""Shared CartPole DQN runner and speed metrics for HPO experiments."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from confgate_meta_hpo import (
    BATCH_SIZE,
    BUFFER_SIZE,
    DEFAULT_TARGET_FREQ,
    DEFAULT_TRAIN_FREQ,
    DQNAgent,
    LEARNING_RATE,
    LEARNING_STARTS,
    ReplayBuffer,
)

CARTPOLE_MAX_RETURN = 500.0
CARTPOLE_SOLVE_RETURN = 475.0


@dataclass
class DQNConfig:
    """The three requested hyperparameter families plus epsilon schedule shape."""

    learning_rate: float = LEARNING_RATE
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_episodes: int = 500
    train_freq: int = DEFAULT_TRAIN_FREQ
    target_freq: int = DEFAULT_TARGET_FREQ

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DQNConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})


def rolling_first_solve(
    returns: np.ndarray,
    threshold: float = CARTPOLE_SOLVE_RETURN,
    window: int = 20,
) -> Optional[int]:
    """Return one-based episode index of first rolling-mean solve."""
    if returns.size < window:
        return None
    means = np.convolve(returns, np.ones(window) / window, mode="valid")
    hits = np.flatnonzero(means >= threshold)
    return int(hits[0] + window) if hits.size else None


def speed_metrics(episode_returns: list[float], budget_episodes: int) -> Dict[str, Any]:
    """Metrics centered on learning speed rather than final return only."""
    values = np.asarray(episode_returns, dtype=np.float64)
    if values.size == 0:
        return {
            "speed_score": 0.0,
            "auc_normalized": 0.0,
            "mean_return": 0.0,
            "last100_mean": 0.0,
            "first_solve_episode": None,
            "episodes": 0,
        }

    normalized = np.clip(values / CARTPOLE_MAX_RETURN, 0.0, 1.0)
    # Missing episodes are zero-valued when a run hits its safety cap.
    auc_normalized = float(normalized.sum() / max(1, budget_episodes))
    first_solve = rolling_first_solve(values)
    # AUC is the primary objective. A small time-to-solve term breaks close ties.
    solve_speed = (
        1.0 - (first_solve / max(1, budget_episodes))
        if first_solve is not None
        else 0.0
    )
    return {
        "speed_score": auc_normalized + 0.05 * max(0.0, solve_speed),
        "auc_normalized": auc_normalized,
        "mean_return": float(values.mean()),
        "last100_mean": float(values[-100:].mean()),
        "first_solve_episode": first_solve,
        "episodes": int(values.size),
    }


class CartPoleDQNRun:
    """Incremental, checkpointable DQN lifetime used by Optuna and PB2."""

    def __init__(
        self,
        config: DQNConfig,
        seed: int,
        device: torch.device,
        env_id: str = "CartPole-v1",
        max_steps: Optional[int] = None,
    ):
        self.config = config
        self.seed = int(seed)
        self.device = device
        self.env_id = env_id
        self.max_steps = max_steps
        self._seed_everything(self.seed)

        self.env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
        obs_shape = self.env.observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        n_actions = int(self.env.action_space.n)
        self.agent = DQNAgent(
            obs_dim,
            n_actions,
            device,
            gamma=config.gamma,
        )
        self.agent.opt.param_groups[0]["lr"] = float(config.learning_rate)
        self.replay = ReplayBuffer(BUFFER_SIZE, obs_shape, device)
        self.episode_returns: list[float] = []
        self.steps = 0
        self.obs, _ = self.env.reset(seed=self.seed)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_config(self, config: DQNConfig) -> None:
        """Apply PB2 mutations without restarting the inner learner."""
        self.config = config
        self.agent.gamma = float(config.gamma)
        self.agent.opt.param_groups[0]["lr"] = float(config.learning_rate)

    def epsilon(self) -> float:
        frac = min(
            1.0,
            len(self.episode_returns)
            / max(1, int(self.config.epsilon_decay_episodes)),
        )
        return float(
            self.config.epsilon_start
            + frac * (self.config.epsilon_final - self.config.epsilon_start)
        )

    def train_to_episode(self, target_episode: int) -> Dict[str, Any]:
        """Advance this lifetime to an absolute episode count."""
        while len(self.episode_returns) < target_episode:
            if self.max_steps is not None and self.steps >= self.max_steps:
                break
            eps = self.epsilon()
            action = self.agent.act(self.obs, eps)
            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            done = terminated or truncated
            real_next = next_obs.copy()
            if truncated and "final_observation" in infos:
                real_next = infos["final_observation"]
            self.replay.add(
                self.obs,
                real_next,
                action,
                float(reward),
                float(done),
            )

            if "episode" in infos:
                ret = float(np.asarray(infos["episode"]["r"]).item())
                self.episode_returns.append(ret)

            self.obs = next_obs
            if done:
                self.obs, _ = self.env.reset()

            if (
                self.steps > LEARNING_STARTS
                and self.steps % max(1, int(self.config.train_freq)) == 0
                and len(self.replay) >= BATCH_SIZE
            ):
                self.agent.td_update(self.replay.sample(BATCH_SIZE))
            if self.steps % max(1, int(self.config.target_freq)) == 0:
                self.agent.sync_target()
            self.steps += 1

        metrics = speed_metrics(self.episode_returns, target_episode)
        metrics.update({"env_steps": self.steps, "epsilon": self.epsilon()})
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "seed": self.seed,
            "env_id": self.env_id,
            "max_steps": self.max_steps,
            "agent_q": self.agent.q_net.state_dict(),
            "agent_target": self.agent.t_net.state_dict(),
            "agent_opt": self.agent.opt.state_dict(),
            "replay": {
                "observations": self.replay.observations,
                "next_observations": self.replay.next_observations,
                "actions": self.replay.actions,
                "rewards": self.replay.rewards,
                "dones": self.replay.dones,
                "pos": self.replay.pos,
                "full": self.replay.full,
            },
            "episode_returns": self.episode_returns,
            "steps": self.steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.set_config(DQNConfig.from_dict(state["config"]))
        self.agent.q_net.load_state_dict(state["agent_q"])
        self.agent.t_net.load_state_dict(state["agent_target"])
        self.agent.opt.load_state_dict(state["agent_opt"])
        replay = state["replay"]
        self.replay.observations[:] = replay["observations"]
        self.replay.next_observations[:] = replay["next_observations"]
        self.replay.actions[:] = replay["actions"]
        self.replay.rewards[:] = replay["rewards"]
        self.replay.dones[:] = replay["dones"]
        self.replay.pos = int(replay["pos"])
        self.replay.full = bool(replay["full"])
        self.episode_returns = [float(v) for v in state["episode_returns"]]
        self.steps = int(state["steps"])
        # PB2 checkpoints are emitted at episode boundaries; start a fresh episode.
        self.obs, _ = self.env.reset(seed=self.seed + self.steps)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False)
        )

    def close(self) -> None:
        self.env.close()
