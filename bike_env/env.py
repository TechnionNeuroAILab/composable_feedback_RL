"""Bike riding Gym environment (from BikeRider DQN notebook).

Uses the classic ``gym`` API (``reset`` -> obs only; ``step`` -> obs, reward, done, info)
for compatibility with stable-baselines3 1.x and its ``DummyVecEnv``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
from gym import spaces


class BikeEnvAdvanced(gym.Env):
    """
    Multi-lane bike environment with holes and stochastic fall mechanics.

    Observation: ``[x_pos, velocity, hole_lane_0, ..., hole_lane_{n-1}]``
    Actions: 0=left, 1=stay, 2=right, 3=accelerate, 4=brake
    """

    metadata = {"render.modes": []}

    def __init__(self, n_lanes: int = 3):
        super().__init__()
        self.n_lanes = n_lanes
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(2 + n_lanes,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.max_speed = 12.0
        self.visibility_range = 25.0
        self.seed()
        self.reset()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self):
        self.x_pos = float(self.n_lanes // 2)
        self.velocity = 0.0
        self.angle = 0.0
        self.time_left = 1000
        self.holes_passed = 0
        self.grace_period_steps = 0
        self.holes = [random.uniform(40, 60) + (i * 25) for i in range(self.n_lanes)]
        return self._get_obs()

    def step(self, action: int):
        self.time_left -= 1
        reward, reason, fell = 0, "None", False
        passed_this_step = False

        if action == 0:
            self.angle = -0.45
        elif action == 2:
            self.angle = 0.45
        else:
            self.angle = 0.0

        if action == 3:
            self.velocity = min(self.max_speed, self.velocity + 1.0)
        elif action == 4:
            self.velocity = max(0.0, self.velocity * 0.9)

        self.x_pos += math.sin(self.angle) * self.velocity * 0.1
        self.x_pos = float(np.clip(self.x_pos, 0, self.n_lanes - 1))

        p_slow = (
            0
            if self.grace_period_steps > 0
            else (
                self.sigmoid_prob(1.5 - self.velocity, threshold=0.5)
                if self.velocity < 1.5
                else 0
            )
        )
        p_turn = self.sigmoid_prob(self.velocity, threshold=8.5) if self.angle != 0 else 0
        p_fast = self.sigmoid_prob(self.velocity, threshold=9.0, steepness=5.0)

        current_lane = int(round(self.x_pos))

        if random.random() < p_turn:
            fell, reason = True, "Turn Slip"
        elif random.random() < p_fast:
            fell, reason = True, "Speed Wobble"
        elif abs(self.holes[current_lane]) < 0.8:
            fell, reason = True, "Hole Collision"
        elif random.random() < p_slow:
            fell, reason = True, "Slow Unstable"

        if self.grace_period_steps > 0:
            self.grace_period_steps -= 1

        if fell:
            self.velocity, self.angle = 0.0, 0.0
            self.time_left -= 50
            self.grace_period_steps = 5
            self.holes = [random.uniform(40, 60) + (i * 25) for i in range(self.n_lanes)]
        else:
            reward = self.velocity

            for i in range(self.n_lanes):
                old_h = self.holes[i]
                new_h = old_h - (self.velocity * 0.1)
                if old_h >= 0 and new_h < 0:
                    passed_this_step = True
                    self.holes_passed += 1

        for i in range(self.n_lanes):
            self.holes[i] -= self.velocity * 0.1
            if self.holes[i] < -5:
                self.holes[i] = random.uniform(40, 60)

        done = self.time_left <= 0
        info = {"fell": fell, "reason": reason, "passed": passed_this_step}
        return self._get_obs(), reward, done, info

    def sigmoid_prob(self, val: float, threshold: float, steepness: float = 3.0) -> float:
        return 1 / (1 + math.exp(-steepness * (val - threshold)))

    def _get_obs(self) -> np.ndarray:
        h_obs = [h if h <= self.visibility_range else 100.0 for h in self.holes]
        return np.array([self.x_pos, self.velocity, *h_obs], dtype=np.float32)


class FallCounterWrapper(gym.Wrapper):
    """
    Counts ``info["fell"]`` per episode and sets ``info["episode_falls"]`` on the terminal step.

    Use with ``Monitor(..., info_keywords=("episode_falls",))`` so falls are written to ``monitor.csv``.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._falls_this_episode = 0

    def reset(self, **kwargs):
        self._falls_this_episode = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("fell"):
            self._falls_this_episode += 1
        if done:
            info = dict(info)
            info["episode_falls"] = int(self._falls_this_episode)
        return obs, reward, done, info
