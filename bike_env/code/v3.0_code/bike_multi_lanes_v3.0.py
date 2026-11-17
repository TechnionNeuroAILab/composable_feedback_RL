#!/usr/bin/env python3
"""Bike board environment v3.0 = v2.1 (route) + optional SPEED and BALANCE factors, each with its own reward term.

Every factor has its own state, observation slots and reward term, and can be switched on or off. Observation and
action sizes never change (a switched-off factor's slots are 0 and its action does nothing), so factors can be
toggled during training. The reward terms are returned in `info` and sum exactly to the reward.

Levels:  L1 = route (identical to v2.1, checked by the smoke test)   L2 = route + speed (with speed-limit zones)
         L4.1 = route + balance with free speed: the speed factor is on (throttle works, speed drives progress and
                balance) but there are NO speed-limit zones, so reward_speeding is always 0; the turn kick at full speed
                is softened to k(5) = 2 (KICK below has 3). Level key "4.1".
         (Earlier levels L3 = route + speed + balance and L4 = L4.1 with k(5) = 3 were removed; the speed/balance/
         speed_limits/kick switches of FactorsWrapper still allow any combination.)

Action: MultiDiscrete([3, 3]) = (steer, throttle). steer 0/1/2 = left/stay/right for the next row (v2.1 semantics);
throttle 0/1/2 = brake/hold/accelerate (speed -1/0/+1, kept in 1..5). Without the speed factor, speed is fixed at 5.

ROUTE (v2.1, unchanged): holes on a board with a known number of fall-free routes; one decision per row.
    reward_progress = the v2.1 fall-free row reward (5 per primitive step, 20 per row) x speed / 5
    reward_hole_fall = -100 per hole fall (v2.1's collision penalty and time loss, unchanged)
SPEED: speed level v in 1..5 changes how much progress a row earns. Speed-limit zones (limit 2 or 3) lie along the
    board and are visible ahead.  reward_speeding = -SPEEDING_COST x max(0, v - limit) per row.
    Speed is abstract: it scales progress and drives balance, it does not change the hole geometry or the number of
    rows per episode.
BALANCE: lean theta in -3..3 (integer, 0 = upright), updated once per row after the throttle and the lane change:
        theta1 = theta + K,  K = -k(v) for a lane change right, +k(v) for left, 0 when the lane is kept
        theta' = theta1 + G(v) * sign(theta1)
        k(v) = 1, 1, 1, 2, 3   (turn kick: fast turns throw you outward; L4.1 uses k(5) = 2)
        G(v) = +1, +1, 0, -1, -1   (slow: a lean grows; 3: it stays; fast: the bike rights itself)
    |theta'| >= 4 is a BALANCE FALL: -100, the same time loss as a hole fall, theta -> 0, speed -> 1.
    reward_balance = -BALANCE_COST x theta'^2 per row, or the balance-fall penalty.
    There is no lean action: the rider balances by speed (stability) and by steering (a counter-steer kicks back).
After any fall (hole or balance) theta resets to 0 and speed to 1 (when those factors are on).

Observation = v2.1's 8 numbers + speed block [v/5, current limit/5, next zone's limit/5, rows to next zone / VIEW]
              + balance block [theta/4]  (13 numbers; blocks of switched-off factors are 0).

    python bike_multi_lanes_v3.0.py --level 4.1 --lanes 5 --solutions 8 --episodes 3     # print episode statistics
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

_HERE = Path(__file__).resolve().parent
_V21 = _HERE.parent / "v2.1_code" / "bike_dqn_multi_lanes_v2.1.py"

KICK = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3}
STABILITY = {1: 1, 2: 1, 3: 0, 4: -1, 5: -1}
THETA_FALL = 4
BALANCE_COST = 0.8          # per row: -0.8 theta^2 (a lean of 3 costs 7.2 against up to 20 progress)
SPEEDING_COST = 8.0         # per row and per speed level over the limit (more than the extra progress of 4)
FALL_PENALTY = -100.0       # = v2.1 collision penalty
VIEW_ROWS = 12              # speed-limit signs are seen this many rows ahead (~ v2.1's 25-unit hole visibility)
START_SPEED = 3
TERMS = ("reward_progress", "reward_hole_fall", "reward_speeding", "reward_balance")
LEVELS = {1: dict(speed=False, balance=False), 2: dict(speed=True, balance=False),
          "4.1": dict(speed=True, balance=True, speed_limits=False, kick={5: 2})}   # kick: overrides of KICK


def _load_v21():
    if "bike21" in sys.modules:
        return sys.modules["bike21"]
    spec = importlib.util.spec_from_file_location("bike21", _V21)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bike21"] = module
    spec.loader.exec_module(module)
    return module


def balance_update(theta: int, v: int, lane_delta: int, kick: dict | None = None) -> tuple[int, bool]:
    """One row of the balance rule. lane_delta = +1 (moved right), -1 (left), 0. Returns (theta', fell)."""
    theta1 = theta - int(np.sign(lane_delta)) * (kick or KICK)[v]
    theta2 = theta1 + STABILITY[v] * int(np.sign(theta1))
    if abs(theta2) >= THETA_FALL:
        return 0, True
    return int(np.clip(theta2, -(THETA_FALL - 1), THETA_FALL - 1)), False


class FactorsWrapper(gym.Wrapper):
    """Adds the speed and balance factors on top of v2.1's RowDecisionWrapper."""

    def __init__(self, env, speed: bool = True, balance: bool = True, speed_limits: bool = True, kick: dict | None = None,
                 zone_fraction: float = 0.3,
                 zone_rows: tuple[int, int] = (8, 20), limits: tuple[int, ...] = (2, 3), seed: int = 0):
        super().__init__(env)
        self.use_speed, self.use_balance = bool(speed), bool(balance)
        self.use_limits = self.use_speed and bool(speed_limits)     # off: no zones, limit 5 everywhere, no speeding cost
        self.kick = {**KICK, **(kick or {})}                         # turn kick per speed (level overrides on top of KICK)
        self.zone_fraction, self.zone_rows, self.limits = zone_fraction, zone_rows, limits
        self._base = env.unwrapped
        self._rng = np.random.default_rng(seed + 7_919)   # own stream: never touches the v2.1 env's random state
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        low = np.concatenate([env.observation_space.low, np.array([0, 0, 0, 0, -1], np.float32)])
        high = np.concatenate([env.observation_space.high, np.array([1, 1, 1, 1, 1], np.float32)])
        self.observation_space = gym.spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        self.speed, self.theta, self.zones = 5, 0, []

    # -- speed-limit zones (in row units along the board) ------------------------------------------------------
    def _make_zones(self):
        cfg = self._base.board.config
        n = cfg.n_rows + int(np.ceil(cfg.start_offset / cfg.row_spacing)) + 1
        zones, r = [], int(self._rng.integers(2, 10))
        mean_len = np.mean(self.zone_rows)
        gap_mean = mean_len * (1 - self.zone_fraction) / max(self.zone_fraction, 1e-6)
        while r < n:
            length = int(self._rng.integers(self.zone_rows[0], self.zone_rows[1] + 1))
            zones.append((r, r + length, int(self._rng.choice(self.limits))))
            r += length + max(1, int(self._rng.exponential(gap_mean)))
        return zones

    def _row(self) -> int:
        cfg = self._base.board.config
        return int(np.floor((self._base.distance_traveled - cfg.start_offset) / cfg.row_spacing)) + 1

    def _limit_info(self):
        row = self._row()
        current, nxt, dist = 5, 5, VIEW_ROWS
        for a, b, lim in self.zones:
            if a <= row < b:
                current = lim
            elif a > row:
                if a - row <= VIEW_ROWS:
                    nxt, dist = lim, a - row
                break
        return current, nxt, dist

    def _obs(self, base_obs):
        sp = np.zeros(4, np.float32)
        if self.use_speed:
            cur, nxt, dist = self._limit_info()
            sp[:] = [self.speed / 5, cur / 5, nxt / 5, dist / VIEW_ROWS]
        bal = np.array([self.theta / THETA_FALL if self.use_balance else 0.0], np.float32)
        return np.concatenate([np.asarray(base_obs, np.float32), sp, bal])

    # -- gym API ----------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + 7_919)
        obs, info = self.env.reset(seed=seed, options=options)
        self.speed = START_SPEED if self.use_speed else 5
        self.theta = 0
        self.zones = self._make_zones() if self.use_limits else []
        self.ep = {"hole_falls": 0, "balance_falls": 0, "speeding_rows": 0, **{t: 0.0 for t in TERMS}}
        return self._obs(obs), dict(info)

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        steer, throttle = int(a[0]), int(a[1]) if a.size > 1 else 1
        if self.use_speed:
            self.speed = int(np.clip(self.speed + throttle - 1, 1, 5))
        v = self.speed
        limit = self._limit_info()[0] if self.use_speed else 5
        lane0 = self._base._current_lane()
        obs, total, terminated, truncated, info = self.env.step(steer)
        lane_delta = self._base._current_lane() - lane0
        hole_falls = int(info.get("row_falls", 0))

        r_hole = hole_falls * FALL_PENALTY
        r_progress = (total - r_hole) * v / 5.0
        r_speeding = -SPEEDING_COST * max(0, v - limit) if self.use_speed else 0.0
        r_balance, balance_fell = 0.0, False
        if hole_falls:
            self.theta = 0
            if self.use_speed:
                self.speed = 1
        elif self.use_balance:
            self.theta, balance_fell = balance_update(self.theta, v, lane_delta, self.kick)
            if balance_fell:
                r_balance = FALL_PENALTY
                self._base.time_left -= self._base.config.fall_time_penalty    # same time loss as a hole fall
                self._base.grace_period_steps = self._base.config.grace_steps
                if self.use_speed:
                    self.speed = 1
                terminated = terminated or self._base.time_left <= 0
            else:
                r_balance = -BALANCE_COST * self.theta ** 2

        reward = r_progress + r_hole + r_speeding + r_balance
        terms = dict(zip(TERMS, (r_progress, r_hole, r_speeding, r_balance)))
        self.ep["hole_falls"] += hole_falls
        self.ep["balance_falls"] += int(balance_fell)
        self.ep["speeding_rows"] += int(r_speeding < 0)
        for k_, x in terms.items():
            self.ep[k_] += x
        info = {**info, **terms, "speed": v, "limit": limit, "theta": self.theta, "lane_delta": lane_delta,
                "balance_fell": balance_fell, **{f"ep_{k_}": x for k_, x in self.ep.items()}}
        return self._obs(obs), float(reward), terminated, truncated, info


def parse_level(level):
    """1 / "1" / "L1" -> 1 ... "4.1" / "L4.1" -> "4.1" (the LEVELS key)."""
    s = str(level).lstrip("L")
    key = int(s) if s.isdigit() else s
    if key not in LEVELS:
        raise ValueError(f"unknown bike v3.0 level {level!r}; known: {list(LEVELS)}")
    return key


def make_env(board_config, level: int | str | None = None, speed: bool = True, balance: bool = True,
             speed_limits: bool = True, new_board_each_episode: bool = True, **kwargs):
    """v3.0 env. `level` (1/2/"4.1") overrides the speed/balance/speed_limits/kick switches. Returns (env, board)."""
    if level is not None:
        cfg = LEVELS[parse_level(level)]
        speed, balance, speed_limits = cfg["speed"], cfg["balance"], cfg.get("speed_limits", True)
        kwargs.setdefault("kick", cfg.get("kick"))
    env, board = _load_v21().make_env(board_config, new_board_each_episode)
    return FactorsWrapper(env, speed=speed, balance=balance, speed_limits=speed_limits, seed=int(board_config.seed),
                          **kwargs), board


def board_config(lanes=5, rows=240, solutions=8, seed=0):
    v20 = _load_v21()._load_v20()
    return v20.BoardConfig(lanes, rows, solutions, 1, 2.0, 10.0, int(seed))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=str, choices=[str(k) for k in LEVELS], default="4.1")
    p.add_argument("--lanes", type=int, default=5)
    p.add_argument("--rows", type=int, default=240)
    p.add_argument("--solutions", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    a = p.parse_args()
    env, _ = make_env(board_config(a.lanes, a.rows, a.solutions, a.seed), level=a.level)
    for i in range(a.episodes):
        env.reset(seed=a.seed + i)
        done, ret, info = False, 0.0, {}
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            ret += r
            done = term or trunc
        print(f"episode {i}: return {ret:.0f}  " + "  ".join(f"{k}={v:.0f}" for k, v in info.items() if k.startswith("ep_")))


if __name__ == "__main__":
    main()
