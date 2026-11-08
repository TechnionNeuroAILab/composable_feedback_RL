"""bike_env v2.1 as a benchmark for every algorithm in this folder.

Board: 5 lanes, 240 rows, 8 solutions (the existing SB3 DQN solves it: greedy return 4900 =
5.0 speed x 980 steps). One agent decision = one row (left/stay/right), observation = 8 numbers,
~240 decisions per perfect episode, so "environment steps" below means agent decisions
(comparable to Atari's frame-skipped agent steps).

Reward scaling: one decision earns at most 5.0 x 4 primitive steps = 20, and a fall costs far
more. All learners train on reward / 20 (a standard preprocessing, applied identically to every
algorithm); reported episode returns are always the RAW, unscaled return (RecordEpisodeStatistics
sits below the scaling wrapper).
"""
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

BIKE_FILE = Path(__file__).resolve().parents[3] / "bike_env/code/v2.1_code/bike_dqn_multi_lanes_v2.1.py"
REWARD_SCALE = 20.0
LANES, ROWS, SOLUTIONS = 5, 240, 8
MAX_RETURN = 4900.0


def load_bike():
    if "bike21" in sys.modules:
        return sys.modules["bike21"]
    spec = importlib.util.spec_from_file_location("bike21", BIKE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bike21"] = module
    spec.loader.exec_module(module)
    _install_board_cache(module._load_v20())
    return module


_EVAL_BOARDS = {"active": False, "cache": {}}


def _install_board_cache(v20):
    """Carving a board is deterministic in its config but costs 0.5-2.3 s at 100 solutions, and every episode draws a new
    board. While greedy_return() runs, boards are memoized by config, so the fixed evaluation boards are carved once per
    process; training episodes (flag off) are untouched, so training behaviour is exactly as before."""
    orig = v20.Board

    def Board(config):
        if not _EVAL_BOARDS["active"]:
            return orig(config)
        cache = _EVAL_BOARDS["cache"]
        if config not in cache:
            cache[config] = orig(config)
        return cache[config]

    v20.Board = Board


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class KeepEpisodeInfo(gym.Wrapper):
    """The bike env's info dict holds None values, which gymnasium's vector envs cannot store; keep
    only what the algorithms read (the 'episode' statistics from RecordEpisodeStatistics)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {k: info[k] for k in ("episode",) if k in info}

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, {}


def make_bike_env(seed, reward_scale=REWARD_SCALE, lanes=LANES, rows=ROWS, solutions=SOLUTIONS):
    b = load_bike()
    cfg = b._load_v20().BoardConfig(lanes, rows, solutions, 1, 2.0, 10.0, int(seed))
    env, _ = b.make_env(cfg, True)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if reward_scale != 1.0:
        env = ScaleReward(env, 1.0 / reward_scale)
    return KeepEpisodeInfo(env)


def greedy_return(policy, lanes=LANES, solutions=SOLUTIONS, n_episodes=10, seed0=5000):
    """Shared evaluation protocol of the bike grid: mean RAW return of an exploration-free policy over n_episodes fresh
    boards. Episode i resets with seed0 + i, so every method sees the same boards. `policy(obs) -> action` must not
    explore. Uses its own environment (never the training env), so nothing is counted as a training step or stored."""
    _EVAL_BOARDS["active"] = True
    try:
        env = make_bike_env(seed0, reward_scale=1.0, lanes=lanes, solutions=solutions)
        returns = []
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed0 + i)
            done, ret = False, 0.0
            while not done:
                obs, r, term, trunc, _ = env.step(int(policy(obs)))
                ret += float(r)
                done = term or trunc
            returns.append(ret)
        env.close()
    finally:
        _EVAL_BOARDS["active"] = False
    return float(np.mean(returns))
