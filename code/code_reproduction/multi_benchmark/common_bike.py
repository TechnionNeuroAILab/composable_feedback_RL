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


def greedy_return(policy, lanes=LANES, solutions=SOLUTIONS, n_episodes=10, seed0=5000, level=None, joint=False, lookahead=0):
    """Shared evaluation protocol of the bike grid: mean RAW return of an exploration-free policy over n_episodes fresh
    boards. Episode i resets with seed0 + i, so every method sees the same boards. `policy(obs) -> action` must not
    explore. Uses its own environment (never the training env), so nothing is counted as a training step or stored.
    level = 1/2/"4.1": bike v3.0 at that level instead of v2.1; policy then returns (steer, throttle), or one joint
    action 0..8 with joint=True. lookahead / joint: the same observation / action wrappers as make_bike3_env."""
    _EVAL_BOARDS["active"] = True
    try:
        env = (make_bike_env(seed0, reward_scale=1.0, lanes=lanes, solutions=solutions) if level is None else
               make_bike3_env(seed0, level=level, reward_scale=1.0, lanes=lanes, solutions=solutions, joint=joint,
                              lookahead=lookahead))
        returns = []
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed0 + i)
            done, ret = False, 0.0
            while not done:
                act = np.asarray(policy(obs))
                obs, r, term, trunc, _ = env.step(int(act) if act.size == 1 and level is None else act.astype(np.int64))
                ret += float(r)
                done = term or trunc
            returns.append(ret)
        env.close()
    finally:
        _EVAL_BOARDS["active"] = False
    return float(np.mean(returns))


# ---------------------------------------------------------------------------------------------------- bike v3.0
BIKE3_FILE = Path(__file__).resolve().parents[3] / "bike_env/code/v3.0_code/bike_multi_lanes_v3.0.py"
BIKE3_TERMS = ("reward_progress", "reward_hole_fall", "reward_speeding", "reward_balance")


def load_bike3():
    load_bike()                                  # v3.0 reuses the v2.1 module (and its board cache) as "bike21"
    if "bike30" in sys.modules:
        return sys.modules["bike30"]
    spec = importlib.util.spec_from_file_location("bike30", BIKE3_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bike30"] = module
    spec.loader.exec_module(module)
    return module


class KeepTermsInfo(KeepEpisodeInfo):
    """KeepEpisodeInfo plus the v3.0 reward terms and per-episode factor counters (needed by reward-term loops)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        keep = ("episode",) + BIKE3_TERMS
        return obs, reward, terminated, truncated, {k: v for k, v in info.items() if k in keep or k.startswith("ep_")}


def make_bike3_env(seed, level="4.1", reward_scale=REWARD_SCALE, lanes=LANES, rows=ROWS, solutions=SOLUTIONS, joint=False,
                   lookahead=0):
    """bike v3.0 at level 1 (route = v2.1), 2 (+ speed with speed limits) or "4.1" (route + balance, free speed, no speed
    limits, turn kick 2 at speed 5); level may be int or str ("L4.1" works); action = (steer, throttle).
    joint=True: one Discrete(9) action a = 3 * steer + throttle (for a vanilla single-head DQN).
    lookahead=K > 0: append the hole grid of the next K rows around the ego lane (LookAhead) to the observation."""
    b = load_bike3()
    env, _ = b.make_env(b.board_config(lanes, rows, solutions, int(seed)), level=level)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if reward_scale != 1.0:
        env = ScaleReward(env, 1.0 / reward_scale)
    env = KeepTermsInfo(env)
    if lookahead:
        env = LookAhead(env, lookahead)
    if joint:
        env = JointAction(env)
    return env


class JointAction(gym.ActionWrapper):
    """MultiDiscrete([3, 3]) (steer, throttle) -> Discrete(9), a = 3 * steer + throttle: one Q-value per joint action."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(9)

    def action(self, a):
        return np.array(divmod(int(np.asarray(a).reshape(-1)[0]), 3), dtype=np.int64)


class LookAhead(gym.ObservationWrapper):
    """Appends a K x 7 grid to the observation: 1 = hole, 0 = free, for board rows row..row+K-1 (row = the row the next
    action decides) at ego-relative lanes -3..+3; lanes outside the board (walls) are 1, rows past the board end 0."""

    def __init__(self, env, k):
        super().__init__(env)
        self.k = int(k)
        b3 = load_bike3()
        fw = env
        while not isinstance(fw, b3.FactorsWrapper):
            fw = fw.env
        self.fw = fw
        low = np.concatenate([env.observation_space.low, np.zeros(7 * self.k, np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(7 * self.k, np.float32)])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        base = self.fw._base
        holes, row, lane = base.board.holes, self.fw._row(), base._current_lane()
        g = np.ones((self.k, 7), np.float32)
        for i in range(self.k):
            r = row + i
            for j, d in enumerate(range(-3, 4)):
                if 0 <= lane + d < holes.shape[1]:
                    g[i, j] = float(holes[r, lane + d]) if 0 <= r < holes.shape[0] else 0.0
        return np.concatenate([np.asarray(obs, np.float32), g.ravel()])
