#!/usr/bin/env python3
"""Smoke test of bike v3.0 (bike_multi_lanes_v3.0.py).

  1. balance rule: the worked examples of the design (slow wobble, medium, fast heal, two fast changes -> fall)
  2. level 1 == v2.1: same boards, same steering -> identical rewards, observations (first 8 numbers) and episode ends
  3. every level (L4.1: no speed-limit zones, no speeding cost): reward terms sum to the reward; observation shape / bounds; switched-off blocks are 0
  4. episode statistics per level for a random policy and a simple hand-written heuristic
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bike30", HERE / "bike_multi_lanes_v3.0.py")
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


# ---------------------------------------------------------------- 1. balance rule
def trace(v, deltas, theta=0):
    out = []
    for d in deltas:
        theta, fell = b3.balance_update(theta, v, d)
        out.append("fall" if fell else theta)
        if fell:
            break
    return out

check(trace(1, [1, 0, 0]) == [-2, -3, "fall"], f"slow v=1, one change right (kick and growth in the same row): {trace(1, [1, 0, 0])}")
check(trace(3, [1, 0, 0]) == [-1, -1, -1], f"medium v=3, one change right: {trace(3, [1, 0, 0])}")
check(trace(5, [1, 0, 0]) == [-2, -1, 0], f"fast v=5, one change right heals: {trace(5, [1, 0, 0])}")
check(trace(5, [1, 1]) == [-2, "fall"], f"fast v=5, two changes right -> fall: {trace(5, [1, 1])}")
check(trace(3, [1, -1]) == [-1, 0], f"counter-steer at v=3 recovers: {trace(3, [1, -1])}")
check(trace(4, [0], theta=2) == [1], "v=4 self-rights a lean of 2 to 1")
_k41 = b3.make_env(b3.board_config(5, 240, 8, 0), level="4.1")[0].kick
check(_k41 == {**b3.KICK, 5: 2} and b3.balance_update(0, 5, 1, _k41) == (-1, False), "L4.1: kick at v=5 is 2 (one change right -> -1)")

# ---------------------------------------------------------------- 2. level 1 == v2.1
v21 = b3._load_v21()
for seed in (0, 1):
    cfg = b3.board_config(5, 240, 8, seed)
    e21, _ = v21.make_env(cfg, True)
    e30, _ = b3.make_env(cfg, level=1)
    rng = np.random.default_rng(seed)
    o21, _ = e21.reset(seed=100 + seed)
    o30, _ = e30.reset(seed=100 + seed)
    same, n, done = np.allclose(o21, o30[:8]), 0, False
    ret21 = ret30 = 0.0
    while not done:
        steer = int(rng.integers(3))
        o21, r21, t21, tr21, _ = e21.step(steer)
        o30, r30, t30, tr30, i30 = e30.step([steer, int(rng.integers(3))])   # throttle must be ignored at level 1
        same &= np.allclose(o21, o30[:8]) and abs(r21 - r30) < 1e-9 and (t21, tr21) == (t30, tr30) and not o30[8:].any()
        ret21, ret30, n, done = ret21 + r21, ret30 + r30, n + 1, t21 or tr21
    check(same, f"level 1 == v2.1 on board seed {seed}: {n} decisions, return {ret21:.0f} vs {ret30:.0f}")

# ---------------------------------------------------------------- 3. reward terms, observation
for level in (1, 2, "4.1"):
    env, _ = b3.make_env(b3.board_config(5, 240, 8, 3), level=level)
    ok, n = True, 0
    for ep in range(2):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, t, tr, info = env.step(env.action_space.sample())
            ok &= abs(sum(info[k] for k in b3.TERMS) - r) < 1e-6
            ok &= obs.shape == (13,) and env.observation_space.contains(obs)
            ok &= (level != 1 or not obs[8:12].any()) and (level != 2 or obs[12] == 0)   # off-blocks are 0
            if level == "4.1":                                 # no speed-limit zones: limit 5 everywhere, never speeding
                ok &= info["reward_speeding"] == 0 and info["limit"] == 5 and np.allclose(obs[9:12], 1)
            n, done = n + 1, t or tr
    check(ok, f"level {level}: reward terms sum to the reward, obs shape/bounds/off-blocks ({n} decisions)")


# ---------------------------------------------------------------- 4. episode statistics
def heuristic(obs, level):
    """Steer to the adjacent lane with the farthest hole (prefer staying); speed 4 (self-righting), under the limit."""
    d = obs[1:8]                                   # hole distance in lane slots -3..+3, -1 = wall; ego = index 3
    opts = {0: d[2], 1: d[3] + 0.5, 2: d[4]}       # left / stay (+0.5 tie-break) / right
    steer = max((k for k in opts if d[2 + k] >= 0 or k == 1), key=lambda k: opts[k])
    if level == 1:
        return [steer, 1]
    v, cur, nxt, dist = (obs[8:12] * np.array([5, 5, 5, b3.VIEW_ROWS])).round()
    target = min(4, cur, nxt if dist <= 2 else 5)
    return [steer, 2 if v < target else 0 if v > target else 1]


print("\nepisode statistics (5 lanes, 8 solutions, 5 boards each; max return at level 1 = 4900)")
print(f"{'level':>5} {'policy':>9} {'return':>8} {'progress':>9} {'hole falls':>10} {'bal. falls':>10} {'speeding rows':>13}")
for level in (1, 2, "4.1"):
    env, _ = b3.make_env(b3.board_config(5, 240, 8, 11), level=level)
    for name in ("random", "heuristic"):
        stats = []
        for ep in range(5):
            obs, _ = env.reset(seed=500 + ep)
            done, ret = False, 0.0
            while not done:
                act = env.action_space.sample() if name == "random" else heuristic(obs, level)
                obs, r, t, tr, info = env.step(act)
                ret, done = ret + r, t or tr
            stats.append([ret, info["ep_reward_progress"], info["ep_hole_falls"], info["ep_balance_falls"], info["ep_speeding_rows"]])
        s = np.mean(stats, 0)
        print(f"{str(level):>5} {name:>9} {s[0]:8.0f} {s[1]:9.0f} {s[2]:10.1f} {s[3]:10.1f} {s[4]:13.1f}")
print("\nSMOKE TEST PASSED")
