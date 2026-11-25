#!/usr/bin/env python
"""Exact best fall-free return of bike v3.0 on the evaluation boards (dynamic programming).

For each evaluation board (the boards of common_bike.greedy_return: reset seeds 5000..5009) and level:
  1. Forward pass over the row decisions with copies of the environment, one copy per lane: for every lane and steer,
     record whether the row is hole-free, the lane reached and the row's base progress (at full speed), plus the
     speed limit of the row. Without falls the rows, their progress and the limits do not depend on the policy.
  2. Backward DP over (decision, lane, speed, lean) with the v3.0 rules for progress x v/5, speeding cost and the
     balance rule (balance falls excluded).
Falls of either kind are excluded: the result is the best fall-free return (a fall costs -100 and time).

Usage: python optimum_bike3.py [levels...]   (default 1 2 4.1; rows of other levels already in the csv are kept)
Also reports the upper bound (every row at full progress, no costs) and writes results/bike3/optimum.csv.
"""
import copy
import os

import numpy as np
import pandas as pd

from common_bike import _EVAL_BOARDS, load_bike3, make_bike3_env

CHECK = 10         # replay the optimal plan in the real env on the first CHECK boards per level
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results/bike3/optimum.csv")
b3 = load_bike3()


def factors_wrapper(env):
    while not isinstance(env, b3.FactorsWrapper):
        env = env.env
    return env


def forward(fw):
    """Per decision i: limit[i], and trans[i][lane] = list of (steer, next_lane, base_reward) for hole-free steers."""
    states, limits, trans = {fw._base._current_lane(): copy.deepcopy(fw)}, [], []
    while states:
        any_c = next(iter(states.values()))
        limits.append(any_c._limit_info()[0] if any_c.use_speed else 5)
        nxt, row = {}, {}
        for lane, c in states.items():
            row[lane] = []
            for steer in range(3):
                if not 0 <= lane + steer - 1 < c._base.config.n_lanes:
                    continue                     # steering into the wall: the env picks a random valid action
                c2 = copy.deepcopy(c)
                _, r, term, trunc, info = c2.env.step(steer)
                if info.get("row_falls", 0):
                    continue
                l2 = c2._base._current_lane()
                row[lane].append((steer, l2, r, term or trunc))
                if not (term or trunc):
                    nxt.setdefault(l2, c2)
        trans.append(row)
        states = nxt
    return limits, trans


def solve(limits, trans, lane0, speed, balance, kick=None):
    vs = range(1, 6) if speed else [5]
    ths = range(-3, 4) if balance else [0]
    V = {}                                   # value-to-go at decision i (before acting) for (lane, v, theta)
    for i in reversed(range(len(trans))):
        Vi = {}
        for lane, opts in trans[i].items():
            for v in vs:
                for th in ths:
                    best = -np.inf
                    for steer, l2, r, done in opts:
                        for u in ((0, 1, 2) if speed else (1,)):
                            v2 = int(np.clip(v + u - 1, 1, 5)) if speed else 5
                            rew = r * v2 / 5 - (b3.SPEEDING_COST * max(0, v2 - limits[i]) if speed else 0.0)
                            th2 = 0
                            if balance:
                                th2, fell = b3.balance_update(th, v2, l2 - lane, kick)
                                if fell:
                                    continue
                                rew -= b3.BALANCE_COST * th2 ** 2
                            if not done:
                                rew += V[i + 1].get((l2, v2, th2), -np.inf)
                            best = max(best, rew)
                    Vi[(lane, v, th)] = best
        V[i] = Vi
    return V[0][(lane0, b3.START_SPEED if speed else 5, 0)], V


def replay(env, seed, limits, trans, V, speed, balance, kick=None):
    """Play the DP-optimal plan in the real environment; returns the real return (check of the DP model)."""
    env.reset(seed=seed)
    fw = factors_wrapper(env)
    ret, i, done = 0.0, 0, False
    while not done:
        lane, v, th = fw._base._current_lane(), fw.speed, fw.theta
        best, act = -np.inf, None
        for steer, l2, r, d in trans[i][lane]:
            for u in ((0, 1, 2) if speed else (1,)):
                v2 = int(np.clip(v + u - 1, 1, 5)) if speed else 5
                rew = r * v2 / 5 - (b3.SPEEDING_COST * max(0, v2 - limits[i]) if speed else 0.0)
                th2 = 0
                if balance:
                    th2, fell = b3.balance_update(th, v2, l2 - lane, kick)
                    if fell:
                        continue
                    rew -= b3.BALANCE_COST * th2 ** 2
                if not d:
                    rew += V[i + 1].get((l2, v2, th2), -np.inf)
                if rew > best:
                    best, act = rew, (steer, u)
        _, r, term, trunc, _ = env.step(np.array(act))
        ret, i, done = ret + r, i + 1, term or trunc
    return ret


def main(levels=(1, 2, "4.1"), n_boards=10, seed0=5000):
    rows = []
    _EVAL_BOARDS["active"] = True
    try:
        for level in levels:
            env = make_bike3_env(seed0, level=level, reward_scale=1.0)
            for k in range(n_boards):
                env.reset(seed=seed0 + k)
                fw = factors_wrapper(env)
                limits, trans = forward(fw)
                lane0 = fw._base._current_lane()
                upper = sum(max(r for _, _, r, _ in o) for t in trans for o in [sum(t.values(), [])] if o)
                opt, V = solve(limits, trans, lane0, fw.use_speed, fw.use_balance, fw.kick)
                real = replay(env, seed0 + k, limits, trans, V, fw.use_speed, fw.use_balance, fw.kick) if k < CHECK else np.nan
                rows.append(dict(level=float(level), board=seed0 + k, decisions=len(trans), optimum=opt, replayed=real, upper_bound=upper,
                                 limited_rows=int(sum(l < 5 for l in limits))))
                print(rows[-1], flush=True)
    finally:
        _EVAL_BOARDS["active"] = False
    df = pd.DataFrame(rows)
    if os.path.exists(OUT):                  # keep the rows of levels not recomputed in this call
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.level.isin([float(l) for l in levels])], df]).sort_values(["level", "board"])
    df.to_csv(OUT, index=False)
    print(df.groupby("level")[["optimum", "upper_bound", "limited_rows"]].agg(["mean", "min", "max"]).round(0))


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or (1, 2, "4.1"))     # e.g. python optimum_bike3.py 4.1
