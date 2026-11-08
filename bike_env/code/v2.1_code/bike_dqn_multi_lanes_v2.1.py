#!/usr/bin/env python3
"""Bike board environment v2.1 = v2.0 board + (3) one decision per row + (7) clean maximum reward.

v2.0 asked the agent for a lateral action at every 0.5-unit step. On a board with rows 2 units
apart, a lane change must then be made in exactly ONE step (the single step after the nearest
row's danger zone and before the next row's), which DQN struggles to learn. v2.1 changes:

  (3) Decide once per row. One agent action = "left / stay / right" for the NEXT row. The
      environment executes it with the unchanged v1.3.1.4 step dynamics, performing the lane
      change in the safe step between two rows (if a safe step exists). Falls can still happen,
      but only when the agent picks a lane whose cell in the next row is a hole (or the lane it
      leaves is a hole in the row that is still ahead). Actions stay adjacent-lane only
      (max_shift = 1), so the board's solution count is still exact.
  (7) Clean maximum reward. The v1.3.1.4 proximity penalty (-1..-3 per step when an ego-lane hole is
      within 5 steps) is removed, so a fall-free step earns exactly the speed (5.0). A perfect episode
      therefore scores speed x steps with no deductions.

Everything else (board generation, v1.3.1.4 collision rule, observation = speed + nearest hole distance in
the 4 ego-relative lane slots, SB3 DQN hyperparameters of v1.3.1.4) is unchanged. Training also reports a
GREEDY evaluation (no exploration, fresh boards) every 100 episodes, because training episodes contain
epsilon-greedy random actions that cause falls even for a perfect policy.

    python bike_dqn_multi_lanes_v2.1.py --train --lanes 3 --rows 240 --solutions 8 --seed 1 --episodes 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_V20 = _HERE.parent / "bike_dqn_multi_lanes_v2.0.py"
_DEFAULT_OUT = _HERE / "training_results_v2_1"

_ACTION_DELTA = {0: -1, 1: 0, 2: 1}   # left, stay, right


def _load_v20():
    if "bike_v2_0" in sys.modules:
        return sys.modules["bike_v2_0"]
    spec = importlib.util.spec_from_file_location("bike_v2_0", _V20)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bike_v2_0"] = module
    spec.loader.exec_module(module)
    return module


class RowDecisionWrapper(gym.Wrapper):
    """One action per row on top of the (v2.0 board, v1.3.1.4 dynamics) environment."""

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        base._proximity_penalty = lambda lane: 0.0           # (7) clean maximum reward
        self._base = base

    # -- geometry helpers (same collision rule as v1.3.1.4's _fall_reason) ------------------
    def _lane_danger(self, lane: int, distance: float) -> bool:
        """True if a step starting at ``distance`` in ``lane`` hits the nearest hole ahead."""
        layout = self._base.hole_layouts[lane]
        i = int(np.searchsorted(layout, distance - 1e-9, side="left"))
        adv = self._base.config.fixed_speed * 0.1
        return i < len(layout) and layout[i] <= distance + adv + self._base.config.collision_distance + 1e-9

    def _plan(self, lane: int, target: int):
        """Return (n_steps, switch_step) for the next row; switch_step is None if no lane change."""
        base, cfg = self._base, self._base.board.config
        adv = base.config.fixed_speed * 0.1
        d0 = base.distance_traveled
        # k may be negative: the hole-free run-in is split into "virtual rows" (one decision each) so the
        # bike can reach a free first-row lane from the middle lane even on wide (10-lane) boards.
        k = int(np.floor((d0 - 0.5 - cfg.start_offset) / cfg.row_spacing + 1e-9)) + 1
        d_end = cfg.start_offset + cfg.row_spacing * k + 0.5          # just after row k
        n_steps = max(1, int(np.ceil((d_end - d0) / adv - 1e-9)))
        if target == lane:
            return n_steps, None
        for s in range(n_steps):                                      # first fully safe switch step
            ok = all(not self._lane_danger(lane, d0 + t * adv) for t in range(s)) and \
                 all(not self._lane_danger(target, d0 + t * adv) for t in range(s, n_steps))
            if ok:
                return n_steps, s
        return n_steps, 0                                             # no safe step: the agent will fall

    def step(self, action):
        base = self._base
        resolved = base._resolve_action(int(np.asarray(action).item()))
        lane = base._current_lane()
        target = lane + _ACTION_DELTA[resolved]
        n_steps, switch = self._plan(lane, target)
        total, info, obs, terminated, truncated = 0.0, {}, None, False, False
        falls_before = base.falls
        for t in range(n_steps):
            primitive = resolved if (switch is not None and t == switch) else 1
            obs, reward, terminated, truncated, info = self.env.step(primitive)
            total += reward
            if terminated or truncated:
                break
        info = dict(info)
        info["row_falls"] = base.falls - falls_before
        info["requested_action"], info["executed_action"] = int(np.asarray(action).item()), resolved
        return obs, total, terminated, truncated, info


def make_env(board_config, new_board_each_episode: bool = True, **env_kwargs):
    v20 = _load_v20()
    env_kwargs.setdefault("visibility_range", 25.0)
    env_kwargs.setdefault("episode_steps", 1000)
    env, board = v20.make_env(board_config, new_board_each_episode, **env_kwargs)
    return RowDecisionWrapper(env), board


def _greedy_eval(model, board_config, episodes: int, seed0: int, env_kwargs) -> dict:
    env, _ = make_env(replace(board_config, seed=seed0), True, **env_kwargs)
    # ValidActionDQN masks wall actions using the env it is attached to (default: the TRAINING env). The
    # evaluation env has its own lane, so attach it here, otherwise the mask is taken from a stale lane.
    previous = getattr(model, "_action_env", None)
    model.attach_action_env(env.unwrapped)
    falls, rewards, finished = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done, total, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            done = term or trunc
        falls.append(info["falls"]); rewards.append(total); finished.append(bool(info.get("finished")))
    model._action_env = previous
    return {"greedy_falls": float(np.mean(falls)), "greedy_reward": float(np.mean(rewards)),
            "greedy_finished": float(np.mean(finished)), "greedy_zero_fall_episodes": float(np.mean(np.array(falls) == 0))}


def train(board_config, *, total_episodes: int, seed: int, out_dir: Path, device: str = "cpu",
          eval_every: int = 100, eval_episodes: int = 10, **env_kwargs) -> Path:
    from stable_baselines3.common.monitor import Monitor
    v = _load_v20()._load_v1314()
    env_kwargs.setdefault("visibility_range", 25.0)
    env_kwargs.setdefault("episode_steps", 1000)
    run_dir = out_dir / f"lanes_{board_config.n_lanes}__sol_{board_config.n_solutions}__rows_{board_config.n_rows}__seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    board_config = replace(board_config, seed=seed)
    env, board = make_env(board_config, True, **env_kwargs)
    total_timesteps = total_episodes * board_config.n_rows       # ~ one decision per row per episode
    (run_dir / "config.json").write_text(json.dumps(
        {"version": "v2.1", "board": board_config.__dict__, "seed": seed, "total_episodes": total_episodes,
         "total_decisions_budget": total_timesteps, "env": env_kwargs, "eval_every": eval_every,
         "eval_episodes": eval_episodes, "initial_board": board.summary()}, indent=2))
    env = Monitor(env, str(run_dir))
    model = v.ValidActionDQN(
        "MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]},
        exploration_fraction=0.25, exploration_final_eps=0.025, learning_rate=1e-4,
        learning_starts=5000, batch_size=128, seed=seed, device=device, verbose=0)

    eval_rows: list[dict] = []

    class Callback(v.TrainingMetricsCallback):
        def _on_step(self) -> bool:
            before = self.episode_count
            keep = super()._on_step()
            if self.episode_count != before and self.episode_count % eval_every == 0:
                eval_rows.append({"episode": self.episode_count, "decisions": self.num_timesteps,
                                  **_greedy_eval(self.model, board_config, eval_episodes, 10_000 + seed * 1000, env_kwargs)})
                pd.DataFrame(eval_rows).to_csv(run_dir / "greedy_eval.csv", index=False)
            return keep

    cb = Callback(run_dir, total_timesteps=total_timesteps, total_episodes=total_episodes, loss_log_freq=100)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(str(run_dir / "model_end"))
    final = _greedy_eval(model, board_config, 50, 50_000 + seed * 1000, env_kwargs)
    eval_rows.append({"episode": total_episodes, "decisions": model.num_timesteps, **final})
    pd.DataFrame(eval_rows).to_csv(run_dir / "greedy_eval.csv", index=False)
    (run_dir / "final_greedy_eval_50ep.json").write_text(json.dumps(final, indent=2))
    env.close()
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lanes", type=int, default=3)
    p.add_argument("--rows", type=int, default=240)
    p.add_argument("--solutions", type=int, default=8)
    p.add_argument("--max-shift", type=int, default=1)
    p.add_argument("--row-spacing", type=float, default=2.0)
    p.add_argument("--start-offset", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train", action="store_true")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    a = p.parse_args()
    v20 = _load_v20()
    cfg = v20.BoardConfig(a.lanes, a.rows, a.solutions, a.max_shift, a.row_spacing, a.start_offset, a.seed)
    if a.train:
        print("trained:", train(cfg, total_episodes=a.episodes, seed=a.seed, out_dir=a.out, device=a.device))
    else:
        board = v20.Board(cfg)
        print(json.dumps(board.summary()))


if __name__ == "__main__":
    main()
