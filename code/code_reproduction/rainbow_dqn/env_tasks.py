"""Vector-observation tasks for the ported agent: the discrete benchmarks of multi_benchmark (bike v2.1, CartPole,
Acrobot, LunarLander, HalfCheetah/Hopper with one 3-way torque choice per joint, Freeway RAM, FrozenLake). Uses the
same environment construction as the other algorithms (multi_benchmark/discrete_halfcheetah.py, common_bike.py)."""
import os
import sys

import gymnasium as gym
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "multi_benchmark"))


class GymTask:
  def __init__(self, env_id, seed, reward_scale=1.0, obs_scale=1.0, lanes=5, solutions=8):
    if env_id == "bike":
      from common_bike import make_bike_env
      env = make_bike_env(seed, reward_scale=1.0, lanes=lanes, solutions=solutions)   # raw rewards; scaling is applied here
    else:
      if env_id.startswith("ALE/"):
        import ale_py
        gym.register_envs(ale_py)    # registers the ALE namespace (as the other scripts do)
      from discrete_halfcheetah import prepare_env
      env = prepare_env(gym.make(env_id), env_id)
    self.env, self.seed, self.first = env, seed, True
    self.reward_scale, self.obs_scale = reward_scale, obs_scale
    space = env.action_space
    self.factored = isinstance(space, gym.spaces.MultiDiscrete)
    self.branches = int(len(space.nvec)) if self.factored else 1
    self.n_actions = int(space.nvec[0]) if self.factored else int(space.n)
    self.obs_dim = int(np.prod(env.observation_space.shape))
    self.true_done, self.raw_return, self.episode_return = False, 0.0, 0.0

  def _obs(self, o):
    return torch.tensor(np.asarray(o, np.float32).reshape(-1) * self.obs_scale)

  def reset(self):
    kwargs = {"seed": self.seed} if self.first else {}
    self.first = False
    obs, _ = self.env.reset(**kwargs)
    self.episode_return, self.true_done = 0.0, False
    return self._obs(obs)

  def step(self, action):
    a = np.asarray(action, dtype=np.int64) if self.factored else int(action)
    obs, r, term, trunc, _ = self.env.step(a)
    self.episode_return += float(r)
    self.true_done = bool(term or trunc)
    return self._obs(obs), float(r), self.true_done      # raw reward; main() scales it

  def close(self):
    self.env.close()
