"""Auto-applied gym legacy patch for SEARL multiprocessing workers (spawn-safe).

Import this before SEARL in workers by placing this directory on PYTHONPATH and
calling `import gym_legacy_patch` from a .pth / or via run_searl_td3_compat.
When using spawn, workers re-import modules; we also register via sitecustomize.
"""
from __future__ import annotations

import numpy as np


def apply() -> None:
    import gym
    from gym import Wrapper

    if getattr(gym.make, "_searl_legacy_patched", False):
        return

    class LegacyAPIWrapper(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._seed_value = None
            # SEARL reads env._max_episode_steps; gym 0.26 blocks private getattr.
            try:
                self._max_episode_steps = int(env.spec.max_episode_steps)
            except Exception:
                self._max_episode_steps = getattr(env, "_max_episode_steps", 1000)

        def __getattribute__(self, name):
            if name == "_max_episode_steps":
                return object.__getattribute__(self, "_max_episode_steps")
            return super().__getattribute__(name)

        def seed(self, seed=None):
            self._seed_value = seed
            try:
                self.action_space.seed(seed)
                self.observation_space.seed(seed)
            except Exception:
                pass
            return [seed]

        def reset(self, **kwargs):
            if self._seed_value is not None and "seed" not in kwargs:
                kwargs["seed"] = self._seed_value
                self._seed_value = None
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, _info = result
                return obs
            return result

        def step(self, action):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
                return obs, reward, done, info
            return result

    _orig_make = gym.make

    def make_legacy(*args, **kwargs):
        return LegacyAPIWrapper(_orig_make(*args, **kwargs))

    make_legacy._searl_legacy_patched = True  # type: ignore[attr-defined]
    gym.make = make_legacy

    def _env_seed(self, seed=None):
        try:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        except Exception:
            pass
        self._np_random = np.random.RandomState(seed)
        return [seed]

    gym.Env.seed = _env_seed


apply()
