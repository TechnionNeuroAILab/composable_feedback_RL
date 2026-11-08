"""Per-joint discrete torques for continuous MuJoCo tasks, plus observation adapters.

HalfCheetah-v4: 6 joints x {-1, 0, +1} = 18 outputs.
Hopper-v4: 3 joints x {-1, 0, +1} = 9 outputs.
Index 0/1/2 on a joint is torque -1/0/+1. Control cost is unchanged.

Also turns a Discrete observation (FrozenLake) into a one-hot float vector, and casts
uint8 observations (Atari RAM) to float32, so the MLP agents can consume them.
"""
import numpy as np
import gymnasium as gym

N_BINS = 3
VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
JOINT_ENVS = {"HalfCheetah-v4": 6, "Hopper-v4": 3}


def torques_from(action, n_joints):
    idx = np.asarray(action, dtype=np.int64).reshape(n_joints)
    return VALUES[idx]


class DiscreteJoints(gym.Wrapper):
    """One 3-way choice per joint, and float32 observations."""

    def __init__(self, env):
        super().__init__(env)
        n = int(env.action_space.shape[0])
        self.n_joints = n
        self.action_space = gym.spaces.MultiDiscrete([N_BINS] * n)
        self.observation_space = gym.spaces.Box(
            np.asarray(env.observation_space.low, np.float32),
            np.asarray(env.observation_space.high, np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.asarray(obs, np.float32), info

    def step(self, act):
        obs, reward, terminated, truncated, info = self.env.step(torques_from(act, self.n_joints))
        return np.asarray(obs, np.float32), reward, terminated, truncated, info


class OneHotObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = int(env.observation_space.n)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n,), np.float32)

    def observation(self, obs):
        v = np.zeros(self.observation_space.shape, np.float32)
        v[int(obs)] = 1.0
        return v


class FloatObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            np.asarray(env.observation_space.low, np.float32),
            np.asarray(env.observation_space.high, np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.asarray(obs, np.float32)


def prepare_env(env, env_id):
    if env_id in JOINT_ENVS and isinstance(env.action_space, gym.spaces.Box):
        env = DiscreteJoints(env)
    if isinstance(env.observation_space, gym.spaces.Discrete):
        env = OneHotObservation(env)
    elif getattr(env.observation_space, "dtype", None) is not None and env.observation_space.dtype != np.float32:
        env = FloatObservation(env)
    return env


def maybe_discretize(env, env_id):
    return prepare_env(env, env_id)
