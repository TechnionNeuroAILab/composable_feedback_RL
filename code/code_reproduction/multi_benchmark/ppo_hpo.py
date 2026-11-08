#!/usr/bin/env python
"""CleanRL discrete-action PPO (cleanrl/ppo_atari.py for Atari, cleanrl/ppo.py for bike v2.1), with an
OPTIONAL hyperparameter scheduler from the HyperController reproduction.

  --hpo none            the CleanRL PPO baseline: same loop/defaults as the CleanRL scripts (lr annealing on).
  --hpo HyperController the verbatim HyperController scheduler (../hypercontroller/hyper_framework.py),
  --hpo PB2             or the verbatim PB2 scheduler, driven exactly like rl_tests.py drives them: after every
                        PPO iteration `scheduler.step(logs, DataFrame(logs))` is called with the iteration's mean
                        per-step training reward, and the returned config is applied to the next iteration:
                          lr -> Adam lr          clip_param -> clip_coef
                          lambda -> gae_lambda   train_batch_size -> PPO minibatch size
                        (bounds as in rl_tests.py, except train_batch_size, which is rescaled to
                        [batch/16, batch/2] because the PPO batch here is 1024 (Atari) / 512 (bike), not 1000.)
                        LR annealing is off in this mode: the scheduler owns the learning rate.

Logs a CSV of finished episodes: global_step, episodic_return (raw game score / raw bike return),
clipped_return (sum of clipped rewards = what the agent trains on; equals the raw return for bike).
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HC_DIR = os.path.join(HERE, "..", "hypercontroller")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariAgent(nn.Module):  # cleanrl/ppo_atari.py
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(), layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU())
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        probs = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class MlpAgent(nn.Module):  # cleanrl/ppo.py
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                    layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), std=1.0))
        self.actor = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                   layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, n_actions), std=0.01))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = Categorical(logits=self.actor(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class FactoredMlpAgent(nn.Module):
    """One categorical per joint (6 x 3 = 18 logits). Log-prob and entropy are summed over joints."""

    def __init__(self, obs_dim, n_joints=6, n_bins=3):
        super().__init__()
        self.n_joints, self.n_bins = n_joints, n_bins
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                    layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), std=1.0))
        self.actor = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                   layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                   layer_init(nn.Linear(64, n_joints * n_bins), std=0.01))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x).view(-1, self.n_joints, self.n_bins)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


def make_atari_env(env_id):
    import ale_py  # noqa: F401  (registers the ALE environments)
    from cleanrl_utils.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv,
                                              NoopResetEnv)

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def make_gym_thunk(env_id):
    def thunk():
        try:
            import ale_py
            gym.register_envs(ale_py)
        except Exception:
            pass
        from discrete_halfcheetah import prepare_env
        return gym.wrappers.RecordEpisodeStatistics(prepare_env(gym.make(env_id), env_id))
    return thunk


def make_bike_thunk(seed):
    from common_bike import make_bike_env
    return lambda: make_bike_env(seed)


@dataclass
class Args:
    task: str = "atari"
    env_id: str = "PongNoFrameskip-v4"
    hpo: str = "none"
    seed: int = 0
    total_timesteps: int = 250_000
    csv_path: str = ""
    t_ready: int = 5
    cuda: bool = True
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["atari", "bike", "gym"], required=True)
    p.add_argument("--env_id", default="PongNoFrameskip-v4")
    p.add_argument("--hpo", choices=["none", "HyperController", "PB2"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_timesteps", type=int, default=250_000)
    p.add_argument("--csv_path", default="")
    p.add_argument("--t_ready", type=int, default=5, help="same value as the HalfCheetah HyperController runs")
    ns = p.parse_args()
    a = Args(task=ns.task, env_id=ns.env_id, hpo=ns.hpo, seed=ns.seed, total_timesteps=ns.total_timesteps,
             csv_path=ns.csv_path, t_ready=ns.t_ready)
    if a.task in ("bike", "gym"):      # cleanrl/ppo.py defaults
        a.num_envs, a.clip_coef, a.cuda = 4, 0.2, False
    return a


def main():
    args = parse()
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.task == "atari":
        thunks = [make_atari_env(args.env_id) for _ in range(args.num_envs)]
    elif args.task == "gym":
        thunks = [make_gym_thunk(args.env_id) for _ in range(args.num_envs)]
    else:
        thunks = [make_bike_thunk(args.seed * 100 + i) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(thunks)
    space = envs.single_action_space
    factored = isinstance(space, gym.spaces.MultiDiscrete)
    n_actions = int(space.nvec.sum()) if factored else space.n
    obs_shape = envs.single_observation_space.shape
    action_shape = (int(len(space.nvec)),) if factored else ()
    if args.task == "atari":
        agent = AtariAgent(n_actions)
    elif factored:
        agent = FactoredMlpAgent(obs_shape[0], int(len(space.nvec)), int(space.nvec[0]))
    else:
        agent = MlpAgent(obs_shape[0], n_actions)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    scheduler = None
    if args.hpo != "none":
        sys.path.insert(0, HC_DIR)
        import hyper_framework  # the verbatim HyperController-reproduction module
        bounds = {"lambda": [0.9, 1.0], "clip_param": [0.1, 0.5], "lr": [1e-5, 1e-3],
                  "train_batch_size": [batch_size // 16, batch_size // 2]}
        scheduler = hyper_framework.Scheduler(bounds, args.t_ready, args.hpo, num_iterations)
        args.anneal_lr = False
    hpo_logs = defaultdict(list)

    csv_file = None
    if args.csv_path:
        csv_file = open(args.csv_path, "w", buffering=1)
        csv_file.write("global_step,episodic_return,clipped_return\n")
    clip_acc = np.zeros(args.num_envs)
    jitter_rng = np.random.RandomState(args.seed + 12345)
    last_cfg, pb2_failed_at = {}, None
    iter_returns, last_signal = [], 0.0   # gym task: episode returns finished in the current PPO iteration

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / num_iterations) * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            clip_acc += reward

            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        raw = float(np.squeeze(info["episode"]["r"]))
                        clipped = clip_acc[i] if args.task == "atari" else raw
                        if csv_file is not None:
                            csv_file.write(f"{global_step},{raw},{clipped}\n")
                        clip_acc[i] = 0.0
                        iter_returns.append(raw)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal, nextvalues = 1.0 - next_done, next_value
                else:
                    nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages, b_returns, b_values = advantages.reshape(-1), returns.reshape(-1), values.reshape(-1)

        b_inds = np.arange(batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, batch_size, minibatch_size):
                mb = b_inds[s:s + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions.long()[mb])
                ratio = (newlogprob - b_logprobs[mb]).exp()
                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if scheduler is not None:
            r = float(rewards.mean().item())      # mean per-step training reward of this batch (as in rl_tests.py)
            if args.task == "gym":
                # CartPole pays +1 on every step, so the per-step mean is the constant 1.0 and carries no information
                # (PB2's GP fails on constant data, HyperController sees R-R0 = 0). Use the mean return of the
                # episodes that finished in this iteration (scaled to [0,1] by the 500-step cap) instead.
                if iter_returns:
                    # CartPole's 500-step cap is the [0, 1] scale. Other gym tasks use a
                    # typical solved return so HyperController / PB2 see an O(1) signal
                    # (FrozenLake is 0/1; dividing by 500 would bury it under PB2's jitter).
                    hpo_scale = {
                        "CartPole-v1": 500.0, "Acrobot-v1": 500.0, "LunarLander-v2": 200.0,
                        "Hopper-v4": 1000.0, "ALE/Freeway-ram-v5": 20.0, "FrozenLake-v1": 1.0,
                    }.get(args.env_id, 500.0)
                    last_signal = float(np.mean(iter_returns)) / hpo_scale
                r = last_signal
                if args.hpo == "PB2":
                    # once PPO saturates at 500 the signal is EXACTLY constant for many iterations and PB2's GP
                    # (verbatim pb2.py) fails with "not pd"; a tiny jitter (0.1% of the 0..1 scale, own RNG) avoids it
                    r += 1e-3 * jitter_rng.randn()
            iter_returns = []
            hpo_logs["reward"].append(r)
            hpo_logs["Reward"].append(r)
            hpo_logs["iteration"].append(iteration - 1)
            try:
                cfg = scheduler.step(hpo_logs, pd.DataFrame(hpo_logs))
                last_cfg = cfg
            except np.linalg.LinAlgError as err:
                # verbatim PB2's GP fit breaks down ("not pd") once its stored data grows past ~1000 rows; the
                # scheduler then cannot propose anything new, so keep its last configuration and say so
                if pb2_failed_at is None:
                    pb2_failed_at = iteration
                    print(f"[{args.hpo}] GP fit failed at iteration {iteration} (step {global_step}): {err}; "
                          f"keeping the last hyperparameters from here on", flush=True)
                cfg = last_cfg
            if all(k in cfg for k in ("lr", "clip_param", "lambda", "train_batch_size")):
                optimizer.param_groups[0]["lr"] = float(cfg["lr"])
                args.clip_coef = float(cfg["clip_param"])
                args.gae_lambda = float(cfg["lambda"])
                minibatch_size = int(min(batch_size, max(8, np.ceil(cfg["train_batch_size"]))))

        if iteration % 20 == 0:
            print(f"iter {iteration}/{num_iterations} step {global_step} sps {int(global_step / (time.time() - start))}",
                  flush=True)

    envs.close()
    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
