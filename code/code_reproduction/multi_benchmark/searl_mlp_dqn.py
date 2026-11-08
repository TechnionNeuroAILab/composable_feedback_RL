#!/usr/bin/env python
"""SEARL and its random-search baseline with an MLP Q-network (DQN), for bike v2.1.

Why this file exists: the original SEARL DQN code (../searl_dqn) is CNN-only (Atari frames); bike v2.1 has an
8-number observation. This adapts the sequential SEARL port in ../searl (EvolvableMLP, tournament selection,
the mutation operators, shared replay memory, evaluation->selection->mutation->training loop, all reused
unchanged) from TD3 (actor + twin critics) to DQN (one Q-network):

  * individual = one EvolvableMLP Q-net + rl_config (lr is the mutated hyperparameter, as in the TD3 setting)
  * mutations   = the same five operators (none / weight / architecture / activation / lr), applied to the Q-net
  * training    = double-DQN update, Huber loss, soft target update, batches from the SHARED replay memory
  * fitness     = return of one epsilon-greedy (0.1) episode on a fresh board; all its transitions go to the
                  shared replay; the first `start_timesteps` frames are random actions

and `run_random_search` is the paper's baseline: 10 sampled configurations (hidden layers {1,2,3}, nodes
[64, 384], activation {relu, tanh, elu}, lr log-uniform [1e-5, 5e-3]; Table 4 minus the CNN part) trained by a plain
DQN, the best one reported with its x-axis multiplied by the number of configurations (fair-evaluation protocol).
Frames = agent decisions (one decision per bike row).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "searl"))

from common_bike import REWARD_SCALE, make_bike_env  # noqa: E402
from config import make_config  # noqa: E402  (../searl)
from evolvable_mlp import EvolvableMLP  # noqa: E402
from mutation import Mutations  # noqa: E402
from tournament_selection import TournamentSelection  # noqa: E402

torch.set_num_threads(1)
OBS_DIM, N_ACTIONS = 8, 3
FACTORED, N_JOINTS = False, 0
ENV_NAME, R_DIV = "bike", REWARD_SCALE   # set from --env in main(); R_DIV = reward divisor used when storing transitions


def _make_env(seed):
    if ENV_NAME == "bike":
        return make_bike_env(seed, reward_scale=1.0)
    import gymnasium as gym
    try:
        import ale_py
        gym.register_envs(ale_py)
    except Exception:
        pass
    from discrete_halfcheetah import prepare_env
    return prepare_env(gym.make(ENV_NAME), ENV_NAME)


class ArrayReplay:
    def __init__(self, capacity, obs_dim=None):
        obs_dim = OBS_DIM if obs_dim is None else obs_dim
        self.capacity, self.n, self.ptr = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, N_JOINTS) if FACTORED else (capacity,), np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.d[idx]))


def get_optimizer(name):
    return {"adam": torch.optim.Adam, "adamax": torch.optim.Adamax, "rmsprop": torch.optim.RMSprop,
            "sdg": torch.optim.SGD}[name]


def dqn_updates(net, target, optimizer, replay, iterations, batch_size, gamma, tau):
    net.train()
    for _ in range(iterations):
        s, a, r, s2, d = replay.sample(batch_size)
        with torch.no_grad():
            if not FACTORED:
                best = net(s2).argmax(1, keepdim=True)
                next_v = target(s2).gather(1, best).squeeze(1)
            else:
                online = net(s2).view(-1, N_JOINTS, 3)
                nxt = target(s2).view(-1, N_JOINTS, 3)
                next_v = nxt.gather(-1, online.argmax(-1, keepdim=True)).squeeze(-1).mean(-1)
            y = r + gamma * (1.0 - d) * next_v
        if not FACTORED:
            taken = net(s).gather(1, a[:, None]).squeeze(1)
        else:
            taken = net(s).view(-1, N_JOINTS, 3).gather(-1, a.long().unsqueeze(-1)).squeeze(-1).mean(-1)
        loss = F.smooth_l1_loss(taken, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def run_episode(env, net, seed, epsilon, replay=None, random_actions=False):
    """One episode; returns (raw_return, n_decisions). Transitions go to `replay` (reward scaled)."""
    obs, _ = env.reset(seed=int(seed))
    done, ret, n = False, 0.0, 0
    with torch.no_grad():
        while not done:
            if random_actions or np.random.rand() < epsilon:
                a = np.random.randint(0, 3, size=N_JOINTS) if FACTORED else np.random.randint(N_ACTIONS)
            else:
                q = net(torch.from_numpy(np.asarray(obs, np.float32))[None])
                a = q.view(N_JOINTS, 3).argmax(1).numpy() if FACTORED else int(q.argmax(1).item())
            nobs, r, term, trunc, _ = env.step(a)
            if replay is not None:
                replay.add(obs, a, r / R_DIV, nobs, float(term))
            obs, ret, n, done = nobs, ret + r, n + 1, term or trunc
    return ret, n


# ----------------------------------------------------------------------------------------------- SEARL
class DQNIndividual:
    def __init__(self, q_config, rl_config, index):
        self.q_config, self.rl_config, self.index = q_config, rl_config, index
        self.q_net = EvolvableMLP(num_inputs=OBS_DIM, num_outputs=N_ACTIONS, output_vanish=False, **q_config)
        self.fitness, self.improvement = [], 0
        self.train_log = {"mutation": None, "parent_index": None}

    def clone(self, index=None):
        c = DQNIndividual(copy.deepcopy(self.q_net.short_dict), copy.deepcopy(self.rl_config),
                          self.index if index is None else index)
        c.fitness = copy.deepcopy(self.fitness)
        c.train_log = copy.deepcopy(self.train_log)
        c.q_net = self.q_net.clone()
        return c


class MutationsDQN(Mutations):
    """The original operators, pointed at the single Q-network."""

    def activation_mutation(self, individual):
        individual.q_net = self._permutate_activation(individual.q_net)
        individual.train_log["mutation"] = "activation"
        return individual

    def parameter_mutation(self, individual):
        individual.q_net = self.classic_parameter_mutation(individual.q_net)
        individual.train_log["mutation"] = "classic_parameter"
        return individual

    def architecture_mutate(self, individual):
        off = individual.q_net.clone()
        if self.rng.uniform(0, 1) < self.cfg.mutation.new_layer_prob:
            off.add_layer()
            individual.train_log["mutation"] = "architecture_new_layer"
        else:
            off.add_node()
            individual.train_log["mutation"] = "architecture_new_node"
        individual.q_net = off
        return individual


def searl_config(seed, pop, frames):
    return make_config({
        "nevo": {"population_size": pop, "tournament_size": 3, "elitism": True},
        "mutation": {"no_mutation": 0.2, "parameters": 0.2, "architecture": 0.2, "activation": 0.2,
                     "rl_hyperparam": 0.2, "rl_hp_selection": ["lr_actor"], "new_layer_prob": 0.2,
                     "mutation_sd": 0.1},
        "train": {"num_frames": frames, "replay_memory_size": frames, "min_train_steps": 250},
        "rl": {"train_frames_fraction": 0.5, "gamma": 0.99, "tau": 0.01, "batch_size": 128, "lr_actor": 3e-4,
               "optimizer": "adam", "start_timesteps": 5000},
        "eval": {"min_eval_steps": 250, "exploration_eps": 0.1},
        "seed": {"mutation": seed},
        "actor": {"hidden_size": [128], "activation": "relu", "output_activation": None, "layer_norm": False},
    })


def run_searl(seed, pop, frames, log=print):
    cfg = searl_config(seed, pop, frames)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    env = _make_env(seed * 100)
    test_env = _make_env(seed * 100 + 7)   # greedy test of the best individual (not counted)
    replay = ArrayReplay(cfg.train.replay_memory_size)
    tournament, mutation = TournamentSelection(cfg), MutationsDQN(cfg)
    population = [DQNIndividual(copy.deepcopy(cfg.actor.get_dict), copy.deepcopy(cfg.rl), i) for i in range(pop)]
    history, num_frames, epoch = [], 0, 0
    while num_frames < frames:
        epoch += 1
        t0 = time.time()
        fit, eval_frames = np.zeros(pop), 0
        for i, indi in enumerate(population):
            indi.q_net.eval()
            ep, n_done = 0, 0
            rets = []
            while ep < 1 or n_done < cfg.eval.min_eval_steps:
                ret, n = run_episode(env, indi.q_net, rng.randint(0, 1_000_000), cfg.eval.exploration_eps, replay,
                                     random_actions=num_frames <= cfg.rl.start_timesteps)
                rets.append(ret)
                ep, n_done = ep + 1, n_done + n
            fit[i] = np.mean(rets)
            eval_frames += n_done
            indi.fitness.append(fit[i])
        num_frames += eval_frames
        best = int(np.argmax(fit))
        # fitness above includes 10% random actions (each can drop the bike into a hole); the paper's elite
        # test evaluation is exploration-free and not counted in env steps -- same here (3 fresh boards)
        population[best].q_net.eval()
        test_return = float(np.mean([run_episode(test_env, population[best].q_net, 900_000 + epoch * 10 + k, 0.0)[0]
                                     for k in range(3)]))
        history.append({"epoch": epoch, "num_frames": num_frames, "mean_fitness": float(fit.mean()),
                        "best_fitness": float(fit[best]), "test_return": test_return, "time": time.time() - t0,
                        "best_lr": population[best].rl_config.lr_actor,
                        "best_hidden": str(population[best].q_net.hidden_size),
                        "best_activation": population[best].q_net.activation})
        if epoch % 10 == 0:
            log(f"[SEARL-MLP] epoch={epoch} frames={num_frames} best={fit[best]:.0f} mean={fit.mean():.0f} test={test_return:.0f}")
        if num_frames >= frames:
            break
        _, population = tournament.select(population)
        population = mutation.mutation(population)
        iterations = max(cfg.train.min_train_steps, int(cfg.rl.train_frames_fraction * eval_frames))
        for indi in population:
            net, a = indi.q_net, indi.rl_config
            target = type(net)(**net.init_dict)
            target.load_state_dict(net.state_dict())
            opt = get_optimizer(a.optimizer)(net.parameters(), lr=a.lr_actor)
            dqn_updates(net, target, opt, replay, iterations, a.batch_size, a.gamma, a.tau)
    return history


# ------------------------------------------------------------------------------------------ random search
def sample_config(rng):
    n_mlp = int(rng.choice([1, 2, 3]))
    return {"hidden_size": [int(rng.randint(64, 385)) for _ in range(n_mlp)],
            "activation": str(rng.choice(["relu", "tanh", "elu"])),
            "lr": float(np.exp(rng.uniform(np.log(1e-5), np.log(5e-3))))}


def train_dqn_config(cfg, seed, frames, eval_every=4000, eval_episodes=3, log=print):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = _make_env(seed * 100 + 1)
    eval_env = _make_env(seed * 100 + 2)
    net = EvolvableMLP(num_inputs=OBS_DIM, num_outputs=N_ACTIONS, hidden_size=cfg["hidden_size"],
                       activation=cfg["activation"], output_vanish=False)
    target = type(net)(**net.init_dict)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    replay = ArrayReplay(frames)
    history, steps, next_eval, ep_seed = [], 0, 0, seed * 1_000_003
    while steps < frames:
        eps = max(0.05, 1.0 - 0.95 * steps / (0.5 * frames))
        ret, n = run_episode(env, net, ep_seed, eps, replay, random_actions=steps < 2000)
        ep_seed += 1
        steps += n
        if steps >= 2000:
            dqn_updates(net, target, opt, replay, max(1, n // 2), 128, 0.99, 0.01)
        if steps >= next_eval:
            net.eval()
            greedy = np.mean([run_episode(eval_env, net, 777_000 + k, 0.0)[0] for k in range(eval_episodes)])
            history.append({"num_frames": steps, "reward": float(greedy)})
            next_eval = steps + eval_every
    return history


def run_random_search(seed, n_configs, frames_per_config, log=print):
    rng = np.random.RandomState(seed)
    results = []
    for c in range(n_configs):
        cfg = sample_config(rng)
        hist = train_dqn_config(cfg, seed * 100 + c, frames_per_config, log=log)
        results.append({"config": cfg, "history": hist, "final": hist[-1]["reward"]})
        log(f"[RS-MLP] config {c + 1}/{n_configs} {cfg} final={hist[-1]['reward']:.0f}")
    best = max(results, key=lambda r: r["final"])
    return results, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["searl", "random_search"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=480_000)
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--n_configs", type=int, default=10)
    p.add_argument("--frames_per_config", type=int, default=48_000)
    p.add_argument("--out", required=True)
    p.add_argument("--env", default="bike", help="'bike' (bike v2.1) or a gymnasium discrete env id, e.g. CartPole-v1")
    a = p.parse_args()
    global ENV_NAME, OBS_DIM, N_ACTIONS, R_DIV
    ENV_NAME = a.env
    if a.env != "bike":
        import gymnasium as gym
        probe = _make_env(0)
        global FACTORED, N_JOINTS
        FACTORED = isinstance(probe.action_space, gym.spaces.MultiDiscrete)
        N_JOINTS = int(len(probe.action_space.nvec)) if FACTORED else 0
        n_actions = int(probe.action_space.nvec.sum()) if FACTORED else int(probe.action_space.n)
        OBS_DIM, N_ACTIONS, R_DIV = probe.observation_space.shape[0], n_actions, 1.0
        probe.close()
    os.makedirs(a.out, exist_ok=True)
    csv = os.path.join(a.out, "history.csv")
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.method == "searl":
        pd.DataFrame(run_searl(a.seed, a.pop, a.frames, log)).to_csv(csv, index=False)
    else:
        results, best = run_random_search(a.seed, a.n_configs, a.frames_per_config, log)
        with open(os.path.join(a.out, "all_configs.json"), "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "history"} for r in results], f, indent=2)
        pd.DataFrame(best["history"]).to_csv(csv, index=False)
        open(os.path.join(a.out, "num_configs_tried.txt"), "w").write(str(a.n_configs))
    print("wrote", csv)


if __name__ == "__main__":
    main()
