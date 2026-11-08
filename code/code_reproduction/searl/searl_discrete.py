#!/usr/bin/env python
"""Discrete-action SEARL (DQN) in the same sequential reproduction style as searl_td3.py.

One implementation for every discrete benchmark in this project, so the SEARL lines are comparable with the
TD3/HalfCheetah reproduction in this folder (same tournament selection, same shared-replay loop, same mutation
operators, no multiprocessing):

  --task atari  (Pong, Enduro, ...): the original Rainbow-style CNN Q-network (evolvable_cnn.py, copied verbatim:
                noisy MLP heads, dueling, 51-atom categorical output), the original CNN mutation operators
                (mutation_cnn.py, verbatim), the original categorical DQN update (training_dqn.py: projection,
                soft target update every 2 iterations, noise reset every 5), original config values
                (configs/searl_dqn_config.yml) except the scale knobs (population, frames) set on the CLI.
                Observation = ONE 84x84 grayscale frame, frame-skip 4, noop-reset, episodic-life, fire-reset,
                clipped rewards -- exactly the original wrap_deepmind(frame_stack=False) pipeline, rebuilt on
                gymnasium (the original needs the unbuildable gym 0.17.1/atari-py stack). Fitness = greedy return of
                one episode (noisy nets in eval mode), transitions of every evaluation go to the shared replay.
  --task gym   (small gymnasium discrete tasks, e.g. CartPole-v1): identical to --task bike (MLP path), reward unscaled.
  --task bike  (bike_env v2.1, 8-number observation): same loop with an MLP Q-network (evolvable_mlp.py), the
                port's mutation operators pointed at that network, a standard double-DQN update (Huber loss, soft
                target) and epsilon-greedy (0.1) fitness; additionally logs an exploration-free `test_return` of the
                best individual each epoch (not counted in env steps, nothing written to the replay).

Memory: frames are kept as uint8 in the replay (the original stores float32 tensors, 4x larger; the values are
identical because frames are integers 0..255 and the network divides by 255).

Frames/steps = agent steps (Atari: after frame-skip 4; bike: one decision per row).
"""
import argparse
import copy
import os
import sys
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "multi_benchmark"))

from config import make_config  # noqa: E402
from evolvable_cnn import EvolvableCnnDQN  # noqa: E402
from evolvable_mlp import EvolvableMLP  # noqa: E402
from mutation import Mutations as MlpMutations  # noqa: E402
from mutation_cnn import Mutations as CnnMutations  # noqa: E402
from tournament_selection import TournamentSelection  # noqa: E402

torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ environments
def make_atari_env(env_id):
    import gymnasium as gym
    import ale_py  # noqa: F401
    from cleanrl_utils.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv,
                                              NoopResetEnv)
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

    class ChannelFirst(gym.ObservationWrapper):
        def __init__(self, e):
            super().__init__(e)
            self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), np.uint8)

        def observation(self, o):
            return np.ascontiguousarray(np.transpose(o, (2, 0, 1)))

    return ChannelFirst(env)


def make_env(task, env_id, seed, lanes=5, solutions=8):
    if task == "atari":
        sys.path.insert(0, os.path.join(HERE, "..", "multi_benchmark"))
        return make_atari_env(env_id)
    if task == "gym":
        import gymnasium as gym
        try:
            import ale_py
            gym.register_envs(ale_py)
        except Exception:
            pass
        from discrete_halfcheetah import prepare_env
        return prepare_env(gym.make(env_id), env_id)
    from common_bike import make_bike_env
    return make_bike_env(seed * 100, reward_scale=1.0, lanes=lanes, solutions=solutions)


# ----------------------------------------------------------------------------------------------------- replay
class ArrayReplay:
    def __init__(self, capacity, obs_shape, obs_dtype, action_shape=()):
        self.capacity, self.n, self.ptr = capacity, 0, 0
        self.s = np.zeros((capacity,) + tuple(obs_shape), obs_dtype)
        self.s2 = np.zeros((capacity,) + tuple(obs_shape), obs_dtype)
        self.a = np.zeros((capacity,) + tuple(action_shape), np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch, device):
        idx = np.random.randint(0, self.n, size=batch)
        t = lambda x: torch.from_numpy(x[idx]).to(device)  # noqa: E731
        return t(self.s).float(), t(self.a), t(self.r), t(self.s2).float(), t(self.d)


# ------------------------------------------------------------------------------------------------- individual
def get_optimizer(name):
    return {"adam": torch.optim.Adam, "adamax": torch.optim.Adamax, "rmsprop": torch.optim.RMSprop,
            "sdg": torch.optim.SGD}[name]


class DQNIndividual:
    def __init__(self, actor, rl_config, index):
        self.actor, self.rl_config, self.index = actor, rl_config, index
        self.fitness, self.improvement = [], 0
        self.train_log = {"mutation": None, "parent_index": None}

    def clone(self, index=None):
        c = DQNIndividual(self.actor.clone(), copy.deepcopy(self.rl_config), self.index if index is None else index)
        c.fitness = copy.deepcopy(self.fitness)
        c.train_log = copy.deepcopy(self.train_log)
        return c


class MlpDqnMutations(MlpMutations):
    """The port's operators (weight / activation / lr / none) with the architecture and activation mutations
    pointed at the single Q-network (the TD3 version mutates actor + critics together)."""

    def activation_mutation(self, individual):
        individual.actor = self._permutate_activation(individual.actor)
        individual.train_log["mutation"] = "activation"
        return individual

    def architecture_mutate(self, individual):
        off = individual.actor.clone()
        if self.rng.uniform(0, 1) < self.cfg.mutation.new_layer_prob:
            off.add_layer()
            individual.train_log["mutation"] = "architecture_new_layer"
        else:
            off.add_node()
            individual.train_log["mutation"] = "architecture_new_node"
        individual.actor = off
        return individual


# ------------------------------------------------------------------------------------------------- training
def train_cnn(indi, replay, iterations, device):
    """Original training_dqn.update_parameters (categorical DQN, noisy nets), batches from the shared replay."""
    a = indi.rl_config
    actor = indi.actor
    target = type(actor)(**actor.init_dict)
    target.load_state_dict(actor.state_dict())
    actor.to(device)
    actor.train()
    target.to(device)
    optim = get_optimizer(a.optimizer)(actor.parameters(), lr=a.lr_actor)
    A = actor.num_atoms
    delta_z = float(actor.Vmax - actor.Vmin) / (A - 1)
    support0 = torch.linspace(actor.Vmin, actor.Vmax, A).to(device)
    losses = []
    for it in range(iterations):
        s, act, r, s2, d = replay.sample(a.batch_size, device)
        with torch.no_grad():
            B = s2.size(0)
            nd = target(s2) * support0
            na = nd.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1).expand(nd.size(0), 1, nd.size(2))
            nd = nd.gather(1, na).squeeze(1)
            rr, dd, sup = r.unsqueeze(1).expand_as(nd), d.unsqueeze(1).expand_as(nd), support0.unsqueeze(0).expand_as(nd)
            Tz = (rr + (1 - dd) * 0.99 * sup).clamp(min=actor.Vmin, max=actor.Vmax)
            b = (Tz - actor.Vmin) / delta_z
            lo, up = b.floor().long(), b.ceil().long()
            offset = torch.linspace(0, (B - 1) * A, B).long().unsqueeze(1).expand(B, A).to(device)
            proj = torch.zeros(nd.size()).to(device)
            proj.view(-1).index_add_(0, (lo + offset).view(-1), (nd * (up.float() - b)).view(-1))
            proj.view(-1).index_add_(0, (up + offset).view(-1), (nd * (b - lo.float())).view(-1))
        dist = actor(s)
        dist = dist.gather(1, act.unsqueeze(1).unsqueeze(1).expand(a.batch_size, 1, A)).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(proj * dist.log()).sum(1).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 5 == 0:
            actor.reset_noise()
            target.reset_noise()
        losses.append(float(loss.detach().cpu()))
        if it % 2 == 0 and it != 0:
            for p, tp in zip(actor.parameters(), target.parameters()):
                tp.data.copy_(a.tau * p.data + (1 - a.tau) * tp.data)
    indi.actor = actor.cpu().clone()
    indi.train_log["train_losses"] = float(np.mean(losses))
    return indi


def train_mlp(indi, replay, iterations, device):
    a = indi.rl_config
    net = indi.actor
    target = type(net)(**net.init_dict)
    target.load_state_dict(net.state_dict())
    opt = get_optimizer(a.optimizer)(net.parameters(), lr=a.lr_actor)
    net.train()
    for _ in range(iterations):
        s, act, r, s2, d = replay.sample(a.batch_size, "cpu")
        with torch.no_grad():
            if act.ndim == 1:
                best = net(s2).argmax(1, keepdim=True)
                next_v = target(s2).gather(1, best).squeeze(1)
            else:
                # one head of 3 Q-values per joint; the mean of the heads estimates the return
                n_joints = act.shape[1]
                online = net(s2).view(-1, n_joints, 3)
                nxt = target(s2).view(-1, n_joints, 3)
                next_v = nxt.gather(-1, online.argmax(-1, keepdim=True)).squeeze(-1).mean(-1)
            y = r + a.gamma * (1.0 - d) * next_v
        if act.ndim == 1:
            taken = net(s).gather(1, act[:, None]).squeeze(1)
        else:
            taken = net(s).view(-1, act.shape[1], 3).gather(-1, act.long().unsqueeze(-1)).squeeze(-1).mean(-1)
        loss = F.smooth_l1_loss(taken, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - a.tau).add_(a.tau * p.data)
    return indi


# ----------------------------------------------------------------------------------------------- evaluation
def run_episode(task, env, net, seed, device, epsilon, replay=None, random_actions=False, reward_scale=1.0):
    obs, _ = env.reset(seed=int(seed))
    done, ret, n = False, 0.0, 0
    factored = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    n_actions = None if factored else env.action_space.n
    with torch.no_grad():
        while not done:
            if random_actions or (epsilon > 0 and np.random.rand() < epsilon):
                a = env.action_space.sample() if factored else np.random.randint(n_actions)
            elif task == "atari":
                a = int(net.act(obs))
            else:
                q = net(torch.from_numpy(obs.astype(np.float32))[None])
                if factored:
                    a = q.view(-1, 3).argmax(1).cpu().numpy()
                else:
                    a = int(q.argmax(1).item())
            nobs, r, term, trunc, _ = env.step(a)
            if replay is not None:
                replay.add(obs, a, r * reward_scale, nobs, float(term))
            obs, ret, n, done = nobs, ret + r, n + 1, term or trunc
    return ret, n


# ---------------------------------------------------------------------------------------------------- SEARL
def make_cfg(task, seed, pop, frames):
    d = {
        "nevo": {"population_size": pop, "tournament_size": 3, "elitism": True},
        "mutation": {"no_mutation": 0.2, "parameters": 0.2, "architecture": 0.2, "activation": 0.2,
                     "rl_hyperparam": 0.2, "rl_hp_selection": ["lr_actor"], "new_layer_prob": 0.2,
                     "mutation_sd": 0.1},
        "seed": {"mutation": seed},
    }
    if task == "atari":   # configs/searl_dqn_config.yml
        d["train"] = {"num_frames": frames, "min_train_steps": 1000, "max_train_steps": 50000}
        d["rl"] = {"train_frames_fraction": 0.5, "gamma": 0.99, "tau": 0.005, "batch_size": 128, "lr_actor": 1e-4,
                   "optimizer": "adam"}
        d["eval"] = {"eval_episodes": 1, "min_eval_steps": 250}
        d["actor"] = {"channel_size": [32, 32], "kernal_size": [8, 4], "stride_size": [4, 2], "hidden_size": [128],
                      "num_atoms": 51, "Vmin": -10, "Vmax": 10, "mlp_activation": "relu", "cnn_activation": "relu",
                      "layer_norm": False}
    else:
        d["train"] = {"num_frames": frames, "min_train_steps": 250, "max_train_steps": 50000}
        d["rl"] = {"train_frames_fraction": 0.5, "gamma": 0.99, "tau": 0.01, "batch_size": 128, "lr_actor": 3e-4,
                   "optimizer": "adam", "start_timesteps": 5000}
        d["eval"] = {"eval_episodes": 1, "min_eval_steps": 250, "exploration_eps": 0.1}
        d["actor"] = {"hidden_size": [128], "activation": "relu", "output_activation": None, "layer_norm": False}
    return make_config(d)


def run_searl_discrete(task, env_id, seed, pop, frames, log=print, lanes=5, solutions=8, eval_episodes=3, eval_seed0=None, eval_every_frames=None):
    cfg = make_cfg(task, seed, pop, frames)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and task == "atari" else "cpu")
    env = make_env(task, env_id, seed, lanes, solutions)
    test_env = make_env(task, env_id, seed + 1000, lanes, solutions) if task != "atari" else None
    factored = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    n_actions = int(env.action_space.nvec.sum()) if factored else env.action_space.n
    obs_shape = env.observation_space.shape
    action_shape = tuple(env.action_space.nvec.shape) if factored else ()
    replay = ArrayReplay(frames, obs_shape, np.uint8 if task == "atari" else np.float32, action_shape)
    tournament = TournamentSelection(cfg)
    mutation = (CnnMutations if task == "atari" else MlpDqnMutations)(cfg)
    reward_scale = 1.0 / 20.0 if task == "bike" else 1.0      # bike: shared reward/20 preprocessing

    def new_actor():
        ac = copy.deepcopy(cfg.actor.get_dict)
        if task == "atari":
            return EvolvableCnnDQN(input_shape=obs_shape, num_actions=n_actions, device="cpu", **ac)
        return EvolvableMLP(num_inputs=obs_shape[0], num_outputs=n_actions, output_vanish=False, **ac)

    population = [DQNIndividual(new_actor(), copy.deepcopy(cfg.rl), i) for i in range(pop)]
    history, num_frames, epoch = [], 0, 0
    last_eval_bucket = 0
    while num_frames < frames:
        epoch += 1
        t0 = time.time()
        fit, eval_frames = np.zeros(pop), 0
        for i, indi in enumerate(population):
            net = indi.actor
            if task == "atari":
                net.to(device)
                net.device = device
            net.eval()
            rets, ep, n_done = [], 0, 0
            while ep < cfg.eval.eval_episodes or n_done < cfg.eval.min_eval_steps:
                ret, n = run_episode(
                    task, env, net, rng.randint(0, 100_000), device,
                    epsilon=0.0 if task == "atari" else cfg.eval.exploration_eps, replay=replay,
                    random_actions=(task != "atari" and num_frames <= cfg.rl.start_timesteps),
                    reward_scale=reward_scale)
                rets.append(ret)
                ep, n_done = ep + 1, n_done + n
            if task == "atari":
                net.to("cpu")
                net.device = "cpu"
            fit[i] = np.mean(rets)
            eval_frames += n_done
            indi.fitness.append(fit[i])
        num_frames += eval_frames
        best = int(np.argmax(fit))
        rec = {"epoch": epoch, "num_frames": num_frames, "mean_fitness": float(fit.mean()),
               "best_fitness": float(fit[best]), "best_lr": population[best].rl_config.lr_actor}
        if task == "bike" and eval_seed0 is not None and eval_every_frames:
            # shared bike-grid protocol: evaluate the best individual only when num_frames passes the next multiple of
            # eval_every_frames (the same spacing as the single-agent runs); other epochs log NaN
            due = num_frames // eval_every_frames > last_eval_bucket
        else:
            due = True
        if task != "atari" and due:
            net = population[best].actor
            net.eval()
            last_eval_bucket = num_frames // eval_every_frames if eval_every_frames else 0
            # default: 3 episodes on seeds that change every epoch; eval_seed0 set: eval_episodes episodes on FIXED seeds
            # (the same boards at every epoch and for every method, see common_bike.greedy_return)
            seeds = ([900_000 + epoch * 10 + k for k in range(3)] if eval_seed0 is None
                     else [eval_seed0 + k for k in range(eval_episodes)])
            if task == "bike" and eval_seed0 is not None:
                from common_bike import greedy_return   # the one evaluation function every method uses

                def _greedy(o):
                    with torch.no_grad():
                        return int(net(torch.from_numpy(np.asarray(o, np.float32))[None]).argmax(1).item())

                rec["test_return"] = greedy_return(_greedy, lanes, solutions, eval_episodes, eval_seed0)
            else:
                rec["test_return"] = float(np.mean([run_episode(task, test_env, net, sd_, device, 0.0)[0] for sd_ in seeds]))
        rec["time"] = time.time() - t0
        history.append(rec)
        if epoch % 5 == 0 or task != "atari" and epoch % 10 == 0:
            log(f"[SEARL-{task}] epoch={epoch} frames={num_frames} best={fit[best]:.1f} mean={fit.mean():.1f}"
                + (f" test={rec['test_return']:.0f}" if "test_return" in rec else ""))
        if num_frames >= frames:
            break
        _, population = tournament.select(population)
        population = mutation.mutation(population)
        iterations = min(max(cfg.train.min_train_steps, int(cfg.rl.train_frames_fraction * eval_frames)),
                         cfg.train.max_train_steps)
        train = train_cnn if task == "atari" else train_mlp
        population = [train(indi, replay, iterations, device) for indi in population]
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["atari", "bike", "gym"], required=True)
    p.add_argument("--env", default="PongNoFrameskip-v4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--lanes", type=int, default=5, help="bike task only: number of lanes")
    p.add_argument("--solutions", type=int, default=8, help="bike task only: number of solutions (fall-free routes)")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-seed0", type=int, default=None, help="if set, greedy test episodes use fixed seeds eval-seed0 + k")
    p.add_argument("--eval-every-frames", type=int, default=None, help="bike + eval-seed0: evaluate about every this many frames")
    p.add_argument("--frames", type=int, default=60_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hist = run_searl_discrete(a.task, a.env, a.seed, a.pop, a.frames, log=lambda m: print(m, flush=True), lanes=a.lanes,
                              solutions=a.solutions, eval_episodes=a.eval_episodes, eval_seed0=a.eval_seed0,
                              eval_every_frames=a.eval_every_frames)
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
    print("wrote", os.path.join(a.out, "history.csv"))


if __name__ == "__main__":
    main()
