#!/usr/bin/env python
"""SEARL-DQN (searl_discrete.py, --task gym) with ConfGate aux heads driving the learning-rate knob.

The base loop, selection, replay, evaluation, mutation operators and the double-DQN update are the unmodified
code of searl_discrete.py; this file only patches in (by rebinding module names, no edits to that file):

  * aux heads   -- one linear Q head per observation group (CartPole: cart = obs[0:2], pole = obs[2:4]) on the
                   Q-network's last hidden layer, fed the group-masked observation, trunk detached. Trained after each
                   TD phase on the full network's TD target. They never act and never change the Q-network's gradient.
  * confidence  -- m = fraction of replay states on which the two heads pick the same greedy action.
  * controller  -- lr_eff = lr_base * (lr_floor + lr_slope * m) for that TD phase; lr_base is SEARL's own mutated
                   lr_actor and is restored afterwards. No gating of the features, no controller gradient.
  * mutation    -- extra `controller` operator that perturbs one coefficient (lr_floor / lr_slope).

--mix (ConfGate-mix): the heads also take part in ACTING. Each individual acts (epsilon-greedy fitness episodes, greedy test
return) with Q_mix = (1 - lam) Q_full + lam * mean_k Q_k, where Q_k is head k on its group-masked view; lam in [0, 1] is a
per-individual coefficient (init --lam-init, 0.2) that SEARL mutates additively (sd 0.1, clipped) through the `controller`
operator, inherited by clones. The TD target / Q update are unchanged. Arms: --mix --no-controller (mix only, lr gate off),
--mix (mix + lr gate). --mix --lam-init 0 --fixed-lam --no-controller must equal plain SEARL.

--no-controller keeps the heads but fixes lr_floor = 1, lr_slope = 0 and disables the controller mutation; this must
reproduce searl_discrete.py exactly (all added random draws use private generators).  --mutations lr_only turns off
the weight, architecture and activation mutations (SEARL's lr mutation and the controller remain).
"""
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import searl_discrete as sd  # noqa: E402

CARTPOLE_GROUPS = {"cart": [0, 1], "pole": [2, 3]}
def bike_groups(lanes):
    """bike obs = [speed] + nearest-hole distance in ego-relative lane slots (offsets -(w//2) .. w - w//2 - 1, w = lanes + lanes//2).
    Left head: speed, ego slot and the slots left of the bike. Right head: speed, ego slot and the slots to its right."""
    width = lanes + lanes // 2
    start = -(width // 2)
    idx = {off: 1 + i for i, off in enumerate(range(start, start + width))}
    return {"left": [0] + [i for off, i in idx.items() if off <= 0], "right": [0] + [i for off, i in idx.items() if off >= 0]}


OBS_GROUPS = {"CartPole-v1": CARTPOLE_GROUPS,
              "HalfCheetah-v4": {"pose": list(range(0, 8)), "velocity": list(range(8, 17))}}  # 6 joints x 3 torques
_HEAD_GEN = torch.Generator().manual_seed(20_000)
_SAMPLE_RNG = np.random.RandomState(30_000)
CONFGATE_LOG = []


def seed_confgate(seed):
    """Private streams for head init and replay sampling, so the global streams stay identical to plain SEARL."""
    global _HEAD_GEN, _SAMPLE_RNG
    _HEAD_GEN = torch.Generator().manual_seed(seed + 20_000)
    _SAMPLE_RNG = np.random.RandomState(seed + 30_000)


def _new_linear(in_features, out_features):
    layer = nn.utils.skip_init(nn.Linear, in_features, out_features)
    bound = 1.0 / np.sqrt(in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=_HEAD_GEN)
        layer.bias.uniform_(-bound, bound, generator=_HEAD_GEN)
    return layer


def trunk_features(net, x):
    """Activations right before the Q-network's output layer."""
    for name, layer in net.net.named_children():
        if name == "linear_layer_output":
            break
        x = layer(x)
    return x


class AuxHeads(nn.Module):
    def __init__(self, feature_dim, n_actions, groups):
        super().__init__()
        self.feature_dim, self.n_actions = feature_dim, n_actions
        self.groups = [list(g) for g in groups.values()]
        self.names = list(groups)
        self.heads = nn.ModuleList([_new_linear(feature_dim, n_actions) for _ in self.groups])

    def mask(self, obs, g):
        out = torch.zeros_like(obs)
        out[..., self.groups[g]] = obs[..., self.groups[g]]
        return out

    def resize(self, feature_dim):
        """Follow an architecture mutation of the last hidden layer (keep overlapping weights)."""
        if feature_dim == self.feature_dim:
            return
        new = nn.ModuleList([_new_linear(feature_dim, self.n_actions) for _ in self.groups])
        k = min(feature_dim, self.feature_dim)
        with torch.no_grad():
            for old, nw in zip(self.heads, new):
                nw.weight[:, :k] = old.weight[:, :k]
                nw.bias.copy_(old.bias)
        self.heads, self.feature_dim = new, feature_dim


class LRController:
    BOUNDS = {"lr_floor": (0.2, 1.0), "lr_slope": (0.0, 1.0)}

    def __init__(self, lr_floor=0.5, lr_slope=0.5, aux_fraction=0.5, lam=0.0, gate_mutable=True, lam_mutable=False):
        self.lr_floor, self.lr_slope, self.aux_fraction = lr_floor, lr_slope, aux_fraction
        self.lam, self.gate_mutable, self.lam_mutable = lam, gate_mutable, lam_mutable   # lam: weight of the heads in the acting policy

    def coefficients(self):
        return {**{k: getattr(self, k) for k in self.BOUNDS}, "lam": self.lam}

    def scale(self, m):
        return self.lr_floor + self.lr_slope * m

    def mutate(self, rng, sd=0.1):
        keys = (list(self.BOUNDS) if self.gate_mutable else []) + (["lam"] if self.lam_mutable else [])
        key = rng.choice(keys)
        if key == "lam":   # additive: a multiplicative step could never move lam off 0
            self.lam = float(np.clip(self.lam + rng.normal(0, sd), 0.0, 1.0))
            return key
        lo, hi = self.BOUNDS[key]
        val = getattr(self, key)
        val = val + rng.normal(0, sd) * (abs(val) if abs(val) > 1e-3 else 1.0)
        setattr(self, key, float(np.clip(val, lo, hi)))
        return key


# ---- ConfGate-mix: the heads take part in acting. Q_mix = (1 - lam) Q_full + lam * mean_k Q_k(group-k view)
NEUTRAL_LAM = False   # control: lam is carried and mutated but never used to act, so selection cannot see it
_REG = {}   # id(actor) -> individual, refreshed every training phase (every individual is trained before it is evaluated)


def register(indi):
    _REG[id(indi.actor)] = indi
    while len(_REG) > 64:
        _REG.pop(next(iter(_REG)))


def policy_q(net, x):
    """Q-values used to act. Plain net(x) unless the individual has lam > 0 (then no extra random draws, no grad)."""
    q = net(x)
    indi = _REG.get(id(net))
    if NEUTRAL_LAM or indi is None or indi.controller is None or indi.controller.lam <= 0 or indi.heads.feature_dim != net.hidden_size[-1]:
        return q
    heads = indi.heads
    qh = torch.stack([heads.heads[g](trunk_features(net, heads.mask(x, g))) for g in range(len(heads.heads))]).mean(0)
    return (1 - indi.controller.lam) * q + indi.controller.lam * qh


def run_episode_mix(task, env, net, seed, device, epsilon, replay=None, random_actions=False, reward_scale=1.0):
    """searl_discrete.run_episode with the Q-values taken from policy_q (identical when lam == 0)."""
    obs, _ = env.reset(seed=int(seed))
    done, ret, n = False, 0.0, 0
    factored = isinstance(env.action_space, sd.gym.spaces.MultiDiscrete)
    n_actions = None if factored else env.action_space.n
    with torch.no_grad():
        while not done:
            if random_actions or (epsilon > 0 and np.random.rand() < epsilon):
                a = env.action_space.sample() if factored else np.random.randint(n_actions)
            else:
                q = policy_q(net, torch.from_numpy(obs.astype(np.float32))[None])
                a = q.view(-1, 3).argmax(1).cpu().numpy() if factored else int(q.argmax(1).item())
            nobs, r, term, trunc, _ = env.step(a)
            if replay is not None:
                replay.add(obs, a, r * reward_scale, nobs, float(term))
            obs, ret, n, done = nobs, ret + r, n + 1, term or trunc
    return ret, n


class CGIndividual(sd.DQNIndividual):
    def __init__(self, actor, rl_config, index, heads=None, controller=None):
        super().__init__(actor, rl_config, index)
        self.heads, self.controller = heads, controller

    def clone(self, index=None):
        c = CGIndividual(self.actor.clone(), copy.deepcopy(self.rl_config), self.index if index is None else index,
                         copy.deepcopy(self.heads), copy.deepcopy(self.controller))
        c.fitness = copy.deepcopy(self.fitness)
        c.train_log = copy.deepcopy(self.train_log)
        return c


class CGMutations(sd.MlpDqnMutations):
    """Parent's operator menu plus `controller`; with cfg.mutation.controller == 0 this is the parent's menu."""

    def controller_mutation(self, individual):
        key = individual.controller.mutate(self.rng, self.cfg.mutation.mutation_sd)
        individual.train_log["mutation"] = "controller_" + key
        return individual

    def mutation(self, population):
        if not getattr(self.cfg.mutation, "controller", 0):
            return super().mutation(population)
        options, proba = [], []
        for name, fn in (("no_mutation", self.no_mutation), ("architecture", self.architecture_mutate),
                         ("parameters", self.parameter_mutation), ("activation", self.activation_mutation),
                         ("rl_hyperparam", self.rl_hyperparam_mutation), ("controller", self.controller_mutation)):
            p = getattr(self.cfg.mutation, name, 0)
            if p:
                options.append(fn)
                proba.append(float(p))
        proba = np.array(proba) / np.sum(proba)
        choice = self.rng.choice(options, len(population), p=proba)
        return [m(i) for m, i in zip(choice, population)]


def pairwise_agreement(greedy, n_choices):
    """Mean over states (and joints) of the share of head pairs that picked the same action.
    With vote counts c_a per action: sum_a c_a (c_a - 1) / (K (K - 1)); for K = 2 this is exactly 1[a_1 = a_2]."""
    votes = F.one_hot(torch.stack(greedy), n_choices).sum(0).float()
    k = len(greedy)
    return float((votes * (votes - 1)).sum(-1).div(k * (k - 1)).mean())


def _sample_states(replay, batch):
    idx = _SAMPLE_RNG.randint(0, replay.n, size=batch)
    return idx


def confidence(indi, replay):
    """m = mean pairwise agreement of the aux heads' greedy actions over replay states."""
    indi.heads.resize(indi.actor.hidden_size[-1])
    idx = _sample_states(replay, indi.rl_config.batch_size)
    s = torch.from_numpy(replay.s[idx]).float()
    with torch.no_grad():
        q = [indi.heads.heads[g](trunk_features(indi.actor, indi.heads.mask(s, g)))
             for g in range(len(indi.heads.heads))]
        if replay.a.ndim == 2:  # one 3-way choice per joint: agreement per joint, averaged
            q = [x.view(-1, replay.a.shape[1], 3) for x in q]
        greedy = [x.argmax(-1) for x in q]
    return pairwise_agreement(greedy, q[0].shape[-1])


def train_heads(indi, replay, iterations, lr):
    """Fit each head on the full network's TD target from group-masked inputs; trunk detached."""
    indi.heads.resize(indi.actor.hidden_size[-1])
    a, net = indi.rl_config, indi.actor
    opt = torch.optim.Adam(indi.heads.parameters(), lr=lr)
    for _ in range(max(1, int(iterations * indi.controller.aux_fraction))):
        idx = _sample_states(replay, a.batch_size)
        s, act = torch.from_numpy(replay.s[idx]).float(), torch.from_numpy(replay.a[idx])
        r, s2, d = (torch.from_numpy(replay.r[idx]), torch.from_numpy(replay.s2[idx]).float(),
                    torch.from_numpy(replay.d[idx]))
        factored = act.ndim == 2
        with torch.no_grad():
            if factored:  # same per-joint double-DQN target as train_mlp
                online = net(s2).view(-1, act.shape[1], 3)
                nxt = net(s2).view(-1, act.shape[1], 3)
                next_v = nxt.gather(-1, online.argmax(-1, keepdim=True)).squeeze(-1).mean(-1)
            else:
                next_v = net(s2).max(1)[0]
            y = r + a.gamma * (1.0 - d) * next_v
        loss = 0.0
        for g in range(len(indi.heads.heads)):
            with torch.no_grad():
                feat = trunk_features(net, indi.heads.mask(s, g))
            qg = indi.heads.heads[g](feat)
            if factored:
                q = qg.view(-1, act.shape[1], 3).gather(-1, act.long().unsqueeze(-1)).squeeze(-1).mean(-1)
            else:
                q = qg.gather(1, act[:, None]).squeeze(1)
            loss = loss + F.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


_plain_train_mlp = sd.train_mlp


def train_mlp_cg(indi, replay, iterations, device):
    base_lr = indi.rl_config.lr_actor
    m = confidence(indi, replay)
    lr_eff = base_lr * indi.controller.scale(m)
    indi.rl_config.lr_actor = lr_eff
    _plain_train_mlp(indi, replay, iterations, device)
    indi.rl_config.lr_actor = base_lr
    train_heads(indi, replay, iterations, base_lr)
    register(indi)
    CONFGATE_LOG.append({"individual": indi.index, "mutation": indi.train_log["mutation"], "m": m,
                         "lr_base": base_lr, "lr_eff": lr_eff, **indi.controller.coefficients()})
    return indi


def run(env_id, seed, pop, frames, controller=True, mutations="full", heads="groups", log=print, task="gym", lanes=5,
        solutions=8, eval_episodes=3, eval_seed0=None, eval_every_frames=None, mix=False, lam_init=0.2, lam_mutable=True,
        neutral_lam=False):
    global NEUTRAL_LAM
    NEUTRAL_LAM = bool(neutral_lam)
    import fastrand
    assert not (mix and task == "bike"), "ConfGate-mix is implemented for the gym tasks (CartPole / HalfCheetah) only"
    seed_confgate(seed)
    fastrand.pcg32_seed(seed)  # weight mutation uses fastrand's own stream; seed it so runs are reproducible
    CONFGATE_LOG.clear()

    probe = sd.make_env(task, env_id, seed, lanes, solutions)
    obs_dim = probe.observation_space.shape[0]
    base_groups = bike_groups(lanes) if task == "bike" else OBS_GROUPS.get(env_id)
    groups = base_groups if heads == "groups" else {f"obs{i}": [i] for i in range(obs_dim)}  # per_obs: one head per entry
    sp = probe.action_space
    n_actions = int(sp.nvec.sum()) if hasattr(sp, "nvec") else sp.n
    probe.close()

    # controller = lr gate on/off. mix = heads also act (lam, mutated by SEARL unless lam_mutable is False).
    ctrl_kwargs = {} if controller else {"lr_floor": 1.0, "lr_slope": 0.0}
    ctrl_kwargs.update(gate_mutable=bool(controller), lam=float(lam_init) if mix else 0.0, lam_mutable=bool(mix and lam_mutable))
    controller_op = bool(controller or (mix and lam_mutable))

    class _Indi(CGIndividual):
        def __init__(self, actor, rl_config, index, heads=None, controller_=None):
            if heads is None:
                heads = AuxHeads(actor.hidden_size[-1], n_actions, groups)
                controller_ = LRController(**ctrl_kwargs)
            super().__init__(actor, rl_config, index, heads, controller_)

    base_make_cfg = sd.make_cfg

    def make_cfg(task, seed_, pop_, frames_):
        cfg = base_make_cfg(task, seed_, pop_, frames_)
        cfg.mutation.set_attr("controller", 0.2 if controller_op else 0)
        if mutations == "lr_only":
            for k in ("parameters", "architecture", "activation"):
                cfg.mutation.set_attr(k, 0)
        return cfg

    base_run_episode = sd.run_episode
    sd.make_cfg, sd.DQNIndividual, sd.MlpDqnMutations, sd.train_mlp = make_cfg, _Indi, CGMutations, train_mlp_cg
    if mix:
        sd.run_episode = run_episode_mix
    _REG.clear()
    try:
        history = sd.run_searl_discrete(task, env_id, seed, pop, frames, log=log, lanes=lanes, solutions=solutions,
                                        eval_episodes=eval_episodes, eval_seed0=eval_seed0, eval_every_frames=eval_every_frames)
    finally:
        sd.run_episode = base_run_episode
        sd.make_cfg, sd.DQNIndividual, sd.MlpDqnMutations, sd.train_mlp = (
            base_make_cfg, CGIndividual.__mro__[1], CGMutations.__mro__[1], _plain_train_mlp)
    return history, pd.DataFrame(CONFGATE_LOG)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--task", choices=["gym", "bike"], default="gym")
    p.add_argument("--lanes", type=int, default=5, help="bike only")
    p.add_argument("--solutions", type=int, default=8, help="bike only")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-seed0", type=int, default=None, help="if set, greedy test episodes use fixed seeds eval-seed0 + k")
    p.add_argument("--eval-every-frames", type=int, default=None, help="bike + eval-seed0: evaluate about every this many frames")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--frames", type=int, default=60_000)
    p.add_argument("--no-controller", action="store_true", help="lr gate off (floor 1, slope 0, no gate mutation)")
    p.add_argument("--mix", action="store_true", help="ConfGate-mix: the heads also take part in acting (Q_mix = (1-lam) Q_full + lam mean_k Q_k)")
    p.add_argument("--lam-init", type=float, default=0.2)
    p.add_argument("--fixed-lam", action="store_true", help="do not let SEARL mutate lam (control runs)")
    p.add_argument("--neutral-lam", action="store_true", help="control: lam mutates and is inherited but is not used to act (selection blind to it)")
    p.add_argument("--mutations", choices=["full", "lr_only"], default="full")
    p.add_argument("--heads", choices=["groups", "per_obs"], default="groups",
                   help="groups: two heads on observation groups; per_obs: one head per observation entry")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hist, cglog = run(a.env, a.seed, a.pop, a.frames, controller=not a.no_controller, mutations=a.mutations, heads=a.heads, task=a.task,
                      lanes=a.lanes, solutions=a.solutions, eval_episodes=a.eval_episodes, eval_seed0=a.eval_seed0,
                      eval_every_frames=a.eval_every_frames, mix=a.mix, lam_init=a.lam_init, lam_mutable=not a.fixed_lam, neutral_lam=a.neutral_lam,
                      log=lambda m: print(m, flush=True))
    h = pd.DataFrame(hist)
    h.to_csv(os.path.join(a.out, "history.csv"), index=False)
    if "test_return" in h:  # shared format (rows without an evaluation hold NaN and are dropped) of the bike grid: one greedy-return curve per run
        h[["num_frames", "test_return"]].dropna().rename(columns={"num_frames": "global_step", "test_return": "greedy_return"}).to_csv(
            os.path.join(a.out, "eval.csv"), index=False)
    cglog.to_csv(os.path.join(a.out, "confgate_log.csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
