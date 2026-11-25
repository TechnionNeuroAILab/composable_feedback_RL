#!/usr/bin/env python
"""SEARL-DQN on discrete HalfCheetah-v4 where each individual's Q-function is a weighted ensemble of independent heads.

Heads (17 + 1 = 18 on HalfCheetah-v4, which has a 17-number observation):
  * full   -- one head on the whole observation (the plain SEARL Q-network)
  * obs<i> -- one head per observation entry; its input is that single number only (input size 1)
Every head is its own EvolvableMLP (own weights, own trunk, nothing shared) and outputs Q-values for all 18 actions
(6 joints x {-1, 0, +1}).

The Q-values that choose the action (epsilon-greedy fitness episodes, greedy test return, the double-DQN target) are

    Q(s, .) = sum_k w_k Q_k(s[inputs_k], .)

w is a per-individual vector, one weight per head, stored with the network (saved, cloned, inherited by offspring).
It is derived from a per-individual on/off mask m (one flag per head): w_k = m_k / sum(m).

Model A (default): every flag stays on, so w_k = 1 / K and Q is the plain mean of all heads. Nothing changes w.
Model B (--toggle-heads): SEARL gets a 6th mutation operator, `head_toggle` (same probability as the other five, 1/6):
  pick one head uniformly, flip its flag (on -> off or off -> on); a flip that would leave no head on picks another head.
  Only the heads that are ON vote (equal share); the masked Q is used for acting AND for the TD target. Switched-off
  heads keep training on that target (they only stop voting), so a head that is switched back on is up to date.
  --protect-full keeps the full-observation head always on (it is never toggled).
Model C (--weight-heads): as B, but the flag is a strength g_k anywhere in [0, 1] instead of 0 or 1 (m above is g; all
  g_k start at 1, so A, B and C are the same individual at generation 0). w_k = g_k / sum(g), used for acting AND for
  the TD target; every head keeps training on that target whatever its g_k. The 6th operator is `head_weight` (1/6):
  pick one head uniformly and give it a new g_k (--weight-mut):
    resample (default)  g_k ~ U[0, 1): the continuous version of B's flip (in practice never exactly 0 or 1)
    step                g_k <- clip(g_k + N(0, --weight-sigma), 0, 1): local; reaches exactly 0 (head off) and 1
    add                 g_k <- min(1, g_k + --weight-add) (default 0.02): the picked head only ever GAINS strength,
                        a fixed amount per mutation; heads already at 1 are not picked. Nothing lowers a g_k, so
                        selection (which lineages survive) is the only brake.
  A draw that would leave sum(g) = 0 is repeated. --protect-full keeps the full head's g at 1.
  --init-gates full: start from the competent end instead: g_full = 1 and every single-observation head at 0, so at
  generation 0 the individual acts and builds its target exactly like plain SEARL (the other heads still train on
  that target). With `add`, each head mutation then gives one single-observation head 0.02 more strength (the first
  one is 0.02 / 1.02 = 2% of the vote). With no selection, a lineage gets one head mutation per 6 generations, so
  the summed strength of the single-observation heads grows by 0.02 / 6 per generation.
  Only the ratios of g matter (g and c * g give the same w), so read w_k or n_eff, not g_k alone, as "how much a
  head votes". With no selection at all, `resample` moves every g_k from 1 towards U[0, 1] (mean 0.5).

Training (--loss):
  own  (default) every head regresses on the SAME double-DQN target, built from the ensemble:
           a*  = argmax_a Q(s', a)                       (online ensemble)
           y   = r + gamma * (1 - d) * Q_target(s', a*)   (target ensemble; per-joint mean as in searl_discrete.py)
           loss = sum_k Huber(Q_k(s, a), y)
       so each head is a Q-estimate from its own view; their mean is what acts.
  mix  one Huber loss on the ensemble Q(s, a) only; the heads are then parts of one Q-function, not separate estimates.
--target full: the double-DQN target (a* and Q_target) is built from the FULL head alone instead of the ensemble, so
  every head (the full head too) learns the value of the full head's greedy policy; the ensemble still acts.

SEARL itself is unchanged (searl_discrete.py loop, tournament selection with elitism, shared replay, evaluation, the
rl_hyperparam / lr mutation). Each mutation operator acts on the individual, which means on every head:
  parameters    SEARL's weight noise, drawn separately in every head
  activation    one new activation, given to every head
  architecture  one new layer, or one node change (same layer and size), given to every head
The patching is done by rebinding names in searl_discrete (no edits to that file), as in searl_discrete_confgate.py.

Outputs in --out: history.csv (as searl_discrete.py), heads_log.csv (per training phase, individual and head: on/off,
weight, mean Huber loss to the target, and how often the head's greedy action per joint equals the ensemble's on the last
batch) and population_log.csv (every evaluated individual, every generation: index, parent, the mutation that produced
it, fitness, number of heads on, the names of the heads that are off, and the on/off string 'm' + one 0/1 per head in
the order full, obs0, ..., obs16; 'on' means g_k > 0; n_eff = 1 / sum_k w_k^2 is the effective number of voting heads,
18 when all vote equally and 1 when one head has all the weight; w_full is the full head's share of the vote).
Per-head strengths, for every model (0/1 in A and B):
  gates_log.csv    one row per evaluated individual per generation: generation, individual, parent, mutation, fitness,
                   n_on, n_eff, then g_<head> and w_<head> for every head. Generation g is epoch g of history.csv
                   (same row order as its num_frames); fitness is that epoch's training-episode return.
  gates_final.csv  the population at the end of the run, one row per (individual, head): fitness, is_best (the
                   individual whose greedy test return is the last row of history.csv), gate g_k, weight w_k.
  config.json      the command-line arguments of the run.
population_log.csv and gates_log.csv include the last epoch (its population is evaluated but never goes through
selection). plot_obs_heads_weights.py turns gates_log.csv into trend figures and tables.
"""
import argparse
import copy
import json
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
from evolvable_mlp import EvolvableMLP  # noqa: E402

ENV_ID = "HalfCheetah-v4"
N_BINS = 3   # actions per joint
LOSS = "own"
TARGET = "ensemble"   # --target full: double-DQN target from the full head only (the ensemble still acts)
TOGGLE, PROTECT_FULL = False, False
WEIGHT, WEIGHT_MUT, WEIGHT_SIGMA = False, "resample", 0.3   # model C (--weight-heads)
WEIGHT_ADD, INIT_GATES = 0.02, "equal"                      # --weight-mut add step; --init-gates full
HEADS_LOG, POP_LOG, GATES_LOG = [], [], []
LAST_POP = []         # the individuals selection returned last (the same objects are then mutated, trained, evaluated)
Q_DUMP = []           # per phase: Q of every head on the same 32 replay states, [K, 32, 18] (see train_obs_heads)
_PHASE, _GEN = [0], [0]
OUT_DIR = [None]      # when set, logs are rewritten after every generation (so a stopped run keeps its logs)


def flush_logs():
    if OUT_DIR[0] is None:
        return
    pd.DataFrame(HEADS_LOG).to_csv(os.path.join(OUT_DIR[0], "heads_log.csv"), index=False)
    pd.DataFrame(POP_LOG).to_csv(os.path.join(OUT_DIR[0], "population_log.csv"), index=False)
    pd.DataFrame(GATES_LOG).to_csv(os.path.join(OUT_DIR[0], "gates_log.csv"), index=False)
    if Q_DUMP:
        np.savez_compressed(os.path.join(OUT_DIR[0], "q_dump.npz"), q=np.stack([d["q"] for d in Q_DUMP]),
                            phase=np.array([d["phase"] for d in Q_DUMP]), individual=np.array([d["individual"] for d in Q_DUMP]),
                            mask=np.stack([d["mask"] for d in Q_DUMP]))


# ---------------------------------------------------------------------------------------------------- network
class ObsHeadsQ(nn.Module):
    """Q(s, .) = sum_k weights[k] * heads[k](s[inputs[k]]). Drop-in for EvolvableMLP inside searl_discrete.py
    (forward, init_dict, clone, state_dict); the mutation operators below edit the heads."""

    def __init__(self, num_inputs, num_outputs, hidden_size=(128,), activation="relu", output_activation=None,
                 layer_norm=False, output_vanish=False, full_head=True, head_dicts=None, mask=None):
        super().__init__()
        self.num_inputs, self.num_outputs, self.full_head = num_inputs, num_outputs, full_head
        self.inputs = ([None] if full_head else []) + [[i] for i in range(num_inputs)]   # None = whole observation
        self.names = (["full"] if full_head else []) + [f"obs{i}" for i in range(num_inputs)]
        if head_dicts is None:
            head_dicts = [dict(num_inputs=num_inputs if ix is None else len(ix), num_outputs=num_outputs,
                               hidden_size=list(hidden_size), activation=activation,
                               output_activation=output_activation, layer_norm=layer_norm) for ix in self.inputs]
        self.heads = nn.ModuleList([EvolvableMLP(output_vanish=output_vanish, **copy.deepcopy(d)) for d in head_dicts])
        k = len(self.heads)
        m = torch.ones(k) if mask is None else torch.as_tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", m.clone())
        self.register_buffer("weights", m / m.sum())

    def set_mask(self, mask):
        self.mask.copy_(torch.as_tensor(mask, dtype=torch.float32))
        self.weights.copy_(self.mask / self.mask.sum())

    def mask_str(self):
        """'m' + one 0/1 per head, 1 = the head votes (g > 0); the prefix keeps the leading zeros in the CSV."""
        return "m" + "".join(str(int(v > 0)) for v in self.mask.tolist())

    def n_eff(self):
        """Effective number of voting heads, 1 / sum_k w_k^2 with w normalised to sum 1."""
        w = self.weights / self.weights.sum()
        return float(1.0 / (w ** 2).sum())

    def off_heads(self):
        return ";".join(n for n, v in zip(self.names, self.mask.tolist()) if not v)

    @property
    def activation(self):
        return self.heads[0].activation

    @property
    def hidden_size(self):
        return self.heads[0].hidden_size

    def head_q(self, x):
        """[K, B, num_outputs]: every head's Q-values."""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        return torch.stack([h(x if ix is None else x[..., ix]) for h, ix in zip(self.heads, self.inputs)])

    def forward(self, x):
        return torch.einsum("k,k...->...", self.weights, self.head_q(x))

    @property
    def init_dict(self):
        return {"num_inputs": self.num_inputs, "num_outputs": self.num_outputs, "full_head": self.full_head,
                "head_dicts": [copy.deepcopy(h.init_dict) for h in self.heads], "mask": self.mask.tolist()}

    def clone(self):
        c = ObsHeadsQ(**copy.deepcopy(self.init_dict))
        c.load_state_dict(self.state_dict())
        return c


def _rebuild_head(head, **changes):
    d = head.init_dict
    d.update(changes)
    new = EvolvableMLP(output_vanish=False, **d)
    new.load_state_dict(head.state_dict())
    return new


# --------------------------------------------------------------------------------------------------- mutation
class ObsHeadsMutations(sd.MlpDqnMutations):
    """SEARL's operators applied to the individual, i.e. to every head (see module docstring), plus `head_toggle`
    (model B) or `head_weight` (model C). With neither set in cfg.mutation the menu and its random draws are exactly
    the parent's (model A); B and C draw the operator from the same 6-entry menu."""

    def mutation(self, population):
        if not (getattr(self.cfg.mutation, "head_toggle", 0) or getattr(self.cfg.mutation, "head_weight", 0)):
            return super().mutation(population)
        options, proba = [], []
        for name, fn in (("no_mutation", self.no_mutation), ("architecture", self.architecture_mutate),
                         ("parameters", self.parameter_mutation), ("activation", self.activation_mutation),
                         ("rl_hyperparam", self.rl_hyperparam_mutation), ("head_toggle", self.head_toggle_mutation),
                         ("head_weight", self.head_weight_mutation)):
            p = getattr(self.cfg.mutation, name, 0)
            if p:
                options.append(fn)
                proba.append(float(p))
        proba = np.array(proba) / np.sum(proba)
        choice = self.rng.choice(options, len(population), p=proba)
        return [fn(ind) for fn, ind in zip(choice, population)]

    def head_toggle_mutation(self, individual):
        net = individual.actor
        mask = net.mask.clone()
        cand = [k for k, n in enumerate(net.names) if not (PROTECT_FULL and n == "full")]
        if mask.sum() <= 1:   # never switch off the last head that is on
            cand = [k for k in cand if mask[k] == 0]
        k = int(self.rng.choice(cand))
        mask[k] = 1.0 - mask[k]
        net.set_mask(mask)
        individual.train_log["mutation"] = f"head_toggle_{net.names[k]}_{'on' if mask[k] else 'off'}"
        return individual

    def head_weight_mutation(self, individual):
        net = individual.actor
        g = net.mask.clone()
        cand = [k for k, n in enumerate(net.names) if not (PROTECT_FULL and n == "full")]
        if WEIGHT_MUT == "add":   # only heads that can still gain
            cand = [k for k in cand if float(g[k]) < 1.0]
            if not cand:
                individual.train_log["mutation"] = "head_weight_none"
                return individual
        k = int(self.rng.choice(cand))
        old = float(g[k])
        while True:   # repeated only if every strength would be 0 (possible with `step`, which clips to exactly 0)
            if WEIGHT_MUT == "resample":
                new = float(self.rng.uniform(0.0, 1.0))
            elif WEIGHT_MUT == "add":
                new = min(1.0, old + WEIGHT_ADD)
            else:
                new = float(np.clip(old + self.rng.normal(0.0, WEIGHT_SIGMA), 0.0, 1.0))
            g[k] = new
            if float(g.sum()) > 0:
                break
        net.set_mask(g)
        individual.train_log["mutation"] = f"head_weight_{net.names[k]}_{old:.3f}_to_{new:.3f}"
        return individual

    def parameter_mutation(self, individual):
        for h in individual.actor.heads:
            self.classic_parameter_mutation(h)
        individual.train_log["mutation"] = "classic_parameter"
        return individual

    def activation_mutation(self, individual):
        net = individual.actor.clone()
        new = self.rng.choice([a for a in ("relu", "elu", "tanh") if a != net.activation])
        for k, h in enumerate(net.heads):
            net.heads[k] = _rebuild_head(h, activation=new)
        individual.actor = net
        individual.train_log["mutation"] = "activation"
        return individual

    def architecture_mutate(self, individual):
        net = individual.actor.clone()
        # all heads share one architecture; at the 3-layer limit EvolvableMLP.add_layer would fall back to a RANDOM
        # add_node per head, so take the node branch explicitly to keep the heads in step
        if self.rng.uniform(0, 1) < self.cfg.mutation.new_layer_prob and len(net.hidden_size) < 3:
            for h in net.heads:
                h.add_layer()
            individual.train_log["mutation"] = "architecture_new_layer"
        else:
            node = net.heads[0].add_node()
            for h in net.heads[1:]:
                h.add_node(**node)
            individual.train_log["mutation"] = "architecture_new_node"
        individual.actor = net
        return individual


# --------------------------------------------------------------------------------------------------- training
def train_obs_heads(indi, replay, iterations, device):
    a, net = indi.rl_config, indi.actor
    target = net.clone()
    opt = sd.get_optimizer(a.optimizer)(net.parameters(), lr=a.lr_actor)
    net.train()
    k = len(net.heads)
    loss_sum = torch.zeros(k)
    for _ in range(iterations):
        s, act, r, s2, d = replay.sample(a.batch_size, "cpu")
        j = act.shape[1]
        with torch.no_grad():
            if TARGET == "full":   # head 0 is the full-observation head (its input is the whole observation)
                online = net.heads[0](s2).view(-1, j, N_BINS)
                nxt = target.heads[0](s2).view(-1, j, N_BINS)
            else:
                online = net(s2).view(-1, j, N_BINS)
                nxt = target(s2).view(-1, j, N_BINS)
            next_v = nxt.gather(-1, online.argmax(-1, keepdim=True)).squeeze(-1).mean(-1)
            y = r + a.gamma * (1.0 - d) * next_v
        qk = net.head_q(s).view(k, -1, j, N_BINS)                                      # [K, B, J, 3]
        idx = act.long()[None, :, :, None].expand(k, -1, -1, 1)
        taken_k = qk.gather(-1, idx).squeeze(-1).mean(-1)                             # [K, B]
        per_head = F.smooth_l1_loss(taken_k, y.expand_as(taken_k), reduction="none").mean(1)   # [K]
        if LOSS == "own":
            loss = per_head.sum()
        else:
            loss = F.smooth_l1_loss((net.weights[:, None] * taken_k).sum(0), y)
        opt.zero_grad()
        loss.backward()
        for h in net.heads:   # per head, so each head sees the same clipping as the plain SEARL network
            torch.nn.utils.clip_grad_norm_(h.parameters(), 10.0)
        opt.step()
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - a.tau).add_(a.tau * p.data)
        loss_sum += per_head.detach()

    with torch.no_grad():   # agreement of every head's greedy action with the ensemble's, per joint, last batch
        qk = net.head_q(s).view(k, -1, j, N_BINS)
        ens = (net.weights[:, None, None, None] * qk).sum(0).argmax(-1)
        agree = (qk.argmax(-1) == ens[None]).float().mean((1, 2))
        # Q-value statistics per head on the last batch: level, spread across states (state sensitivity),
        # and per-state agreement with the full head (correlation over states x action-bins)
        flat = qk.reshape(k, qk.shape[1], -1)                                           # [K, B, J*3]
        q_mean = flat.mean((1, 2))
        q_state_std = flat.std(1).mean(1)
        q_max = qk.max(-1).values.mean((1, 2))
        c = flat - flat.mean(1, keepdim=True)
        c = c.reshape(k, -1)
        corr_full = (c @ c[0]) / (c.norm(dim=1) * c[0].norm() + 1e-8)
        # per-state preference over the 3 torques, centred per state (removes the common level): what drives argmax
        adv = (qk - qk.mean(-1, keepdim=True)).reshape(k, -1)
        adv_corr_full = (adv @ adv[0]) / (adv.norm(dim=1) * adv[0].norm() + 1e-8)
    _PHASE[0] += 1
    Q_DUMP.append({"phase": _PHASE[0], "individual": indi.index, "q": flat[:, :32].numpy().astype(np.float32),
                   "mask": net.mask.numpy().copy()})
    for h in range(k):
        HEADS_LOG.append({"phase": _PHASE[0], "individual": indi.index, "mutation": indi.train_log["mutation"],
                          "lr": a.lr_actor, "head": net.names[h], "on": int(net.mask[h] > 0), "gate": float(net.mask[h]),
                          "weight": float(net.weights[h]),
                          "loss": float(loss_sum[h] / max(1, iterations)), "agree": float(agree[h]),
                          "q_mean": float(q_mean[h]), "q_max": float(q_max[h]), "q_state_std": float(q_state_std[h]),
                          "corr_full": float(corr_full[h]), "adv_corr_full": float(adv_corr_full[h])})
    return indi


# -------------------------------------------------------------------------------------------------------- run
class LoggingSelection(sd.TournamentSelection):
    """Unchanged tournament selection; first records every evaluated individual (no random draws)."""

    def select(self, population):
        log_population(population)
        flush_logs()
        elite, new_population = super().select(population)
        LAST_POP[:] = new_population
        return elite, new_population


def log_population(population):
    """One row per individual in population_log and gates_log; call once per epoch, after the fitness episodes."""
    _GEN[0] += 1
    for ind in population:
        net = ind.actor
        row = {"generation": _GEN[0], "individual": ind.index, "parent": ind.train_log["parent_index"],
               "mutation": ind.train_log["mutation"], "fitness": float(ind.fitness[-1]),
               "n_on": int((net.mask > 0).sum()), "n_eff": net.n_eff()}
        w = (net.weights / net.weights.sum()).tolist()
        POP_LOG.append({**row, "w_full": w[0] if net.names[0] == "full" else float("nan"), "off": net.off_heads(),
                        "mask": net.mask_str()})
        GATES_LOG.append({**row, **{f"g_{n}": v for n, v in zip(net.names, net.mask.tolist())},
                          **{f"w_{n}": v for n, v in zip(net.names, w)}})


def final_gates():
    """The population at the end of the run, one row per (individual, head)."""
    if not LAST_POP:
        return pd.DataFrame()
    best = int(np.argmax([ind.fitness[-1] for ind in LAST_POP]))
    rows = []
    for i, ind in enumerate(LAST_POP):
        net = ind.actor
        w = (net.weights / net.weights.sum()).tolist()
        for n, g, wk in zip(net.names, net.mask.tolist(), w):
            rows.append({"individual": ind.index, "parent": ind.train_log["parent_index"],
                         "fitness": float(ind.fitness[-1]), "is_best": int(i == best), "n_eff": net.n_eff(),
                         "head": n, "gate": g, "weight": wk})
    return pd.DataFrame(rows)


def run(seed, pop, frames, loss="own", full_head=True, toggle=False, protect_full=False, log=print, target="ensemble",
        weight=False, weight_mut="resample", weight_sigma=0.3, weight_add=0.02, init_gates="equal"):
    global LOSS, TOGGLE, PROTECT_FULL, TARGET, WEIGHT, WEIGHT_MUT, WEIGHT_SIGMA, WEIGHT_ADD, INIT_GATES
    assert target == "ensemble" or full_head, "--target full needs the full head"
    assert not (toggle and weight), "model B (--toggle-heads) and model C (--weight-heads) are exclusive"
    assert weight_mut in ("resample", "step", "add")
    assert init_gates == "equal" or (init_gates == "full" and full_head), "--init-gates full needs the full head"
    assert 0 < weight_add <= 1
    WEIGHT, WEIGHT_MUT, WEIGHT_SIGMA = bool(weight), weight_mut, float(weight_sigma)
    WEIGHT_ADD, INIT_GATES = float(weight_add), init_gates
    TARGET = target
    import fastrand
    LOSS, TOGGLE, PROTECT_FULL = loss, bool(toggle), bool(protect_full)
    fastrand.pcg32_seed(seed)   # weight mutation draws from fastrand's own stream
    HEADS_LOG.clear()
    POP_LOG.clear()
    GATES_LOG.clear()
    LAST_POP.clear()
    Q_DUMP.clear()
    _PHASE[0] = _GEN[0] = 0

    def make_net(**kw):
        # --init-gates full: only the full head votes at the start (head 0); offspring inherit the mask by cloning
        mask = [1.0] + [0.0] * kw["num_inputs"] if INIT_GATES == "full" else None
        return ObsHeadsQ(full_head=full_head, mask=mask, **kw)

    base_make_cfg = sd.make_cfg

    def make_cfg(*args):
        cfg = base_make_cfg(*args)
        cfg.mutation.set_attr("head_toggle", 0.2 if TOGGLE else 0)   # model B: 6 operators, 1/6 each
        cfg.mutation.set_attr("head_weight", 0.2 if WEIGHT else 0)   # model C: 6 operators, 1/6 each
        return cfg

    saved = sd.EvolvableMLP, sd.MlpDqnMutations, sd.train_mlp, sd.make_cfg, sd.TournamentSelection
    (sd.EvolvableMLP, sd.MlpDqnMutations, sd.train_mlp, sd.make_cfg,
     sd.TournamentSelection) = make_net, ObsHeadsMutations, train_obs_heads, make_cfg, LoggingSelection
    try:
        history = sd.run_searl_discrete("gym", ENV_ID, seed, pop, frames, log=log)
    finally:
        sd.EvolvableMLP, sd.MlpDqnMutations, sd.train_mlp, sd.make_cfg, sd.TournamentSelection = saved
    if LAST_POP and len(LAST_POP[0].fitness) == len(history):   # the last epoch: evaluated, but never selected
        log_population(LAST_POP)
    return history, pd.DataFrame(HEADS_LOG), pd.DataFrame(POP_LOG)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--frames", type=int, default=1_000_000)
    p.add_argument("--loss", choices=["own", "mix"], default="own",
                   help="own: every head fits the ensemble's TD target; mix: one loss on the weighted ensemble")
    p.add_argument("--no-full-head", action="store_true", help="only the per-observation heads (17 heads)")
    p.add_argument("--toggle-heads", action="store_true",
                   help="model B: SEARL switches heads on/off (head_toggle operator); only heads that are on vote")
    p.add_argument("--weight-heads", action="store_true",
                   help="model C: SEARL changes a strength g_k in [0, 1] per head (head_weight operator); w = g / sum(g)")
    p.add_argument("--weight-mut", choices=["resample", "step", "add"], default="resample",
                   help="model C: resample = new g_k ~ U[0, 1); step = g_k + N(0, --weight-sigma), clipped to [0, 1]; "
                        "add = g_k + --weight-add, capped at 1 (a head only gains)")
    p.add_argument("--weight-sigma", type=float, default=0.3, help="model C, --weight-mut step: step size")
    p.add_argument("--weight-add", type=float, default=0.02, help="model C, --weight-mut add: strength gained per mutation")
    p.add_argument("--init-gates", choices=["equal", "full"], default="equal",
                   help="equal: every head starts at g = 1; full: the full head at 1 and every other head at 0 "
                        "(the individual starts as plain SEARL)")
    p.add_argument("--protect-full", action="store_true",
                   help="models B, C: the full-observation head is never switched off / always keeps g = 1")
    p.add_argument("--target", choices=["ensemble", "full"], default="ensemble",
                   help="ensemble: double-DQN target from the voting ensemble; full: from the full-observation head only")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.toggle_heads and a.weight_heads:
        p.error("--toggle-heads (model B) and --weight-heads (model C) are exclusive")
    os.makedirs(a.out, exist_ok=True)
    OUT_DIR[0] = a.out
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist, heads, popl = run(a.seed, a.pop, a.frames, loss=a.loss, full_head=not a.no_full_head, toggle=a.toggle_heads,
                            protect_full=a.protect_full, target=a.target, log=lambda m: print(m, flush=True),
                            weight=a.weight_heads, weight_mut=a.weight_mut, weight_sigma=a.weight_sigma,
                            weight_add=a.weight_add, init_gates=a.init_gates)
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
    heads.to_csv(os.path.join(a.out, "heads_log.csv"), index=False)
    popl.to_csv(os.path.join(a.out, "population_log.csv"), index=False)
    flush_logs()
    final_gates().to_csv(os.path.join(a.out, "gates_final.csv"), index=False)
    print("wrote", os.path.join(a.out, "history.csv"))


if __name__ == "__main__":
    main()
