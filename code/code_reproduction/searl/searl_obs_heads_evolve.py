#!/usr/bin/env python
"""SEARL shapes the 18 observation heads (discrete HalfCheetah-v4). Built on searl_obs_heads_halfcheetah.py (imported,
not edited): same independent heads (full + one per observation entry), same per-head training on one double-DQN
target, same logs. Two new, independent mechanisms, each a SEARL mutation operator (probability 0.2 before
normalisation, like SEARL's own five):

Idea 2 -- SEARL evolves what each head sees (--evolve-inputs)
  Every observation head k has an input set I_k (initially {k}); the full head always sees everything.
  `head_input` mutation: pick one observation head; add one observation entry it does not see, or remove one (never
  the last), with probability 1/2 each when both are possible. The head's first layer is resized: weights of entries
  it keeps are copied, a new entry gets a zero column, so after an `add` the head computes exactly the same Q and
  learns to use the new entry by training (a `remove` drops that entry's column, which does change the head's Q). Inherited by offspring like any weight.

Idea 3 -- SEARL evolves how much each head's opinion changes training (--gate); the heads never act
  Each individual has a gain g_k >= 0 per observation head (initially 1). Before every training phase, on one replay
  batch, a_k = share of (state, joint) where head k's greedy torque equals the full head's, and
      c = sum_k g_k a_k / sum_k g_k                          (weighted agreement with the full head)
      lr_eff = lr * (lr_floor + lr_slope * c)              (lr_floor = lr_slope = 0.5, the ConfGate defaults, fixed)
  is that phase's learning rate (SEARL's own lr is restored afterwards, so its lr mutation is unchanged).
  `head_gain` mutation: pick one observation head, g_k *= exp(N(0, 0.5)), clipped to [0.01, 100].
  --blind-gate (control): gains mutate and are inherited exactly the same, but c uses equal gains, so selection
  cannot act through them; differences from --gate are then due to selection on g, not to drift.

Idea 4 -- start competent, SEARL only prunes (--n-heads K --init-inputs all --input-mode remove)
  K heads: head 0 = full head (never mutated in its inputs), heads h1..h(K-1) start with ALL observation entries
  (own random init). `head_input` in remove mode: pick one non-full head and drop a block of
  max(1, round(--drop-frac * n)) of its n entries (never the last). Control: same K full heads, no --evolve-inputs.
  Arms: I4_K4_remove = --vote equal --n-heads 4 --init-inputs all --evolve-inputs --input-mode remove
        I4_K4_ctrl   = --vote equal --n-heads 4 --init-inputs all

  --prune-full: the full head can be pruned too (it is then called h0 and starts with all entries like the others).

Without SEARL (--no-searl): population 1, no selection, no SEARL mutation (fixed lr 3e-4); same networks, vote,
  target, replay, training schedule (0.5 updates per collected frame, like one SEARL lineage) and evaluation. With
  --evolve-inputs --input-mode remove, the same block-prune is applied on a fixed schedule instead: every
  --prune-every environment steps (default 24000 = one prune per lineage per 6 SEARL generations of 4000 steps),
  one random prunable head loses a random 25% of its entries. No selection: which entries go is pure chance.
  Arms: DQN1 = --no-searl --n-heads 1 --init-inputs all; DQN4_ctrl = --no-searl --n-heads 4 --init-inputs all;
        DQN4_prune3 = DQN4_ctrl + --evolve-inputs --input-mode remove; DQN4_pruneall = DQN4_prune3 + --prune-full;
        I4_K4_remove_all = I4_K4_remove + --prune-full (with SEARL).

Limb heads (--limbs split|overlap|random, with --loss mix): heads own body parts and joints; their Q-values are
  SUMMED (weight 1 each, not averaged) and trained jointly on the sum (one Huber loss on the summed Q).
  HalfCheetah: torso = obs 0,1,8,9,10 (rootz, rooty, vx, vz, vangle); back leg = obs 2-4, 11-13; front leg =
  obs 5-7, 14-16; actions = 6 joints x 3 torques, joints 0-2 back leg, 3-5 front leg.
  split   back head (back-leg obs -> Q of joints 0-2) + front head (front-leg obs -> joints 3-5)
          + torso head (torso obs -> a term for all 6 joints). Every observation goes to exactly one head.
  overlap back head (back-leg + torso obs -> joints 0-2) + front head (front-leg + torso obs -> joints 3-5).
  random  as split, but the 17 observation entries are dealt at random into groups of 6 / 6 / 5 (fixed per seed).
  Arms: L3_split / L2_overlap / L3_random (+ --no-searl for the versions without SEARL).

Who acts (--vote):
  equal   all 18 heads vote with weight 1/18 (model A).
  full    only the full head acts (mask = full only), so the acting policy and the TD target are the full head's,
          i.e. plain SEARL plus heads that learn on the side (needed for --gate: the heads never act).

Suggested arms:  I3 = --vote full --gate        I3_blind = --vote full --gate --blind-gate
                 I2 = --vote equal --evolve-inputs          I23 = --vote full --gate --evolve-inputs
Outputs in --out: the files of searl_obs_heads_halfcheetah.py plus gate_log.csv (per training phase: individual,
lr_base, lr_eff, c, gains, inputs); population_log.csv gets `inputs` ('|'-separated per head, entries '+'-joined)
and `gains` columns.
"""
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import searl_discrete as sd  # noqa: E402
import searl_obs_heads_halfcheetah as oh  # noqa: E402
from evolvable_mlp import EvolvableMLP  # noqa: E402

N_BINS = oh.N_BINS
EVOLVE_INPUTS, GATE, BLIND_GATE = False, False, False
INPUT_MODE, DROP_FRAC = "both", 0.25      # head_input mutation: "both" = add or remove one entry (I2); "remove" = drop a block
N_HEADS, INIT_INPUTS = None, "single"     # None = full + one head per entry; "all" = every non-full head sees everything
PRUNE_FULL = False                        # the full head is prunable too (inputs = explicit list, named h0)
NO_SEARL, PRUNE_EVERY = False, 24_000
LIMBS, LIMB_SEED = None, 0
TORSO, BACK, FRONT = [0, 1, 8, 9, 10], [2, 3, 4, 11, 12, 13], [5, 6, 7, 14, 15, 16]


def limb_layout(mode, seed):
    """(names, input index lists, output joint lists or None = all joints) of the limb heads."""
    if mode == "split":
        return ["back", "front", "torso"], [BACK, FRONT, TORSO], [[0, 1, 2], [3, 4, 5], None]
    if mode == "overlap":
        return ["back", "front"], [sorted(BACK + TORSO), sorted(FRONT + TORSO)], [[0, 1, 2], [3, 4, 5]]
    if mode == "random":
        perm = np.random.RandomState(1000 + seed).permutation(17).tolist()
        return (["rand_back", "rand_front", "rand_all"], [sorted(perm[:6]), sorted(perm[6:12]), sorted(perm[12:])],
                [[0, 1, 2], [3, 4, 5], None])
    raise ValueError(mode)
ENV_FRAMES = [0]                          # training-episode frames collected so far (counted by the run_episode wrapper)
LR_FLOOR, LR_SLOPE = 0.5, 0.5
GAIN_SD, GAIN_BOUNDS = 0.5, (0.01, 100.0)
GATE_LOG = []
_GATE_RNG = np.random.RandomState(0)   # private stream for the gate's replay batch (re-seeded in run)


# ---------------------------------------------------------------------------------------------------- network
class EvoHeadsQ(oh.ObsHeadsQ):
    """ObsHeadsQ with an evolvable input set per head (head_inputs) and a gain per head (gains, used by the gate)."""

    def __init__(self, num_inputs, num_outputs, hidden_size=(128,), activation="relu", output_activation=None,
                 layer_norm=False, output_vanish=False, full_head=True, head_dicts=None, mask=None, head_inputs=None,
                 gains=None, vote="equal", head_outputs=None, head_names=None, combine="mean"):
        if head_inputs is None and LIMBS:
            head_names, head_inputs, head_outputs = limb_layout(LIMBS, LIMB_SEED)
            combine = "sum"
            mask = [1.0] * len(head_inputs)          # every limb head contributes, whatever --vote says
        if head_inputs is None:
            if INIT_INPUTS == "all":
                assert full_head and N_HEADS, "--init-inputs all needs --n-heads K (full head + K-1 heads)"
                head_inputs = [list(range(num_inputs)) if PRUNE_FULL else None] + [
                    list(range(num_inputs)) for _ in range(N_HEADS - 1)]
            else:
                head_inputs = ([None] if full_head else []) + [[i] for i in range(num_inputs)]
        head_inputs = [None if ix is None else sorted(int(i) for i in ix) for ix in head_inputs]
        if head_outputs is None:
            head_outputs = [None] * len(head_inputs)
        if head_dicts is None:
            head_dicts = [dict(num_inputs=num_inputs if ix is None else len(ix),
                               num_outputs=num_outputs if jo is None else N_BINS * len(jo),
                               hidden_size=list(hidden_size), activation=activation,
                               output_activation=output_activation, layer_norm=layer_norm)
                          for ix, jo in zip(head_inputs, head_outputs)]
        k = len(head_inputs)
        if mask is None:
            mask = [1.0] * k if vote == "equal" else [1.0] + [0.0] * (k - 1)   # "full": only head 0 acts
        super().__init__(num_inputs, num_outputs, hidden_size, activation, output_activation, layer_norm,
                         output_vanish, full_head, head_dicts, mask)
        self.inputs = head_inputs
        self.names = (["full"] if full_head else []) + (
            [f"obs{i}" for i in range(num_inputs)] if len(head_inputs) == num_inputs + int(full_head)
            and INIT_INPUTS != "all" else [f"h{j}" for j in range(1, k + 1 - int(full_head))])
        if INIT_INPUTS == "all" and head_inputs[0] is not None:
            self.names[0] = "h0"
        if head_names is not None:
            self.names = list(head_names)
        self.outputs = [None if jo is None else list(jo) for jo in head_outputs]
        self.combine = combine
        if combine == "sum":     # Q = sum of the heads' terms (each head covers its own joints / all joints)
            assert bool((self.mask == 1).all()), "summed limb heads must all be on"
            self.weights.copy_(self.mask)
        self.vote = vote
        g = torch.ones(k) if gains is None else torch.as_tensor(gains, dtype=torch.float32)
        self.register_buffer("gains", g.clone())

    @property
    def init_dict(self):
        d = super().init_dict
        d.update(head_inputs=copy.deepcopy(self.inputs), gains=self.gains.tolist(), vote=self.vote,
                 head_outputs=copy.deepcopy(self.outputs), head_names=list(self.names), combine=self.combine)
        return d

    def head_q(self, x):
        """[K, B, num_outputs]; a head that owns only some joints is zero on the other joints' outputs."""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        out = []
        for h, ix, jo in zip(self.heads, self.inputs, self.outputs):
            q = h(x if ix is None else x[..., ix])
            if jo is not None:
                full = q.new_zeros(q.shape[:-1] + (self.num_outputs,))
                cols = [N_BINS * j + b for j in jo for b in range(N_BINS)]
                full[..., cols] = q
                q = full
            out.append(q)
        return torch.stack(out)

    def clone(self):
        c = EvoHeadsQ(**copy.deepcopy(self.init_dict))
        c.load_state_dict(self.state_dict())
        return c

    def inputs_str(self):
        return "|".join("all" if ix is None else "+".join(map(str, ix)) for ix in self.inputs)

    def gains_str(self):
        return "|".join(f"{g:.3g}" for g in self.gains.tolist())


def resize_head_input(head, old_ix, new_ix):
    """New EvolvableMLP with input set new_ix; first-layer columns of kept entries copied, new entries zero."""
    d = head.init_dict
    d["num_inputs"] = len(new_ix)
    new = EvolvableMLP(output_vanish=False, **d)
    old_sd, new_sd = head.state_dict(), new.state_dict()
    key = [k for k in old_sd if k.endswith("linear_layer_0.weight")][0]
    for k in new_sd:
        if k != key:
            new_sd[k] = old_sd[k].clone()
    w = torch.zeros_like(new_sd[key])
    for c_new, i in enumerate(new_ix):
        if i in old_ix:
            w[:, c_new] = old_sd[key][:, old_ix.index(i)]
    new_sd[key] = w
    new.load_state_dict(new_sd)
    return new


# --------------------------------------------------------------------------------------------------- mutation
class EvoMutations(oh.ObsHeadsMutations):
    def mutation(self, population):
        menu = [("no_mutation", self.no_mutation), ("architecture", self.architecture_mutate),
                ("parameters", self.parameter_mutation), ("activation", self.activation_mutation),
                ("rl_hyperparam", self.rl_hyperparam_mutation)]
        if EVOLVE_INPUTS:
            menu.append(("head_input", self.head_input_mutation))
        if GATE:
            menu.append(("head_gain", self.head_gain_mutation))
        options, proba = [], []
        for name, fn in menu:
            p = getattr(self.cfg.mutation, name, 0.2 if name in ("head_input", "head_gain") else 0)
            if p:
                options.append(fn)
                proba.append(float(p))
        proba = np.array(proba) / np.sum(proba)
        choice = self.rng.choice(options, len(population), p=proba)
        return [fn(ind) for fn, ind in zip(choice, population)]

    def _obs_heads(self, net):
        return [k for k, ix in enumerate(net.inputs) if ix is not None]

    def head_input_mutation(self, individual):
        net = individual.actor.clone()
        k = int(self.rng.choice(self._obs_heads(net)))
        old = list(net.inputs[k])
        absent = [i for i in range(net.num_inputs) if i not in old]
        if INPUT_MODE == "remove":
            if len(old) <= 1:
                individual.train_log["mutation"] = f"head_input_{net.names[k]}_none"
                return individual
            n_drop = min(len(old) - 1, max(1, int(round(DROP_FRAC * len(old)))))
            drop = sorted(int(i) for i in self.rng.choice(old, n_drop, replace=False))
            new = [x for x in old if x not in drop]
            net.heads[k] = resize_head_input(net.heads[k], old, new)
            net.inputs[k] = new
            individual.actor = net
            individual.train_log["mutation"] = f"head_input_{net.names[k]}_remove{'+'.join(map(str, drop))}"
            return individual
        can_add, can_remove = bool(absent), len(old) > 1
        add = can_add and (not can_remove or self.rng.uniform() < 0.5)
        if add:
            i = int(self.rng.choice(absent))
            new = sorted(old + [i])
        else:
            i = int(self.rng.choice(old))
            new = [x for x in old if x != i]
        net.heads[k] = resize_head_input(net.heads[k], old, new)
        net.inputs[k] = new
        individual.actor = net
        individual.train_log["mutation"] = f"head_input_{net.names[k]}_{'add' if add else 'remove'}{i}"
        return individual

    def head_gain_mutation(self, individual):
        net = individual.actor
        k = int(self.rng.choice(self._obs_heads(net)))
        g = float(net.gains[k]) * float(np.exp(self.rng.normal(0, GAIN_SD)))
        net.gains[k] = float(np.clip(g, *GAIN_BOUNDS))
        individual.train_log["mutation"] = f"head_gain_{net.names[k]}"
        return individual


class NoSearlMutations(EvoMutations):
    """--no-searl: no SEARL operator; only the scheduled random block-prune (when --evolve-inputs)."""
    next_prune = [PRUNE_EVERY]

    def mutation(self, population):
        out = []
        for ind in population:
            if EVOLVE_INPUTS and ENV_FRAMES[0] >= self.next_prune[0]:
                self.next_prune[0] += PRUNE_EVERY
                out.append(self.head_input_mutation(ind))
            else:
                out.append(self.no_mutation(ind))
        return out


# --------------------------------------------------------------------------------------------------- training
def head_agreement_with_full(net, s):
    """a_k: share of (state, joint) where head k's greedy torque equals the full head's (head 0)."""
    with torch.no_grad():
        q = net.head_q(s).view(len(net.heads), s.shape[0], -1, N_BINS).argmax(-1)
        return (q == q[0:1]).float().mean((1, 2))


def train_gated(indi, replay, iterations, device):
    net, a = indi.actor, indi.rl_config
    base_lr = a.lr_actor
    c, lr_eff = float("nan"), base_lr
    if GATE:
        idx = _GATE_RNG.randint(0, replay.n, size=a.batch_size)
        s = torch.from_numpy(replay.s[idx]).float()
        agree = head_agreement_with_full(net, s)
        obs = torch.tensor([ix is not None for ix in net.inputs])
        g = torch.ones_like(net.gains) if BLIND_GATE else net.gains
        g = g * obs
        c = float((g * agree).sum() / g.sum())
        lr_eff = base_lr * (LR_FLOOR + LR_SLOPE * c)
        a.lr_actor = lr_eff
    try:
        oh.train_obs_heads(indi, replay, iterations, device)
    finally:
        a.lr_actor = base_lr
    GATE_LOG.append({"phase": oh._PHASE[0], "individual": indi.index, "mutation": indi.train_log["mutation"],
                     "lr_base": base_lr, "lr_eff": lr_eff, "c": c, "gains": indi.actor.gains_str(),
                     "inputs": indi.actor.inputs_str(), "mean_input_size": float(np.mean(
                         [len(ix) for ix in indi.actor.inputs if ix is not None]))})
    return indi


# -------------------------------------------------------------------------------------------------------- run
class EvoSelection(oh.LoggingSelection):
    def select(self, population):
        extra = [(ind.actor.inputs_str(), ind.actor.gains_str()) for ind in population]
        n0 = len(oh.POP_LOG)   # the parent appends one row per individual, then selects
        out = super().select(population)
        for row, (ins, gs) in zip(oh.POP_LOG[n0:], extra):
            row["inputs"], row["gains"] = ins, gs
        flush_logs()
        return out


def flush_logs():
    oh.flush_logs()
    if oh.OUT_DIR[0] is not None:
        pd.DataFrame(GATE_LOG).to_csv(os.path.join(oh.OUT_DIR[0], "gate_log.csv"), index=False)


def run(seed, pop, frames, vote="equal", evolve_inputs=False, gate=False, blind_gate=False, log=print,
        n_heads=None, init_inputs="single", input_mode="both", drop_frac=0.25, prune_full=False, no_searl=False,
        prune_every=24_000, hidden=None, loss="own", limbs=None):
    global EVOLVE_INPUTS, GATE, BLIND_GATE, _GATE_RNG, N_HEADS, INIT_INPUTS, INPUT_MODE, DROP_FRAC
    global PRUNE_FULL, NO_SEARL, PRUNE_EVERY
    N_HEADS, INIT_INPUTS, INPUT_MODE, DROP_FRAC = n_heads, init_inputs, input_mode, drop_frac
    PRUNE_FULL, NO_SEARL, PRUNE_EVERY = bool(prune_full), bool(no_searl), int(prune_every)
    assert not (prune_full and init_inputs != "all"), "--prune-full needs --init-inputs all"
    if NO_SEARL:
        pop = 1
        NoSearlMutations.next_prune[0] = PRUNE_EVERY
    ENV_FRAMES[0] = 0
    import fastrand
    assert not (gate and vote == "equal"), "--gate: the heads must not act (use --vote full)"
    EVOLVE_INPUTS, GATE, BLIND_GATE = bool(evolve_inputs), bool(gate), bool(blind_gate)
    _GATE_RNG = np.random.RandomState(seed + 12345)
    global LIMBS, LIMB_SEED
    LIMBS, LIMB_SEED = limbs, seed
    assert not limbs or loss == "mix", "--limbs: the heads' sum must be trained jointly (--loss mix)"
    oh.LOSS, oh.TARGET, oh.TOGGLE, oh.PROTECT_FULL = loss, "ensemble", False, False
    fastrand.pcg32_seed(seed)
    for lst in (oh.HEADS_LOG, oh.POP_LOG, oh.Q_DUMP, GATE_LOG):
        lst.clear()
    oh._PHASE[0] = oh._GEN[0] = 0

    def make_net(**kw):
        kw.setdefault("vote", vote)
        return EvoHeadsQ(**kw)

    base_make_cfg = sd.make_cfg

    def make_cfg(*args):
        cfg = base_make_cfg(*args)
        cfg.mutation.set_attr("head_input", 0.2 if EVOLVE_INPUTS else 0)
        cfg.mutation.set_attr("head_gain", 0.2 if GATE else 0)
        if hidden:   # width of every head's hidden layer (default 128); DQN1_wide uses 512 ~ 4 heads x 128 in weights
            cfg.actor.set_attr("hidden_size", [int(hidden)])
        return cfg

    base_run_episode = sd.run_episode

    def counting_run_episode(*args, **kw):
        ret, n = base_run_episode(*args, **kw)
        if kw.get("replay") is not None:          # training/fitness episodes only (test episodes pass no replay)
            ENV_FRAMES[0] += n
        return ret, n

    saved = sd.EvolvableMLP, sd.MlpDqnMutations, sd.train_mlp, sd.make_cfg, sd.TournamentSelection, sd.run_episode
    (sd.EvolvableMLP, sd.MlpDqnMutations, sd.train_mlp, sd.make_cfg, sd.TournamentSelection,
     sd.run_episode) = (make_net, NoSearlMutations if NO_SEARL else EvoMutations, train_gated, make_cfg,
                        EvoSelection, counting_run_episode)
    try:
        history = sd.run_searl_discrete("gym", oh.ENV_ID, seed, pop, frames, log=log)
    finally:
        (sd.EvolvableMLP, sd.MlpDqnMutations, sd.train_mlp, sd.make_cfg, sd.TournamentSelection,
         sd.run_episode) = saved
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--frames", type=int, default=400_000)
    p.add_argument("--loss", choices=["own", "mix"], default="own",
                   help="own: every head fits the target alone; mix: one loss on the combined Q (needed by --limbs)")
    p.add_argument("--limbs", choices=["split", "overlap", "random"], default=None, help="limb heads, Q summed")
    p.add_argument("--vote", choices=["equal", "full"], default="full")
    p.add_argument("--evolve-inputs", action="store_true", help="idea 2: head_input mutation")
    p.add_argument("--gate", action="store_true", help="idea 3: lr gate from gain-weighted head agreement + head_gain mutation")
    p.add_argument("--blind-gate", action="store_true", help="control: gains mutate but the gate uses equal gains")
    p.add_argument("--n-heads", type=int, default=None, help="idea 4: K heads (full + K-1), needs --init-inputs all")
    p.add_argument("--init-inputs", choices=["single", "all"], default="single")
    p.add_argument("--input-mode", choices=["both", "remove"], default="both")
    p.add_argument("--drop-frac", type=float, default=0.25)
    p.add_argument("--prune-full", action="store_true", help="the full head can be pruned too")
    p.add_argument("--no-searl", action="store_true", help="population 1, no selection, no SEARL mutation")
    p.add_argument("--prune-every", type=int, default=24_000, help="--no-searl: env steps between random prunes")
    p.add_argument("--hidden", type=int, default=None, help="hidden units per head (default 128)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    oh.OUT_DIR[0] = a.out
    hist = run(a.seed, a.pop, a.frames, vote=a.vote, evolve_inputs=a.evolve_inputs, gate=a.gate,
               blind_gate=a.blind_gate, log=lambda m: print(m, flush=True), n_heads=a.n_heads,
               init_inputs=a.init_inputs, input_mode=a.input_mode, drop_frac=a.drop_frac, prune_full=a.prune_full,
               no_searl=a.no_searl, prune_every=a.prune_every, hidden=a.hidden, loss=a.loss, limbs=a.limbs)
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
    flush_logs()
    print("wrote", os.path.join(a.out, "history.csv"))


if __name__ == "__main__":
    main()
