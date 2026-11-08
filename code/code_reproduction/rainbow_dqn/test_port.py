"""Checks the port against Kaixhin's original code (a clone of https://github.com/Kaixhin/Rainbow at --ref):
  1. agent: one learn() step on identical weights / batch / random numbers gives identical priorities and parameters
  2. memory: identical append sequence + RNG state gives identical sampled batches (pixel mode, history 4, n 3)
  3. factored branches: with every branch carrying the same data, the loss equals the single-branch loss
Run: python test_port.py --ref /path/to/Rainbow
"""
import argparse
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    for m in ("model", "agent", "memory"):
        sys.modules.pop(m, None)
    sys.path.insert(0, path)
    mods = tuple(importlib.import_module(m) for m in ("model", "agent", "memory"))
    sys.path.remove(path)
    return mods


ref_model, ref_agent, ref_memory = load(a.ref)
my_model, my_agent, my_memory = load(HERE)

args = SimpleNamespace(atoms=51, V_min=-10, V_max=10, history_length=4, architecture="canonical", hidden_size=512,
                       noisy_std=0.1, multi_step=3, discount=0.99, batch_size=32, norm_clip=10, learning_rate=6.25e-5,
                       adam_eps=1.5e-4, device=torch.device("cpu"), model=None, priority_weight=0.4,
                       priority_exponent=0.5, trunk_size=128)


class FakeMem:
    def __init__(self, batch, actions_fn):
        self.b, self.captured = batch, None
        self.actions_fn = actions_fn

    def sample(self, n):
        idxs, s, act, R, ns, nt, w = self.b
        return idxs, s, self.actions_fn(act), R, ns, nt, w

    def update_priorities(self, idxs, pr):
        self.captured = pr


# ---- 1. agent -------------------------------------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
B, A = 32, 6
batch = (np.arange(B), torch.rand(B, 4, 84, 84, generator=g), torch.randint(0, A, (B,), generator=g),
         torch.randn(B, generator=g), torch.rand(B, 4, 84, 84, generator=g), (torch.rand(B, 1, generator=g) > 0.2).float(),
         torch.rand(B, generator=g) + 0.1)
torch.manual_seed(7)
ref = ref_agent.Agent(args, SimpleNamespace(action_space=lambda: A))
torch.manual_seed(7)
mine = my_agent.Agent(args, A, 1, None)
same_init = all(torch.equal(x, y) for x, y in zip(ref.online_net.state_dict().values(), mine.online_net.state_dict().values()))
fm_ref, fm_my = FakeMem(batch, lambda x: x), FakeMem(batch, lambda x: x.view(-1, 1))
torch.manual_seed(11); ref.learn(fm_ref)
torch.manual_seed(11); mine.learn(fm_my)
d_pr = np.abs(fm_ref.captured - fm_my.captured).max()
d_par = max((x - y).abs().max().item() for x, y in zip(ref.online_net.state_dict().values(), mine.online_net.state_dict().values()))
print(f"1. agent  : same initial weights={same_init}  max|priority diff|={d_pr:.2e}  max|param diff after step|={d_par:.2e}")
assert same_init and d_pr < 1e-5 and d_par < 1e-6

# ---- 2. memory ------------------------------------------------------------------------------------------------
rng = np.random.RandomState(3)
rm = ref_memory.ReplayMemory(args, 64)
mm = my_memory.ReplayMemory(args, 64)
for ep_len in (14, 9, 20, 12, 18):
    for t in range(ep_len):
        s = torch.from_numpy(rng.randint(0, 255, (4, 84, 84)).astype(np.float32) / 255.0)
        act, r, term = int(rng.randint(0, 6)), float(rng.randn()), t == ep_len - 1
        rm.append(s, act, r, term)
        mm.append(s, act, r, term)
np.random.seed(5); out_r = rm.sample(8)
np.random.seed(5); out_m = mm.sample(8)
names = ["tree_idxs", "states", "actions", "returns", "next_states", "nonterminals", "weights"]
diffs = []
for n, x, y in zip(names, out_r, out_m):
    x, y = torch.as_tensor(np.asarray(x)).float(), torch.as_tensor(np.asarray(y)).float().reshape(x.shape if hasattr(x, "shape") else -1)
    diffs.append((n, (x - y).abs().max().item()))
print("2. memory :", "  ".join(f"{n}={d:.1e}" for n, d in diffs))
assert all(d < 1e-6 for _, d in diffs)

# ---- 3. factored branches ---------------------------------------------------------------------------------------
nb = 3
torch.manual_seed(7)
mine_f = my_agent.Agent(args, A, nb, None)
loss_single = []
class Grab(FakeMem):
    pass
fm_f = FakeMem(batch, lambda x: x.view(-1, 1).expand(-1, nb).contiguous())
# make all branches identical copies of branch 0 so the mean over branches equals the single-branch loss
with torch.no_grad():
    for net in (mine_f.online_net, mine_f.target_net):
        for name in ("fc_z_v", "fc_z_a"):
            lin = getattr(net, name)
            per = lin.weight_mu.shape[0] // nb
            for t in (lin.weight_mu, lin.weight_sigma, lin.weight_epsilon):
                for k in range(1, nb):
                    t[k * per:(k + 1) * per] = t[:per]
            for t in (lin.bias_mu, lin.bias_sigma, lin.bias_epsilon):
                for k in range(1, nb):
                    t[k * per:(k + 1) * per] = t[:per]
# reference: nb=1 agent whose heads equal branch 0 of mine_f
torch.manual_seed(7)
single = my_agent.Agent(args, A, 1, None)
with torch.no_grad():
    for net_s, net_f in ((single.online_net, mine_f.online_net), (single.target_net, mine_f.target_net)):
        for name, p_s in net_s.named_parameters():
            p_f = dict(net_f.named_parameters())[name]
            p_s.copy_(p_f[: p_s.shape[0]] if p_s.shape[0] != p_f.shape[0] else p_f)
        for name, b_s in net_s.named_buffers():
            b_f = dict(net_f.named_buffers())[name]
            b_s.copy_(b_f[: b_s.shape[0]] if b_s.shape[0] != b_f.shape[0] else b_f)
fm_s = FakeMem(batch, lambda x: x.view(-1, 1))
torch.manual_seed(11); single.learn(fm_s)
torch.manual_seed(11); mine_f.learn(fm_f)
d = np.abs(fm_s.captured - fm_f.captured).max()
print(f"3. branches: max|per-sample loss difference| single vs 3 identical branches = {d:.2e}")
assert d < 1e-4
print("ALL CHECKS PASSED")
