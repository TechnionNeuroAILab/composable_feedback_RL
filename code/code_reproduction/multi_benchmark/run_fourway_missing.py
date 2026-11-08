#!/usr/bin/env python
"""Trains ONE missing (modality, seed) of code/conf_gate/confgate_fourway.py with exactly the arguments
main() would use for `--total-episodes 5000` (the budget of the stored fourway results: epsilon decay 3500
episodes, learning_starts 10000, train every 10 steps), into the same checkpoint folder, so each seed can run as
its own process. No plotting, nothing else in paper/ is touched.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "conf_gate"))
import confgate_fourway as cf  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--modality", choices=["analytical", "learnable", "rnn"], required=True)
p.add_argument("--seed", type=int, required=True)
a = p.parse_args()

EPISODES, EPS_DECAY = 5000, 3500
n_step = max(500_000, 15_000_000)          # main(): step cap when --total-episodes is given
ckpt_dir = ROOT / "paper" / "_tmp_b_feedb_cg" / f"ckpt_fourway_decay{EPS_DECAY}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
common = dict(checkpoint_dir=ckpt_dir, epsilon_decay_episodes=EPS_DECAY, total_episodes=EPISODES,
              learning_starts=cf.LEARNING_STARTS, train_frequency=cf.TRAIN_FREQUENCY)

if a.modality == "analytical":
    if cf.cg_analytical._ckpt_path(ckpt_dir, a.seed, True).exists():
        sys.exit("checkpoint exists, skipping")
    cf.cg_analytical.run_one(a.seed, n_step, True, device, **common)
elif a.modality == "learnable":
    if cf.cg_learnable._ckpt_path_learnable(ckpt_dir, a.seed).exists():
        sys.exit("checkpoint exists, skipping")
    cf.cg_learnable.run_one(a.seed, n_step, True, device, learnable_gate=True, **common)
else:
    if cf._ckpt_path_rnn(ckpt_dir, a.seed).exists():
        sys.exit("checkpoint exists, skipping")
    cf.run_one_rnn(a.seed, n_step, device, **common)
