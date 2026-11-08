#!/usr/bin/env python
"""Entry point for one (method, seed) run of the SEARL Figure 2a
(HalfCheetah) reproduction. Writes a history CSV compatible with
make_figure2a.py.

Reduced-scale budgets (see base_config.py and the module docstrings in
searl_td3.py / modified_pbt.py / random_search.py for exactly what was
changed and why). This keeps the paper's qualitative story -- SEARL's
shared-replay population needs meaningfully fewer total environment steps
than modified PBT or random search to reach comparable reward -- while
cutting absolute step counts drastically (paper: SEARL 2M, PBT/RS
~10-20M env steps *per seed*) so a full (3-seed x 3-method) suite finishes
in well under an hour on this machine instead of requiring hundreds of
GPU-hours. Because these tiny (128-hidden-unit) networks are Python-loop-
and BLAS-call-bound rather than FLOP-bound, wall-clock cost tracks total
gradient steps, not total env frames -- and SEARL's shared replay makes it
pay *more* gradient steps per env frame than the baselines (that's the
whole point of the algorithm), so matching the paper's ~10x frame gap
exactly would make this reproduction's random-search baseline alone take
~2 hours. We settled for a ~2-4x frame gap instead, which still shows the
same ordering:

    SEARL:        population=3, num_frames=50_000        (~20 min)
    ModifiedPBT:  population=3, generations=67, steps/gen=200
                  (~200k frames, ~10 min)
    RandomSearch: 4 configs x 25_000 frames/config
                  (~100k frames total, ~25 min)

To scale toward the paper's exact settings, override the relevant fields in
base_config.py (population_size, train.num_frames, nevo.generations/
steps_per_generation, and the num_configs/frames_per_config CLI args below)
-- and see td3_training.py/td3_baseline.py if you want to remove the
Python-loop tensor-construction bottleneck first (batch the replay-memory
sample as one vectorized array-to-tensor conversion instead of per-transition
torch.Tensor(...) calls); that alone should give an order of magnitude more
speed for the same step counts.
"""
import argparse
import json
import os

import torch
# These tiny (128-hidden-unit) networks do not benefit from PyTorch's default
# CPU intra-op thread pool (one thread per core on this 128-core machine) --
# they only pay its per-call synchronization overhead, which otherwise makes
# every single forward/backward pass ~10-25x slower than it needs to be.
torch.set_num_threads(4)

import pandas as pd

from base_config import make_base_config
from searl_td3 import run_searl_td3
from modified_pbt import run_modified_pbt
from random_search import run_random_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["searl", "modified_pbt", "random_search"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folder", type=str, default="results")
    parser.add_argument("--num_configs", type=int, default=4, help="random_search only")
    parser.add_argument("--frames_per_config", type=int, default=25_000, help="random_search only")
    parser.add_argument("--population-size", type=int, default=None,
                        help="Override nevo.population_size. Default stays 3.")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Override train.num_frames. Default stays 50000.")
    parser.add_argument("--heldout-interval", type=int, default=0,
                        help="Frames between greedy held-out evals. 0 disables them.")
    parser.add_argument("--heldout-episodes", type=int, default=10)
    args = parser.parse_args()

    out_dir = os.path.join(args.folder, f"HalfCheetah-v4_{args.method}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "history.csv")
    if os.path.exists(csv_path):
        print(f"{csv_path} already exists, skipping")
        return

    cfg = make_base_config(seed=args.seed)
    if args.population_size is not None:
        cfg.nevo.set_attr("population_size", args.population_size)
    if args.num_frames is not None:
        cfg.train.set_attr("num_frames", args.num_frames)
    if args.method == "searl" and args.heldout_interval > 0:
        cfg.eval.set_attr("heldout_interval", args.heldout_interval)
        cfg.eval.set_attr("heldout_episodes", args.heldout_episodes)

    def log(msg):
        print(msg, flush=True)

    if args.method == "searl":
        history = run_searl_td3(cfg, log=log)
        pd.DataFrame(history).to_csv(csv_path, index=False)
        heldout = [
            {"num_frames": int(row["num_frames"]), "return": float(row["heldout_return"])}
            for row in history
            if row.get("heldout_return") is not None
        ]
        if heldout:
            with open(os.path.join(out_dir, "heldout_evaluations.json"), "w") as handle:
                json.dump(heldout, handle, indent=2)
                handle.write("\n")

    elif args.method == "modified_pbt":
        history = run_modified_pbt(cfg, log=log)
        pd.DataFrame(history).to_csv(csv_path, index=False)

    elif args.method == "random_search":
        results, best = run_random_search(cfg, num_configs=args.num_configs,
                                           frames_per_config=args.frames_per_config,
                                           seed=args.seed, log=log)
        with open(os.path.join(out_dir, "all_configs.json"), "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "history"} for r in results], f, indent=2)
        pd.DataFrame(best["history"]).to_csv(csv_path, index=False)
        # fair-evaluation-protocol bookkeeping (Section 4.3): the real cost of
        # reporting this best-config curve is num_configs x frames_per_config,
        # not just the winning config's own frame count. make_figure2a.py
        # rescales the x-axis using this file.
        with open(os.path.join(out_dir, "num_configs_tried.txt"), "w") as f:
            f.write(str(args.num_configs))

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
