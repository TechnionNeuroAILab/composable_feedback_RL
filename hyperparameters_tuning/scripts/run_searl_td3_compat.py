#!/usr/bin/env python3
"""Launch SEARL TD3 with a gym-0.26 compatibility shim (old seed/reset/step API).

Does not modify vendor/searl sources. SEARL uses multiprocessing spawn; we inject
a Pool initializer that applies the gym patch in each worker.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _worker_init() -> None:
    import gym_legacy_patch  # noqa: F401


def _patch_spawn_pools() -> None:
    """Ensure every spawn Pool runs gym_legacy_patch in workers."""
    import multiprocessing as mp

    orig_get_context = mp.get_context

    def get_context(method=None):
        ctx = orig_get_context(method)
        if getattr(ctx, "_searl_pool_patched", False):
            return ctx
        orig_pool = ctx.Pool

        def Pool(*args, **kwargs):
            kwargs.setdefault("initializer", _worker_init)
            return orig_pool(*args, **kwargs)

        ctx.Pool = Pool
        ctx._searl_pool_patched = True
        return ctx

    mp.get_context = get_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--expt_dir", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    vendor = root / "vendor" / "searl"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(vendor))
    os.environ["PYTHONPATH"] = f"{scripts}:{vendor}:{os.environ.get('PYTHONPATH', '')}"

    import gym_legacy_patch  # noqa: F401

    _patch_spawn_pools()

    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:"
        f"{Path.home()}/.mujoco/mujoco200/bin:/usr/lib/nvidia-384"
    )

    from searl.neuroevolution.searl_td3 import start_searl_td3_run

    with open(args.config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    start_searl_td3_run(config_dict, expt_dir=args.expt_dir)


if __name__ == "__main__":
    main()
