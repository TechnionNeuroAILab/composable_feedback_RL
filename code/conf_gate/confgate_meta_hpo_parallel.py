#!/usr/bin/env python3
"""
Parallel multi-seed launcher for confgate_meta_hpo.

Spawns N independent worker processes (default 4), each running one seed's
meta-train + eval. Speeds wall-clock of the outer loop campaign by filling
the GPU(s). Checkpoints are tagged and resume-safe so you can:

  - re-run the same command to continue incomplete seeds
  - raise --meta-iters later to extend outer training
  - add more --seeds later and keep the same --tag
  - --aggregate-only after workers finish to build multi-seed plots

Examples:
  # Fresh 4-seed CartPole smoke (1000 eps × 20 outer)
  python code/confgate_meta_hpo_parallel.py \\
      --tasks CartPole-v1 --meta-iters 20 --total-episodes 1000 \\
      --seeds 1,2,3,4 --workers 4 --tag cartpole_ep1000_m20

  # Extend outer loop later (resume from checkpoints)
  python code/confgate_meta_hpo_parallel.py \\
      --tasks CartPole-v1 --meta-iters 40 --total-episodes 1000 \\
      --seeds 1,2,3,4 --workers 4 --tag cartpole_ep1000_m20

  # Aggregate plots only
  python code/confgate_meta_hpo_parallel.py --aggregate-only \\
      --tag cartpole_ep1000_m20 --total-episodes 1000 --seeds 1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "confgate_meta_hpo.py"
CKPT_DIR = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_meta_hpo"
FIG_DIR = ROOT / "paper" / "figures" / "conf_meta_hpo_v2"
RUNS_DIR = CKPT_DIR / "parallel_runs"


def _parse_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_seeds(s: str) -> List[int]:
    return sorted({int(p.strip()) for p in s.split(",") if p.strip()})


def _parse_gpus(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _manifest_path(tag: str) -> Path:
    return RUNS_DIR / f"{tag}_manifest.json"


def _seed_status(seed: int, tag: str, meta_iters: int) -> Dict:
    tag_sfx = f"_{tag}" if tag else ""
    ckpt = CKPT_DIR / f"meta_controller_seed{seed}{tag_sfx}.pt"
    eval_p = CKPT_DIR / f"eval_vs_baseline_seed{seed}{tag_sfx}.pt"
    done = 0
    if ckpt.exists():
        try:
            import torch

            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            done = int(blob.get("meta_iters_done", len(blob.get("history", []))))
        except Exception as exc:  # noqa: BLE001
            return {
                "seed": seed,
                "meta_done": 0,
                "meta_target": meta_iters,
                "ckpt": str(ckpt),
                "eval_done": eval_p.exists(),
                "error": str(exc),
            }
    return {
        "seed": seed,
        "meta_done": done,
        "meta_target": meta_iters,
        "train_complete": done >= meta_iters,
        "eval_done": eval_p.exists(),
        "ckpt": str(ckpt),
        "eval": str(eval_p),
    }


def _write_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _build_worker_cmd(
    seed: int,
    args: argparse.Namespace,
    python_exe: str,
) -> List[str]:
    cmd = [
        python_exe,
        str(WORKER),
        "--tasks",
        args.tasks,
        "--meta-iters",
        str(args.meta_iters),
        "--inner-steps",
        str(args.inner_steps),
        "--seeds",
        str(seed),
        "--tag",
        args.tag,
        "--train-missing-only",
        "--aggregate-only",
    ]
    if args.total_episodes is not None:
        cmd.extend(["--total-episodes", str(args.total_episodes)])
    if args.skip_eval:
        cmd.append("--skip-eval")
    if args.no_resume:
        cmd.append("--no-resume")
    return cmd


def _run_workers(
    seeds: Sequence[int],
    gpus: Sequence[str],
    workers: int,
    args: argparse.Namespace,
) -> List[Dict]:
    python_exe = sys.executable
    pending = list(seeds)
    active: Dict[int, Dict] = {}  # pid -> info
    finished: List[Dict] = []
    log_dir = RUNS_DIR / args.tag / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _launch_one(seed: int, slot: int) -> None:
        gpu = gpus[slot % len(gpus)]
        cmd = _build_worker_cmd(seed, args, python_exe)
        log_path = log_dir / f"seed{seed}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # Avoid contention on BLAS threads inside each worker
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        print(
            f"[launcher] start seed={seed} gpu={gpu} log={log_path}",
            flush=True,
        )
        log_f = open(log_path, "a", buffering=1)
        log_f.write(
            f"\n===== launch {datetime.now(timezone.utc).isoformat()} "
            f"cmd={' '.join(cmd)} CUDA_VISIBLE_DEVICES={gpu} =====\n"
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        active[proc.pid] = {
            "proc": proc,
            "seed": seed,
            "gpu": gpu,
            "log": str(log_path),
            "log_f": log_f,
            "started": time.time(),
        }

    slot = 0
    while pending or active:
        while pending and len(active) < workers:
            seed = pending.pop(0)
            st = _seed_status(seed, args.tag, args.meta_iters)
            if (
                st.get("train_complete")
                and (args.skip_eval or st.get("eval_done"))
                and not args.no_resume
            ):
                print(
                    f"[launcher] seed={seed} already complete "
                    f"(meta {st['meta_done']}/{args.meta_iters}, "
                    f"eval={st.get('eval_done')}); skip",
                    flush=True,
                )
                finished.append({**st, "exit_code": 0, "skipped": True})
                continue
            _launch_one(seed, slot)
            slot += 1

        if not active:
            break

        # Poll
        time.sleep(2.0)
        done_pids = []
        for pid, info in active.items():
            rc = info["proc"].poll()
            if rc is None:
                continue
            info["log_f"].close()
            st = _seed_status(info["seed"], args.tag, args.meta_iters)
            rec = {
                **st,
                "exit_code": rc,
                "gpu": info["gpu"],
                "log": info["log"],
                "elapsed_s": round(time.time() - info["started"], 1),
                "skipped": False,
            }
            finished.append(rec)
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(
                f"[launcher] finish seed={info['seed']} {status} "
                f"meta={st.get('meta_done')}/{args.meta_iters} "
                f"eval={st.get('eval_done')} ({rec['elapsed_s']}s)",
                flush=True,
            )
            done_pids.append(pid)
        for pid in done_pids:
            del active[pid]

        # Refresh manifest while running
        _write_manifest(
            _manifest_path(args.tag),
            {
                "tag": args.tag,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "tasks": args.tasks,
                    "meta_iters": args.meta_iters,
                    "total_episodes": args.total_episodes,
                    "inner_steps": args.inner_steps,
                    "seeds": list(seeds),
                    "workers": workers,
                    "gpus": list(gpus),
                },
                "active": [
                    {"seed": i["seed"], "gpu": i["gpu"], "pid": pid, "log": i["log"]}
                    for pid, i in active.items()
                ],
                "finished": finished,
                "pending": list(pending),
            },
        )

    return finished


def _aggregate(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    """Call plot-only path in a single process over all seeds."""
    cmd = [
        sys.executable,
        str(WORKER),
        "--plot-only",
        "--aggregate-only",
        "--tasks",
        args.tasks,
        "--inner-steps",
        str(args.inner_steps),
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--tag",
        args.tag,
    ]
    if args.total_episodes is not None:
        cmd.extend(["--total-episodes", str(args.total_episodes)])
    print(f"[launcher] aggregate: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=str, default="CartPole-v1")
    p.add_argument("--meta-iters", type=int, default=20)
    p.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        metavar="E",
        help="Optional episode cap per inner run (omit to use --inner-steps only)",
    )
    p.add_argument(
        "--inner-steps",
        type=int,
        default=50_000,
        help="Safety cap (also raised to total_episodes*2000 in worker)",
    )
    p.add_argument("--seeds", type=str, default="1,2,3,4")
    p.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    p.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids; workers round-robin across them",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="cartpole_ep1000_m20",
        help="Run id used in checkpoints/figures/manifest (keep stable to resume)",
    )
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only build multi-seed plots from existing eval checkpoints",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print per-seed progress from checkpoints and exit",
    )
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    gpus = _parse_gpus(args.gpus) or ["0"]
    workers = max(1, min(args.workers, len(seeds)))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.status or args.aggregate_only:
        statuses = [_seed_status(s, args.tag, args.meta_iters) for s in seeds]
        print(json.dumps({"tag": args.tag, "seeds": statuses}, indent=2))
        if args.aggregate_only:
            missing = [s for s in statuses if not s.get("eval_done")]
            if missing:
                raise SystemExit(
                    f"Cannot aggregate; missing eval for seeds "
                    f"{[m['seed'] for m in missing]}"
                )
            _aggregate(args, seeds)
        return

    print("=" * 60, flush=True)
    print("confgate_meta_hpo_parallel", flush=True)
    print(f"  tag         : {args.tag}", flush=True)
    print(f"  tasks       : {args.tasks}", flush=True)
    print(f"  meta_iters  : {args.meta_iters}", flush=True)
    print(f"  inner_steps : {args.inner_steps}", flush=True)
    print(
        f"  episodes    : {args.total_episodes if args.total_episodes is not None else '(none — step budget)'}",
        flush=True,
    )
    print(f"  seeds       : {seeds}", flush=True)
    print(f"  workers     : {workers}", flush=True)
    print(f"  gpus        : {gpus}", flush=True)
    print(f"  manifest    : {_manifest_path(args.tag)}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    finished = _run_workers(seeds, gpus, workers, args)
    elapsed = time.time() - t0

    fails = [f for f in finished if f.get("exit_code", 1) != 0]
    _write_manifest(
        _manifest_path(args.tag),
        {
            "tag": args.tag,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "config": {
                "tasks": args.tasks,
                "meta_iters": args.meta_iters,
                "total_episodes": args.total_episodes,
                "inner_steps": args.inner_steps,
                "seeds": seeds,
                "workers": workers,
                "gpus": gpus,
            },
            "finished": finished,
            "failed_seeds": [f["seed"] for f in fails],
            "status": [_seed_status(s, args.tag, args.meta_iters) for s in seeds],
        },
    )

    if fails:
        print(f"[launcher] FAILED seeds: {[f['seed'] for f in fails]}", flush=True)
        raise SystemExit(1)

    if not args.skip_eval:
        try:
            _aggregate(args, seeds)
        except subprocess.CalledProcessError as exc:
            print(f"[launcher] aggregate failed: {exc}", flush=True)
            raise SystemExit(1)

    print(f"[launcher] all done in {elapsed:.1f}s", flush=True)
    print(
        f"[launcher] extend later with same --tag and higher --meta-iters "
        f"or extra --seeds",
        flush=True,
    )


if __name__ == "__main__":
    main()
