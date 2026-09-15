# HalfCheetah AutoRL reproduction (SEARL · HyperController · HOOF)

Paper-faithful wrappers around the **official** codebases. All files for this effort live here.

**Do not compare the three methods against each other.** They use different inner algorithms, budgets, and metrics:

| Framework | Paper | Inner algo | Env | Budget / metric |
|-----------|-------|------------|-----|-----------------|
| SEARL | Franke et al., ICLR 2021 | TD3 + evolutionary HPO | `HalfCheetah-v2` | 2M **population** env steps (fair protocol) |
| HOOF | Paul et al., NeurIPS 2019 | A2C / TNPG | `HalfCheetah-v2` | 5M steps; **median ± IQR** |
| HyperController | Gornet et al., 2025 | PPO | `HalfCheetah-v4` | 1000 iters / 1M frames; vs **HPO wall-clock** |

## Layout

```text
hyperparameters_tuning/
  configs/          # SEARL paper + smoke YAMLs
  scripts/          # clone + run wrappers + smoke_all
  plot/             # paper-style plotting
  vendor/           # official clones (gitignored; clone via script)
  results/          # run logs (gitignored)
  figures/          # generated plots
```

## Setup

```bash
cd composable_feedback_RL/hyperparameters_tuning
bash scripts/clone_vendors.sh
```

Pinned commits are listed in [`vendor/README.md`](vendor/README.md). See [`requirements-notes.md`](requirements-notes.md) for MuJoCo / Docker / torchrl deps.

## Run (cheap → expensive)

```bash
# Smoke (does not claim paper numbers)
bash scripts/smoke_all.sh

# HyperController HalfCheetah-v4 full paper sweep (10 seeds × 6 methods)
bash scripts/run_hypercontroller_halfcheetah.sh

# HOOF (MuJoCo free — no personal key; free unlocked mjkey auto-fetched)
bash scripts/ensure_hoof_mujoco_free.sh
bash scripts/run_hoof_halfcheetah.sh --mode a2c
bash scripts/run_hoof_halfcheetah.sh --mode npg

# SEARL TD3 HalfCheetah-v2 paper config (pop=20, 2M frames)
bash scripts/run_searl_halfcheetah.sh
```

## Plot

```bash
python plot/plot_hypercontroller_halfcheetah.py
python plot/plot_hoof_halfcheetah.py
python plot/plot_searl_halfcheetah.py
```

## HalfCheetah checklist

### SEARL (must)

- [ ] Fig 2a — vs modified PBT and random search; **x = all workers’ env steps**; RS ×20 configs
- [ ] Fig 4 — population network size + actor/critic LR schedules
- [ ] Fig 5b — ablations (shared replay / NAS / LR mutation)
- [ ] Table 3 — final actor ≈ **2.8 layers**, **~1019 nodes**

Optional: Fig 3a, Apps E/G/H.

Config: [`configs/searl_td3_halfcheetah.yml`](configs/searl_td3_halfcheetah.yml) (overrides vendor Walker2d / pop=10 defaults).

### HOOF (must)

- [ ] Fig 1a — HOOF-A2C LR vs baseline A2C vs tuned meta-gradients
- [ ] Table 1 — KL ε row (ε=0.03 median **1524**)
- [ ] Fig 2a — HOOF-TNPG vs TRPO
- [ ] Fig 3 — `(δ, γ, λ)` schedules

Then appendix: no-KL, SGD, entropy grid (Table 2).

### HyperController (must)

- [ ] Fig 1 — median train / eval reward sum vs HPO wall-clock (6 methods)
- [ ] Fig 2 — boxplots at t=1000
- [ ] Table I — 10/10 seeds finish without NaN

Paper does not publish scalar reward tables — match curve **shape** and success rate; absolute wall-clock depends on hardware (see `results/.../machine_info.txt`).

## Fair evaluation (SEARL)

Interaction counter starts at the first meta-optimization step. Population methods sum **all** workers’ steps. Offline random search multiplies the best config’s curve by the number of configs (20).

## Success criteria

- Main HalfCheetah figures regenerate from our logs.
- HOOF median @ 5e6 within ~1 IQR of Table 1 / Fig 1a.
- SEARL sample-efficiency story holds under fair step counting; architecture in Table 3 ballpark.
- HyperController 10/10 seeds; Fig 1/2 shapes match (wall-clock may differ).

See [`RESULTS.md`](RESULTS.md) for expected reference numbers.
