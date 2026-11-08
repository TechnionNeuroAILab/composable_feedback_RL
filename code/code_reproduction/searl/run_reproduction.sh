#!/usr/bin/env bash
# Reproduces Figure 2 panel (a) (HalfCheetah) of Franke, Koehler, Biedenkapp
# & Hutter, "Sample-Efficient Automated Deep Reinforcement Learning"
# (SEARL, ICLR 2021): SEARL vs. modified PBT vs. random search, reward
# against total environment steps.
#
# Reduced-scale reproduction -- see run_reproduction.py's module docstring
# for exactly which paper numbers were cut and why (short version: the
# paper's baselines need up to ~20M env steps per seed; this uses ~100k-500k
# so 3 seeds x 3 methods finishes in well under an hour).
#
# Runs under the system's default Python (gymnasium + mujoco + torch already
# installed there); no special conda env needed, unlike the HyperController
# side.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# see run_reproduction.py: tiny networks + this machine's 128 cores means the
# default BLAS/torch thread pool is pure overhead, not speedup.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SEEDS="${SEEDS:-3}"
FOLDER="${FOLDER:-results}"
NUM_CONFIGS="${NUM_CONFIGS:-4}"
FRAMES_PER_CONFIG="${FRAMES_PER_CONFIG:-25000}"
MAX_PARALLEL="${MAX_PARALLEL:-9}"

mkdir -p "$FOLDER" logs
njobs=0
for seed in $(seq 0 $((SEEDS - 1))); do
  for method in searl modified_pbt random_search; do
    logfile="logs/${method}_seed${seed}.log"
    (
      python3 run_reproduction.py \
        --method "$method" \
        --seed "$seed" \
        --folder "$FOLDER" \
        --num_configs "$NUM_CONFIGS" \
        --frames_per_config "$FRAMES_PER_CONFIG"
    ) > "$logfile" 2>&1 &
    njobs=$((njobs + 1))
    if (( njobs % MAX_PARALLEL == 0 )); then
      wait -n
    fi
  done
done
wait
echo "All runs finished. Results in $FOLDER/, stdout captured in logs/."
