#!/usr/bin/env bash
# CleanRL dqn.py (defaults, 480k decisions) on bike v2.1, 3 seeds.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p results/cleanrl_dqn logs
for s in 0 1 2; do
  [ -f results/cleanrl_dqn/bike_seed$s.csv ] && continue
  ( python cleanrl_dqn_bike.py --seed $s --no-cuda --csv_path results/cleanrl_dqn/bike_seed$s.csv.tmp \
      > logs/cleanrl_dqn_bike_seed$s.log 2>&1 && mv results/cleanrl_dqn/bike_seed$s.csv.tmp results/cleanrl_dqn/bike_seed$s.csv ) &
done
wait; echo "ALL DONE"
