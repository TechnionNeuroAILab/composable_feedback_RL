#!/usr/bin/env bash
# SEARL (pop 4, 480k decisions) and random search (10 configs x 48k decisions, x-axis x10) with MLP DQN on bike v2.1.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/bike_searl results/bike_rs logs
for s in 0 1 2; do
  [ -f results/bike_searl/seed$s/history.csv ] || python searl_mlp_dqn.py --method searl --seed $s --out results/bike_searl/seed$s > logs/bike_searl_seed$s.log 2>&1 &
  [ -f results/bike_rs/seed$s/history.csv ] || python searl_mlp_dqn.py --method random_search --seed $s --out results/bike_rs/seed$s > logs/bike_rs_seed$s.log 2>&1 &
done
wait; echo "ALL DONE"
