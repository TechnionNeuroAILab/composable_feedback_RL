#!/usr/bin/env bash
# Single-agent DQN + ConfGate aux heads (cleanrl_dqn_confgate.py; no SEARL, no population) for 1M steps.
#   CartPole-v1: seeds 0-9 (same seeds as results_cartpole/cleanrl_dqn)     HalfCheetah-v4 (discrete): seeds 0-2
# lr_floor / lr_slope stay at their defaults (0.5 / 0.5). Writes <results>/cleanrl_dqn_confgate/seed<k>.csv (+ _conf.csv).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
STEPS=${STEPS:-1000000}
mkdir -p results_cartpole/cleanrl_dqn_confgate results_halfcheetah_1M/cleanrl_dqn_confgate logs_confgate_dqn
run() { env=$1; dir=$2; s=$3
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
    python cleanrl_dqn_confgate.py --env_id $env --seed $s --total_timesteps $STEPS --no-cuda \
    --csv_path $dir/seed$s.csv.tmp --conf-log-path $dir/seed${s}_conf.csv > logs_confgate_dqn/${env}_seed$s.log 2>&1 \
    && mv $dir/seed$s.csv.tmp $dir/seed$s.csv; }
export -f run; export STEPS
{
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results_cartpole/cleanrl_dqn_confgate/seed$s.csv ] || echo "CartPole-v1 results_cartpole/cleanrl_dqn_confgate $s"; done
for s in 0 1 2; do [ -f results_halfcheetah_1M/cleanrl_dqn_confgate/seed$s.csv ] || echo "HalfCheetah-v4 results_halfcheetah_1M/cleanrl_dqn_confgate $s"; done
} | xargs -P 13 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
