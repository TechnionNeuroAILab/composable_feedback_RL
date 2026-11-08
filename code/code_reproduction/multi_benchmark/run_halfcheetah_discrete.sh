#!/usr/bin/env bash
# HalfCheetah-v4 with each joint chosen separately (6 x {-1, 0, +1} = 18 outputs):
# SEARL DQN, CleanRL DQN, discrete CleanRL PPO. 500k env steps, 3 seeds.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=results_halfcheetah
mkdir -p $R/searl $R/cleanrl_dqn $R/ppo logs_halfcheetah
run() { kind=$1; s=$2
  export CUDA_VISIBLE_DEVICES=$(( RANDOM % 2 ))
  case $kind in
    searl) (cd ../searl && python searl_discrete.py --task gym --env HalfCheetah-v4 --seed $s --pop 4 --frames 500000 \
              --out ../multi_benchmark/$R/searl/seed$s) ;;
    dqn)   python cleanrl_dqn_bike.py --env_id HalfCheetah-v4 --seed $s --total_timesteps 500000 --no-cuda \
              --csv_path $R/cleanrl_dqn/seed$s.csv.tmp && mv $R/cleanrl_dqn/seed$s.csv.tmp $R/cleanrl_dqn/seed$s.csv ;;
    ppo)   python ppo_hpo.py --task gym --env_id HalfCheetah-v4 --hpo none --seed $s --total_timesteps 500000 \
              --csv_path $R/ppo/none_seed$s.csv.tmp && mv $R/ppo/none_seed$s.csv.tmp $R/ppo/none_seed$s.csv ;;
  esac > logs_halfcheetah/${kind}_seed$s.log 2>&1; }
export -f run; export R
{
for s in 0 1 2; do
  [ -f $R/searl/seed$s/history.csv ] || echo "searl $s"
  [ -f $R/cleanrl_dqn/seed$s.csv ] || echo "dqn $s"
  [ -f $R/ppo/none_seed$s.csv ] || echo "ppo $s"
done
} | xargs -P "${PARALLEL:-9}" -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
