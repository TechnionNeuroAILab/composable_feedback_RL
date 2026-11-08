#!/usr/bin/env bash
# Single-agent DQN + ConfGate with one aux head per observation entry (--heads per_obs), 1M steps, seeds 0-2,
# CartPole-v1 and discrete HalfCheetah-v4 (4 and 17 heads). Output: <results>/cleanrl_dqn_confgate_perobs/seed<k>.csv
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
STEPS=${STEPS:-1000000}
CP=results_cartpole/cleanrl_dqn_confgate_perobs; HC=results_halfcheetah_1M/cleanrl_dqn_confgate_perobs
mkdir -p $CP $HC logs_confgate_dqn
run() { env=$1; dir=$2; s=$3
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
    python cleanrl_dqn_confgate.py --env_id $env --seed $s --total_timesteps $STEPS --no-cuda --heads per_obs \
    --csv_path $dir/seed$s.csv.tmp --conf-log-path $dir/seed${s}_conf.csv > logs_confgate_dqn/perobs_${env}_seed$s.log 2>&1 \
    && mv $dir/seed$s.csv.tmp $dir/seed$s.csv; }
export -f run; export STEPS
{
for s in 0 1 2; do [ -f $CP/seed$s.csv ] || echo "CartPole-v1 $CP $s"; [ -f $HC/seed$s.csv ] || echo "HalfCheetah-v4 $HC $s"; done
} | xargs -P 6 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
