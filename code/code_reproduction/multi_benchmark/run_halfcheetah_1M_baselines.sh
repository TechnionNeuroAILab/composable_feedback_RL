#!/usr/bin/env bash
# Rainbow (Kaixhin port) and CleanRL DQN on discrete HalfCheetah-v4 (6 joints x {-1,0,+1}) for 1M steps, seeds 0-2.
# Same commands and settings as run_halfcheetah_discrete.sh / rainbow_dqn/run_discrete_tasks.sh; only the step budget
# changes (note: epsilon decay / PER beta schedules are defined relative to the total budget, so they stretch too).
# Results do not overwrite the 500k runs: results_halfcheetah_1M/cleanrl_dqn/ and ../rainbow_dqn/results_1M/HalfCheetah-v4/.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
STEPS=${STEPS:-1000000}
CL=results_halfcheetah_1M/cleanrl_dqn; RB=../rainbow_dqn/results_1M/HalfCheetah-v4
mkdir -p $CL $RB logs_halfcheetah_1M ../rainbow_dqn/logs_1M
run() { kind=$1; s=$2
  case $kind in
    dqn) OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
           python cleanrl_dqn_bike.py --env_id HalfCheetah-v4 --seed $s --total_timesteps $STEPS --no-cuda \
           --csv_path $CL/seed$s.csv.tmp > logs_halfcheetah_1M/dqn_seed$s.log 2>&1 && mv $CL/seed$s.csv.tmp $CL/seed$s.csv ;;
    rainbow) (cd ../rainbow_dqn && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
           python main.py --task gym --env HalfCheetah-v4 --seed $s --T-max $STEPS --learn-start 2000 --target-update 1000 \
           --memory-capacity 100000 --replay-frequency 2 --learning-rate 2.5e-4 --reward-scale 0.05 --V-min -5 --V-max 10 \
           --obs-scale 1 --csv-path results_1M/HalfCheetah-v4/seed$s.csv.tmp > logs_1M/HalfCheetah-v4_seed$s.log 2>&1 \
           && mv results_1M/HalfCheetah-v4/seed$s.csv.tmp results_1M/HalfCheetah-v4/seed$s.csv) ;;
  esac; }
export -f run; export CL RB STEPS
for s in 0 1 2; do
  [ -f $CL/seed$s.csv ] || echo "dqn $s"
  [ -f $RB/seed$s.csv ] || echo "rainbow $s"
done | xargs -P 6 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
