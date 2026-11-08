#!/usr/bin/env bash
# Single-agent DQN + ConfGate-mix: the aux heads also take part in ACTING (Q_mix = (1-L) Q_full + L mean_k Q_k, L = 0.2 fixed,
# no evolution). Arms: mix = mix only (lr gate off), mixgate = mix + lr gate (floor 0.5, slope 0.5); heads 2h (groups) / obs (per_obs).
# 1M steps, seeds 0-2, CartPole-v1 and discrete HalfCheetah-v4. Output: <results>/cleanrl_dqn_confgate_<arm>_<heads>/seed<k>.csv
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
STEPS=${STEPS:-1000000}; LAM=${LAM:-0.2}; SEEDS=${SEEDS:-"0 1 2"}
mkdir -p logs_confgate_dqn
run() { env=$1; arm=$2; hd=$3; s=$4
  base=results_cartpole; [ $env = HalfCheetah-v4 ] && base=results_halfcheetah_1M
  dir=$base/cleanrl_dqn_confgate_${arm}_${hd}; mkdir -p $dir
  [ -f $dir/seed$s.csv ] && return 0
  gate="--lr-floor 1 --lr-slope 0"; [ $arm = mixgate ] && gate="--lr-floor 0.5 --lr-slope 0.5"
  heads=groups; [ $hd = obs ] && heads=per_obs
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
    python cleanrl_dqn_confgate.py --env_id $env --seed $s --total_timesteps $STEPS --no-cuda --heads $heads $gate --mix-lambda $LAM \
    --csv_path $dir/seed$s.csv.tmp --conf-log-path $dir/seed${s}_conf.csv > logs_confgate_dqn/${arm}_${hd}_${env}_seed$s.log 2>&1 \
    && mv $dir/seed$s.csv.tmp $dir/seed$s.csv; }
export -f run; export STEPS LAM
{ for s in $SEEDS; do for env in CartPole-v1 HalfCheetah-v4; do for arm in mix mixgate; do for hd in 2h obs; do echo "$env $arm $hd $s"; done; done; done; done
} | xargs -P ${PAR:-24} -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
