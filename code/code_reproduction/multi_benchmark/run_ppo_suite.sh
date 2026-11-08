#!/usr/bin/env bash
# CleanRL PPO baseline (--hpo none) vs HyperController vs PB2 on Pong, Enduro (250k agent steps) and
# bike v2.1 (480k decisions = 2000 episodes x 240 rows), 3 seeds each. See ppo_hpo.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=results/ppo; mkdir -p $OUT logs
run() { task=$1; env=$2; hpo=$3; s=$4; steps=$5
  export CUDA_VISIBLE_DEVICES=$(( RANDOM % 2 ))
  name="${env}_${hpo}_seed${s}"
  python ppo_hpo.py --task "$task" --env_id "$env" --hpo "$hpo" --seed "$s" --total_timesteps "$steps" \
    --csv_path "results/ppo/${name}.csv.tmp" > "logs/ppo_${name}.log" 2>&1 \
    && mv "results/ppo/${name}.csv.tmp" "results/ppo/${name}.csv"; }
export -f run
{
for s in 0 1 2; do
  for hpo in none HyperController PB2; do
    [ -f "$OUT/bike_${hpo}_seed${s}.csv" ] || echo "bike bike $hpo $s 480000"
    for env in PongNoFrameskip-v4 EnduroNoFrameskip-v4; do
      [ -f "$OUT/${env}_${hpo}_seed${s}.csv" ] || echo "atari $env $hpo $s 250000"
    done
  done
done
} | xargs -P "${PARALLEL:-9}" -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
