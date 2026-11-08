#!/usr/bin/env bash
# Unified discrete SEARL (searl_discrete.py) on Pong, Enduro (60k frames) and bike v2.1 (480k decisions), pop 4, 3 seeds.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=results_discrete; mkdir -p $OUT logs_discrete
run() { task=$1; env=$2; s=$3; frames=$4; name=$5
  export CUDA_VISIBLE_DEVICES=$(( RANDOM % 2 ))
  python searl_discrete.py --task "$task" --env "$env" --seed "$s" --pop 4 --frames "$frames" --out "results_discrete/${name}_seed${s}" \
    > "logs_discrete/${name}_seed${s}.log" 2>&1; }
export -f run
{ for s in 0 1 2; do
    [ -f $OUT/bike_seed${s}/history.csv ] || echo "bike bike $s 480000 bike"
    for env in PongNoFrameskip-v4 EnduroNoFrameskip-v4; do
      [ -f $OUT/${env}_seed${s}/history.csv ] || echo "atari $env $s 60000 $env"
    done
  done; } | xargs -P 9 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
