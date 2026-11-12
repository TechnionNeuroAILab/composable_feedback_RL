#!/bin/bash
# SEARL + ConfGate heads (lr gate) on bike_env v2.1, lanes 3/5/7/10, 8 solutions, pop 4, 480k steps, seeds 0-2:
# the same settings and default evaluation as run_searl_lanes.sh (plain SEARL). Heads: 2h = left/right groups, perobs = one per entry.
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"; mkdir -p results_lanes_confgate logs_extra
for H in 2h perobs; do for L in 3 5 7 10; do for s in 0 1 2; do echo "$H $L $s"; done; done; done | xargs -P 24 -L 1 bash -c '
  H=$0; L=$1; s=$2; o=results_lanes_confgate/$H/lanes$L/seed$s; f=""; [ $H = perobs ] && f="--heads per_obs"
  [ -f $o/history.csv ] || (cd ../searl && python searl_discrete_confgate.py --task bike --lanes $L --seed $s --pop 4 --frames 480000 $f --out ../multi_benchmark/$o) > logs_extra/searl_lanes_confgate_${H}_${L}_seed$s.log 2>&1'
echo ALL DONE
