#!/bin/bash
# SEARL DQN (reproduction port) on bike_env v2.1 with a varying number of lanes; 8 solutions, pop 4, 480k steps.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate cleanrl_dqn_repro
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"; mkdir -p results_lanes logs_extra
for L in 3 5 7 10; do for s in 0 1 2; do echo "$L $s"; done; done | xargs -P 12 -L 1 bash -c '
  L=$0; s=$1; o=results_lanes/lanes$L/seed$s
  [ -f $o/history.csv ] || (cd ../searl && python searl_discrete.py --task bike --lanes $L --seed $s --pop 4 --frames 480000 --out ../multi_benchmark/$o) > logs_extra/searl_lanes${L}_seed$s.log 2>&1'
echo ALL DONE > run_searl_lanes.log
