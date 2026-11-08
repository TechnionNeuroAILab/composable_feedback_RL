#!/usr/bin/env bash
# More SEARL DQN seeds with the UNCHANGED settings (pop 4): CartPole-v1 seeds 10-19 (1M frames) and Hopper-v4 seeds 3-9
# (480k frames), pooled with the existing seeds in make_multi_figure.py (it globs seed*).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
mkdir -p logs_extra
run() { env=$1; frames=$2; s=$3; out=$4
  (cd ../searl && python searl_discrete.py --task gym --env "$env" --seed "$s" --pop 4 --frames "$frames" \
      --out "../multi_benchmark/$out/seed$s") > "logs_extra/searl_more_${env}_seed$s.log" 2>&1; }
export -f run
{
for s in $(seq 10 19); do [ -f results_cartpole/searl/seed$s/history.csv ] || echo "CartPole-v1 1000000 $s results_cartpole/searl"; done
for s in $(seq 3 9);  do [ -f results_extra/Hopper-v4/searl/seed$s/history.csv ] || echo "Hopper-v4 480000 $s results_extra/Hopper-v4/searl"; done
} | xargs -P 17 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
