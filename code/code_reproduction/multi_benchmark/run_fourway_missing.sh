#!/usr/bin/env bash
# seeds 2..10 of analytical / learnable / RNN gating (seed 1 already stored), one process per (method, seed)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
run() { m=$1; s=$2; export CUDA_VISIBLE_DEVICES=""; python run_fourway_missing.py --modality $m --seed $s > logs/fourway_${m}_seed${s}.log 2>&1; }
export -f run
for s in 2 3 4 5 6 7 8 9 10; do for m in analytical learnable rnn; do echo "$m $s"; done; done | xargs -P 14 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
