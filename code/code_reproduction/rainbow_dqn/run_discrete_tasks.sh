#!/usr/bin/env bash
# Ported Rainbow on the vector tasks of multi_benchmark_discrete.png, with the same step budgets and seeds as the
# other methods (CartPole 1M x 10 seeds, FrozenLake 100k, HalfCheetah 500k, the rest 480k; 3 seeds).
# Rainbow settings are Kaixhin's, scaled to the budget (same for every task): replay every 2 steps, lr 2.5e-4, target update
# every 1000 steps, learn from step 2000, 100k replay, n=3, PER omega 0.5 / beta 0.4->1, 51 atoms, MLP trunk 128 + 512-unit
# dueling noisy streams. Per task only the reward scale and the C51 support [Vmin,Vmax] change, so that the support covers the
# discounted return (rewards are multiplied by the scale before learning; reported returns are the raw ones).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
mkdir -p results logs
run() { env=$1; steps=$2; s=$3; rs=$4; vmin=$5; vmax=$6; obs=$7
  slug=${env//\//_}; mkdir -p results/$slug
  python main.py --task gym --env "$env" --seed "$s" --T-max "$steps" --learn-start 2000 --target-update 1000 \
    --memory-capacity 100000 --replay-frequency 2 --learning-rate 2.5e-4 --reward-scale "$rs" --V-min "$vmin" --V-max "$vmax" \
    --obs-scale "$obs" --csv-path "results/$slug/seed$s.csv.tmp" > "logs/${slug}_seed$s.log" 2>&1 \
    && mv "results/$slug/seed$s.csv.tmp" "results/$slug/seed$s.csv"; }
export -f run
{
# env steps seeds reward_scale Vmin Vmax obs_scale
while read -r env steps nseeds rs vmin vmax obs; do
  slug=${env//\//_}
  for s in $(seq 0 $((nseeds - 1))); do
    [ -f "results/$slug/seed$s.csv" ] || echo "$env $steps $s $rs $vmin $vmax $obs"
  done
done <<'TASKS'
CartPole-v1 1000000 10 0.1 0 10 1
bike 480000 3 0.005 -10 10 1
Acrobot-v1 480000 3 0.1 -10 0 1
LunarLander-v2 480000 3 0.05 -10 10 1
HalfCheetah-v4 500000 3 0.05 -5 10 1
Hopper-v4 480000 3 0.03 -1 10 1
ALE/Freeway-ram-v5 480000 3 1 0 10 0.00392157
FrozenLake-v1 100000 3 1 0 1 1
TASKS
} | xargs -P "${PARALLEL:-24}" -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
